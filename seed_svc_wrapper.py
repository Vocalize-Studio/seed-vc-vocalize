import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
import yaml
from modules.commons import build_model, load_checkpoint, recursive_munch
from hf_utils import load_custom_model_from_hf
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.audio import mel_spectrogram
from modules.rmvpe import RMVPE
from transformers import AutoFeatureExtractor, WhisperModel

from dataclasses import dataclass
import time
from typing import Optional, Callable, Dict, Any

@dataclass
class Progress:
    # Normalized progress in [0, 1]
    pct: float
    # Short status for UI (e.g., "Preparing features", "Converting 3/12", "Finalizing")
    status: str
    # Seconds remaining (smoothed); None when unknown
    eta_sec: Optional[float] = None
    # Optional details for power users / logs
    meta: Optional[Dict[str, Any]] = None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class SeedVCWrapper:
    def __init__(self, device=None):
        """
        Initialize the Seed-VC wrapper with all necessary models and configurations.
        
        Args:
            device: torch device to use. If None, will be automatically determined.
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Load DiT model and configuration
        self._load_dit_model()
        
        # Load additional modules
        self._load_additional_modules()
        
        # Set streaming parameters
        self.overlap_frame_len = 16
        self.bitrate = "320k"
        
    def _load_dit_model(self):
        """Load the DiT model for voice conversion."""
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
        )
        config = yaml.safe_load(open(dit_config_path, 'r'))
        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, stage='DiT')
        self.hop_length = config['preprocess_params']['spect_params']['hop_length']
        self.sr = config['preprocess_params']['sr']
        
        # Load checkpoints
        self.model, _, _, _ = load_checkpoint(
            self.model, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Set up mel spectrogram function
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        # Load whisper model
        whisper_name = self.model_params.speech_tokenizer.name
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
        def semantic_fn(waves_16k):
            ori_inputs = self.whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   sampling_rate=16000)
            ori_input_features = self.whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(self.device)
            with torch.no_grad():
                ori_outputs = self.whisper_model.encoder(
                    ori_input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
        self.semantic_fn = semantic_fn
        
    def _load_additional_modules(self):
        """Load additional modules like CAMPPlus, BigVGAN, and RMVPE."""
        # Load CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)
        
        self.vocoder_fn = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
        self.vocoder_fn.remove_weight_norm()
        self.vocoder_fn = self.vocoder_fn.eval().to(self.device)
        
        # Load RMVPE for F0 extraction
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)

        
        
    @staticmethod
    def adjust_f0_semitones(f0_sequence, n_semitones):
        """Adjust F0 values by a number of semitones."""
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor
    
    @staticmethod
    def crossfade(chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2
    
    

    def _process_whisper_features(self, audio_16k, is_source=True):
        """Process audio through Whisper model to extract features."""
        if audio_16k.size(-1) <= 16000 * 30:
            # If audio is short enough, process in one go
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, :audio_16k.size(-1) // 320 + 1]
        else:
            # Process long audio in chunks
            overlapping_time = 5  # 5 seconds
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = audio_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer, 
                        audio_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features, attention_mask=inputs.attention_mask
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, :chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            features = torch.cat(features_list, dim=1)
        
        return features
    
    @torch.inference_mode()
    def convert_voice(self, source, target, diffusion_steps=10, length_adjust=1.0,
                    inference_cfg_rate=0.7, auto_f0_adjust=True,
                    pitch_shift=0, stream_output=False):
        """
        Convert both timbre and voice from source to target.
        Returns:
            If stream_output is True: list[(mp3_bytes, (sr, chunk_wave_np))]
            Else: np.ndarray shape [N, 1] of the full waveform.
        """
        import torch.nn.functional as F

        inference_module = self.model
        mel_fn = self.to_mel

        # ---------- tiny helpers ----------
        def _device_type():
            return getattr(self.device, "type", str(self.device))

        def _amp_dtype():
            dt = _device_type()
            # float16 for CUDA/MPS, bfloat16 for CPU (if fp16-ish path is desired),
            # but we’ll still request fp16 autocast only when it helps.
            return torch.float16 if dt in ("cuda", "mps") else torch.bfloat16

        def _semantic_chunked(waves_16k, chunk_s=30, overlap_s=5):
            """Chunked semantic features to reduce memory for long inputs; stitches with a small skip."""
            total = waves_16k.size(-1)
            if total <= 16000 * chunk_s:
                return self.semantic_fn(waves_16k)
            S_list, buf, t = [], None, 0
            step = 16000 * (chunk_s - overlap_s)
            while t < total:
                if buf is None:
                    chunk = waves_16k[:, t : t + 16000 * chunk_s]
                    S = self.semantic_fn(chunk)
                    S_list.append(S)
                    buf = chunk[:, -16000 * overlap_s:]
                    t += 16000 * chunk_s
                else:
                    nxt = torch.cat([buf, waves_16k[:, t : t + step]], dim=-1)
                    S = self.semantic_fn(nxt)
                    # Heuristic skip of first 'overlap' portion in feature frames (~50 frames/s at 16k)
                    S_list.append(S[:, 50 * overlap_s :])
                    buf = nxt[:, -16000 * overlap_s:]
                    t += step
            return torch.cat(S_list, dim=1)

        # ---------- load audio (mono) ----------
        source_audio_np, _ = librosa.load(source, sr=self.sr, mono=True)
        ref_audio_np, _    = librosa.load(target, sr=self.sr, mono=True)

        # Keep reference short (≤ 25 s) so prompt won't eat the whole context
        ref_audio_np = ref_audio_np[: self.sr * 25]

        # to tensors on device
        source_audio = torch.tensor(source_audio_np).unsqueeze(0).float().to(self.device)
        ref_audio    = torch.tensor(ref_audio_np).unsqueeze(0).float().to(self.device)

        # ---------- compute mels BEFORE sizing (fixes mel2-before-assign bug) ----------
        mel  = mel_fn(source_audio.float())
        mel2 = mel_fn(ref_audio.float())

        # ---------- (re)derive window params once; or compute if not set ----------
        if not hasattr(self, "max_context_window"):
            self.max_context_window = self.sr // self.hop_length * 30  # ≈30 s in mel frames
        if not hasattr(self, "overlap_wave_len"):
            self.overlap_wave_len = self.overlap_frame_len * self.hop_length  # samples
        max_context_window = self.max_context_window
        overlap_wave_len   = self.overlap_wave_len

        # leave some space (≥ 4×overlap frames) for source after the prompt; trim prompt tail if needed
        min_src_space = max(1, 4 * self.overlap_frame_len)
        available_for_prompt = max(1, max_context_window - min_src_space)
        if mel2.size(2) > available_for_prompt:
            mel2 = mel2[:, :, -available_for_prompt:]

        max_source_window = max(1, max_context_window - mel2.size(2))

        # ---------- resample to 16 kHz on CPU for stability; move back to device ----------
        ref_waves_16k_cpu       = torchaudio.functional.resample(ref_audio.cpu(),    self.sr, 16000)
        converted_waves_16k_cpu = torchaudio.functional.resample(source_audio.cpu(), self.sr, 16000)
        ref_waves_16k       = ref_waves_16k_cpu.to(self.device)
        converted_waves_16k = converted_waves_16k_cpu.to(self.device)

        # ---------- semantic features (chunked for long source) ----------
        S_alt = _semantic_chunked(converted_waves_16k)  # source (can be long)
        S_ori = self.semantic_fn(ref_waves_16k)         # prompt (short; single pass)

        # ---------- target lengths AFTER any prompt trimming ----------
        target_lengths  = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # ---------- style features (kaldi fbank expects CPU) ----------
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2  = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0).to(self.device))

        # ---------- F0 (audio already 16 kHz) ----------
        F0_ori_np = self.rmvpe.infer_from_audio(ref_waves_16k[0],      thred=0.03)
        F0_alt_np = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        # device-aware casting
        if _device_type() in ("mps", "cuda"):
            F0_ori = torch.from_numpy(F0_ori_np).float().to(self.device)[None]
            F0_alt = torch.from_numpy(F0_alt_np).float().to(self.device)[None]
        else:
            F0_ori = torch.from_numpy(F0_ori_np).to(self.device)[None]
            F0_alt = torch.from_numpy(F0_alt_np).to(self.device)[None]

        # robust F0 normalization (skip if unvoiced)
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        if voiced_F0_ori.numel() == 0 or voiced_F0_alt.numel() == 0:
            shifted_f0_alt = F0_alt.clone()
        else:
            log_f0_alt = torch.log(F0_alt + 1e-5)
            median_log_f0_ori = torch.median(torch.log(voiced_F0_ori + 1e-5))
            median_log_f0_alt = torch.median(torch.log(voiced_F0_alt + 1e-5))
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                mask = F0_alt > 1
                shifted_log_f0_alt[mask] = log_f0_alt[mask] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)

        if pitch_shift != 0:
            mask = F0_alt > 1
            shifted_f0_alt[mask] = self.adjust_f0_semitones(shifted_f0_alt[mask], pitch_shift)

        # ---------- length regulation ----------
        cond, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _, _, _ = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )

        # (Optional) match F0 to cond length if later modules expect exact alignment
        # interpolated_shifted_f0_alt = F.interpolate(
        #     shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode="nearest"
        # ).squeeze(1)

        # ---------- chunked inference with safe overlap-add ----------
        current_sr = self.sr
        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None
        streamed_outputs = []

        # autocast config
        _dt = _device_type()
        _dtype = _amp_dtype()

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(device_type=_dt, dtype=_dtype):
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                # drop prompt portion from target mel
                vc_target = vc_target[:, :, mel2.size(-1):]

            vc_wave = self.vocoder_fn(vc_target.float())[0]  # [1, T]

            # Clamp overlap to avoid underflow on tiny chunks
            ov = min(overlap_wave_len, vc_wave.shape[-1])

            if processed_frames == 0:
                if is_last_chunk:
                    current_chunk_output_wave = vc_wave[0].cpu().numpy()
                else:
                    current_chunk_output_wave = vc_wave[0, :-ov].cpu().numpy()
                    previous_chunk = vc_wave[0, -ov:]
            elif is_last_chunk:
                current_chunk_output_wave = self.crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), ov
                )
            else:
                current_chunk_output_wave = self.crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0, :-ov].cpu().numpy(), ov
                )
                previous_chunk = vc_wave[0, -ov:]

            generated_wave_chunks.append(current_chunk_output_wave)
            step = vc_target.size(2) - self.overlap_frame_len if not is_last_chunk else vc_target.size(2)
            processed_frames += step

            if stream_output:
                # clamp to [-1,1] before int16 convert for stability
                wav = np.clip(current_chunk_output_wave, -1.0, 1.0)
                output_wave_int16 = (wav * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(),
                    frame_rate=current_sr,
                    sample_width=output_wave_int16.dtype.itemsize,
                    channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()
                streamed_outputs.append((mp3_bytes, (current_sr, wav.reshape(-1, 1))))

        if stream_output:
            return streamed_outputs
        else:
            return np.concatenate(generated_wave_chunks).reshape(-1, 1)

    @torch.inference_mode()
    def convert_voice_stream(
        self,
        source,
        target,
        diffusion_steps=10,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        auto_f0_adjust=True,
        pitch_shift=0,
        progress_cb: Optional[Callable[[Progress], None]] = None,
        progress_weights: Optional[Dict[str, float]] = None,
        # NEW: control heartbeat cadence (seconds) to keep UI responsive
        progress_heartbeat_s: float = 0.5,
    ):
        """
        Streaming conversion yielding (mp3_bytes, maybe_full, processed_frames, total_frames)
        with rich progress updates via progress_cb.
        """
        pw = {"prep": 0.15, "infer": 0.80, "final": 0.05}
        if progress_weights:
            # normalize but keep simple merge; we’ll clamp later via pct
            pw.update(progress_weights)

        # ---- progress helpers ------------------------------------------------
        t0 = time.time()
        last_beat = 0.0
        # Simple EMA for throughput and ETA
        ema_alpha = 0.15
        ema_fps = None
        total_frames = None
        processed_frames = 0

        def _emit(pct: float, status: str, meta: Optional[Dict[str, Any]] = None):
            nonlocal last_beat, ema_fps
            now = time.time()
            # Update EMA FPS if we can
            if total_frames and processed_frames > 0:
                dt = max(now - t0, 1e-6)
                inst_fps = processed_frames / dt
                ema_fps = inst_fps if ema_fps is None else (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps

            # Compute ETA when possible
            eta_sec = None
            if ema_fps and total_frames:
                remain = max(total_frames - processed_frames, 0)
                if ema_fps > 1e-6:
                    eta_sec = remain / ema_fps

            prog = Progress(pct=_clamp01(pct), status=status, eta_sec=eta_sec, meta=meta or {})

            # Heartbeat throttle for very tight loops
            if progress_cb: # Temporarily remove throttle for debugging
                last_beat = now
                try:
                    progress_cb(prog)
                except Exception:
                    # Never break conversion on UI-side callback errors
                    pass

        def _phase(pct_from: float, pct_to: float, local: float) -> float:
            # map local [0..1] within [pct_from..pct_to]
            return pct_from + (pct_to - pct_from) * _clamp01(local)

        # ---- start -----------------------------------------------------------
        _emit(0.01, "Starting…")

        if progress_cb:
            progress_cb(Progress(pct=0.03, status="Loading models…", eta_sec=None, meta={}))

        # PREP PHASE ───────────────────────────────────────────────────────────
        prep_start = time.time()
        _emit(_phase(0.00, pw["prep"], 0.05), "Loading audio")

        import torch.nn.functional as F
        inference_module = self.model
        mel_fn = self.to_mel

        def _device_type():
            return getattr(self.device, "type", str(self.device))
        def _amp_dtype():
            dt = _device_type()
            return torch.float16 if dt in ("cuda", "mps") else torch.bfloat16
        def _semantic_chunked(waves_16k, chunk_s=30, overlap_s=5):
            total = waves_16k.size(-1)
            if total <= 16000 * chunk_s:
                return self.semantic_fn(waves_16k)
            S_list, buf, t = [], None, 0
            step = 16000 * (chunk_s - overlap_s)
            while t < total:
                if buf is None:
                    chunk = waves_16k[:, t : t + 16000 * chunk_s]
                    S = self.semantic_fn(chunk)
                    S_list.append(S)
                    buf = chunk[:, -16000 * overlap_s:]
                    t += 16000 * chunk_s
                else:
                    nxt = torch.cat([buf, waves_16k[:, t : t + step]], dim=-1)
                    S = self.semantic_fn(nxt)
                    S_list.append(S[:, 50 * overlap_s :])
                    buf = nxt[:, -16000 * overlap_s:]
                    t += step
            return torch.cat(S_list, dim=1)

        # load audio (model SR)
        source_audio_np, _ = librosa.load(source, sr=self.sr, mono=True)
        ref_audio_np, _    = librosa.load(target, sr=self.sr, mono=True)
        ref_audio_np = ref_audio_np[: self.sr * 25]
        source_audio = torch.tensor(source_audio_np).unsqueeze(0).float().to(self.device)
        ref_audio    = torch.tensor(ref_audio_np).unsqueeze(0).float().to(self.device)

        # Make sure to pass sampling_rate where relevant to silence warnings
        # e.g., feature_extractor(audio, sampling_rate=16000)

        _emit(_phase(0.00, pw["prep"], 0.25), "Extracting features (Whisper/semantics)")
        # features/encoders…

        # mels
        mel  = mel_fn(source_audio.float())
        mel2 = mel_fn(ref_audio.float())

        # windows & overlap
        if not hasattr(self, "max_context_window"):
            self.max_context_window = self.sr // self.hop_length * 30
        if not hasattr(self, "overlap_wave_len"):
            self.overlap_wave_len = self.overlap_frame_len * self.hop_length
        max_context_window = self.max_context_window
        overlap_wave_len   = self.overlap_wave_len

        # keep prompt short enough
        min_src_space = max(1, 4 * self.overlap_frame_len)
        available_for_prompt = max(1, max_context_window - min_src_space)
        if mel2.size(2) > available_for_prompt:
            mel2 = mel2[:, :, -available_for_prompt:]
        max_source_window = max(1, max_context_window - mel2.size(2))

        # 16k resample for semantic/style/F0
        ref_waves_16k       = torchaudio.functional.resample(ref_audio.cpu(),    self.sr, 16000).to(self.device)
        converted_waves_16k = torchaudio.functional.resample(source_audio.cpu(), self.sr, 16000).to(self.device)

        # semantic
        S_alt = _semantic_chunked(converted_waves_16k)
        S_ori = self.semantic_fn(ref_waves_16k)

        # lengths
        target_lengths  = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        _emit(_phase(0.00, pw["prep"], 0.55), "Estimating F0 / style")
        # F0/style…

        # style (kaldi fbank on CPU)
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2  = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0).to(self.device))

        # F0 (already 16k)
        F0_ori_np = self.rmvpe.infer_from_audio(ref_waves_16k[0],      thred=0.03)
        F0_alt_np = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori_np).float().to(self.device)[None]
        F0_alt = torch.from_numpy(F0_alt_np).float().to(self.device)[None]

        # F0 normalize/shift
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        if voiced_F0_ori.numel() == 0 or voiced_F0_alt.numel() == 0:
            shifted_f0_alt = F0_alt.clone()
        else:
            log_f0_alt = torch.log(F0_alt + 1e-5)
            median_log_f0_ori = torch.median(torch.log(voiced_F0_ori + 1e-5))
            median_log_f0_alt = torch.median(torch.log(voiced_F0_alt + 1e-5))
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                mask = F0_alt > 1
                shifted_log_f0_alt[mask] = log_f0_alt[mask] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            mask = F0_alt > 1
            shifted_f0_alt[mask] = self.adjust_f0_semitones(shifted_f0_alt[mask], pitch_shift)

        _emit(_phase(0.00, pw["prep"], 0.80), "Length regulation")
        # length regulator outputs:
        # cond, prompt_condition = ...
        # IMPORTANT: define total_frames once cond is ready:
        # length regulation
        cond, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _, _, _ = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
        total_frames = int(cond.size(1))

        _emit(pw["prep"], "Preparation complete", meta={"total_frames": total_frames,
                                                        "diffusion_steps": diffusion_steps})

        prep_end = time.time()

        # INFER (STREAM) PHASE ────────────────────────────────────────────────
        infer_start = time.time()
        processed_frames = 0
        chunk_idx = 0
        est_chunks = max(total_frames // 2048, 1)  # heuristic if you use 2k frame chunks

        _dt = _device_type()
        _dtype = _amp_dtype()

        generated_wave_chunks = []

        # streaming overlap-add
        current_sr = self.sr

        while processed_frames < total_frames:
            chunk_idx += 1
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=_dt, dtype=_dtype):
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = self.vocoder_fn(vc_target.float())[0]  # [1, T]
            ov = min(self.overlap_wave_len, vc_wave.shape[-1])

            if processed_frames == 0:
                if is_last_chunk:
                    out = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(out)
                    wav16 = (np.clip(out, -1.0, 1.0) * 32768.0).astype(np.int16)
                    mp3 = AudioSegment(wav16.tobytes(), frame_rate=current_sr,
                                       sample_width=wav16.dtype.itemsize, channels=1
                                      ).export(format="mp3", bitrate=self.bitrate).read()
                    infer_local = processed_frames / max(total_frames, 1)
                    pct = _phase(pw["prep"], pw["prep"] + pw["infer"], infer_local)
                    _emit(pct,
                          f"Converting {chunk_idx}/{est_chunks}",
                          meta={"processed_frames": processed_frames,
                                "total_frames": total_frames,
                                "fps_ema": ema_fps})
                    yield mp3, (current_sr, np.concatenate(generated_wave_chunks).reshape(-1,1))
                    break
                out = vc_wave[0, :-ov].cpu().numpy()
                generated_wave_chunks.append(out)
                previous_chunk = vc_wave[0, -ov:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                wav16 = (np.clip(out, -1.0, 1.0) * 32768.0).astype(np.int16)
                mp3 = AudioSegment(wav16.tobytes(), frame_rate=current_sr,
                                   sample_width=wav16.dtype.itemsize, channels=1
                                  ).export(format="mp3", bitrate=self.bitrate).read()
                infer_local = processed_frames / max(total_frames, 1)
                pct = _phase(pw["prep"], pw["prep"] + pw["infer"], infer_local)
                _emit(pct,
                      f"Converting {chunk_idx}/{est_chunks}",
                      meta={"processed_frames": processed_frames,
                            "total_frames": total_frames,
                            "fps_ema": ema_fps})
                yield mp3, None
            elif is_last_chunk:
                out = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), ov)
                generated_wave_chunks.append(out)
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                wav16 = (np.clip(out, -1.0, 1.0) * 32768.0).astype(np.int16)
                mp3 = AudioSegment(wav16.tobytes(), frame_rate=current_sr,
                                   sample_width=wav16.dtype.itemsize, channels=1
                                  ).export(format="mp3", bitrate=self.bitrate).read()
                infer_local = processed_frames / max(total_frames, 1)
                pct = _phase(pw["prep"], pw["prep"] + pw["infer"], infer_local)
                _emit(pct,
                      f"Converting {chunk_idx}/{est_chunks}",
                      meta={"processed_frames": processed_frames,
                            "total_frames": total_frames,
                            "fps_ema": ema_fps})
                yield mp3, (current_sr, np.concatenate(generated_wave_chunks).reshape(-1,1))
                break
            else:
                out = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-ov].cpu().numpy(), ov)
                generated_wave_chunks.append(out)
                previous_chunk = vc_wave[0, -ov:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                wav16 = (np.clip(out, -1.0, 1.0) * 32768.0).astype(np.int16)
                mp3 = AudioSegment(wav16.tobytes(), frame_rate=current_sr,
                                   sample_width=wav16.dtype.itemsize, channels=1
                                  ).export(format="mp3", bitrate=self.bitrate).read()
                infer_local = processed_frames / max(total_frames, 1)
                pct = _phase(pw["prep"], pw["prep"] + pw["infer"], infer_local)
                _emit(pct,
                      f"Converting {chunk_idx}/{est_chunks}",
                      meta={"processed_frames": processed_frames,
                            "total_frames": total_frames,
                            "fps_ema": ema_fps})
                yield mp3, None

        infer_end = time.time()

        # FINAL PHASE ─────────────────────────────────────────────────────────
        final_start = time.time()
        _emit(1.0 - pw["final"] * 0.6, "Finalizing (stitch/encode)")
        # … assemble final waveform / file write …

        _emit(1.0 - pw["final"] * 0.2, "Writing output")
        # … write to disk or return final …

        _emit(1.0, "Done", meta={
            "t_prep_s": round(prep_end - prep_start, 3),
            "t_infer_s": round(infer_end - infer_start, 3),
            "t_final_s": round(time.time() - final_start, 3),
        })


