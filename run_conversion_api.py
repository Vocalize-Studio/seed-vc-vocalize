import os
import torch
import soundfile as sf
from seed_svc_wrapper import SeedVCWrapper, Progress
from tqdm import tqdm

def main():
    # Initialize the SeedVCWrapper
    print("Initializing the Seed-VC wrapper...")
    # You can specify a device here, e.g., device="cuda" or device="cpu"
    # If None, it will automatically determine based on availability.
    vc_wrapper = SeedVCWrapper(device=None) 

    # Define input and conversion settings
    source_audio_path = "./examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav"
    reference_audio_path = "./examples/reference/azuma_0.wav"

    # Conversion parameters (matching convert_voice arguments)
    diffusion_steps = 50
    length_adjust = 1.0
    inference_cfg_rate = 0.7
    auto_f0_adjust = False
    pitch_shift = 1
    stream_output = False # Set to True if you want to stream output

    # Run the conversion
    print("Starting voice conversion (streaming)...")
    os.makedirs("./output", exist_ok=True)

    final_sr, final_wave = None, None
    
    pbar = tqdm(total=100, desc="Overall", unit="%")

    def on_progress(progress: 'Progress'):
        # p in [0,1]
        new_val = int(round(progress.pct * 100))
        if new_val > pbar.n:
            pbar.update(new_val - pbar.n)
        pbar.set_description(f"Overall ({progress.status})")
        if progress.eta_sec is not None:
            pbar.set_postfix_str(f"ETA: {int(progress.eta_sec)}s")
        else:
            pbar.set_postfix_str("")

    for mp3_bytes, maybe_full in vc_wrapper.convert_voice_stream(
        source=source_audio_path,
        target=reference_audio_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        progress_cb=on_progress,
        progress_heartbeat_s=0.1,
    ):
        # send mp3_bytes to client if needed
        if maybe_full is not None:
            final_sr, final_wave = maybe_full

    pbar.close()

    if final_wave is not None:
        out_path = "./output/converted_audio.wav"
        sf.write(out_path, final_wave.squeeze(-1), final_sr)
        print(f"Conversion complete! Output saved to {out_path}")
    else:
        print("Conversion complete! (streamed only)")


if __name__ == "__main__":
    main()
