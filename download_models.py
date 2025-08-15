"""
This script pre-downloads all the necessary model weights for the Seed-VC pipeline.
Running this script will ensure that all models are cached locally, so you don't
have to wait for them to download when you run the main application for the first time.
"""

import os

from hf_utils import load_custom_model_from_hf
from modules.bigvgan import bigvgan
from transformers import AutoFeatureExtractor, WhisperModel

def download_models():
    """Downloads all the models required for the Seed-VC pipeline."""
    print("Starting model download...")

    # Set the cache directory
    os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

    # Models to download using load_custom_model_from_hf
    custom_models = [
        (
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        ),
        ("funasr/campplus", "campplus_cn_common.bin", None),
        ("lj1995/VoiceConversionWebUI", "rmvpe.pt", None),
    ]

    for repo_id, model_filename, config_filename in custom_models:
        print(f"Downloading {model_filename} from {repo_id}...")
        load_custom_model_from_hf(repo_id, model_filename, config_filename)
        print(f"Downloaded {model_filename}.")

    # Models to download using from_pretrained
    pretrained_models = [
        "nvidia/bigvgan_v2_44khz_128band_512x",
        "openai/whisper-small",
    ]

    for model_name in pretrained_models:
        print(f"Downloading {model_name}...")
        if "bigvgan" in model_name:
            bigvgan.BigVGAN.from_pretrained(model_name)
        elif "whisper" in model_name:
            WhisperModel.from_pretrained(model_name)
            AutoFeatureExtractor.from_pretrained(model_name)
        print(f"Downloaded {model_name}.")

    print("\nAll models have been downloaded successfully!")


if __name__ == "__main__":
    download_models()
