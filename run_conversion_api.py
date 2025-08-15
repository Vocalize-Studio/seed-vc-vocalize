import os
import torch
import soundfile as sf
from seed_svc_wrapper import SeedVCWrapper

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
    auto_f0_adjust = True
    pitch_shift = 1
    stream_output = False # Set to True if you want to stream output

    # Run the conversion
    print("Starting voice conversion...")
    # If stream_output is True, this will be a generator
    # For simplicity, I'm assuming stream_output is False for direct return
    output_audio_np = vc_wrapper.convert_voice(
        source=source_audio_path,
        target=reference_audio_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        stream_output=stream_output
    )

    # Save the output audio (if not streaming)
    if not stream_output:
        output_filename = "./output/converted_audio.wav"
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        # Assuming the output_audio_np is a numpy array and sr is available from the wrapper
        current_sr = vc_wrapper.sr
        print(f"Shape of output_audio_np: {output_audio_np.shape}")
        sf.write(output_filename, output_audio_np, current_sr)
        print(f"\nConversion complete! Output audio saved to: {output_filename}")
    else:
        print("\nConversion complete! Output was streamed.")


if __name__ == "__main__":
    main()
