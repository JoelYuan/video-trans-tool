import os
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple():
    import torch
    import voxcpm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model_path = config.VOXCPM_MODEL_PATH
    logger.info(f"Loading VoxCPM model from: {model_path}")

    model = voxcpm.VoxCPM.from_pretrained(
        model_path,
        optimize=True,
        load_denoiser=False,
    )
    logger.info("Model loaded successfully")

    ref_audio = "抖音女播音.wav"
    text = "She is the sacred symbol of the kingdom."

    logger.info(f"Reference: {ref_audio}")
    logger.info(f"Text: {text}")

    if not os.path.exists(ref_audio):
        logger.error(f"Reference audio not found: {ref_audio}")
        return

    logger.info("Generating speech...")

    wav = model.generate(
        text=text,
        reference_wav_path=ref_audio,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=False,
    )

    output_path = "output_english/test_segment.wav"
    wavfile.write(output_path, model.tts_model.sample_rate, wav)
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Duration: {len(wav)/model.tts_model.sample_rate:.2f}s")


if __name__ == "__main__":
    test_simple()