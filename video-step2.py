import os
import sys
import argparse
import logging
import json
import re
from pathlib import Path
import numpy as np
import torch
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_srt(srt_path: str) -> list:
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")

    for block in content:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                index_line = lines[0].strip()
                time_line = lines[1].strip()
                text = "\n".join(lines[2:])

                index = index_line

                time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)', time_line)
                if not time_match:
                    continue

                start_time = parse_srt_time(time_match.group(1))
                end_time = parse_srt_time(time_match.group(2))

                role = None
                clean_text = text
                role_match = re.search(r'\[Role:\s*([^\]]+)\]', time_line)
                if not role_match:
                    role_match = re.search(r'\[Role:\s*([^\]]+)\]', text)
                if role_match:
                    role = role_match.group(1).strip()
                    clean_text = re.sub(r'\[Role:\s*[^\]]+\]', '', text).strip()

                segments.append({
                    "index": index,
                    "start": start_time,
                    "end": end_time,
                    "text": clean_text,
                    "role": role,
                    "duration": end_time - start_time
                })
            except (ValueError, IndexError):
                continue

    return segments

def parse_srt_time(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def find_reference_audio(reference_dir: str, role: str) -> str:
    role_path = os.path.join(reference_dir, role)
    if not os.path.exists(role_path):
        logger.warning(f"Reference directory not found for role '{role}': {role_path}")
        return None

    wav_files = sorted([f for f in os.listdir(role_path) if f.endswith('.wav')])
    if not wav_files:
        logger.warning(f"No reference audio files found for role '{role}'")
        return None

    ref_path = os.path.join(role_path, wav_files[0])
    logger.info(f"Using reference audio for role '{role}': {ref_path}")
    return ref_path

def init_voxcpm_model(model_path: str = None):
    import voxcpm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing VoxCPM model on device: {device}")

    if model_path is None:
        model_path = config.VOXCPM_MODEL_PATH

    model = voxcpm.VoxCPM.from_pretrained(model_path, optimize=True, load_denoiser=False)
    logger.info("VoxCPM model loaded successfully")
    return model, device

def generate_speech(
    model,
    text: str,
    reference_wav: str,
    device: str = "cuda",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> np.ndarray:
    generate_kwargs = dict(
        text=text,
        reference_wav_path=reference_wav,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=True,
        denoise=False,
    )

    wav = model.generate(**generate_kwargs)
    return wav

def pad_audio_to_target(audio: np.ndarray, target_length: int, sample_rate: int) -> np.ndarray:
    if len(audio) >= target_length:
        return audio[:target_length]
    padding = target_length - len(audio)
    return np.pad(audio, (0, padding), mode='constant')

def mix_audio(background: np.ndarray, foreground: np.ndarray, foreground_start: int) -> np.ndarray:
    result = background.copy()
    foreground_end = foreground_start + len(foreground)

    if foreground_end > len(result):
        foreground = foreground[:len(result) - foreground_start]
        foreground_end = len(result)

    result[foreground_start:foreground_end] += foreground[:foreground_end - foreground_start]
    return result

def generate_voiceovers(
    model,
    segments: list,
    reference_dir: str,
    original_audio: str,
    device: str,
    output_dir: str,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    save_individual: bool = True,
) -> tuple:
    import scipy.io.wavfile as wavfile
    sample_rate = model.tts_model.sample_rate

    individual_dir = Path(output_dir) / "individual"
    if save_individual:
        individual_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Individual audio files will be saved to: {individual_dir}")

    if original_audio and os.path.exists(original_audio):
        logger.info(f"Loading original audio for background: {original_audio}")
        _, original_wav = wavfile.read(original_audio)
        if len(original_wav.shape) > 1:
            original_wav = original_wav[:, 0]
        original_wav = original_wav.astype(np.float32)
        if original_wav.max() > 1.0:
            original_wav = original_wav / 32768.0
        total_duration = len(original_wav) / sample_rate
        logger.info(f"Original audio duration: {total_duration:.2f}s")
    else:
        max_end = max(seg["end"] for seg in segments if seg["text"].strip())
        total_duration = max_end + 1.0
        original_wav = np.zeros(int(total_duration * sample_rate), dtype=np.float32)
        logger.info(f"No original audio found, creating silence for {total_duration:.2f}s")

    generated_audio = np.zeros(len(original_wav), dtype=np.float32)

    for i, seg in enumerate(segments):
        if not seg["text"].strip():
            logger.info(f"[{i+1}] Skipping empty segment")
            continue

        role = seg.get("role")
        if not role:
            logger.warning(f"[{i+1}] No role specified for segment, skipping")
            continue

        ref_audio = find_reference_audio(reference_dir, role)
        if not ref_audio:
            logger.warning(f"[{i+1}] No reference audio for role '{role}', skipping")
            continue

        start_sample = int(seg["start"] * sample_rate)

        logger.info(f"[{i+1}] Generating: {seg['text'][:50]}... (role: {role})")

        try:
            wav = generate_speech(
                model,
                seg["text"],
                ref_audio,
                device,
                cfg_value,
                inference_timesteps
            )

            generated_audio = mix_audio(generated_audio, wav, start_sample)

            seg["generated"] = True
            seg["generated_length"] = len(wav) / sample_rate
            logger.info(f"[{i+1}] Generated {len(wav)/sample_rate:.2f}s audio")

            if save_individual:
                safe_role = role.replace("/", "_").replace("\\", "_")
                safe_text = re.sub(r'[\\/:*?"<>|]', '', seg["text"][:30])
                filename = f"seg{i+1:03d}_{safe_role}_{safe_text}.wav"
                filepath = individual_dir / filename
                wavfile.write(str(filepath), sample_rate, wav.astype(np.float32))
                seg["individual_file"] = str(filepath)
                logger.info(f"[{i+1}] Saved individual: {filepath}")

        except Exception as e:
            logger.error(f"[{i+1}] Generation failed: {e}")
            seg["generated"] = False

    return generated_audio, sample_rate

def main():
    parser = argparse.ArgumentParser(description="Generate English voiceover from SRT using VoxCPM2")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing english.srt")
    parser.add_argument("-s", "--srt", default="english.srt", help="SRT filename (default: english.srt)")
    parser.add_argument("-r", "--reference", default=None, help="Reference audio directory (contains role folders). Default: ../segments/segments/reference")
    parser.add_argument("-a", "--audio", default=None, help="Original audio file for background (optional)")
    parser.add_argument("-o", "--output", default="segments_english", help="Output directory for voiceover")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale (default: 2.0)")
    parser.add_argument("--steps", type=int, default=10, help="Inference timesteps (default: 10)")
    parser.add_argument("--model", type=str, default=None, help="VoxCPM model path (optional)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    srt_path = input_dir / args.srt

    if args.reference:
        reference_dir = Path(args.reference)
    else:
        reference_dir = input_dir / "segments" / "reference"

    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return

    if not reference_dir.exists():
        logger.warning(f"Reference directory not found: {reference_dir}")
        logger.info(f"Please check reference directory path or run video-step1.5.py first")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading SRT file: {srt_path}")
    segments = parse_srt(str(srt_path))
    logger.info(f"Found {len(segments)} subtitle entries")

    roles = set(seg.get("role") for seg in segments if seg.get("role"))
    logger.info(f"Found {len(roles)} roles: {', '.join(roles)}")

    model, device = init_voxcpm_model(args.model)

    logger.info("\n" + "="*50)
    logger.info("Starting voiceover generation...")
    logger.info("="*50 + "\n")

    generated_audio, sample_rate = generate_voiceovers(
        model,
        segments,
        str(reference_dir),
        args.audio,
        device,
        str(output_dir),
        args.cfg,
        args.steps
    )

    output_wav = output_dir / "voiceover.wav"
    import scipy.io.wavfile as wavfile
    wavfile.write(str(output_wav), sample_rate, generated_audio.astype(np.float32))
    logger.info(f"Voiceover saved: {output_wav}")

    metadata = {
        "srt_path": str(srt_path),
        "reference_dir": str(reference_dir),
        "original_audio": args.audio,
        "total_segments": len(segments),
        "segments": segments,
        "sample_rate": sample_rate,
        "duration": len(generated_audio) / sample_rate,
        "cfg": args.cfg,
        "inference_steps": args.steps
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "="*50)
    logger.info("Complete!")
    logger.info(f"  - Voiceover: {output_wav}")
    logger.info(f"  - Duration: {len(generated_audio)/sample_rate:.2f}s")
    logger.info(f"  - Metadata: {metadata_path}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
