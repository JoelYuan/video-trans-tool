import os
import logging
from faster_whisper import WhisperModel
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def test_whisper(audio_path: str, model_path: str, device: str = "cuda", compute_type: str = "float16"):
    logger.info(f"Testing Whisper with audio: {audio_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Device: {device}, Compute type: {compute_type}")

    try:
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        logger.info("Whisper model loaded successfully!")

        logger.info("Starting transcription...")
        segments, info = model.transcribe(
            audio_path,
            language="zh",
            beam_size=5,
            word_timestamps=True
        )

        logger.info(f"Detected language: {info.language} ({info.language_probability:.2f})")
        logger.info("Transcription result:")
        
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start_time = segment.start
            end_time = segment.end
            text = segment.text

            start_h = int(start_time // 3600)
            start_m = int((start_time % 3600) // 60)
            start_s = start_time % 60
            start_str = f"{start_h:02d}:{start_m:02d}:{start_s:05.2f}".replace(".", ",")

            end_h = int(end_time // 3600)
            end_m = int((end_time % 3600) // 60)
            end_s = end_time % 60
            end_str = f"{end_h:02d}:{end_m:02d}:{end_s:05.2f}".replace(".", ",")

            srt_content += f"{i}\n"
            srt_content += f"{start_str} --> {end_str}\n"
            srt_content += f"{text}\n\n"

            logger.info(f"[{i}] {start_str} --> {end_str}: {text}")

        output_srt = "test_output.srt"
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)
        logger.info(f"SRT file saved: {output_srt}")

        return True

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    audio_path = "test.wav"
    model_path = config.WHISPER_MODEL_PATH
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        exit(1)

    logger.info("=" * 50)
    logger.info("Testing Whisper ASR with local model")
    logger.info("=" * 50)

    success = test_whisper(audio_path, model_path, device="cuda", compute_type="float16")
    
    if success:
        logger.info("=" * 50)
        logger.info("Test completed successfully!")
        logger.info("=" * 50)
    else:
        logger.info("=" * 50)
        logger.info("Test failed!")
        logger.info("=" * 50)