import os

# 模型下载 (ModelScope 魔搭社区)
# Whisper 模型: modelscope download --model pengzhendong/faster-whisper-medium --save ~/.cache/modelscope/hub/models/pengzhendong/faster-whisper-medium
# VoxCPM 模型: modelscope download --model OpenBMB/VoxCPM2 --save ~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2

WHISPER_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/pengzhendong/faster-whisper-medium")
VOXCPM_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2")

WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "float16"
