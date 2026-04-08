# ============================================================
# LOCAL LAUNCHER — Audio Model Training (PC 2)
# ============================================================
# Usage:
#   1. Open terminal in VS Code
#   2. venv\Scripts\activate
#   3. python run_audio_training.py
# ============================================================

import os, sys, types, torch

print("="*60)
print("  Deepfake Audio Training — Local PC Launcher")
print("="*60)

# ── 1. Verify GPU ─────────────────────────────────────────────
if not torch.cuda.is_available():
    print("\n⚠️  WARNING: No CUDA GPU detected!")
    resp = input("   Continue on CPU? (y/n): ")
    if resp.lower() != 'y':
        sys.exit(0)
    BATCH_SIZE, GRAD_ACCUM = 2, 16
else:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {vram:.1f} GB")

    # Audio is heavier than vision — use smaller batches
    if vram >= 20:
        BATCH_SIZE, GRAD_ACCUM = 16, 2
        print("   → A100/RTX 3090 mode (batch=16, accum=2)")
    elif vram >= 10:
        BATCH_SIZE, GRAD_ACCUM = 8,  4
        print("   → RTX 3080 mode (batch=8, accum=4)")
    elif vram >= 8:
        BATCH_SIZE, GRAD_ACCUM = 4,  8
        print("   → RTX 3070 / 2080 mode (batch=4, accum=8)")
    else:
        BATCH_SIZE, GRAD_ACCUM = 2, 16
        print("   → GTX 1060/1080 mode (batch=2, accum=16)")

# ── 2. Setup local paths ──────────────────────────────────────
BASE = r"C:\deepfake_training"

CHECKPOINT_DIR = os.path.join(BASE, "models", "audio", "checkpoints")
BEST_MODEL_DIR = os.path.join(BASE, "models", "audio", "best_model")
LOG_DIR        = os.path.join(BASE, "logs",   "audio")
DATASET_ROOT   = os.path.join(BASE, "datasets", "audio", "organized")

for d in [CHECKPOINT_DIR, BEST_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(DATASET_ROOT):
    print(f"\n❌ Dataset not found: {DATASET_ROOT}")
    print("   Run: python organize_datasets.py --mode audio")
    sys.exit(1)

print(f"\n📁 Dataset : {DATASET_ROOT}")
print(f"📁 Models  : {BEST_MODEL_DIR}")
print(f"📁 Logs    : {LOG_DIR}")
print(f"📦 Batch   : {BATCH_SIZE}  |  Grad Accum: {GRAD_ACCUM}")

# Verify ffmpeg available (needed for FakeAVCeleb audio extraction)
import shutil
if shutil.which("ffmpeg") is None:
    print("\n⚠️  ffmpeg not found in PATH. FakeAVCeleb extraction will be skipped.")
    print("   Install from: https://github.com/BtbN/FFmpeg-Builds/releases")
else:
    print("✅ ffmpeg found")

# ── 3. Stub out google.colab ──────────────────────────────────
colab_stub = types.ModuleType("google.colab")
colab_stub.drive = types.SimpleNamespace(mount=lambda p: print("[Local] Drive mount skipped"))
colab_stub.files = types.SimpleNamespace(upload=lambda: {})
google_stub = types.ModuleType("google")
google_stub.colab = colab_stub
sys.modules["google"]            = google_stub
sys.modules["google.colab"]      = colab_stub
sys.modules["google.colab.drive"]= types.SimpleNamespace(mount=lambda p: None)
sys.modules["google.colab.files"]= types.SimpleNamespace(upload=lambda: {})

IS_A100 = (torch.cuda.get_device_properties(0).total_memory > 20e9
           if torch.cuda.is_available() else False)

# ── 4. Patch and execute training script ──────────────────────
script_path = os.path.join(BASE, "audio_model_training.py")
if not os.path.exists(script_path):
    print(f"\n❌ Script not found: {script_path}")
    print("   Copy audio_model_training.py to C:\\deepfake_training\\")
    sys.exit(1)

with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

replacements = {
    "'/content/drive/MyDrive/deepfake_models/audio'": f"r'{BASE}\\models\\audio'",
    "/content/drive/MyDrive/deepfake_models/audio/checkpoints": CHECKPOINT_DIR.replace("\\","\\\\"),
    "/content/drive/MyDrive/deepfake_models/audio/best_model":  BEST_MODEL_DIR.replace("\\","\\\\"),
    "/content/drive/MyDrive/deepfake_models/audio/logs":        LOG_DIR.replace("\\","\\\\"),
    '"/audio_dataset"':        f'r"{DATASET_ROOT}"',
    "'/audio_dataset'":        f'r"{DATASET_ROOT}"',
    "Path(\"/audio_dataset\")": f'Path(r"{DATASET_ROOT}")',
    "drive.mount('/content/drive')": "pass  # local: no drive mount",
    # ASVspoof paths
    "'/content/drive/MyDrive/datasets/ASVspoof2019_LA.zip'":
        f"r'{BASE}\\datasets\\audio\\ASVspoof2019_LA.zip'",
    "'/content/drive/MyDrive/datasets/ASVspoof2021_LA.zip'":
        f"r'{BASE}\\datasets\\audio\\ASVspoof2021_LA.zip'",
    # Temp audio extraction paths
    '"/content/tmp_': f'r"{BASE}\\tmp_',
}
for old, new in replacements.items():
    code = code.replace(old, new)

print("\n🚀 Starting audio model training …\n")

exec(compile(code, script_path, "exec"), {
    "__name__":       "__main__",
    "__file__":       script_path,
    "CHECKPOINT_DIR": CHECKPOINT_DIR,
    "BEST_MODEL_DIR": BEST_MODEL_DIR,
    "LOG_DIR":        LOG_DIR,
    "IS_A100":        IS_A100,
})
