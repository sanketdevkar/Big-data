# ============================================================
# LOCAL LAUNCHER — Vision Model Training (PC 1)
# ============================================================
# Usage:
#   1. Open terminal in VS Code
#   2. venv\Scripts\activate
#   3. python run_vision_training.py
# ============================================================

import os, sys, types, torch

print("="*60)
print("  Deepfake Vision Training — Local PC Launcher")
print("="*60)

# ── 1. Verify GPU ─────────────────────────────────────────────
if not torch.cuda.is_available():
    print("\n⚠️  WARNING: No CUDA GPU detected!")
    print("   Training on CPU will be very slow.")
    print("   Check CUDA installation: nvcc --version")
    resp = input("   Continue on CPU? (y/n): ")
    if resp.lower() != 'y':
        sys.exit(0)
else:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {vram:.1f} GB")

    # Auto-tune batch size based on VRAM
    if vram >= 20:
        BATCH_SIZE, GRAD_ACCUM = 32, 1
        print("   → A100/RTX 3090 mode (batch=32, no accum)")
    elif vram >= 10:
        BATCH_SIZE, GRAD_ACCUM = 16, 2
        print("   → RTX 3080 mode (batch=16, accum=2)")
    elif vram >= 8:
        BATCH_SIZE, GRAD_ACCUM = 8, 4
        print("   → RTX 3070 / 2080 mode (batch=8, accum=4)")
    else:
        BATCH_SIZE, GRAD_ACCUM = 4, 8
        print("   → GTX 1060/1080 mode (batch=4, accum=8)")

# ── 2. Setup local paths ──────────────────────────────────────
BASE = r"C:\deepfake_training"

CHECKPOINT_DIR = os.path.join(BASE, "models", "vision", "checkpoints")
BEST_MODEL_DIR = os.path.join(BASE, "models", "vision", "best_model")
LOG_DIR        = os.path.join(BASE, "logs",   "vision")
DATASET_ROOT   = os.path.join(BASE, "datasets", "vision", "organized")

for d in [CHECKPOINT_DIR, BEST_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(DATASET_ROOT):
    print(f"\n❌ Dataset not found: {DATASET_ROOT}")
    print("   Run: python organize_datasets.py --mode vision")
    sys.exit(1)

print(f"\n📁 Dataset : {DATASET_ROOT}")
print(f"📁 Models  : {BEST_MODEL_DIR}")
print(f"📁 Logs    : {LOG_DIR}")
print(f"📦 Batch   : {BATCH_SIZE}  |  Grad Accum: {GRAD_ACCUM}")

# ── 3. Stub out google.colab (not available locally) ─────────
colab_stub = types.ModuleType("google.colab")
colab_stub.drive = types.SimpleNamespace(mount=lambda p: print(f"[Local] Drive mount skipped"))
colab_stub.files = types.SimpleNamespace(upload=lambda: {})

google_stub = types.ModuleType("google")
google_stub.colab = colab_stub
sys.modules["google"]           = google_stub
sys.modules["google.colab"]     = colab_stub
sys.modules["google.colab.drive"]= types.SimpleNamespace(mount=lambda p: None)
sys.modules["google.colab.files"]= types.SimpleNamespace(upload=lambda: {})

# ── 4. Inject local overrides into the training script namespace ─
IS_A100        = (torch.cuda.get_device_properties(0).total_memory > 20e9
                  if torch.cuda.is_available() else False)
__builtins_local__ = {"CHECKPOINT_DIR": CHECKPOINT_DIR,
                       "BEST_MODEL_DIR": BEST_MODEL_DIR,
                       "LOG_DIR":        LOG_DIR,
                       "BATCH_SIZE":     BATCH_SIZE,
                       "GRAD_ACCUM":     GRAD_ACCUM,
                       "IS_A100":        IS_A100}

# Patch the training script: replace Colab-specific paths inline
script_path = os.path.join(BASE, "vision_model_training.py")
if not os.path.exists(script_path):
    print(f"\n❌ Script not found: {script_path}")
    print("   Copy vision_model_training.py to C:\\deepfake_training\\")
    sys.exit(1)

with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace Colab Drive paths with local paths
replacements = {
    "'/content/drive/MyDrive/deepfake_models/vision'": f"r'{BASE}\\models\\vision'",
    "/content/drive/MyDrive/deepfake_models/vision/checkpoints": CHECKPOINT_DIR.replace("\\","\\\\"),
    "/content/drive/MyDrive/deepfake_models/vision/best_model":  BEST_MODEL_DIR.replace("\\","\\\\"),
    "/content/drive/MyDrive/deepfake_models/vision/logs":        LOG_DIR.replace("\\","\\\\"),
    '"/dataset"':        f'r"{DATASET_ROOT}"',
    "'/dataset'":        f'r"{DATASET_ROOT}"',
    "Path(\"/dataset\")": f'Path(r"{DATASET_ROOT}")',
    "drive.mount('/content/drive')": "pass  # local: no drive mount",
}
for old, new in replacements.items():
    code = code.replace(old, new)

print("\n🚀 Starting vision model training …\n")

# Execute the (patched) training script
exec(compile(code, script_path, "exec"), {
    "__name__":        "__main__",
    "__file__":        script_path,
    "CHECKPOINT_DIR":  CHECKPOINT_DIR,
    "BEST_MODEL_DIR":  BEST_MODEL_DIR,
    "LOG_DIR":         LOG_DIR,
    "IS_A100":         IS_A100,
})
