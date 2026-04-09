# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISION MODEL — CLIP ViT-Large  |  ALL DATASETS  |  MAX ACCURACY        ║
# ║  Datasets : Celeb-DF v2 · FF++ · DFDC · DFD · WildDeepfake · ForgeryNet ║
# ║  Platform : Google Colab T4/A100 GPU                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import torch, subprocess, sys, os, json, math, time, copy, random, shutil
import urllib.request, zipfile, hashlib
from pathlib import Path
import gdown
import numpy as np
from PIL import Image
from tqdm import tqdm as _tqdm
from collections import defaultdict

def setup_colab_env():
    """Verify GPU, install dependencies and mount Drive."""
    print("🔬 Verifying environment...")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Runtime → Change Runtime Type → GPU (T4/A100)")

    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "numpy<2",  # Critical fix for OpenCV compatibility
        "transformers==4.40.0", "accelerate==0.29.3",
        "grad-cam==1.5.0", "opencv-python-headless==4.9.0.80",
        "Pillow==10.3.0", "scikit-learn==1.4.2", "matplotlib==3.8.4",
        "seaborn==0.13.2", "tqdm==4.66.2", "huggingface_hub==0.23.0",
        "gdown==5.1.0", "kaggle==1.6.11", "timm==0.9.16",
        "albumentations==1.4.3",
    ], check=True)
    
    from google.colab import drive
    drive.mount('/content/drive')
    for d in [CHECKPOINT_DIR, BEST_MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Setup complete. Drive: {DRIVE_BASE}")

DATASET_ROOT = Path("/dataset")
VAL_RATIO    = 0.15
DS_STATS     = defaultdict(lambda: {"real": 0, "fake": 0})

for split in ["train","val"]:
    for label in ["real","fake"]:
        (DATASET_ROOT/split/label).mkdir(parents=True, exist_ok=True)


def copy_to_dataset(src: Path, label: str, source_name: str, val_ratio=VAL_RATIO):
    """Copy a single image file to train or val split, tracking per-dataset stats."""
    split = "val" if random.random() < val_ratio else "train"
    # Use hash-based unique name to avoid collisions across datasets
    uid   = hashlib.md5(str(src).encode()).hexdigest()[:10]
    dst   = DATASET_ROOT / split / label / f"{source_name}_{uid}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
    DS_STATS[source_name][label] += 1


def extract_video_frames(video_path: Path, out_dir: Path, n_frames: int = 25):
    """Extract n uniformly sampled frames from a video."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release(); return
    idxs  = np.linspace(0, total-1, min(n_frames, total), dtype=int)
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame)
    cap.release()


# ── 1. Celeb-DF v2 ────────────────────────────────────────────────────────────
def download_celebdf():
    CELEBDF_ID  = "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
    CELEBDF_ZIP = "/content/Celeb-DF-v2.zip"
    CELEBDF_EXT = "/content/Celeb-DF-v2"
    try:
        if not os.path.exists(CELEBDF_EXT):
            print("  📥 Downloading Celeb-DF v2 …")
            gdown.download(id=CELEBDF_ID, output=CELEBDF_ZIP, quiet=False)
            with zipfile.ZipFile(CELEBDF_ZIP) as zf: zf.extractall("/content/")
        real_src = Path(CELEBDF_EXT)/"YouTube-real"
        fake_src = Path(CELEBDF_EXT)/"Celeb-synthesis"
        for src, label in [(real_src,"real"),(fake_src,"fake")]:
            for f in _tqdm(list(src.rglob("*.jpg"))+list(src.rglob("*.png")),
                           desc=f"  CelebDF {label}"):
                copy_to_dataset(f, label, "celebdf")
        print("  ✅ Celeb-DF v2 done")
    except Exception as e:
        print(f"  ⚠️  Celeb-DF failed: {e}")

# ── 2. FaceForensics++ c23 ────────────────────────────────────────────────────
def download_faceforensics():
    """
    Requires: Place faceforensics_download.py in /content/
    Obtain from: https://github.com/ondyari/FaceForensics
    Uncomment the subprocess calls below after placing the script.
    """
    FF_SCRIPT = "/content/faceforensics_download.py"
    FF_OUT    = Path("/content/FF++")
    if not os.path.exists(FF_SCRIPT):
        print("  ⚠️  FF++ download script not found — skipping")
        print("       → Get it from github.com/ondyari/FaceForensics")
        return
    METHODS_FAKE = ["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]
    METHODS_REAL = ["original_sequences/actors","original_sequences/youtube"]
    for m in METHODS_FAKE:
        subprocess.run(["python", FF_SCRIPT, str(FF_OUT), "-d", m,
                        "-c", "c23", "-t", "videos"], check=True)
    for m in METHODS_REAL:
        subprocess.run(["python", FF_SCRIPT, str(FF_OUT), "-d", m,
                        "-c", "raw", "-t", "videos"], check=True)
    # Extract frames
    for m in METHODS_FAKE:
        for vid in (FF_OUT/f"manipulated_sequences/{m}/c23/videos").glob("*.mp4"):
            extract_video_frames(vid, Path(f"/content/ff_frames/fake/{vid.stem}"))
    for vid in (FF_OUT/"original_sequences/youtube/raw/videos").glob("*.mp4"):
        extract_video_frames(vid, Path(f"/content/ff_frames/real/{vid.stem}"))
    # Copy frames
    for label in ["real","fake"]:
        for f in _tqdm(list(Path(f"/content/ff_frames/{label}").rglob("*.jpg")),
                       desc=f"  FF++ {label}"):
            copy_to_dataset(f, label, "ff_plus_plus")
    print("  ✅ FaceForensics++ done")

# ── 3. DFDC (Facebook Deepfake Detection Challenge) ───────────────────────────
def download_dfdc():
    """
    Requires Kaggle API credentials.
    Setup: Upload kaggle.json to /content/ OR set env vars.
    kaggle.json → {"username":"YOUR_USER","key":"YOUR_KEY"}
    """
    KAGGLE_JSON = "/content/kaggle.json"
    DFDC_OUT    = Path("/content/dfdc")

    if not os.path.exists(KAGGLE_JSON):
        print("  ⚠️  kaggle.json not found — skipping DFDC")
        print("       → Upload kaggle.json from kaggle.com/settings → API")
        return

    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy(KAGGLE_JSON, os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    DFDC_OUT.mkdir(exist_ok=True)
    print("  📥 Downloading DFDC (this may take a while — ~470 GB full / use part-0..4) …")
    # Download only parts 0–4 to keep storage manageable (~50 GB)
    for part in range(5):
        subprocess.run(["kaggle", "competitions", "download",
                        "deepfake-detection-challenge",
                        "-f", f"dfdc_train_part_{part}.zip",
                        "-p", str(DFDC_OUT)], check=True)
        zip_path = DFDC_OUT / f"dfdc_train_part_{part}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf: zf.extractall(DFDC_OUT)
            zip_path.unlink()

    # Process DFDC metadata JSON
    for meta_file in _tqdm(list(DFDC_OUT.rglob("metadata.json")),
                            desc="  DFDC metadata"):
        with open(meta_file) as f:
            meta = json.load(f)
        vid_dir = meta_file.parent
        for vid_name, info in meta.items():
            label  = "fake" if info["label"] == "FAKE" else "real"
            vid_fp = vid_dir / vid_name
            if not vid_fp.exists(): continue
            frames_dir = Path(f"/content/dfdc_frames/{label}/{vid_fp.stem}")
            extract_video_frames(vid_fp, frames_dir, n_frames=15)
            for fr in frames_dir.glob("*.jpg"):
                copy_to_dataset(fr, label, "dfdc")
    print("  ✅ DFDC done")

# ── 4. WildDeepfake ───────────────────────────────────────────────────────────
def download_wild_deepfake():
    """WildDeepfake — collected from internet, more realistic distribution."""
    WILD_ID  = "1bHt9Zp7IG1DJhpvD6J6xrW6z3hLDVqTm"   # update if changed
    WILD_ZIP = "/content/WildDeepfake.zip"
    WILD_EXT = "/content/WildDeepfake"
    try:
        if not os.path.exists(WILD_EXT):
            print("  📥 Downloading WildDeepfake …")
            gdown.download(id=WILD_ID, output=WILD_ZIP, quiet=False)
            with zipfile.ZipFile(WILD_ZIP) as zf: zf.extractall("/content/")
        for label in ["real","fake"]:
            src_dir = Path(WILD_EXT) / label
            if not src_dir.exists(): continue
            for f in _tqdm(list(src_dir.rglob("*.jpg"))+list(src_dir.rglob("*.png")),
                           desc=f"  WildDeepfake {label}"):
                copy_to_dataset(f, label, "wilddeepfake")
        print("  ✅ WildDeepfake done")
    except Exception as e:
        print(f"  ⚠️  WildDeepfake failed: {e}")

# ── 5. DFD — Google DeepFakeDetection ────────────────────────────────────────
def download_dfd():
    """Requires FF++ download script — same as FaceForensics++."""
    FF_SCRIPT = "/content/faceforensics_download.py"
    if not os.path.exists(FF_SCRIPT):
        print("  ⚠️  DFD skipped — FF++ script required"); return
    DFD_OUT = Path("/content/DFD")
    subprocess.run(["python", FF_SCRIPT, str(DFD_OUT),
                    "-d", "DeepFakeDetection", "-c", "c23", "-t", "videos"], check=True)
    for vid in _tqdm(list(DFD_OUT.rglob("*.mp4")), desc="  DFD frames"):
        label = "fake" if "manipulated" in str(vid) else "real"
        frames_dir = Path(f"/content/dfd_frames/{label}/{vid.stem}")
        extract_video_frames(vid, frames_dir)
        for fr in frames_dir.glob("*.jpg"):
            copy_to_dataset(fr, label, "dfd_google")
    print("  ✅ DFD (Google) done")

# ── 6. ForgeryNet ────────────────────────────────────────────────────────────
def download_forgerynet():
    """
    ForgeryNet requires manual download from:
    https://yinanhe.github.io/projects/forgerynet.html
    Upload to Drive at: /MyDrive/datasets/ForgeryNet.zip
    """
    FNET_ZIP = '/content/drive/MyDrive/datasets/ForgeryNet.zip'
    FNET_EXT = '/content/ForgeryNet'
    if not os.path.exists(FNET_ZIP):
        print("  ⚠️  ForgeryNet.zip not on Drive — skipping")
        print("       → Download from yinanhe.github.io/projects/forgerynet.html")
        return
    if not os.path.exists(FNET_EXT):
        with zipfile.ZipFile(FNET_ZIP) as zf: zf.extractall("/content/")
    for label in ["real","fake"]:
        src = Path(FNET_EXT)/label
        if not src.exists(): continue
        for f in _tqdm(list(src.rglob("*.jpg"))+list(src.rglob("*.png")),
                       desc=f"  ForgeryNet {label}"):
            copy_to_dataset(f, label, "forgerynet")
    print("  ✅ ForgeryNet done")

def download_all_datasets():
    """Download all datasets if they don't exist, skip on failure."""
    print("\n═══ Starting multi-dataset download ═══")
    try:
        download_celebdf()
    except Exception as e:
        print(f"  ⚠️ Celeb-DF download failed: {e}")
        
    try:
        download_faceforensics()
    except Exception as e:
        print(f"  ⚠️ FaceForensics++ download failed: {e}")
        
    try:
        download_dfdc()
    except Exception as e:
        print(f"  ⚠️ DFDC download failed: {e}")
        
    try:
        download_wild_deepfake()
    except Exception as e:
        print(f"  ⚠️ WildDeepfake download failed: {e}")
        
    try:
        download_dfd()
    except Exception as e:
        print(f"  ⚠️ DFD download failed: {e}")
        
    try:
        download_forgerynet()
    except Exception as e:
        print(f"  ⚠️ ForgeryNet download failed: {e}")

def get_dataset_summary():
    """Print the current dataset distribution."""
    total_imgs = sum(len(list((DATASET_ROOT/sp/lb).glob("*")))
                     for sp in ["train","val"] for lb in ["real","fake"])
    
    if total_imgs < 100:
        print("\n⚠️  No real data found — generating synthetic dataset for pipeline testing …")
        for split in ["train","val"]:
            for label in ["real","fake"]:
                p = DATASET_ROOT/split/label
                for i in range(80):
                    arr = (np.random.rand(224,224,3)*255).astype(np.uint8)
                    Image.fromarray(arr).save(p/f"synth_{i:04d}.jpg")

    print("\n═══ Dataset Summary ═══")
    for ds_name, counts in DS_STATS.items():
        print(f"  {ds_name:20s} real={counts['real']:6d}  fake={counts['fake']:6d}")
    for split in ["train","val"]:
        for label in ["real","fake"]:
            n = len(list((DATASET_ROOT/split/label).glob("*")))
            print(f"  {split}/{label}: {n} images")


def plot_dataset_dist():
    """Visualize per-dataset distribution."""
    if not DS_STATS: return
    import matplotlib.pyplot as plt
    names = list(DS_STATS.keys())
    reals = [DS_STATS[n]["real"] for n in names]
    fakes = [DS_STATS[n]["fake"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names)*2), 5))
    ax.bar([i-0.2 for i in x], reals, 0.4, label="REAL", color="#3498db")
    ax.bar([i+0.2 for i in x], fakes, 0.4, label="FAKE", color="#e74c3c")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylabel="Image Count", title="Per-Dataset Distribution"); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{LOG_DIR}/dataset_dist.png", dpi=150)
    plt.show()


# ── CELL 5 : Imports & Config ─────────────────────────────────────────────────
"""## ⚙️ Hyperparameter Config"""
import cv2
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report)
from huggingface_hub import HfApi, login as hf_login
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda")
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

@dataclass
class Cfg:
    # Model
    model_name:       str   = "openai/clip-vit-large-patch14"
    n_unfreeze_blocks:int   = 8          # unfreeze top-8 (more data = unfreeze more)
    # Data
    image_size:       int   = 224
    batch_size:       int   = 16 if not IS_A100 else 64
    num_workers:      int   = 4
    label_smoothing:  float = 0.1
    # Optimiser
    lr:               float = 2e-5
    weight_decay:     float = 0.01
    warmup_steps:     int   = 500
    # Training
    num_epochs:       int   = 20
    grad_accum:       int   = 4 if not IS_A100 else 1
    early_stop:       int   = 5
    save_every:       int   = 5
    fp16:             bool  = True
    seed:             int   = 42
    # Augmentation
    use_mixup:        bool  = True
    mixup_alpha:      float = 0.2
    use_cutmix:       bool  = True
    cutmix_prob:      float = 0.3
    # Curriculum
    use_curriculum:   bool  = True
    curriculum_epochs: int  = 5   # epochs before adding harder datasets
    easy_datasets:    tuple = ("celebdf", "ff_plus_plus")
    hard_datasets:    tuple = ("dfdc", "wilddeepfake", "dfd_google", "forgerynet")
    # Paths
    dataset_root:     str   = "/dataset"
    checkpoint_dir:   str   = CHECKPOINT_DIR
    best_model_dir:   str   = BEST_MODEL_DIR
    log_dir:          str   = LOG_DIR

CFG = Cfg()
random.seed(CFG.seed); np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed); torch.cuda.manual_seed_all(CFG.seed)
print("✅ Config:", asdict(CFG))


# ── CELL 6 : Advanced Transforms ─────────────────────────────────────────────
"""## 🎨 Advanced Augmentation Pipeline (Albumentations + JPEG Compression)"""
CASCADE_PATH = "/content/haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        CASCADE_PATH)
FACE_DET = cv2.CascadeClassifier(CASCADE_PATH)

def detect_crop_face(img: Image.Image, pad=0.15) -> Image.Image:
    bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dets = FACE_DET.detectMultiScale(gray, 1.1, 5, minSize=(48,48))
    if len(dets) == 0: return img
    x,y,w,h = sorted(dets, key=lambda r:r[2]*r[3], reverse=True)[0]
    H,W = bgr.shape[:2]
    pw,ph = int(w*pad), int(h*pad)
    return img.crop((max(0,x-pw),max(0,y-ph),min(W,x+w+pw),min(H,y+h+ph)))

# Strong augmentation for training
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.GaussNoise(var_limit=(10,50), p=0.3),
    A.RandomGrayscale(p=0.05),
    # Simulate JPEG compression artifacts (common in deepfakes)
    A.ImageCompression(quality_lower=40, quality_upper=95, p=0.5),
    # Simulate video codec blocks
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
    A.RandomShadow(p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    # Erase random patches to prevent lazy artifact shortcuts
    A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2(),
])

def apply_aug(img: Image.Image, aug) -> torch.Tensor:
    img_c  = detect_crop_face(img)
    img_r  = img_c.resize((CFG.image_size,)*2, Image.BILINEAR)
    img_np = np.array(img_r)
    out    = aug(image=img_np)["image"]  # (C,H,W) tensor
    return out

print("✅ Augmentation pipeline ready (JPEG compression + coarse dropout)")


# ── CELL 7 : Dataset with Curriculum Support ──────────────────────────────────
"""## 📂 Dataset with Source Tagging & Balanced Sampler"""
class DeepfakeDS(Dataset):
    EXT = {".jpg",".jpeg",".png",".bmp",".webp"}
    def __init__(self, root, split, aug, allowed_sources=None):
        """
        allowed_sources: set of source name substrings; None = all sources.
        Used for curriculum training to restrict early epochs to easy data.
        """
        self.aug = aug
        self.items: List[Tuple[Path,int]] = []
        for li, ln in enumerate(["real","fake"]):
            d = Path(root)/split/ln
            for f in d.iterdir():
                if f.suffix.lower() not in self.EXT: continue
                if allowed_sources is not None:
                    src_tag = f.stem.split("_")[0]
                    if not any(src_tag.startswith(s) for s in allowed_sources):
                        continue
                self.items.append((f, li))
        real_n = sum(1 for _,l in self.items if l==0)
        fake_n = sum(1 for _,l in self.items if l==1)
        print(f"  [{split}|{'all' if allowed_sources is None else 'curriculum'}] "
              f"{len(self.items)} imgs  real={real_n} fake={fake_n}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fp, lbl = self.items[i]
        try:    img = Image.open(fp).convert("RGB")
        except: img = Image.new("RGB",(CFG.image_size,)*2,(128,)*3)
        return apply_aug(img, self.aug), torch.tensor(lbl, dtype=torch.long)
    def get_sampler(self):
        """Balanced WeightedRandomSampler to handle class imbalance."""
        labels = [lbl for _, lbl in self.items]
        counts = [labels.count(0), labels.count(1)]
        weights= [1.0/counts[l] for l in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)

def make_loaders(source_filter=None):
    tr = DeepfakeDS(CFG.dataset_root,"train",train_aug,source_filter)
    va = DeepfakeDS(CFG.dataset_root,"val",  val_aug,  None)
    sampler = tr.get_sampler()
    kw = dict(num_workers=CFG.num_workers, pin_memory=True)
    return (DataLoader(tr, CFG.batch_size, sampler=sampler, **kw),
            DataLoader(va, CFG.batch_size*2, shuffle=False, **kw))

train_loader, val_loader = make_loaders()
print(f"✅ Loaders: train={len(train_loader)} batches  val={len(val_loader)} batches")


# ── CELL 8 : Model (CLIP ViT-Large) ──────────────────────────────────────────
"""## 🤖 CLIP ViT-Large + Classification Head (Top-8 blocks unfrozen)"""
class CLIPDeepfake(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip = CLIPModel.from_pretrained(cfg.model_name)
        self.encoder  = clip.vision_model
        self.vis_proj = clip.visual_projection
        proj_dim      = clip.config.projection_dim  # 768
        # Freeze all, unfreeze top N blocks
        for p in self.encoder.parameters(): p.requires_grad = False
        for layer in self.encoder.encoder.layers[-cfg.n_unfreeze_blocks:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in self.encoder.post_layernorm.parameters(): p.requires_grad = True
        # Multi-layer head
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.25),
            nn.Linear(proj_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        out  = self.encoder(pixel_values=x)
        proj = self.vis_proj(out.pooler_output)
        return self.head(proj), proj

model = CLIPDeepfake(CFG).to(DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model — Total:{total/1e6:.1f}M  Trainable:{trainable/1e6:.1f}M")


# ── CELL 9 : Loss / Optimizer / Scheduler ────────────────────────────────────
"""## ⚡ Label-Smoothing Loss + AdamW + Cosine Warmup"""
class SmoothCE(nn.Module):
    def __init__(self, s=0.1):
        super().__init__(); self.s = s
    def forward(self, logits, tgt):
        n = logits.size(-1)
        lp = F.log_softmax(logits,-1)
        with torch.no_grad():
            d = torch.full_like(lp, self.s/(n-1))
            d.scatter_(1, tgt.unsqueeze(1), 1-self.s)
        return -(d*lp).sum(-1).mean()

backbone_p = [p for n,p in model.named_parameters() if "head" not in n and p.requires_grad]
head_p     = [p for n,p in model.named_parameters() if "head" in n]
optimizer  = torch.optim.AdamW(
    [{"params":backbone_p,"lr":CFG.lr},
     {"params":head_p,    "lr":CFG.lr*10}],
    weight_decay=CFG.weight_decay)
total_steps= (len(train_loader)//CFG.grad_accum)*CFG.num_epochs
scheduler  = get_cosine_schedule_with_warmup(optimizer, CFG.warmup_steps, total_steps)
criterion  = SmoothCE(CFG.label_smoothing)
scaler     = GradScaler(enabled=CFG.fp16)
print(f"✅ Optimizer ready — total steps: {total_steps}")


# ── CELL 10 : MixUp & CutMix ─────────────────────────────────────────────────
"""## 🎲 MixUp & CutMix Augmentation"""
def mixup_batch(imgs, labels, alpha=0.2):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(imgs.size(0))
    mixed = lam * imgs + (1-lam) * imgs[idx]
    return mixed, labels, labels[idx], lam

def cutmix_batch(imgs, labels, beta=1.0):
    lam = np.random.beta(beta, beta)
    idx = torch.randperm(imgs.size(0))
    B,C,H,W = imgs.shape
    cut  = int(H * math.sqrt(1-lam))
    cx   = random.randint(0, W-1)
    cy   = random.randint(0, H-1)
    x1,x2 = max(0,cx-cut//2), min(W,cx+cut//2)
    y1,y2 = max(0,cy-cut//2), min(H,cy+cut//2)
    mixed = imgs.clone()
    mixed[:,:,y1:y2,x1:x2] = imgs[idx,:,y1:y2,x1:x2]
    lam_adj = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mixed, labels, labels[idx], lam_adj

def mixed_loss(criterion, logits, lbl_a, lbl_b, lam):
    return lam*criterion(logits, lbl_a) + (1-lam)*criterion(logits, lbl_b)

print("✅ MixUp & CutMix ready")


# ── CELL 11 : Metrics ─────────────────────────────────────────────────────────
"""## 📊 Metrics"""
def compute_metrics(labels, preds, probs):
    return {
        "accuracy":  accuracy_score(labels, preds)*100,
        "auc_roc":   roc_auc_score(labels, probs[:,1]),
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
    }

# ── CELL 12 : GradCAM ────────────────────────────────────────────────────────
"""## 🔥 GradCAM"""
class ViTGradCAM:
    def __init__(self, mdl):
        self.mdl = mdl; self._acts = self._grads = None
        last = mdl.encoder.encoder.layers[-1]
        last.register_forward_hook(lambda m,i,o: setattr(self,"_acts",o[0].detach()))
        last.register_backward_hook(lambda m,gi,go: setattr(self,"_grads",go[0].detach()))
    def generate(self, img_t, cls=1):
        self.mdl.eval()
        x = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        lg,_ = self.mdl(x); self.mdl.zero_grad(); lg[0,cls].backward()
        grads = self._grads[0,1:]; acts = self._acts[0,1:]
        cam = F.relu((grads.mean(-1,keepdim=True)*acts).sum(-1))
        n   = int(math.sqrt(cam.shape[0]))
        cam = cam.reshape(n,n).cpu().detach().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cv2.resize(cam,(CFG.image_size,)*2)

gradcam = ViTGradCAM(model)
print("✅ GradCAM ready")

# ── CELL 13 : Checkpoints ────────────────────────────────────────────────────
"""## 💾 Checkpoint Utilities"""
HISTORY_PATH = f"{CFG.log_dir}/history_vision.json"

def save_ckpt(epoch, is_best=False):
    st = {"epoch":epoch,"model":model.state_dict(),
          "optim":optimizer.state_dict(),"sched":scheduler.state_dict(),
          "scaler":scaler.state_dict()}
    torch.save(st, f"{CFG.checkpoint_dir}/ckpt_ep{epoch:03d}.pt")
    if is_best: torch.save(st, f"{CFG.checkpoint_dir}/best_ckpt.pt")
    print(f"  💾 Ckpt saved ep{epoch}")

def load_latest_ckpt():
    ckpts = sorted(Path(CFG.checkpoint_dir).glob("ckpt_ep*.pt"))
    if not ckpts: print("  ℹ️  Fresh start"); return None
    st = torch.load(ckpts[-1], map_location=DEVICE)
    model.load_state_dict(st["model"])
    optimizer.load_state_dict(st["optim"])
    scheduler.load_state_dict(st["sched"])
    scaler.load_state_dict(st["scaler"])
    print(f"  ♻️  Resumed from {ckpts[-1]}"); return st

def load_history(): return json.load(open(HISTORY_PATH)) if os.path.exists(HISTORY_PATH) else []
def save_history(h): json.dump(h, open(HISTORY_PATH,"w"), indent=2)


# ── CELL 14 : Train / Val Loops ───────────────────────────────────────────────
"""## 🏋️ Train & Validate Loops"""
from tqdm import tqdm

def train_one_epoch(loader, ga):
    model.train(); optimizer.zero_grad()
    loss_sum, correct, total = 0., 0, 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train", leave=False)
    for step, (imgs, labels) in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # MixUp / CutMix
        use_mix = random.random()
        if CFG.use_cutmix and use_mix < CFG.cutmix_prob:
            imgs, la, lb, lam = cutmix_batch(imgs, labels)
            with autocast(enabled=CFG.fp16):
                logits,_ = model(imgs)
                loss = mixed_loss(criterion, logits, la, lb, lam) / ga
        elif CFG.use_mixup and use_mix < CFG.cutmix_prob + 0.3:
            imgs, la, lb, lam = mixup_batch(imgs, labels, CFG.mixup_alpha)
            with autocast(enabled=CFG.fp16):
                logits,_ = model(imgs)
                loss = mixed_loss(criterion, logits, la, lb, lam) / ga
        else:
            with autocast(enabled=CFG.fp16):
                logits,_ = model(imgs)
                loss = criterion(logits, labels) / ga

        scaler.scale(loss).backward()
        if (step+1) % ga == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()

        loss_sum += loss.item()*ga
        preds     = logits.argmax(-1)
        correct  += (preds==labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=f"{loss_sum/(step+1):.4f}", acc=f"{100*correct/total:.1f}%")
    return loss_sum/len(loader), 100*correct/total

@torch.no_grad()
def validate(loader):
    model.eval()
    all_l, all_p, loss_sum = [], [], 0.
    for imgs, labels in tqdm(loader, desc="  Val", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast(enabled=CFG.fp16):
            logits,_ = model(imgs)
            loss = criterion(logits, labels)
        loss_sum += loss.item()
        all_p.append(F.softmax(logits,-1).cpu().numpy())
        all_l.append(labels.cpu().numpy())
    all_l = np.concatenate(all_l); all_p = np.concatenate(all_p,0)
    m = compute_metrics(all_l, all_p.argmax(1), all_p)
    m["loss"] = loss_sum/len(loader)
    return m

print("✅ Training loops ready")


# ── CELL 15 : Curriculum Training Main Loop ───────────────────────────────────
"""## 🚀 Curriculum Training Loop
  Epochs 1–{curriculum_epochs} : easy datasets only (celebdf, ff_plus_plus)
  Epochs {curriculum_epochs+1}+ : ALL datasets added
"""
def train_vision_model(config=None):
    """Main training entry point."""
    if config:
        for k, v in config.items():
            if hasattr(CFG, k):
                setattr(CFG, k, v)
    
    # Identify datasets
    download_all_datasets()
    get_dataset_summary()

    # Init
    ckpt       = load_latest_ckpt()
    global start_ep, history, best_auc, patience, best_state, current_curriculum, train_loader, val_loader, scheduler
    start_ep   = (ckpt["epoch"]+1) if ckpt else 0
    history    = load_history()
    best_auc   = max((h.get("val_auc_roc",0) for h in history), default=0.)
    patience   = 0
    best_state = None
    current_curriculum = "easy"

    print(f"\n{'='*60}")
    print(f"  Vision Training — ALL 6 DATASETS + Curriculum + MixUp")
    print(f"  Epochs {start_ep+1}→{CFG.num_epochs}  |  Best AUC: {best_auc:.4f}")
    print(f"{'='*60}")

    for epoch in range(start_ep, CFG.num_epochs):
        t0 = time.time()

        # ── Curriculum phase switch ──────────────────────────────────────────────
        if CFG.use_curriculum and epoch == CFG.curriculum_epochs and current_curriculum == "easy":
            print(f"\n🎓 Epoch {epoch+1}: Switching to FULL dataset (all sources)")
            train_loader, val_loader = make_loaders(source_filter=None)
            total_steps_new = (len(train_loader)//CFG.grad_accum)*(CFG.num_epochs-epoch)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, CFG.warmup_steps//2, total_steps_new)
            current_curriculum = "full"
        elif epoch == 0 and CFG.use_curriculum:
            print(f"\n🎓 Epoch {epoch+1}: Curriculum — easy datasets only")
            train_loader, val_loader = make_loaders(source_filter=set(CFG.easy_datasets))

        print(f"\n[Epoch {epoch+1}/{CFG.num_epochs}] [{current_curriculum.upper()}]")

        try:
            tl, ta = train_one_epoch(train_loader, CFG.grad_accum)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            CFG.grad_accum = min(CFG.grad_accum*2, 16)
            model.encoder.gradient_checkpointing_enable()
            print(f"⚠️  OOM — grad_accum={CFG.grad_accum}, grad checkpointing ON")
            tl, ta = train_one_epoch(train_loader, CFG.grad_accum)

        vm      = validate(val_loader)
        is_best = vm["auc_roc"] > best_auc

        row = {"epoch":epoch+1,"train_loss":tl,"train_acc":ta, "curriculum":current_curriculum,
               **{f"val_{k}":v for k,v in vm.items()}, "elapsed":time.time()-t0}
        history.append(row); save_history(history)

        print(f"  Train │ loss={tl:.4f}  acc={ta:.2f}%")
        print(f"  Val   │ loss={vm['loss']:.4f}  acc={vm['accuracy']:.2f}%  "
              f"AUC={vm['auc_roc']:.4f}  F1={vm['f1']:.4f}")
        print(f"  ⏱ {row['elapsed']:.1f}s  {'🏆 BEST' if is_best else ''}")

        if (epoch+1)%CFG.save_every==0 or is_best:
            save_ckpt(epoch+1, is_best)

        if is_best:
            best_auc = vm["auc_roc"]; best_state = copy.deepcopy(model.state_dict()); patience=0
            # Save Best Model format
            model.load_state_dict(best_state); model.eval()
            CLIPModel.from_pretrained(CFG.model_name).save_pretrained(CFG.best_model_dir)
            torch.save(model.head.state_dict(), f"{CFG.best_model_dir}/head_weights.pt")
            CLIPProcessor.from_pretrained(CFG.model_name).save_pretrained(CFG.best_model_dir)
            cfg_out = {**asdict(CFG), "best_val_auc": best_auc, "labels":["REAL","FAKE"]}
            json.dump(cfg_out, open(f"{CFG.best_model_dir}/train_config.json","w"), indent=2)
        else:
            patience += 1
            if patience >= CFG.early_stop:
                print(f"\n⏹  Early stopping at epoch {epoch+1} (patience={CFG.early_stop})")
                break

    print("\n✅ Vision training complete!")

def evaluate_vision_model():
    """Run full evaluation and plot results."""
    all_l, all_p, cam_imgs = [], [], []
    for imgs, labels in tqdm(val_loader, desc="Final eval"):
        with torch.no_grad(), autocast(enabled=CFG.fp16):
            logits,_ = model(imgs.to(DEVICE))
        all_p.append(F.softmax(logits,-1).cpu().numpy())
        all_l.append(labels.numpy())
        if len(cam_imgs)<25: cam_imgs.extend(imgs[:25-len(cam_imgs)])

    all_l = np.concatenate(all_l); all_p = np.concatenate(all_p,0)
    fm = compute_metrics(all_l, all_p.argmax(1), all_p)
    print("\n════ FINAL EVALUATION ════")
    print(f"  Accuracy : {fm['accuracy']:.2f}%")
    print(f"  AUC-ROC  : {fm['auc_roc']:.4f}")
    
    # Save curves
    hist=load_history(); ep=[h["epoch"] for h in hist]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(ep,[h["train_loss"] for h in hist],label="train")
    axes[0].plot(ep,[h["val_loss"]   for h in hist],label="val")
    axes[0].set(title="Loss"); axes[0].legend()
    axes[1].plot(ep,[h["val_auc_roc"] for h in hist],color="purple")
    axes[1].set(title="Val AUC-ROC")
    plt.show()

def predict_vision(media_path:str, mdir:str=CFG.best_model_dir, max_frames:int=16)->dict:
    """Run inference on image/video."""
    # Ensure model is on device
    m = CLIPDeepfake(CFG).to(DEVICE)
    try:
        m.head.load_state_dict(torch.load(f"{mdir}/head_weights.pt",map_location=DEVICE))
    except:
        print("️⚠️ Best weights not found, using current weights.")
    m.eval(); cam=ViTGradCAM(m); ext=Path(media_path).suffix.lower()
    
    def _inf(img):
        t=apply_aug(img,val_aug).unsqueeze(0).to(DEVICE)
        with torch.no_grad(),autocast(enabled=True):
            lg,_=m(t)
        p=F.softmax(lg,-1)[0,1].item(); hm=cam.generate(apply_aug(img,val_aug),cls=1)
        return p,hm

    if ext in {".mp4",".avi",".mov",".mkv"}:
        cap=cv2.VideoCapture(media_path); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs=np.linspace(0,tot-1,min(max_frames,tot),dtype=int); scores,hms=[],[]
        for fi in tqdm(idxs, desc="Analyzing video"):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ret,fr=cap.read()
            if not ret: continue
            p,hm=_inf(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
            scores.append(p); hms.append(hm)
        cap.release(); conf=float(np.mean(scores)) if scores else 0.5
    else:
        conf,avg_hm=_inf(Image.open(media_path).convert("RGB"))
        
    return {"confidence":round(conf*100,2), "verdict": "FAKE" if conf > 0.5 else "REAL"}

if __name__ == "__main__":
    # If run directly as a script (not imported), start default training
    train_vision_model()
