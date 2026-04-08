# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AUDIO MODEL — WavLM-Large + CNN-Mel  |  ALL DATASETS  |  MAX ACCURACY  ║
# ║  Datasets : ASVspoof 2019 · 2021 · WaveFake · FakeAVCeleb · MLAAD       ║
# ║  Platform : Google Colab T4/A100 GPU                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELL 1 : GPU Check ───────────────────────────────────────────────────────
"""## 🖥️ GPU Verification"""
import torch, subprocess, sys, os, json, math, time, copy, random, shutil, zipfile
from pathlib import Path

if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime → Change Runtime Type → GPU")

print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"CUDA : {torch.version.cuda}")
IS_A100 = torch.cuda.get_device_properties(0).total_memory > 30e9
print(f"Mode : {'A100 — full batch' if IS_A100 else 'T4 — grad_accum mode'}")


# ── CELL 2 : Install Dependencies ────────────────────────────────────────────
"""## 📦 Install All Audio Dependencies"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0", "accelerate==0.29.3",
    "torchaudio==2.3.0", "librosa==0.10.2", "soundfile==0.12.1",
    "scikit-learn==1.4.2", "matplotlib==3.8.4", "seaborn==0.13.2",
    "tqdm==4.66.2", "huggingface_hub==0.23.0", "gdown==5.1.0",
    "datasets==2.19.0",
], check=True)
print("✅ Audio dependencies installed")


# ── CELL 3 : Mount Drive & Paths ─────────────────────────────────────────────
"""## 💾 Mount Google Drive"""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE     = '/content/drive/MyDrive/deepfake_models/audio'
CHECKPOINT_DIR = f'{DRIVE_BASE}/checkpoints'
BEST_MODEL_DIR = f'{DRIVE_BASE}/best_model'
LOG_DIR        = f'{DRIVE_BASE}/logs'
for d in [CHECKPOINT_DIR, BEST_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"✅ Drive mounted → {DRIVE_BASE}")


# ── CELL 4 : Multi-Dataset Download & Organisation ───────────────────────────
"""## 📥 All Audio Datasets Download
  1. ASVspoof 2019 LA  — manual upload to Drive required
  2. ASVspoof 2021 LA  — manual upload to Drive required
  3. WaveFake          — AUTO via Zenodo
  4. FakeAVCeleb       — manual (Google Form) or Drive upload
  5. MLAAD             — AUTO via HuggingFace datasets
"""
import numpy as np
import soundfile as sf
import gdown
from tqdm import tqdm as _tqdm
from collections import defaultdict
from datasets import load_dataset

AUDIO_ROOT  = Path("/audio_dataset")
SAMPLE_RATE = 16_000
MAX_SECS    = 6.0  # extended from 4s (WaveFake clips are longer)
DS_STATS    = defaultdict(lambda: {"real": 0, "fake": 0})
VAL_RATIO   = 0.15

for split in ["train","val"]:
    for label in ["real","fake"]:
        (AUDIO_ROOT/split/label).mkdir(parents=True, exist_ok=True)


def register_audio(src: Path, label: str, source_name: str):
    """Copy audio file to unified layout."""
    split = "val" if random.random() < VAL_RATIO else "train"
    uid   = os.urandom(4).hex()
    dst   = AUDIO_ROOT/split/label/f"{source_name}_{uid}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
    DS_STATS[source_name][label] += 1


# ── 1. ASVspoof 2019 LA ───────────────────────────────────────────────────────
def organize_asvspoof(zip_drive_path: str, version: str = "2019"):
    """
    Place the zip at /MyDrive/datasets/ASVspoof{year}_LA.zip
    Download link: https://datashare.ed.ac.uk/handle/10283/3336  (2019)
                   https://www.asvspoof.org/index2021.html        (2021)
    """
    extract_dir = f"/content/ASVspoof{version}_LA"
    if not os.path.exists(zip_drive_path):
        print(f"  ⚠️  ASVspoof {version} zip not on Drive — skipping")
        print(f"       Path expected: {zip_drive_path}")
        return
    if not os.path.exists(extract_dir):
        print(f"  📦 Extracting ASVspoof {version} …")
        with zipfile.ZipFile(zip_drive_path) as zf:
            zf.extractall("/content/")

    ext_p = Path(extract_dir)
    # Locate protocol & audio directories
    proto_files = list(ext_p.rglob("*.txt"))
    for pf in proto_files:
        if "train" in pf.name.lower() or "trn" in pf.name.lower():
            split_key = "train"
        elif "dev" in pf.name.lower() or "val" in pf.name.lower():
            split_key = "val"
        else:
            continue
        for line in _tqdm(pf.read_text().strip().splitlines(),
                          desc=f"  ASVspoof{version} {split_key}"):
            parts = line.split()
            if len(parts) < 5: continue
            fname, lbl_str = parts[1], parts[4].upper()
            label  = "real" if lbl_str == "BONAFIDE" else "fake"
            # Search for audio file
            for ext in [".flac",".wav"]:
                candidates = list(ext_p.rglob(f"{fname}{ext}"))
                if candidates:
                    register_audio(candidates[0], label, f"asvspoof{version}")
                    break
    print(f"  ✅ ASVspoof {version} organized")


# ── 2. WaveFake (Zenodo — fully automatic) ────────────────────────────────────
def download_wavefake():
    """
    WaveFake: https://zenodo.org/record/5642694
    Contains real LJSpeech + 6 neural vocoder variants as fakes.
    ~117k utterances total.
    """
    WAVEFAKE_URL  = "https://zenodo.org/record/5642694/files/WaveFake.zip?download=1"
    WAVEFAKE_ZIP  = "/content/WaveFake.zip"
    WAVEFAKE_EXT  = "/content/WaveFake"
    try:
        if not os.path.exists(WAVEFAKE_EXT):
            print("  📥 Downloading WaveFake (~35 GB) …")
            subprocess.run(["wget", "-q", "-O", WAVEFAKE_ZIP, WAVEFAKE_URL], check=True)
            with zipfile.ZipFile(WAVEFAKE_ZIP) as zf:
                zf.extractall("/content/")

        # Real: LJSpeech originals
        ljspeech_real = Path(WAVEFAKE_EXT)/"real"
        if not ljspeech_real.exists():
            ljspeech_real = Path(WAVEFAKE_EXT)/"LJSpeech-1.1/wavs"

        for f in _tqdm(list(ljspeech_real.rglob("*.wav"))[:20000],
                        desc="  WaveFake REAL"):
            register_audio(f, "real", "wavefake")

        # Fake: all vocoder subdirs
        vocoder_dirs = [d for d in Path(WAVEFAKE_EXT).iterdir()
                        if d.is_dir() and d.name not in ["real","LJSpeech-1.1"]]
        print(f"  WaveFake vocoders found: {[d.name for d in vocoder_dirs]}")
        for vdir in vocoder_dirs:
            for f in _tqdm(list(vdir.rglob("*.wav"))[:15000],
                           desc=f"  WaveFake FAKE/{vdir.name}"):
                register_audio(f, "fake", "wavefake")
        print("  ✅ WaveFake done")
    except Exception as e:
        print(f"  ⚠️  WaveFake failed: {e}")


# ── 3. MLAAD — Auto via HuggingFace Datasets ─────────────────────────────────
def download_mlaad():
    """
    MLAAD: Multilingual Audio Anti-spoofing Dataset
    Available via HuggingFace datasets hub.
    Contains 76+ languages, modern TTS systems.
    """
    try:
        print("  📥 Loading MLAAD from HuggingFace …")
        # Load subset for manageable size
        ds = load_dataset("muhtasham/MLAAD", split="train", streaming=True)
        real_n, fake_n = 0, 0
        for i, sample in enumerate(_tqdm(ds, desc="  MLAAD", total=50000)):
            if i >= 50000: break
            label    = "fake" if sample.get("label", 1) == 1 else "real"
            audio    = sample["audio"]
            wave_arr = np.array(audio["array"], dtype=np.float32)
            sr       = audio["sampling_rate"]
            # Resample to 16kHz
            import librosa
            if sr != SAMPLE_RATE:
                wave_arr = librosa.resample(wave_arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            out_path = AUDIO_ROOT/"train"/label/f"mlaad_{i:06d}.wav"
            sf.write(str(out_path), wave_arr, SAMPLE_RATE)
            DS_STATS["mlaad"][label] += 1
            if label == "real": real_n += 1
            else: fake_n += 1
        print(f"  ✅ MLAAD done — real={real_n} fake={fake_n}")
    except Exception as e:
        print(f"  ⚠️  MLAAD failed: {e}")


# ── 4. FakeAVCeleb Audio ──────────────────────────────────────────────────────
def load_fakeavceleb():
    """
    FakeAVCeleb audio tracks.
    Upload to Drive at: /MyDrive/datasets/FakeAVCeleb.zip
    Request from: https://github.com/DASH-Lab/FakeAVCeleb
    """
    FAVC_ZIP = "/content/drive/MyDrive/datasets/FakeAVCeleb.zip"
    FAVC_EXT = "/content/FakeAVCeleb"
    if not os.path.exists(FAVC_ZIP):
        print("  ⚠️  FakeAVCeleb zip not found — skipping")
        print("       Request from: github.com/DASH-Lab/FakeAVCeleb")
        return
    if not os.path.exists(FAVC_EXT):
        with zipfile.ZipFile(FAVC_ZIP) as zf: zf.extractall("/content/")
    # Audio extraction from videos
    for vid in _tqdm(list(Path(FAVC_EXT).rglob("*.mp4")), desc="  FakeAVCeleb"):
        label = "fake" if "fake" in str(vid).lower() else "real"
        wav_out = Path(f"/content/fakeavc_audio/{label}_{vid.stem}.wav")
        wav_out.parent.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg","-y","-i",str(vid),"-ac","1","-ar","16000",
                        str(wav_out)], capture_output=True)
        if wav_out.exists():
            register_audio(wav_out, label, "fakeavceleb")
    print("  ✅ FakeAVCeleb done")


# ── Run all downloaders ───────────────────────────────────────────────────────
print("\n═══ Starting multi-dataset audio download ═══")
organize_asvspoof('/content/drive/MyDrive/datasets/ASVspoof2019_LA.zip', "2019")
organize_asvspoof('/content/drive/MyDrive/datasets/ASVspoof2021_LA.zip', "2021")
download_wavefake()
download_mlaad()
load_fakeavceleb()

# Synthetic fallback
total_files = sum(len(list((AUDIO_ROOT/sp/lb).glob("*")))
                  for sp in ["train","val"] for lb in ["real","fake"])
if total_files < 100:
    print("\n⚠️  No real audio found — generating synthetic fallback …")
    for split in ["train","val"]:
        for label in ["real","fake"]:
            p = AUDIO_ROOT/split/label
            for i in range(80):
                wave = (np.random.randn(int(SAMPLE_RATE*MAX_SECS))*0.05).astype(np.float32)
                sf.write(str(p/f"synth_{i:04d}.wav"), wave, SAMPLE_RATE)

print("\n═══ Audio Dataset Summary ═══")
for ds_name, counts in DS_STATS.items():
    print(f"  {ds_name:20s} real={counts['real']:6d}  fake={counts['fake']:6d}")
for split in ["train","val"]:
    for label in ["real","fake"]:
        n = len(list((AUDIO_ROOT/split/label).glob("*")))
        print(f"  {split}/{label}: {n} files")


# ── CELL 4b : Dataset Distribution ───────────────────────────────────────────
"""## 📊 Dataset Distribution Chart"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

if DS_STATS:
    names = list(DS_STATS.keys())
    reals = [DS_STATS[n]["real"] for n in names]
    fakes = [DS_STATS[n]["fake"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(8,len(names)*2),5))
    ax.bar([i-0.2 for i in x], reals, 0.4, label="REAL", color="#3498db")
    ax.bar([i+0.2 for i in x], fakes, 0.4, label="FAKE", color="#e74c3c")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylabel="File Count", title="Per-Dataset Real/Fake Audio Distribution")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{LOG_DIR}/dataset_distribution.png",dpi=150,bbox_inches="tight"); plt.show()
    print("✅ Dataset distribution saved")


# ── CELL 5 : Imports & Config ─────────────────────────────────────────────────
"""## ⚙️ Config"""
import librosa, seaborn as sns
from dataclasses import dataclass, asdict
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import torchaudio.transforms as AT
from transformers import WavLMModel, get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report, roc_curve)
from huggingface_hub import HfApi, login as hf_login

DEVICE = torch.device("cuda")
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECS)

@dataclass
class AudioCfg:
    model_name:      str   = "microsoft/wavlm-large"
    sample_rate:     int   = 16_000
    max_secs:        float = 6.0
    n_mels:          int   = 128      # increased from 80
    hop_length:      int   = 160
    n_fft:           int   = 512
    # Training
    batch_size:      int   = 8 if not IS_A100 else 32
    num_workers:     int   = 4
    lr:              float = 1e-5
    weight_decay:    float = 0.01
    warmup_steps:    int   = 300
    num_epochs:      int   = 30
    grad_accum:      int   = 8 if not IS_A100 else 2
    early_stop:      int   = 7
    save_every:      int   = 5
    fp16:            bool  = True
    seed:            int   = 42
    # Augmentation
    use_specaugment: bool  = True
    freq_mask_param: int   = 20
    time_mask_param: int   = 50
    use_mixup:       bool  = True
    mixup_alpha:     float = 0.3
    # Dual-stream
    use_mel_branch:  bool  = True     # CNN on mel spectrogram
    mel_fusion_w:    float = 0.35     # weight of mel branch in fusion
    # Paths
    dataset_root:    str   = "/audio_dataset"
    checkpoint_dir:  str   = CHECKPOINT_DIR
    best_model_dir:  str   = BEST_MODEL_DIR
    log_dir:         str   = LOG_DIR

CFG = AudioCfg()
random.seed(CFG.seed); np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed); torch.cuda.manual_seed_all(CFG.seed)
print("✅ Config:", asdict(CFG))


# ── CELL 6 : Audio Preprocessing + SpecAugment ───────────────────────────────
"""## 🎙️ Preprocessing: Load → Trim → Normalise + SpecAugment"""
mel_tf = AT.MelSpectrogram(
    sample_rate=CFG.sample_rate, n_fft=CFG.n_fft,
    hop_length=CFG.hop_length,   n_mels=CFG.n_mels, power=2.0
).to(DEVICE)
db_tf = AT.AmplitudeToDB(top_db=80).to(DEVICE)

# SpecAugment transforms
freq_mask = AT.FrequencyMasking(freq_mask_param=CFG.freq_mask_param)
time_mask = AT.TimeMasking(time_mask_param=CFG.time_mask_param)

def load_audio(path: str) -> np.ndarray:
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        try:
            w, sr = torchaudio.load(path)
            w = w.mean(0).numpy()
        except Exception:
            return np.zeros(MAX_SAMPLES, dtype=np.float32)
    if w.ndim > 1: w = w.mean(-1)
    if sr != CFG.sample_rate:
        w = librosa.resample(w, orig_sr=sr, target_sr=CFG.sample_rate)
    w, _ = librosa.effects.trim(w, top_db=30)
    # Normalise to [-1,1]
    pk = np.abs(w).max()
    if pk > 1e-6: w /= pk
    # Pad / truncate
    if len(w) < MAX_SAMPLES:
        w = np.pad(w, (0, MAX_SAMPLES-len(w)))
    return w[:MAX_SAMPLES].astype(np.float32)

def audio_augment(wave: np.ndarray) -> np.ndarray:
    """Time-domain augmentations for training."""
    # Random gain
    gain = random.uniform(0.7, 1.3)
    wave = (wave * gain).clip(-1,1)
    # Random noise injection
    if random.random() < 0.4:
        wave += np.random.randn(*wave.shape).astype(np.float32) * 0.005
        wave = wave.clip(-1,1)
    # Random time shift
    if random.random() < 0.3:
        shift = random.randint(0, MAX_SAMPLES//8)
        wave  = np.roll(wave, shift)
    return wave

def compute_log_mel(wave_t: torch.Tensor, augment=False) -> torch.Tensor:
    """(MAX_SAMPLES,) → log-mel (n_mels, T) with optional SpecAugment."""
    with torch.no_grad():
        mel = mel_tf(wave_t.unsqueeze(0))   # (1, n_mels, T)
        lm  = db_tf(mel)                     # (1, n_mels, T)
        if augment and CFG.use_specaugment:
            lm = freq_mask(lm)
            lm = time_mask(lm)
    return lm.squeeze(0)                     # (n_mels, T)

print("✅ Audio preprocessing + SpecAugment ready")


# ── CELL 7 : Dataset ──────────────────────────────────────────────────────────
"""## 📂 AudioDeepfakeDataset — all sources"""
class AudioDeepfakeDS(Dataset):
    EXT = {".wav",".flac",".mp3",".ogg",".m4a"}
    def __init__(self, root, split, is_train=False):
        self.is_train = is_train
        self.items: List[Tuple[Path,int]] = []
        for li, ln in enumerate(["real","fake"]):
            d = Path(root)/split/ln
            self.items += [(f,li) for f in d.iterdir() if f.suffix.lower() in self.EXT]
        real_n = sum(1 for _,l in self.items if l==0)
        fake_n = sum(1 for _,l in self.items if l==1)
        print(f"  [{split}] {len(self.items)} files  real={real_n} fake={fake_n}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fp, lbl = self.items[i]
        wave    = load_audio(str(fp))
        if self.is_train:
            wave = audio_augment(wave)
        wave_t  = torch.from_numpy(wave)
        return wave_t, torch.tensor(lbl, dtype=torch.long)
    def get_sampler(self):
        labels  = [l for _,l in self.items]
        counts  = [labels.count(0), labels.count(1)]
        weights = [1.0/counts[l] for l in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)

def make_loaders():
    tr = AudioDeepfakeDS(CFG.dataset_root,"train",is_train=True)
    va = AudioDeepfakeDS(CFG.dataset_root,"val",  is_train=False)
    sampler = tr.get_sampler()
    kw = dict(num_workers=CFG.num_workers, pin_memory=True)
    return (DataLoader(tr, CFG.batch_size, sampler=sampler, **kw),
            DataLoader(va, CFG.batch_size, shuffle=False, **kw))

train_loader, val_loader = make_loaders()
print(f"✅ Loaders: train={len(train_loader)} batches  val={len(val_loader)} batches")


# ── CELL 8 : Dual-Stream Model ────────────────────────────────────────────────
"""## 🤖 Dual-Stream Model: WavLM-Large + CNN-Mel → Fusion Head
Stream 1: WavLM-Large processes raw waveform → captures phonemic/temporal artifacts
Stream 2: Small CNN on log-mel spectrogram → captures spectral vocoder fingerprints
Fusion:   Weighted concat → linear classifier
"""
class MelCNN(nn.Module):
    """Lightweight CNN for log-mel spectrogram classification."""
    def __init__(self, n_mels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.GELU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*16, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):  # x: (B, n_mels, T)
        return self.net(x.unsqueeze(1))  # (B, out_dim)


class DualStreamDeepfake(nn.Module):
    """WavLM-Large (waveform) + MelCNN (spectrogram) → fusion classifier."""
    def __init__(self, cfg: AudioCfg):
        super().__init__()
        # ─── WavLM stream ────────────────────────────────────────────────────
        self.wavlm  = WavLMModel.from_pretrained(cfg.model_name)
        hidden      = self.wavlm.config.hidden_size  # 1024
        n_layers    = len(self.wavlm.encoder.layers)
        freeze_up   = int(n_layers * 0.75)           # unfreeze top-25%
        for p in self.wavlm.parameters(): p.requires_grad = False
        for layer in self.wavlm.encoder.layers[freeze_up:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in self.wavlm.feature_projection.parameters(): p.requires_grad = True

        # ─── Mel-CNN stream ──────────────────────────────────────────────────
        self.mel_cnn   = MelCNN(cfg.n_mels, out_dim=256) if cfg.use_mel_branch else None
        self.mel_w     = cfg.mel_fusion_w

        # ─── Fusion head ─────────────────────────────────────────────────────
        wavlm_dim  = hidden
        fusion_dim = wavlm_dim + (256 if cfg.use_mel_branch else 0)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, wave: torch.Tensor, mel: torch.Tensor = None):
        # WavLM stream
        out    = self.wavlm(input_values=wave)
        w_feat = out.last_hidden_state.mean(1)      # (B, hidden)

        if self.mel_cnn is not None and mel is not None:
            m_feat = self.mel_cnn(mel)              # (B, 256)
            feat   = torch.cat([w_feat, m_feat], -1)
        else:
            feat = w_feat

        logits = self.fusion_head(feat)
        return logits, feat

    def enable_grad_ckpt(self):
        self.wavlm.gradient_checkpointing_enable()


model = DualStreamDeepfake(CFG).to(DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Dual-stream model — Total:{total/1e6:.1f}M  Trainable:{trainable/1e6:.1f}M")


# ── CELL 9 : Loss / Optimizer / Scheduler ────────────────────────────────────
"""## ⚡ Class-Weighted Loss + AdamW + Linear Warmup"""
from collections import Counter

def build_class_weights(ds_path, split="train"):
    items = []
    for li, ln in enumerate(["real","fake"]):
        d = Path(ds_path)/split/ln
        items += [li] * len([f for f in d.iterdir()])
    counts = Counter(items)
    total  = sum(counts.values())
    return torch.tensor([total/(2*counts[c]) for c in [0,1]], dtype=torch.float32).to(DEVICE)

weights   = build_class_weights(CFG.dataset_root)
print(f"  Class weights → real={weights[0]:.3f}  fake={weights[1]:.3f}")
criterion = nn.CrossEntropyLoss(weight=weights)
scaler    = GradScaler(enabled=CFG.fp16)

backbone_p = [p for n,p in model.named_parameters()
              if "fusion_head" not in n and "mel_cnn" not in n and p.requires_grad]
head_p     = [p for n,p in model.named_parameters()
              if "fusion_head" in n or "mel_cnn" in n]
optimizer  = torch.optim.AdamW(
    [{"params":backbone_p,"lr":CFG.lr},
     {"params":head_p,    "lr":CFG.lr*10}],
    weight_decay=CFG.weight_decay)
total_steps = (len(train_loader)//CFG.grad_accum)*CFG.num_epochs
scheduler   = get_linear_schedule_with_warmup(
    optimizer, CFG.warmup_steps, total_steps)
print(f"✅ Optimizer ready — total steps: {total_steps}")


# ── CELL 10 : Metrics with EER ────────────────────────────────────────────────
"""## 📊 Metrics: Accuracy, AUC-ROC, EER, F1, P, R"""
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr-fnr))
    return float((fpr[idx]+fnr[idx])/2*100)

def compute_metrics(labels, preds, probs):
    return {"accuracy": accuracy_score(labels,preds)*100,
            "auc_roc":  roc_auc_score(labels, probs[:,1]),
            "eer":      compute_eer(labels, probs[:,1]),
            "f1":       f1_score(labels,preds,average="binary"),
            "precision":precision_score(labels,preds,average="binary",zero_division=0),
            "recall":   recall_score(labels,preds,average="binary",zero_division=0)}
print("✅ Metrics with EER ready")


# ── CELL 11 : Checkpoints ─────────────────────────────────────────────────────
"""## 💾 Checkpoint Utilities"""
HISTORY_PATH = f"{CFG.log_dir}/history_audio.json"

def save_ckpt(epoch, is_best=False):
    st = {"epoch":epoch,"model":model.state_dict(),
          "optim":optimizer.state_dict(),"sched":scheduler.state_dict(),
          "scaler":scaler.state_dict()}
    torch.save(st, f"{CFG.checkpoint_dir}/ckpt_ep{epoch:03d}.pt")
    if is_best: torch.save(st, f"{CFG.checkpoint_dir}/best_ckpt.pt")
    print(f"  💾 Audio ckpt saved ep{epoch}")

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
print("✅ Checkpoint utilities ready")


# ── CELL 12 : MixUp ───────────────────────────────────────────────────────────
"""## 🎲 Audio MixUp"""
def mixup_audio(waves, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(waves.size(0))
    return lam*waves + (1-lam)*waves[idx], labels, labels[idx], lam


# ── CELL 13 : Train / Val Loops ───────────────────────────────────────────────
"""## 🏋️ Train & Validate"""
from tqdm import tqdm

def train_one_epoch(loader, ga):
    model.train(); optimizer.zero_grad()
    loss_sum, correct, total = 0., 0, 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train", leave=False)
    for step, (waves, labels) in pbar:
        waves, labels = waves.to(DEVICE), labels.to(DEVICE)
        # Compute mel for CNN branch
        mel = None
        if CFG.use_mel_branch:
            with torch.no_grad():
                mel = torch.stack([compute_log_mel(w, augment=True) for w in waves])
        # MixUp
        if CFG.use_mixup and random.random() < 0.4:
            waves, la, lb, lam = mixup_audio(waves, labels, CFG.mixup_alpha)
            if mel is not None:
                mel_mix = lam*mel + (1-lam)*mel[torch.randperm(mel.size(0))]
                mel = mel_mix
            with autocast(enabled=CFG.fp16):
                logits,_ = model(waves, mel)
                loss = (lam*criterion(logits,la) + (1-lam)*criterion(logits,lb)) / ga
        else:
            with autocast(enabled=CFG.fp16):
                logits,_ = model(waves, mel)
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
    for waves, labels in tqdm(loader, desc="  Val", leave=False):
        waves, labels = waves.to(DEVICE), labels.to(DEVICE)
        mel = None
        if CFG.use_mel_branch:
            mel = torch.stack([compute_log_mel(w, augment=False) for w in waves])
        with autocast(enabled=CFG.fp16):
            logits,_ = model(waves, mel)
            loss = criterion(logits, labels)
        loss_sum += loss.item()
        all_p.append(F.softmax(logits,-1).cpu().numpy())
        all_l.append(labels.cpu().numpy())
    all_l = np.concatenate(all_l); all_p = np.concatenate(all_p,0)
    m = compute_metrics(all_l, all_p.argmax(1), all_p)
    m["loss"] = loss_sum/len(loader)
    return m

print("✅ Train/val loops ready")


# ── CELL 14 : Main Training Loop ─────────────────────────────────────────────
"""## 🚀 Training (30 epochs, dual-stream, all datasets)"""
ckpt       = load_latest_ckpt()
start_ep   = (ckpt["epoch"]+1) if ckpt else 0
history    = load_history()
best_auc   = max((h.get("val_auc_roc",0) for h in history), default=0.)
best_eer   = min((h.get("val_eer",100.) for h in history), default=100.)
patience   = 0; best_state = None

print(f"\n{'='*60}")
print(f"  WavLM + CNN-Mel Dual-Stream — ALL 5 AUDIO DATASETS")
print(f"  Epochs {start_ep+1}→{CFG.num_epochs}  |  Best AUC:{best_auc:.4f}  EER:{best_eer:.2f}%")
print(f"{'='*60}")

for epoch in range(start_ep, CFG.num_epochs):
    t0 = time.time()
    print(f"\n[Epoch {epoch+1}/{CFG.num_epochs}]")
    try:
        tl, ta = train_one_epoch(train_loader, CFG.grad_accum)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        CFG.grad_accum = min(CFG.grad_accum*2, 32)
        model.enable_grad_ckpt()
        print(f"⚠️  OOM — grad_accum={CFG.grad_accum} + grad ckpt enabled")
        tl, ta = train_one_epoch(train_loader, CFG.grad_accum)

    vm      = validate(val_loader)
    is_best = vm["auc_roc"] > best_auc

    row = {"epoch":epoch+1,"train_loss":tl,"train_acc":ta,
           **{f"val_{k}":v for k,v in vm.items()},"elapsed":time.time()-t0}
    history.append(row); save_history(history)

    print(f"  Train │ loss={tl:.4f}  acc={ta:.2f}%")
    print(f"  Val   │ loss={vm['loss']:.4f}  acc={vm['accuracy']:.2f}%  "
          f"AUC={vm['auc_roc']:.4f}  EER={vm['eer']:.2f}%  F1={vm['f1']:.4f}")
    print(f"  ⏱ {row['elapsed']:.1f}s  {'🏆 BEST' if is_best else ''}")

    if (epoch+1)%CFG.save_every==0 or is_best:
        save_ckpt(epoch+1, is_best)

    if is_best:
        best_auc = vm["auc_roc"]; best_eer = vm["eer"]
        best_state = copy.deepcopy(model.state_dict()); patience = 0
    else:
        patience += 1
        if patience >= CFG.early_stop:
            print(f"\n⏹  Early stopping at epoch {epoch+1}"); break

print("\n✅ Audio training complete!")


# ── CELL 15 : Save Best Model ─────────────────────────────────────────────────
"""## 💾 Save (HuggingFace Format)"""
model.load_state_dict(best_state); model.eval()
model.wavlm.save_pretrained(CFG.best_model_dir)
torch.save(model.fusion_head.state_dict(), f"{CFG.best_model_dir}/fusion_head.pt")
if CFG.use_mel_branch:
    torch.save(model.mel_cnn.state_dict(), f"{CFG.best_model_dir}/mel_cnn.pt")
cfg_out = {**asdict(CFG), "best_auc":best_auc,"best_eer":best_eer,
           "labels":["REAL","FAKE"],"training_datasets":list(DS_STATS.keys())}
json.dump(cfg_out, open(f"{CFG.best_model_dir}/train_config.json","w"), indent=2)
print(f"✅ Audio model saved → {CFG.best_model_dir}  (AUC={best_auc:.4f} EER={best_eer:.2f}%)")


# ── CELL 16 : Full Evaluation ─────────────────────────────────────────────────
"""## 📊 Full Evaluation"""
all_l, all_p = [], []
for waves, labels in tqdm(val_loader, desc="Final eval"):
    waves, labels = waves.to(DEVICE), labels.to(DEVICE)
    mel = torch.stack([compute_log_mel(w) for w in waves]) if CFG.use_mel_branch else None
    with torch.no_grad(), autocast(enabled=CFG.fp16):
        logits,_ = model(waves, mel)
    all_p.append(F.softmax(logits,-1).cpu().numpy())
    all_l.append(labels.cpu().numpy())

all_l = np.concatenate(all_l); all_p = np.concatenate(all_p,0)
fm    = compute_metrics(all_l, all_p.argmax(1), all_p)
print("\n════ FINAL AUDIO EVALUATION — ALL 5 DATASETS ════")
print(f"  Accuracy : {fm['accuracy']:.2f}%")
print(f"  AUC-ROC  : {fm['auc_roc']:.4f}")
print(f"  EER      : {fm['eer']:.2f}%  (lower is better)")
print(f"  F1       : {fm['f1']:.4f}")
print(f"  Precision: {fm['precision']:.4f}")
print(f"  Recall   : {fm['recall']:.4f}")
print(); print(classification_report(all_l,all_p.argmax(1),target_names=["REAL","FAKE"]))

cm = confusion_matrix(all_l,all_p.argmax(1))
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Purples",xticklabels=["REAL","FAKE"],
            yticklabels=["REAL","FAKE"],ax=ax)
ax.set(xlabel="Predicted",ylabel="True",title="Confusion Matrix — All Audio Datasets")
fig.savefig(f"{CFG.log_dir}/confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.show()


# ── CELL 17 : Confidence Histogram + ROC ──────────────────────────────────────
"""## 📊 Confidence Distribution & ROC with EER"""
real_sc = all_p[all_l==0,1]; fake_sc = all_p[all_l==1,1]
fig,axes = plt.subplots(1,2,figsize=(14,5))
axes[0].hist(real_sc,bins=50,alpha=0.7,color="#3498db",label="REAL")
axes[0].hist(fake_sc,bins=50,alpha=0.7,color="#e74c3c",label="FAKE")
axes[0].set(xlabel="P(FAKE)",ylabel="Count",title="Confidence Score Distribution"); axes[0].legend()
fpr,tpr,_ = roc_curve(all_l,all_p[:,1])
axes[1].plot(fpr,tpr,color="#8e44ad",lw=2,label=f"AUC={fm['auc_roc']:.4f}")
axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
eer_x = fm["eer"]/100
axes[1].scatter([eer_x],[1-eer_x],color="red",s=80,zorder=5,label=f"EER={fm['eer']:.2f}%")
axes[1].set(xlabel="FPR",ylabel="TPR",title="ROC Curve — Audio Deepfake"); axes[1].legend()
plt.suptitle("WavLM+MelCNN Dual-Stream — All Datasets"); plt.tight_layout()
fig.savefig(f"{CFG.log_dir}/confidence_roc.png",dpi=150,bbox_inches="tight"); plt.show()
print("✅ Confidence & ROC saved")


# ── CELL 18 : Training Curves ─────────────────────────────────────────────────
"""## 📈 Training Curves (4 panels)"""
hist=load_history(); ep=[h["epoch"] for h in hist]
fig,axes=plt.subplots(1,4,figsize=(22,5))
axes[0].plot(ep,[h["train_loss"] for h in hist],label="train")
axes[0].plot(ep,[h["val_loss"]   for h in hist],label="val")
axes[0].set(title="Loss",xlabel="Epoch"); axes[0].legend()
axes[1].plot(ep,[h["train_acc"]    for h in hist],label="train")
axes[1].plot(ep,[h["val_accuracy"] for h in hist],label="val")
axes[1].set(title="Accuracy %",xlabel="Epoch"); axes[1].legend()
axes[2].plot(ep,[h["val_auc_roc"] for h in hist],color="#8e44ad")
axes[2].set(title="Val AUC-ROC",xlabel="Epoch")
axes[3].plot(ep,[h["val_eer"] for h in hist],color="#e74c3c")
axes[3].set(title="Val EER % (↓ better)",xlabel="Epoch")
plt.suptitle("WavLM+MelCNN Dual-Stream — All Dataset Training"); plt.tight_layout()
fig.savefig(f"{CFG.log_dir}/training_curves.png",dpi=150,bbox_inches="tight"); plt.show()
print("✅ Training curves saved")


# ── CELL 19 : Spectrogram Visualisation ──────────────────────────────────────
"""## 🎵 Spectrogram Samples — Real vs Fake"""
sample_files = (list((AUDIO_ROOT/"val"/"real").glob("*"))[:3] +
                list((AUDIO_ROOT/"val"/"fake").glob("*"))[:3])
labels_show  = ["REAL"]*3 + ["FAKE"]*3
fig,axes = plt.subplots(2,3,figsize=(18,8))
for i,(fp,lbl) in enumerate(zip(sample_files,labels_show)):
    ax = axes[i//3][i%3]
    w  = torch.from_numpy(load_audio(str(fp))).to(DEVICE)
    lm = compute_log_mel(w).cpu().numpy()
    im = ax.imshow(lm,origin="lower",aspect="auto",cmap="magma")
    ax.set(title=f"{lbl} — {fp.name[:22]}",xlabel="Frame",ylabel="Mel Bin")
    plt.colorbar(im,ax=ax,format="%.0f dB")
plt.suptitle("Log-Mel Spectrograms: REAL vs FAKE (128-bin)"); plt.tight_layout()
fig.savefig(f"{CFG.log_dir}/spectrograms.png",dpi=150,bbox_inches="tight"); plt.show()
print("✅ Spectrograms saved")


# ── CELL 20 : HuggingFace Upload ──────────────────────────────────────────────
"""## 🤗 Upload to HuggingFace Hub"""
HF_TOKEN = "YOUR_HF_TOKEN"
HF_USER  = "your-hf-username"
REPO_ID  = f"{HF_USER}/wavlm-large-mel-cnn-deepfake-detector"
hf_login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=REPO_ID, exist_ok=True)
api.upload_folder(folder_path=CFG.best_model_dir, repo_id=REPO_ID,
                  commit_message=f"Dual-stream audio deepfake AUC={best_auc:.4f} EER={best_eer:.2f}%")
print(f"✅ Uploaded → https://huggingface.co/{REPO_ID}")


# ── CELL 21 : Inference ───────────────────────────────────────────────────────
"""## 🔮 Audio Inference"""
def load_best_audio_model(mdir=CFG.best_model_dir):
    m = DualStreamDeepfake(CFG).to(DEVICE)
    m.fusion_head.load_state_dict(torch.load(f"{mdir}/fusion_head.pt",map_location=DEVICE))
    if CFG.use_mel_branch and os.path.exists(f"{mdir}/mel_cnn.pt"):
        m.mel_cnn.load_state_dict(torch.load(f"{mdir}/mel_cnn.pt",map_location=DEVICE))
    m.eval(); return m

def predict_audio(media_path:str, mdir:str=CFG.best_model_dir)->dict:
    mdl = load_best_audio_model(mdir)
    ext = Path(media_path).suffix.lower()
    ap  = media_path
    if ext in {".mp4",".avi",".mov",".mkv"}:
        ap = f"/content/tmp_audio_{Path(media_path).stem}.wav"
        subprocess.run(["ffmpeg","-y","-i",media_path,"-ac","1","-ar","16000",ap],
                       capture_output=True)
    try: wave = load_audio(ap)
    except Exception:
        return {"model":"wavlm-large-melcnn-finetuned","confidence":None,
                "eer_score":None,"feature_breakdown":[],"spectrogram_path":None}

    wav_t = torch.from_numpy(wave).unsqueeze(0).to(DEVICE)
    mel   = compute_log_mel(torch.from_numpy(wave).to(DEVICE),augment=False).unsqueeze(0) if CFG.use_mel_branch else None
    with torch.no_grad(), autocast(enabled=True):
        logits,_ = mdl(wav_t, mel)
    probs = F.softmax(logits,-1)[0].cpu().numpy()
    conf  = float(probs[1])

    spec_out = f"{CFG.log_dir}/spec_{Path(media_path).stem}.png"
    lm = compute_log_mel(torch.from_numpy(wave).to(DEVICE)).cpu().numpy()
    fig,ax=plt.subplots(figsize=(8,3))
    ax.imshow(lm,origin="lower",aspect="auto",cmap="magma")
    ax.set(xlabel="Frame",ylabel="Mel Bin",title=f"Log-Mel — {Path(media_path).name}")
    fig.savefig(spec_out,dpi=100,bbox_inches="tight"); plt.close(fig)

    n_seg=8; seg_len=len(wave)//n_seg; breakdown=[]
    for si in range(n_seg):
        seg=np.pad(wave[si*seg_len:(si+1)*seg_len],
                   (0,max(0,MAX_SAMPLES-seg_len)))[:MAX_SAMPLES]
        st=torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        mt=compute_log_mel(torch.from_numpy(seg).to(DEVICE)).unsqueeze(0) if CFG.use_mel_branch else None
        with torch.no_grad(),autocast(enabled=True):
            lg2,_=mdl(st,mt)
        p2=F.softmax(lg2,-1)[0,1].item()
        breakdown.append({"segment":si+1,"start_ms":round(si*seg_len/SAMPLE_RATE*1000),
                          "fake_prob":round(p2*100,2)})

    eer_approx = abs(conf-(1-conf))*50
    return {"model":"wavlm-large-melcnn-finetuned-all-datasets",
            "confidence":round(conf*100,2),"eer_score":round(eer_approx,2),
            "feature_breakdown":breakdown,"spectrogram_path":spec_out,
            "training_datasets":list(DS_STATS.keys())}

_s=str(next((AUDIO_ROOT/"val"/"fake").glob("*")))
print(json.dumps(predict_audio(_s),indent=2))
print("\n✅ Audio notebook complete!")
