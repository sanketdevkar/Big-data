# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMBINED INFERENCE — All Models + Extended JSON Schema                  ║
# ║  Vision  : CLIP ViT-Large (trained on 6 datasets)                       ║
# ║  Audio   : WavLM-Large + CNN-Mel dual stream (trained on 5 datasets)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELL 1 : Setup ───────────────────────────────────────────────────────────
"""## 🔗 Combined Multimodal Inference
Load both trained models from Google Drive and analyze any media file.
Returns the full extended JSON verdict.
"""
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, math, subprocess, time, random
import torch, cv2, numpy as np
from pathlib import Path
from PIL import Image

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0", "torchaudio==2.3.0", "librosa==0.10.2",
    "soundfile==0.12.1", "opencv-python-headless==4.9.0.80",
    "Pillow==10.3.0", "matplotlib==3.8.4", "albumentations==1.4.3",
], check=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
import torchaudio
import torchaudio.transforms as AT
import librosa
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import CLIPModel, WavLMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")

VISION_DIR = '/content/drive/MyDrive/deepfake_models/vision/best_model'
AUDIO_DIR  = '/content/drive/MyDrive/deepfake_models/audio/best_model'
OUTPUT_DIR = '/content/drive/MyDrive/deepfake_models/inference_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Weights for final fusion
FUSION_W = {
    "clip_vit":  0.45,   # strongest single model
    "wavlm":     0.30,   # raw waveform context
    "mel_cnn":   0.25,   # spectral vocoder fingerprint
}
print(f"  Fusion weights: {FUSION_W}")


# ── CELL 2 : Model Definitions ────────────────────────────────────────────────
"""## 🤖 Model Architecture Definitions"""
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE  = 224
SAMPLE_RATE = 16_000
N_MELS      = 128
MAX_SAMPLES = SAMPLE_RATE * 6

# ── Vision model ─────────────────────────────────────────────────────────────
class CLIPDeepfake(nn.Module):
    def __init__(self, clip_dir):
        super().__init__()
        clip = CLIPModel.from_pretrained(clip_dir)
        self.encoder  = clip.vision_model
        self.vis_proj = clip.visual_projection
        proj_dim      = clip.config.projection_dim
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim), nn.Dropout(0.25),
            nn.Linear(proj_dim,1024), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(1024,256),     nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256,2),
        )
    def forward(self, x):
        return self.head(self.vis_proj(self.encoder(pixel_values=x).pooler_output))

# ── Audio model (dual-stream) ─────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(128*16,512),nn.GELU(),nn.Dropout(0.3),nn.Linear(512,256),
        )
    def forward(self, x): return self.net(x.unsqueeze(1))

class DualStreamDeepfake(nn.Module):
    def __init__(self, wavlm_dir):
        super().__init__()
        self.wavlm   = WavLMModel.from_pretrained(wavlm_dir)
        self.mel_cnn = MelCNN()
        hidden       = self.wavlm.config.hidden_size  # 1024
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden+256), nn.Dropout(0.3),
            nn.Linear(hidden+256,512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512,128),        nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128,2),
        )
    def forward(self, wave, mel=None):
        w_feat = self.wavlm(input_values=wave).last_hidden_state.mean(1)
        if mel is not None:
            m_feat = self.mel_cnn(mel)
            feat   = torch.cat([w_feat, m_feat], -1)
        else:
            feat = w_feat
        return self.fusion_head(feat), feat

print("✅ Model classes defined")


# ── CELL 3 : Load Models ──────────────────────────────────────────────────────
"""## 💾 Load Trained Models from Drive"""
# Vision
vision_model = CLIPDeepfake(VISION_DIR).to(DEVICE)
vision_model.head.load_state_dict(
    torch.load(f"{VISION_DIR}/head_weights.pt", map_location=DEVICE))
vision_model.eval()
print("✅ Vision model (CLIP ViT-Large, 6 datasets) loaded")

# Audio
audio_model = DualStreamDeepfake(AUDIO_DIR).to(DEVICE)
audio_model.fusion_head.load_state_dict(
    torch.load(f"{AUDIO_DIR}/fusion_head.pt", map_location=DEVICE))
if os.path.exists(f"{AUDIO_DIR}/mel_cnn.pt"):
    audio_model.mel_cnn.load_state_dict(
        torch.load(f"{AUDIO_DIR}/mel_cnn.pt", map_location=DEVICE))
audio_model.eval()
print("✅ Audio model (WavLM-Large + MelCNN, 5 datasets) loaded")

# Load training configs
v_cfg = json.load(open(f"{VISION_DIR}/train_config.json"))
a_cfg = json.load(open(f"{AUDIO_DIR}/train_config.json"))
print(f"\n  Vision trained on: {v_cfg.get('training_datasets', ['celebdf','ff++','dfdc','dfd','wilddeepfake','forgerynet'])}")
print(f"  Audio  trained on: {a_cfg.get('training_datasets', ['asvspoof2019','asvspoof2021','wavefake','fakeavceleb','mlaad'])}")


# ── CELL 4 : Preprocessing Helpers ───────────────────────────────────────────
"""## 🔧 Shared Preprocessing"""
import urllib.request

CASCADE_PATH = "/content/haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        CASCADE_PATH)
FACE_DET = cv2.CascadeClassifier(CASCADE_PATH)

val_aug_np = lambda img_np: img_np  # identity; full aug only at train time

def detect_crop_face(img: Image.Image) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray= cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dets= FACE_DET.detectMultiScale(gray,1.1,5,minSize=(48,48))
    if len(dets)==0: return img
    x,y,w,h = sorted(dets,key=lambda r:r[2]*r[3],reverse=True)[0]
    H,W = bgr.shape[:2]
    pw,ph=int(w*.15),int(h*.15)
    return img.crop((max(0,x-pw),max(0,y-ph),min(W,x+w+pw),min(H,y+h+ph)))

vision_tf = T.Compose([
    T.Lambda(detect_crop_face), T.Resize((IMG_SIZE,)*2),
    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
])
UNNORM = T.Compose([T.Normalize([0]*3,[1/s for s in CLIP_STD]),
                    T.Normalize([-m for m in CLIP_MEAN],[1]*3)])

mel_tf = AT.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=512,
                            hop_length=160,n_mels=N_MELS).to(DEVICE)
db_tf  = AT.AmplitudeToDB(top_db=80).to(DEVICE)

def load_audio(path:str)->np.ndarray:
    try:    w,sr = sf.read(path,dtype="float32",always_2d=False)
    except Exception:
        try:
            w,sr = torchaudio.load(path)
            w = w.mean(0).numpy()
        except Exception:
            return np.zeros(MAX_SAMPLES,dtype=np.float32)
    if w.ndim>1: w=w.mean(-1)
    if sr!=SAMPLE_RATE: w=librosa.resample(w,orig_sr=sr,target_sr=SAMPLE_RATE)
    w,_=librosa.effects.trim(w,top_db=30)
    pk=np.abs(w).max()
    if pk>1e-6: w/=pk
    if len(w)<MAX_SAMPLES: w=np.pad(w,(0,MAX_SAMPLES-len(w)))
    return w[:MAX_SAMPLES].astype(np.float32)

def wave_to_mel(wave:np.ndarray)->torch.Tensor:
    with torch.no_grad():
        wt = torch.from_numpy(wave).to(DEVICE)
        return db_tf(mel_tf(wt.unsqueeze(0))).squeeze(0)  # (n_mels, T)


# ── GradCAM for ViT ──────────────────────────────────────────────────────────
class ViTGradCAM:
    def __init__(self, mdl):
        self.mdl=mdl; self._acts=self._grads=None
        last=mdl.encoder.encoder.layers[-1]
        last.register_forward_hook(lambda m,i,o: setattr(self,"_acts",o[0].detach()))
        last.register_backward_hook(lambda m,gi,go: setattr(self,"_grads",go[0].detach()))
    def generate(self, img_t, cls=1):
        self.mdl.eval()
        x=img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        lg=self.mdl(x); self.mdl.zero_grad(); lg[0,cls].backward()
        grads=self._grads[0,1:]; acts=self._acts[0,1:]
        cam=F.relu((grads.mean(-1,keepdim=True)*acts).sum(-1))
        n=int(math.sqrt(cam.shape[0]))
        cam=cam.reshape(n,n).cpu().detach().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cv2.resize(cam,(IMG_SIZE,)*2)

gradcam=ViTGradCAM(vision_model)
print("✅ All helpers ready")


# ── CELL 5 : Vision Predictor ─────────────────────────────────────────────────
"""## 🖼️ Vision Modality Predictor"""
def _infer_image(img:Image.Image):
    t = vision_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), autocast(enabled=True):
        lg = vision_model(t)
    p  = F.softmax(lg,-1)[0,1].item()
    hm = gradcam.generate(vision_tf(img), cls=1)
    return p, hm

def predict_vision_section(media_path:str, max_frames:int=16)->dict:
    ext = Path(media_path).suffix.lower()
    stem= Path(media_path).stem

    if ext in {".mp4",".avi",".mov",".mkv"}:
        cap=cv2.VideoCapture(media_path); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs=np.linspace(0,tot-1,min(max_frames,tot),dtype=int)
        scores,hms=[],[]
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ret,fr=cap.read()
            if not ret: continue
            p,hm=_infer_image(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
            scores.append(p); hms.append(hm)
        cap.release()
        conf=float(np.mean(scores)) if scores else 0.5
        avg_hm=np.mean(hms,0) if hms else np.zeros((IMG_SIZE,)*2)
    elif ext in {".jpg",".jpeg",".png",".bmp",".webp"}:
        conf,avg_hm=_infer_image(Image.open(media_path).convert("RGB")); scores=[conf]
    else:
        return {"model":"clip-vit-large(6-datasets)","confidence":None,
                "heatmap_path":None,"top_regions_flagged":[],"frame_scores":[]}

    hm_out=f"{OUTPUT_DIR}/heatmap_{stem}.png"
    cv2.imwrite(hm_out,cv2.applyColorMap((avg_hm*255).astype(np.uint8),cv2.COLORMAP_JET))
    g=avg_hm.reshape(3,-1,3,-1).mean((1,3))
    REG=["top-left","top-center","top-right","mid-left","mid-center","mid-right",
         "bot-left","bot-center","bot-right"]
    top=[REG[i] for i in g.flatten().argsort()[::-1][:3]]
    return {"model":"clip-vit-large-finetuned(6-datasets)",
            "confidence":round(conf*100,2),"heatmap_path":hm_out,
            "top_regions_flagged":top,"frame_scores":[round(s*100,2) for s in scores],
            "training_datasets":v_cfg.get("training_datasets",[])}


# ── CELL 6 : Audio Predictor ──────────────────────────────────────────────────
"""## 🎙️ Audio Modality Predictor"""
ATTACK_FINGERPRINTS = {
    (0.00,0.30): "likely genuine speech",
    (0.30,0.50): "borderline — possible compression artifact",
    (0.50,0.70): "probable synthetic vocoder (WaveFake-class)",
    (0.70,0.85): "strong TTS signal — neural vocoder detected",
    (0.85,1.00): "high-confidence GAN/diffusion voice clone",
}

def classify_attack(conf:float)->str:
    for (lo,hi),desc in ATTACK_FINGERPRINTS.items():
        if lo<=conf<hi: return desc
    return "unknown"

def predict_audio_section(media_path:str)->dict:
    stem=Path(media_path).stem; ext=Path(media_path).suffix.lower()
    ap=media_path
    if ext in {".mp4",".avi",".mov",".mkv"}:
        ap=f"/content/tmp_{stem}.wav"
        subprocess.run(["ffmpeg","-y","-i",media_path,
                        "-ac","1","-ar","16000",ap],capture_output=True)
        if not os.path.exists(ap):
            return {"model":"wavlm-large-melcnn(5-datasets)","confidence":None,
                    "eer_score":None,"feature_breakdown":[],"spectrogram_path":None,
                    "attack_type_prediction":"N/A (audio extraction failed)"}
    try:
        wave=load_audio(ap)
    except Exception:
        return {"model":"wavlm-large-melcnn(5-datasets)","confidence":None,
                "eer_score":None,"feature_breakdown":[],"spectrogram_path":None,
                "attack_type_prediction":"N/A (load error)"}

    wav_t=torch.from_numpy(wave).unsqueeze(0).to(DEVICE)
    mel  =wave_to_mel(wave).unsqueeze(0)  # (1, N_MELS, T)
    with torch.no_grad(),autocast(enabled=True):
        logits,_=audio_model(wav_t,mel)
    probs=F.softmax(logits,-1)[0].cpu().numpy(); conf=float(probs[1])

    # Spectrogram save
    spec_out=f"{OUTPUT_DIR}/spec_{stem}.png"
    lm=wave_to_mel(wave).cpu().numpy()
    fig,ax=plt.subplots(figsize=(10,3))
    ax.imshow(lm,origin="lower",aspect="auto",cmap="magma")
    ax.set(xlabel="Frame",ylabel=f"Mel Bin (N={N_MELS})",
           title=f"Log-Mel Spectrogram — {Path(media_path).name}")
    plt.colorbar(ax.images[0],ax=ax,format="%.0f dB")
    fig.savefig(spec_out,dpi=120,bbox_inches="tight"); plt.close(fig)

    # Segment breakdown
    n_seg=8; seg_len=len(wave)//n_seg; breakdown=[]
    for si in range(n_seg):
        seg=np.pad(wave[si*seg_len:(si+1)*seg_len],
                   (0,max(0,MAX_SAMPLES-seg_len)))[:MAX_SAMPLES]
        st =torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        mt =wave_to_mel(seg).unsqueeze(0)
        with torch.no_grad(),autocast(enabled=True):
            lg2,_=audio_model(st,mt)
        p2=F.softmax(lg2,-1)[0,1].item()
        breakdown.append({"segment":si+1,
                          "start_ms":round(si*seg_len/SAMPLE_RATE*1000),
                          "end_ms":  round((si+1)*seg_len/SAMPLE_RATE*1000),
                          "fake_prob":round(p2*100,2)})

    eer_approx=abs(conf-(1-conf))*50
    return {"model":"wavlm-large-melcnn-finetuned(5-datasets)",
            "confidence":round(conf*100,2),"eer_score":round(eer_approx,2),
            "attack_type_prediction":classify_attack(conf),
            "feature_breakdown":breakdown,"spectrogram_path":spec_out,
            "training_datasets":a_cfg.get("training_datasets",[])}


# ── CELL 7 : Combined Inference (Full Extended JSON) ──────────────────────────
"""## 🔮 Full Multimodal Analysis — Extended JSON Output"""

def is_visual(path:str)->bool:
    return Path(path).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",
                                          ".mp4",".avi",".mov",".mkv"}

def is_audio_capable(path:str)->bool:
    return Path(path).suffix.lower() in {".wav",".flac",".mp3",".ogg",".m4a",
                                          ".mp4",".avi",".mov",".mkv"}


def analyze_media(media_path:str, max_frames:int=16)->dict:
    """
    Full multimodal deepfake analysis.

    Extended JSON schema:
    {
      "verdict":          "FAKE" | "REAL",
      "confidence_score": float (0-100),
      "vision": {
        "model":                 str,
        "confidence":            float,
        "heatmap_path":          str,
        "top_regions_flagged":   list[str],
        "frame_scores":          list[float],
        "training_datasets":     list[str]
      },
      "audio": {
        "model":                   str,
        "confidence":              float,
        "eer_score":               float,
        "attack_type_prediction":  str,
        "feature_breakdown":       list[dict],
        "spectrogram_path":        str,
        "training_datasets":       list[str]
      },
      "combined_score": float,
      "fusion_weights":  dict,
      "analysis_metadata": {
        "media_path":             str,
        "media_type":             str,
        "timestamp":              str,
        "vision_datasets_count":  int,
        "audio_datasets_count":   int
      }
    }
    """
    import datetime
    t_start = time.time()
    print(f"\n{'━'*60}")
    print(f" 🔍 Analyzing: {Path(media_path).name}")
    print(f"{'━'*60}")

    v_result, a_result = {}, {}
    v_conf, a_conf     = None, None

    if is_visual(media_path):
        print("  🖼  Vision model running …")
        v_result = predict_vision_section(media_path, max_frames)
        v_conf   = v_result.get("confidence")
        print(f"     FAKE confidence: {v_conf}%")

    if is_audio_capable(media_path):
        print("  🎙  Audio model running …")
        a_result = predict_audio_section(media_path)
        a_conf   = a_result.get("confidence")
        print(f"     FAKE confidence: {a_conf}%  | Attack: {a_result.get('attack_type_prediction','?')}")

    # ── Weighted fusion ──────────────────────────────────────────────────────
    scores, total_w = [], 0.
    if v_conf is not None:
        w = FUSION_W["clip_vit"]
        scores.append(v_conf * w); total_w += w
    if a_conf is not None:
        w = FUSION_W["wavlm"] + FUSION_W["mel_cnn"]
        scores.append(a_conf * w); total_w += w

    combined = sum(scores)/total_w if total_w>0 else 50.0
    verdict  = "FAKE" if combined >= 50.0 else "REAL"

    result = {
        "verdict":          verdict,
        "confidence_score": round(combined, 2),
        "vision":           v_result,
        "audio":            a_result,
        "combined_score":   round(combined, 2),
        "fusion_weights":   FUSION_W,
        "analysis_metadata": {
            "media_path":            media_path,
            "media_type":            "video" if Path(media_path).suffix.lower()
                                     in {".mp4",".avi",".mov",".mkv"} else
                                     "audio" if Path(media_path).suffix.lower()
                                     in {".wav",".flac",".mp3"} else "image",
            "timestamp":             datetime.datetime.now().isoformat(),
            "inference_time_s":      round(time.time()-t_start, 2),
            "vision_datasets_count": len(v_cfg.get("training_datasets",[])),
            "audio_datasets_count":  len(a_cfg.get("training_datasets",[])),
        }
    }

    out_path = f"{OUTPUT_DIR}/result_{Path(media_path).stem}.json"
    json.dump(result, open(out_path,"w"), indent=2)

    print(f"\n  {'🔴 FAKE' if verdict=='FAKE' else '🟢 REAL'} — {combined:.1f}% fake confidence")
    print(f"  📄 Saved → {out_path}")
    print(f"  ⏱  {result['analysis_metadata']['inference_time_s']}s")
    return result


# ── CELL 8 : Visual Report Generator ─────────────────────────────────────────
"""## 📊 Visual Forensic Report"""
def generate_report(result:dict):
    """Generate a full-page visual forensic report from analyze_media() result."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    title_color  = "#e74c3c" if result["verdict"]=="FAKE" else "#2ecc71"
    verdict_icon = "🔴 DEEPFAKE DETECTED" if result["verdict"]=="FAKE" else "🟢 AUTHENTIC"

    fig.suptitle(f"{verdict_icon}  —  {result['confidence_score']:.1f}% confidence",
                 fontsize=20, color=title_color, fontweight="bold", y=0.98)

    def style(ax, title):
        ax.set_facecolor('#16213e'); ax.set_title(title,color="#aaaaaa",fontsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor('#333355')
        ax.tick_params(colors="#aaaaaa"); ax.yaxis.label.set_color("#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")

    # ── Panel 1 : GradCAM heatmap ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    hm_path = result["vision"].get("heatmap_path")
    if hm_path and os.path.exists(hm_path):
        ax1.imshow(cv2.cvtColor(cv2.imread(hm_path), cv2.COLOR_BGR2RGB))
    else:
        ax1.text(0.5,0.5,"No heatmap",ha="center",va="center",color="white")
    ax1.axis("off"); ax1.set_title("GradCAM — Face Heatmap",color="#aaaaaa",fontsize=9)

    # ── Panel 2 : Frame scores ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fs  = result["vision"].get("frame_scores",[])
    if fs:
        ax2.bar(range(len(fs)), fs, color=[("#e74c3c" if s>=50 else "#3498db") for s in fs])
        ax2.axhline(50, color="white", linestyle="--", linewidth=0.8)
        ax2.set(ylim=(0,100), xlabel="Frame", ylabel="FAKE %")
    style(ax2, "Frame-level Scores")

    # ── Panel 3 : Spectrogram ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sp  = result["audio"].get("spectrogram_path")
    if sp and os.path.exists(sp):
        ax3.imshow(cv2.cvtColor(cv2.imread(sp), cv2.COLOR_BGR2RGB), aspect="auto")
    else:
        ax3.text(0.5,0.5,"No spectrogram",ha="center",va="center",color="white")
    ax3.axis("off"); ax3.set_title("Log-Mel Spectrogram",color="#aaaaaa",fontsize=9)

    # ── Panel 4 : Audio segment breakdown ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    bd  = result["audio"].get("feature_breakdown",[])
    if bd:
        segs   = [b["segment"] for b in bd]
        fprobs = [b["fake_prob"] for b in bd]
        ax4.bar(segs, fprobs, color=[("#e74c3c" if p>=50 else "#3498db") for p in fprobs])
        ax4.axhline(50,color="white",linestyle="--",linewidth=0.8)
        ax4.set(xlabel="Segment",ylabel="FAKE %",ylim=(0,100))
    style(ax4, "Audio Segment Analysis — WavLM+MelCNN")

    # ── Panel 5 : Confidence gauge ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    comb   = result["combined_score"]
    v_c    = result["vision"].get("confidence",0) or 0
    a_c    = result["audio"].get("confidence",0) or 0
    labels = ["Vision\n(CLIP ViT-L)","Audio\n(WavLM+Mel)","Combined"]
    values = [v_c, a_c, comb]
    colors = [("#e74c3c" if v>=50 else "#3498db") for v in values]
    bars   = ax5.barh(labels, values, color=colors)
    ax5.set(xlim=(0,100),xlabel="FAKE Confidence %")
    for bar,val in zip(bars,values):
        ax5.text(val+1,bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%",va="center",color="white",fontsize=8)
    style(ax5, "Model Confidence Comparison")

    # ── Panel 6 : Metadata text panel ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off"); ax6.set_facecolor('#16213e')
    meta = result.get("analysis_metadata",{})
    v_ds = result["vision"].get("training_datasets",[])
    a_ds = result["audio"].get("training_datasets",[])
    atk  = result["audio"].get("attack_type_prediction","N/A")
    top  = result["vision"].get("top_regions_flagged",[])
    info = (f"  File: {meta.get('media_path','?')}   |   "
            f"Type: {meta.get('media_type','?').upper()}   |   "
            f"Inference: {meta.get('inference_time_s','?')}s   |   "
            f"Timestamp: {meta.get('timestamp','?')[:19]}\n\n"
            f"  Vision datasets ({len(v_ds)}): {', '.join(v_ds) or 'N/A'}\n"
            f"  Audio  datasets ({len(a_ds)}): {', '.join(a_ds) or 'N/A'}\n\n"
            f"  Attack signature: {atk}\n"
            f"  Top flagged regions: {', '.join(top) or 'N/A'}\n"
            f"  EER estimate: {result['audio'].get('eer_score','N/A')}%")
    ax6.text(0.01, 0.95, info, transform=ax6.transAxes, va="top",
             color="#cccccc", fontsize=8.5, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",facecolor="#0f0f1a",alpha=0.8))

    report_path = f"{OUTPUT_DIR}/forensic_report_{Path(meta.get('media_path','media')).stem}.png"
    fig.savefig(report_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig)
    print(f"✅ Forensic report → {report_path}")
    return report_path


# ── CELL 9 : Interactive Analysis ────────────────────────────────────────────
"""## ▶️ Upload & Analyze"""
from google.colab import files as colab_files

print("📁 Upload any media file (image / video / audio):")
uploaded = colab_files.upload()

if uploaded:
    for fname in uploaded:
        result = analyze_media(f"/content/{fname}")
        print(json.dumps(result, indent=2))
        report = generate_report(result)
else:
    print("No file uploaded. Run manually:")
    print('  result = analyze_media("/path/to/file.mp4")')
    print('  generate_report(result)')


# ── CELL 10 : Batch Analysis ──────────────────────────────────────────────────
"""## 📦 Batch Analysis"""
def batch_analyze(directory:str, exts=None)->list:
    if exts is None:
        exts={".jpg",".jpeg",".png",".mp4",".avi",".mov",".wav",".flac",".mp3"}
    files = [f for f in Path(directory).iterdir() if f.suffix.lower() in exts]
    results=[]
    for fp in files:
        try:
            r=analyze_media(str(fp)); r["file"]=str(fp); results.append(r)
        except Exception as e:
            print(f"  ⚠️  {fp.name}: {e}")
    # Summary table
    n_fake=sum(1 for r in results if r["verdict"]=="FAKE")
    print(f"\n{'━'*50}")
    print(f"  Batch Summary: {len(results)} files")
    print(f"  FAKE : {n_fake}  ({100*n_fake/max(1,len(results)):.1f}%)")
    print(f"  REAL : {len(results)-n_fake}")
    print(f"{'━'*50}")
    # Save summary
    summary_path = f"{OUTPUT_DIR}/batch_summary.json"
    json.dump(results, open(summary_path,"w"), indent=2)
    print(f"  📄 Batch summary → {summary_path}")
    return results

# Uncomment to run batch:
# results = batch_analyze("/content/drive/MyDrive/test_media/")
print("\n✅ Combined inference notebook complete!")
print("   Call analyze_media('/path/to/file') → full forensic JSON + visual report")
