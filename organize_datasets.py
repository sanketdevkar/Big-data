"""
organize_datasets.py — Run this AFTER downloading datasets

Usage:
  python organize_datasets.py --mode vision    (PC 1)
  python organize_datasets.py --mode audio     (PC 2)
"""
import os, shutil, random, argparse, json
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Organize deepfake datasets")
parser.add_argument("--mode", choices=["vision","audio"], required=True,
                    help="vision (PC1) or audio (PC2)")
parser.add_argument("--base", default=r"C:\deepfake_training",
                    help="Base training directory")
parser.add_argument("--val-ratio", type=float, default=0.15,
                    help="Fraction of data for validation (default 0.15)")
args = parser.parse_args()

BASE      = Path(args.base)
VAL_RATIO = args.val_ratio
random.seed(42)

# ════════════════════════════════════════════════════════════
# VISION DATASETS  (PC 1)
# ════════════════════════════════════════════════════════════
if args.mode == "vision":
    IN_ROOT  = BASE / "datasets" / "vision"
    OUT_ROOT = BASE / "datasets" / "vision" / "organized"
    IMG_EXT  = {".jpg", ".jpeg", ".png"}
    stats    = {}

    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    def copy_images(src_dir, label, tag, limit=None):
        src  = Path(src_dir)
        if not src.exists():
            print(f"  ⚠️  Skipped (not found): {src}"); return
        files = []
        for ext in IMG_EXT:
            files += list(src.rglob(f"*{ext}"))
        if limit:
            files = files[:limit]
        random.shuffle(files)
        n_val = int(len(files) * VAL_RATIO)
        copied = 0
        for i, f in enumerate(tqdm(files, desc=f"  {tag}/{label}")):
            split = "val" if i < n_val else "train"
            uid   = f"{tag}_{i:07d}{f.suffix}"
            dst   = OUT_ROOT / split / label / uid
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        stats[f"{tag}_{label}"] = copied

    print("\n═══ Organizing Vision Datasets ═══")

    # 1. Celeb-DF v2
    celebdf = IN_ROOT / "Celeb-DF-v2"
    copy_images(celebdf / "YouTube-real",    "real", "celebdf")
    copy_images(celebdf / "Celeb-synthesis", "fake", "celebdf")

    # 2. DFDC — needs frame extraction first
    dfdc = IN_ROOT / "dfdc"
    if dfdc.exists():
        import cv2
        meta_files = list(dfdc.rglob("metadata.json"))
        print(f"\n  DFDC: found {len(meta_files)} video folders")
        for meta_file in tqdm(meta_files, desc="  DFDC extracting frames"):
            with open(meta_file) as f:
                meta = json.load(f)
            vid_dir = meta_file.parent
            for vid_name, info in meta.items():
                label   = "fake" if info.get("label","REAL") == "FAKE" else "real"
                vid_fp  = vid_dir / vid_name
                if not vid_fp.exists():
                    continue
                cap  = cv2.VideoCapture(str(vid_fp))
                tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if tot < 1:
                    cap.release(); continue
                # Sample 10 frames per video
                import numpy as np
                idxs = np.linspace(0, tot-1, min(10,tot), dtype=int)
                split = "val" if random.random() < VAL_RATIO else "train"
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ret, frame = cap.read()
                    if not ret: continue
                    out_p = OUT_ROOT/split/label/f"dfdc_{vid_fp.stem}_{fi:04d}.jpg"
                    if not out_p.exists():
                        cv2.imwrite(str(out_p), frame)
                cap.release()
    else:
        print("  ⚠️  DFDC not found — skipping (download with Kaggle)")

    # 3. WildDeepfake
    wdf = IN_ROOT / "WildDeepfake"
    copy_images(wdf / "real", "real", "wilddeepfake")
    copy_images(wdf / "fake", "fake", "wilddeepfake")

    # 4. FaceForensics++
    ffpp = IN_ROOT / "FF++"
    if ffpp.exists():
        import cv2, numpy as np
        for method in ["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]:
            vid_dir = ffpp/f"manipulated_sequences/{method}/c23/videos"
            if not vid_dir.exists(): continue
            for vid in tqdm(list(vid_dir.glob("*.mp4"))[:500], desc=f"  FF++/{method}"):
                cap = cv2.VideoCapture(str(vid))
                tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if tot < 1: cap.release(); continue
                idxs = np.linspace(0,tot-1,min(15,tot),dtype=int)
                split = "val" if random.random()<VAL_RATIO else "train"
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi))
                    ret,frame=cap.read()
                    if not ret: continue
                    out_p=OUT_ROOT/split/"fake"/f"ffpp_{method}_{vid.stem}_{fi:04d}.jpg"
                    if not out_p.exists(): cv2.imwrite(str(out_p),frame)
                cap.release()
        # Real sequences
        for real_dir in ["original_sequences/youtube/raw/videos",
                         "original_sequences/actors/raw/videos"]:
            rd = ffpp/real_dir
            if not rd.exists(): continue
            for vid in tqdm(list(rd.glob("*.mp4"))[:300], desc="  FF++/real"):
                cap=cv2.VideoCapture(str(vid)); tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if tot<1: cap.release(); continue
                idxs=np.linspace(0,tot-1,min(10,tot),dtype=int)
                split="val" if random.random()<VAL_RATIO else "train"
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ret,frame=cap.read()
                    if not ret: continue
                    out_p=OUT_ROOT/split/"real"/f"ffpp_real_{vid.stem}_{fi:04d}.jpg"
                    if not out_p.exists(): cv2.imwrite(str(out_p),frame)
                cap.release()
    else:
        print("  ⚠️  FaceForensics++ not found — skipping")

    print("\n═══ Vision Dataset Summary ═══")
    for split in ["train","val"]:
        for label in ["real","fake"]:
            n = len(list((OUT_ROOT/split/label).glob("*")))
            print(f"  {split}/{label}: {n:,} images")

# ════════════════════════════════════════════════════════════
# AUDIO DATASETS  (PC 2)
# ════════════════════════════════════════════════════════════
elif args.mode == "audio":
    IN_ROOT  = BASE / "datasets" / "audio"
    OUT_ROOT = BASE / "datasets" / "audio" / "organized"
    AUDIO_EXT= {".wav", ".flac"}
    stats    = {}

    for split in ["train","val"]:
        for label in ["real","fake"]:
            (OUT_ROOT/split/label).mkdir(parents=True, exist_ok=True)

    def copy_audio(src_dir, label, tag, limit=None):
        src = Path(src_dir)
        if not src.exists():
            print(f"  ⚠️  Skipped (not found): {src}"); return
        files = []
        for ext in AUDIO_EXT:
            files += list(src.rglob(f"*{ext}"))
        if limit:
            files = files[:limit]
        random.shuffle(files)
        n_val  = int(len(files)*VAL_RATIO)
        copied = 0
        for i,f in enumerate(tqdm(files, desc=f"  {tag}/{label}")):
            split  = "val" if i<n_val else "train"
            dst    = OUT_ROOT/split/label/f"{tag}_{i:07d}{f.suffix}"
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        stats[f"{tag}_{label}"] = copied

    print("\n═══ Organizing Audio Datasets ═══")

    # 1. ASVspoof 2019 LA (via protocol file)
    asv19 = IN_ROOT / "ASVspoof2019_LA"
    if asv19.exists():
        proto_dir = asv19 / "LA" / "ASVspoof2019_LA_cm_protocols"
        for pf in (asv19.rglob("*.txt") if not proto_dir.exists()
                   else proto_dir.glob("*.txt")):
            if "train" not in pf.name.lower(): continue
            print(f"  ASVspoof2019 — reading {pf.name}")
            for line in tqdm(open(pf).read().splitlines(), desc="  ASVspoof2019"):
                parts = line.split()
                if len(parts)<5: continue
                fname, lbl = parts[1], parts[4].upper()
                label = "real" if lbl=="BONAFIDE" else "fake"
                for ext in [".flac",".wav"]:
                    cands = list(asv19.rglob(f"{fname}{ext}"))
                    if cands:
                        split = "val" if random.random()<VAL_RATIO else "train"
                        dst   = OUT_ROOT/split/label/f"asv19_{fname}{ext}"
                        if not dst.exists(): shutil.copy2(cands[0], dst)
                        break
    else:
        print("  ⚠️  ASVspoof 2019 not found. Download from:")
        print("       https://datashare.ed.ac.uk/handle/10283/3336")

    # 2. ASVspoof 2021 LA
    asv21 = IN_ROOT / "ASVspoof2021_LA"
    if asv21.exists():
        for pf in asv21.rglob("*.txt"):
            if "train" not in pf.name.lower(): continue
            for line in tqdm(open(pf).read().splitlines(), desc="  ASVspoof2021"):
                parts = line.split()
                if len(parts)<5: continue
                fname, lbl = parts[1], parts[4].upper()
                label = "real" if lbl=="BONAFIDE" else "fake"
                for ext in [".flac",".wav"]:
                    cands = list(asv21.rglob(f"{fname}{ext}"))
                    if cands:
                        split = "val" if random.random()<VAL_RATIO else "train"
                        dst   = OUT_ROOT/split/label/f"asv21_{fname}{ext}"
                        if not dst.exists(): shutil.copy2(cands[0], dst)
                        break
    else:
        print("  ⚠️  ASVspoof 2021 not found — skipping")

    # 3. WaveFake
    wavefake = IN_ROOT / "WaveFake"
    if wavefake.exists():
        # Reals: LJSpeech
        for ld in ["LJSpeech-1.1/wavs","real","wavs"]:
            ld_p = wavefake/ld
            if ld_p.exists():
                copy_audio(ld_p, "real", "wavefake_real", limit=20000)
                break
        # Fakes: all vocoder subdirs
        for vd in wavefake.iterdir():
            if vd.is_dir() and vd.name not in ["LJSpeech-1.1","real","wavs"]:
                copy_audio(vd, "fake", f"wavefake_{vd.name}", limit=15000)
    else:
        print("  ⚠️  WaveFake not found. Download from:")
        print("       https://zenodo.org/record/5642694")

    print("\n═══ Audio Dataset Summary ═══")
    for split in ["train","val"]:
        for label in ["real","fake"]:
            n = len(list((OUT_ROOT/split/label).glob("*")))
            print(f"  {split}/{label}: {n:,} files")

print("\n✅ Organization complete!")
print(f"   Output: {OUT_ROOT}")
