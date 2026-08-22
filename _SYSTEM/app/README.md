# 🎨 IG-AUTOMATIK v2.0

Professional Instagram content grading engine. Drop photos, get beautiful results.

---

## ⚡ 3-Second Start

```bash
python watch.py
```

Then drop photos in `1_EINGANG/` → Results appear in `2_FERTIG/`

---

## 📁 Your Project Structure

```
IG-AUTOMATIK/
├─ 🚀 main.py           ← Process all files
├─ 🚀 watch.py          ← Auto-watch folder (recommended)
├─ 📖 README.md         ← You are here
├─ ⚡ QUICK_START.md    ← 2-minute guide
│
├─ 📥 1_EINGANG/        ← DROP YOUR FILES HERE
├─ 📤 2_FERTIG/         ← YOUR RESULTS HERE
│  ├─ POSTS/            (Instagram posts 4:5)
│  ├─ STORIES/          (Instagram stories 9:16)
│  └─ REELS/            (Instagram reels 9:16)
├─ 📦 3_ARCHIV/         ← Backups (safe, never deleted)
└─ ⚙️ _SYSTEM/          ← Config & logs
```

---

## 🎯 How to Use

### Option 1: Auto-Watch (Recommended)
```bash
python watch.py
```
Keeps running. Drop photos, they auto-process.

### Option 2: Batch Process
```bash
python main.py              # Process all
python main.py --limit 5    # Process first 5
```

---

## 📊 What You Get

Each photo produces **2 variants**:

| Variant | Look | Use For |
|---------|------|---------|
| **A** Natural | Clean, balanced | Professional photos |
| **B** Cinematic | Dramatic, vibrant | Instagram content ⭐ |

---

## 🎨 Example Output

```
2_FERTIG/POSTS/
├─ photo_A.jpg           (Natural variant)
├─ photo_B.jpg           (Cinematic variant - better for IG!)
├─ photo_A_archiv.png    (Full resolution)
├─ photo_B_archiv.png    (Full resolution)
└─ POSTS_manifest.json   (Processing info)
```

---

## 📸 Supported Files

**Photos**: JPG, PNG, GIF, WebP, DNG, NEF, CR2, ARW, HEIC, TIFF, BMP  
**Videos**: MP4, MOV, MKV, WebM, AVI, 3GP, M4V

---

## ⚙️ Configuration

Edit `_SYSTEM/config/config.json`:

```json
{
  "export_quality": 98,
  "output_width_post": 1080,
  "produce_formats": ["POSTS", "STORIES"],
  "produce_archives": true,
  "auto_move_sources": true
}
```

---

## 🎛️ Tune the Look

Edit `ig_automatik/config/constants.py`:

```python
# More saturated colors
CINEMATIC_SAT_BASE = 6  # Try 4-8

# Stronger contrast
CINEMATIC_CONTRAST_BASE = 8  # Try 6-10

# Adjust teal-orange effect
CINEMATIC_TEAL_ORANGE_ENHANCED = 2.2  # Try 1.8-3.0
```

Then run `python main.py` - changes apply immediately.

---

## 🔧 Setup (First Time)

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Enable AI scene analysis
# Edit .env and add: OPENROUTER_API_KEY=sk-or-v1-...
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| No output | Check `1_EINGANG/` has photos with supported extensions |
| Module error | Run from project root: `cd IG-AUTOMATIK` |
| Slow processing | Normal: 2-5 sec/photo, 1-2 min/video |
| Want to see logs | Check `_SYSTEM/logs/ig_YYYYMMDD.jsonl` |

---

## 📖 More Help

- **Quick Start**: Read `QUICK_START.md` (2 minutes)
- **Full Guide**: Read `QUICK_START.md` then this file
- **Logs**: Check `_SYSTEM/logs/`

---

## ✨ Features

✅ 2 professional grading variants  
✅ Instagram-optimized sizes  
✅ Auto-watch or batch processing  
✅ Safe archiving (originals never deleted)  
✅ Raw photo support  
✅ Video support (9:16 reels)  
✅ Teal-orange cinematic effect  
✅ Adaptive saturation  

---

**Ready?** Run: `python watch.py` 🚀
