# ⚡ Quick Start Guide

## What You Have

A professional Instagram content grading pipeline with:
- ✨ 2 grading variants (Natural & Cinematic)
- 📁 Organized code structure
- 🚀 2 simple commands to use it

---

## Your 2 Commands

### 1️⃣ Batch Process (One-time)
```bash
python main.py
```
Processes all files in `1_EINGANG/` folder

### 2️⃣ Auto-Watch (Recommended)
```bash
python watch.py
```
Watches folder and processes new files automatically

---

## Simple Workflow

```
1. Drop files in:     1_EINGANG/
                            ↓
2. Run:               python watch.py  (keeps running)
                            ↓
3. Check results in:  2_FERTIG/
```

That's it! ✅

---

## File Organization

```
IG-AUTOMATIK/
├─ main.py              ← Run this (batch)
├─ watch.py             ← Or this (auto)
├─ README.md            ← Full documentation
├─ ig_automatik/        ← Internal (organized code)
├─ 1_EINGANG/           ← Put files here
├─ 2_FERTIG/            ← Get results here
│  ├─ POSTS/
│  ├─ STORIES/
│  └─ REELS/
├─ 3_ARCHIV/            ← Backup of originals
└─ _SYSTEM/
   ├─ config/
   └─ logs/
```

---

## Results

Each photo creates 2 versions:

| Variant | Look | Best For |
|---------|------|----------|
| **A** (Natural) | Balanced, clean | Photos that look good as-is |
| **B** (Cinematic) | Dramatic, vibrant | Instagram-ready, pop, color |

---

## Setup (First Time Only)

1. **Install Python packages** (if not already done):
   ```bash
   pip install numpy opencv-python Pillow pillow-heif watchdog
   ```

2. **Optional: AI Analysis**
   - Edit `.env` file
   - Add your OpenRouter API key
   - Let AI analyze each photo

3. **That's it!** Just run `python watch.py`

---

## Check Results

| Folder | What's Inside |
|--------|---------------|
| `2_FERTIG/POSTS/` | Instagram posts (4:5) |
| `2_FERTIG/STORIES/` | Instagram stories (9:16) |
| `2_FERTIG/REELS/` | Instagram reels (9:16 video) |

Each has:
- `photo_A.jpg` - Natural variant
- `photo_B.jpg` - **Cinematic variant** (better for IG)
- `*_archiv.png` - Full resolution backup

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No output files | Check `1_EINGANG/` has files with supported extensions |
| "Module not found" | Run from project folder: `cd IG-AUTOMATIK` |
| Files not processing | Make sure `watch.py` is still running (or run `python main.py`) |
| Slow processing | Large videos take time - this is normal |

---

## Tips

✅ Use `python watch.py` to let it run in background  
✅ Check `2_FERTIG/` for results (usually 5-10 seconds per photo)  
✅ Try variant B first (better for Instagram)  
✅ All originals safe in `3_ARCHIV/` (never deleted)

---

## Next Steps

1. **Now**: Run `python watch.py`
2. **Drop** a photo in `1_EINGANG/`
3. **Wait** 5-10 seconds
4. **Check** `2_FERTIG/` for your graded photos!

---

**Need more info?** Read `README.md`

Happy grading! 🎨
