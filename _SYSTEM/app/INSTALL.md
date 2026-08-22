# ⚙️ Installation & Setup

## 1️⃣ Install Python Packages

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` - Image processing
- `opencv-python` - Photo grading
- `Pillow` - Image formats
- `pillow-heif` - iPhone HEIC photos
- `watchdog` - Folder monitoring

---

## 2️⃣ (Optional) AI Scene Analysis

To use AI-powered scene analysis:

1. Get API key from [OpenRouter](https://openrouter.ai)
2. Create/edit `.env` file in project root
3. Add your key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   OPENROUTER_MODEL=google/gemini-2.5-flash-lite
   ```

---

## 3️⃣ Run!

**Auto-watch mode** (recommended):
```bash
python watch.py
```

**Batch mode**:
```bash
python main.py
```

---

## 📁 Folders Created Automatically

If they don't exist, they'll be created:
- `1_EINGANG/` - Input folder
- `2_FERTIG/` - Output folder
- `3_ARCHIV/` - Archive folder
- `_SYSTEM/logs/` - Log files

---

## ✅ Test It

1. Run: `python watch.py`
2. Drop a photo in `1_EINGANG/`
3. Wait 5-10 seconds
4. Check `2_FERTIG/` for results

---

**Done!** 🎉
