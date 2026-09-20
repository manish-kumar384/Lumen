# Lumen · Photo Studio

A calm, off-white, all-in-one photo toolkit built with Streamlit.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tools (16)
| Group | Tools |
|---|---|
| Optimise | Compress (quality or target file size) · Resize (pixels, %, social presets, quality, max file size, DPI) · Crop (ratio / free / circle / rounded) · Rotate & Flip (with straighten) |
| Edit | Photo Editor (14 adjustments + 15 filters, before/after split) · Enhance (auto-fix, low-light, denoise, sharpen, clarity, 2-4x upscale) · Remove Background (AI, smart cut-out, colour key; transparent / colour / gradient / blur / image backdrops) · Watermark (text or logo, tiled) · Text & Meme · Blur & Pixelate (auto face detection) · Borders & Corners |
| Create | Collage |
| Convert | Convert (JPG, PNG, WEBP, GIF, BMP, TIFF, ICO, PDF) · Images to PDF · PDF to Images |
| Inspect | Metadata (EXIF, GPS warning, strip) & colour Palette |

Most tools accept several files at once and return a ZIP.

## Files
- `app.py`: interface, theme and tool screens
- `imgtools.py`: all image processing (no Streamlit, easy to test)
- `.streamlit/config.toml`: off-white theme

## Notes
- **Background removal:** "Smart cut-out" is built in and works well on plain backdrops. For busy scenes, hair and fur,
  install the optional AI engine (`pip install "rembg[cpu]"`); a new "AI" option then appears automatically.
- **Large photos:** denoise and 3-4x upscale are the slowest operations; previews use a downsized copy so sliders stay
  responsive, and the full-resolution file is made when you press the export button.
- Requires Streamlit 1.50 or newer.
