"""imgtools.py - image-processing helpers for Lumen Photo Studio.

Pure Pillow / NumPy / OpenCV. No Streamlit imports here, so everything can be tested on its own.
"""
from __future__ import annotations

import importlib.util
import io
import math
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

try:  # optional: iPhone HEIC/HEIF support
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

LANCZOS = Image.Resampling.LANCZOS
BICUBIC = Image.Resampling.BICUBIC
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
FORMATS = {  # label: (pillow format, extension, mime)
    "JPG": ("JPEG", "jpg", "image/jpeg"),
    "PNG": ("PNG", "png", "image/png"),
    "WEBP": ("WEBP", "webp", "image/webp"),
    "GIF": ("GIF", "gif", "image/gif"),
    "BMP": ("BMP", "bmp", "image/bmp"),
    "TIFF": ("TIFF", "tiff", "image/tiff"),
    "ICO": ("ICO", "ico", "image/x-icon"),
    "PDF": ("PDF", "pdf", "application/pdf"),
}
LOSSY = ("JPG", "WEBP")
EXT_TO_LABEL = {"jpg": "JPG", "jpeg": "JPG", "png": "PNG", "webp": "WEBP", "gif": "PNG",
                "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF", "heic": "JPG", "heif": "JPG"}
INPUT_TYPES = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"] + (["heic", "heif"] if HEIF_OK else [])


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def fmt_bytes(n: float) -> str:
    n = float(n)
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} GB"


def out_name(name: str, suffix: str, label: str) -> str:
    return f"{Path(name).stem}{suffix}.{FORMATS[label][1]}"


def make_zip(files: Sequence[tuple[str, bytes]]) -> bytes:
    buf, seen = io.BytesIO(), {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            if name in seen:
                seen[name] += 1
                p = Path(name)
                name = f"{p.stem}_{seen[name]}{p.suffix}"
            else:
                seen[name] = 0
            z.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Loading / basic helpers
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> Image.Image:
    """Decode bytes into an upright RGB / RGBA image (first frame of animations)."""
    img = Image.open(io.BytesIO(data))
    try:
        img.seek(0)
    except Exception:
        pass
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGB", "RGBA"):
        return img
    if img.mode in ("P", "LA", "PA") or "transparency" in img.info:
        return img.convert("RGBA")
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img).astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6) * 255
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def has_alpha(img: Image.Image) -> bool:
    return img.mode == "RGBA" and img.getchannel("A").getextrema()[0] < 255


def flatten(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if img.mode == "RGBA":
        base = Image.new("RGB", img.size, tuple(bg[:3]))
        base.paste(img, mask=img.getchannel("A"))
        return base
    return img.convert("RGB")


def downsize(img: Image.Image, max_side: int = 1400) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), LANCZOS)


def _split_alpha(img: Image.Image):
    if img.mode == "RGBA":
        return img.convert("RGB"), img.getchannel("A")
    return img.convert("RGB"), None


def _merge_alpha(rgb: Image.Image, a):
    if a is None:
        return rgb
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


def _restore(out: Image.Image, had_alpha: bool) -> Image.Image:
    return out if had_alpha or out.mode == "RGB" else out.convert("RGB")


def on_checker(img: Image.Image) -> Image.Image:
    """Composite a transparent image on a light checkerboard so transparency is visible."""
    w, h = img.size
    tile = max(8, min(w, h) // 36)
    yy, xx = np.indices((h, w))
    board = np.where(((yy // tile) + (xx // tile)) % 2 == 0, 255, 234).astype(np.uint8)
    bg = Image.fromarray(board).convert("RGB")
    rgba = img.convert("RGBA")
    bg.paste(rgba, (0, 0), rgba.getchannel("A"))
    return bg


def split_compare(before: Image.Image, after: Image.Image, pos: float = 0.5) -> Image.Image:
    """Left = before, right = after, thin divider line."""
    a, b = before.convert("RGB"), after.convert("RGB").resize(before.size, LANCZOS)
    x = int(a.width * pos)
    out = a.copy()
    out.paste(b.crop((x, 0, a.width, a.height)), (x, 0))
    d = ImageDraw.Draw(out)
    lw = max(2, a.width // 400)
    d.rectangle((x - lw // 2, 0, x + lw // 2, a.height), fill=(255, 255, 255))
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode(img: Image.Image, label: str = "JPG", quality: int = 90, optimize: bool = True,
           dpi: int | None = None, bg=(255, 255, 255), ico_sizes: Sequence[int] | None = None,
           fast: bool = False) -> bytes:
    label = label.upper()
    fmt = FORMATS[label][0]
    buf = io.BytesIO()
    kw: dict = {}
    if dpi:
        kw["dpi"] = (int(dpi), int(dpi))
    if fmt == "JPEG":
        out = flatten(img, bg)
        kw.update(quality=int(quality), optimize=optimize, progressive=True)
        if quality >= 90:
            kw["subsampling"] = 0
    elif fmt == "WEBP":
        out = img if img.mode in ("RGB", "RGBA") else img.convert("RGB")
        kw.pop("dpi", None)
        kw.update(quality=int(quality), method=1 if fast else 4)
    elif fmt == "PNG":
        out = img
        kw.update(optimize=optimize)
    elif fmt == "GIF":
        out = flatten(img, bg).convert("P", palette=Image.Palette.ADAPTIVE)
        kw.pop("dpi", None)
    elif fmt == "BMP":
        out = flatten(img, bg)
    elif fmt == "TIFF":
        out = img
        kw["compression"] = "tiff_lzw"
    elif fmt == "ICO":
        sq = ImageOps.pad(img.convert("RGBA"), (256, 256), LANCZOS, color=(0, 0, 0, 0))
        sq.save(buf, format="ICO", sizes=[(s, s) for s in (ico_sizes or (16, 32, 48, 64, 128, 256))])
        return buf.getvalue()
    else:  # PDF
        out = flatten(img, bg)
        kw.pop("dpi", None)
        kw["resolution"] = float(dpi or 150)
    out.save(buf, format=fmt, **kw)
    return buf.getvalue()


def quantize(img: Image.Image, colors: int = 256) -> Image.Image:
    """Palette-reduce an image (keeps alpha). Returns a 'P' image ready to save as PNG."""
    if img.mode == "RGBA":
        return img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    return img.convert("RGB").quantize(colors=colors)


def _png_bytes(p_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    p_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_to_target(img: Image.Image, label: str, target_bytes: int, min_quality: int = 40,
                     allow_resize: bool = True, **kw) -> tuple[bytes, dict]:
    """Encode so the file is <= target_bytes. Lowers quality first, then dimensions.

    Returns (bytes, info) where info = {quality, colors, scale, size, met}.
    """
    label = label.upper()
    target = max(int(target_bytes), 1024)
    w, h = img.size
    info = {"quality": None, "colors": None, "scale": 1.0, "size": (w, h), "met": False}

    if label in LOSSY:
        cand, scale, floor = img, 1.0, b""
        for _ in range(14):
            lo, hi, best = min_quality, 95, None
            while lo <= hi:
                mid = (lo + hi) // 2
                data = encode(cand, label, mid, fast=True, **kw)
                if len(data) <= target:
                    best, lo = (data, mid), mid + 1
                else:
                    hi = mid - 1
            if best:
                final = encode(cand, label, best[1], **kw) if label == "WEBP" else best[0]
                if len(final) > target:
                    final = best[0]
                info.update(quality=best[1], scale=scale, size=cand.size, met=True)
                return final, info
            floor = encode(cand, label, min_quality, fast=True, **kw)
            info.update(quality=min_quality, scale=scale, size=cand.size)
            if not allow_resize or min(cand.size) <= 32:
                return floor, info
            scale *= max(0.35, min(0.92, math.sqrt(target / len(floor)) * 0.96))
            cand = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), LANCZOS)
        return floor, info

    # lossless formats: try as-is, then palette reduction (PNG), then shrink dimensions
    data = encode(img, label, **kw)
    if len(data) <= target:
        info["met"] = True
        return data, info
    if label == "PNG":
        for colors in (256, 128, 64, 32):
            data = _png_bytes(quantize(img, colors))
            info["colors"] = colors
            if len(data) <= target:
                info["met"] = True
                return data, info
    scale = 1.0
    for _ in range(14):
        if not allow_resize or min(img.size) * scale <= 32:
            break
        scale *= max(0.35, min(0.92, math.sqrt(target / len(data)) * 0.96))
        cand = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), LANCZOS)
        data = _png_bytes(quantize(cand, 256)) if label == "PNG" else encode(cand, label, **kw)
        info.update(scale=scale, size=cand.size)
        if len(data) <= target:
            info["met"] = True
            break
    return data, info


# ---------------------------------------------------------------------------
# Resize / crop / rotate
# ---------------------------------------------------------------------------
def fit_dims(size: tuple[int, int], width: int | None = None, height: int | None = None) -> tuple[int, int]:
    w, h = size
    if width and not height:
        height = round(h * width / w)
    elif height and not width:
        width = round(w * height / h)
    return max(1, int(width or w)), max(1, int(height or h))


def resize_to(img: Image.Image, w: int, h: int, mode: str = "stretch", bg=(255, 255, 255)) -> Image.Image:
    """mode: stretch | fit (inside, keep ratio) | fill (cover & crop) | pad (fit + pad to exact size)."""
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "fit":
        return ImageOps.contain(img, (w, h), LANCZOS)
    if mode == "fill":
        return ImageOps.fit(img, (w, h), LANCZOS)
    if mode == "pad":
        color = tuple(bg[:3]) if img.mode == "RGB" else (*bg[:3], 255)
        return ImageOps.pad(img, (w, h), LANCZOS, color=color)
    return img.resize((w, h), LANCZOS)


def aspect_box(size: tuple[int, int], ratio: tuple[float, float], zoom: float = 1.0,
               cx: float = 0.5, cy: float = 0.5) -> tuple[int, int, int, int]:
    """Largest box of the given aspect ratio inside `size`, shrunk by `zoom`, positioned by cx/cy (0..1)."""
    W, H = size
    r = ratio[0] / ratio[1]
    bw, bh = (H * r, H) if W / H > r else (W, W / r)
    bw, bh = bw / zoom, bh / zoom
    left, top = (W - bw) * cx, (H - bh) * cy
    return round(left), round(top), round(left + bw), round(top + bh)


def frac_box(size: tuple[int, int], fx: tuple[float, float], fy: tuple[float, float]) -> tuple[int, int, int, int]:
    W, H = size
    l, r = round(W * fx[0]), round(W * fx[1])
    t, b = round(H * fy[0]), round(H * fy[1])
    return l, t, max(r, l + 1), max(b, t + 1)


def draw_crop_overlay(img: Image.Image, box, dim: float = 0.6) -> Image.Image:
    base = img.convert("RGBA")
    shade = Image.new("RGBA", base.size, (18, 18, 18, int(255 * dim)))
    out = Image.alpha_composite(base, shade)
    out.paste(base.crop(box), box[:2])
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    for i in (1, 2):
        d.line([(x0 + (x1 - x0) * i / 3, y0), (x0 + (x1 - x0) * i / 3, y1)], fill=(255, 255, 255, 120), width=1)
        d.line([(x0, y0 + (y1 - y0) * i / 3), (x1, y0 + (y1 - y0) * i / 3)], fill=(255, 255, 255, 120), width=1)
    d.rectangle(box, outline=(255, 255, 255, 255), width=max(2, round(min(base.size) * 0.004)))
    return out.convert("RGB")


def round_corners(img: Image.Image, radius: float = 0.15, circle: bool = False) -> Image.Image:
    """radius is a fraction of the shorter side. circle=True crops to a centred circle."""
    img = img.convert("RGBA")
    if circle:
        s = min(img.size)
        img = ImageOps.fit(img, (s, s), LANCZOS)
    w, h = img.size
    m = Image.new("L", (w * 3, h * 3), 0)
    d = ImageDraw.Draw(m)
    if circle:
        d.ellipse((0, 0, w * 3 - 1, h * 3 - 1), fill=255)
    else:
        d.rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), radius=int(min(w, h) * radius * 3), fill=255)
    m = m.resize((w, h), LANCZOS)
    img.putalpha(ImageChops.multiply(img.getchannel("A"), m))
    return img


def rotate_flip(img: Image.Image, angle: float = 0, flip_h: bool = False, flip_v: bool = False,
                expand: bool = True, fill: str = "Transparent") -> Image.Image:
    """angle is clockwise degrees."""
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    if round(angle, 3) % 360:
        if fill == "Transparent":
            img = img.convert("RGBA")
            color = (0, 0, 0, 0)
        else:
            color = (255, 255, 255) if fill == "White" else (0, 0, 0)
            if img.mode == "RGBA":
                color = (*color, 255)
        img = img.rotate(-angle, resample=BICUBIC, expand=expand, fillcolor=color)
    return img


# ---------------------------------------------------------------------------
# Tone, colour and filters
# ---------------------------------------------------------------------------
def adjust(img: Image.Image, exposure=0, brightness=0, contrast=0, highlights=0, shadows=0, saturation=0,
           vibrance=0, warmth=0, tint=0, fade=0, vignette=0, grain=0, clarity=0, sharpness=0) -> Image.Image:
    """Every argument is -100..100 (0 = unchanged)."""
    if not any((exposure, brightness, contrast, highlights, shadows, saturation, vibrance, warmth, tint,
                fade, vignette, grain, clarity, sharpness)):
        return img
    rgb, a = _split_alpha(img)
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if exposure:
        x = x * (2.0 ** (exposure / 50.0))
    if brightness:
        x = x + brightness / 250.0
    if highlights or shadows:
        lum = np.clip((x @ LUMA)[..., None], 0, 1)
        x = x + (1 - lum) ** 2 * (shadows / 100.0) * 0.30 + lum ** 2 * (highlights / 100.0) * 0.30
    if contrast:
        c = 1.0 + (contrast / 100.0 if contrast > 0 else contrast / 125.0)
        x = (x - 0.5) * c + 0.5
    if saturation or vibrance:
        gray = (x @ LUMA)[..., None]
        if saturation:
            x = gray + (x - gray) * (1 + saturation / 100.0)
        if vibrance:
            spread = np.clip(x.max(-1, keepdims=True) - x.min(-1, keepdims=True), 0, 1)
            x = gray + (x - gray) * (1 + (vibrance / 100.0) * (1 - spread))
    if warmth:
        x[..., 0] *= 1 + 0.18 * warmth / 100.0
        x[..., 2] *= 1 - 0.18 * warmth / 100.0
    if tint:
        x[..., 1] *= 1 - 0.12 * tint / 100.0
    if fade:
        f = fade / 100.0
        x = x * (1 - 0.25 * f) + 0.12 * f
    if vignette:
        H, W = x.shape[:2]
        yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
        v = np.clip(np.sqrt(xx * xx + yy * yy) / math.sqrt(2), 0, 1) ** 2.2
        x = x * (1 - (vignette / 100.0) * 0.8 * v)[..., None]
    if grain:
        rng = np.random.default_rng(1234)
        x = x + rng.normal(0, grain / 100.0 * 0.09, x.shape[:2]).astype(np.float32)[..., None]
    out = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    long_side = max(out.size)
    if clarity > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=max(2.0, long_side * 0.012), percent=int(clarity * 1.3), threshold=0))
    elif clarity < 0:
        out = Image.blend(out, out.filter(ImageFilter.GaussianBlur(long_side * 0.004)), -clarity / 100 * 0.7)
    if sharpness > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=max(0.6, long_side / 1800), percent=int(sharpness * 2.2), threshold=1))
    elif sharpness < 0:
        out = Image.blend(out, out.filter(ImageFilter.GaussianBlur(max(0.6, long_side / 1500))), -sharpness / 100)
    return _merge_alpha(out, a)


FILTERS = ["None", "Grayscale", "Noir", "Sepia", "Vintage", "Warm", "Cool", "Fade", "Vivid", "Dramatic",
           "Cinematic", "Sketch", "Emboss", "Posterize", "Invert"]

_SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], dtype=np.float32)


def _sketch(rgb: Image.Image) -> Image.Image:
    g = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    k = max(3, int(max(rgb.size) * 0.008) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(255 - g, (k, k), 0)
    out = cv2.divide(g, np.clip(255 - blur, 1, 255).astype(np.uint8), scale=256)
    return Image.fromarray(out).convert("RGB")


def _filter_core(rgb: Image.Image, name: str) -> Image.Image:
    if name == "Grayscale":
        return ImageOps.grayscale(rgb).convert("RGB")
    if name == "Noir":
        g = ImageOps.autocontrast(ImageOps.grayscale(rgb), cutoff=2).convert("RGB")
        return adjust(g, contrast=35, vignette=25)
    if name == "Sepia":
        x = np.asarray(rgb, np.float32) @ _SEPIA.T
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if name == "Vintage":
        return adjust(rgb, warmth=28, fade=30, contrast=-8, saturation=-22, vignette=30, grain=12)
    if name == "Warm":
        return adjust(rgb, warmth=38, saturation=8, brightness=4)
    if name == "Cool":
        return adjust(rgb, warmth=-38, saturation=4)
    if name == "Fade":
        return adjust(rgb, fade=40, contrast=-14, saturation=-16)
    if name == "Vivid":
        return adjust(rgb, saturation=30, vibrance=30, contrast=14)
    if name == "Dramatic":
        return adjust(rgb, contrast=34, shadows=-22, highlights=-12, saturation=-12, clarity=45, vignette=25)
    if name == "Cinematic":
        x = np.asarray(rgb, np.float32) / 255.0
        lum = (x @ LUMA)[..., None]
        x = x + (1 - lum) * np.array([-0.05, 0.02, 0.07], np.float32) + lum * np.array([0.07, 0.02, -0.06], np.float32)
        out = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
        return adjust(out, contrast=16, saturation=-6, vignette=18)
    if name == "Sketch":
        return _sketch(rgb)
    if name == "Emboss":
        return rgb.filter(ImageFilter.EMBOSS)
    if name == "Posterize":
        return ImageOps.posterize(rgb, 3)
    if name == "Invert":
        return ImageOps.invert(rgb)
    return rgb


def apply_filter(img: Image.Image, name: str, intensity: float = 1.0) -> Image.Image:
    if name in ("None", "", None) or intensity <= 0:
        return img
    rgb, a = _split_alpha(img)
    out = _filter_core(rgb, name)
    if intensity < 1:
        out = Image.blend(rgb, out, float(intensity))
    return _merge_alpha(out, a)


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------
def clahe(img: Image.Image, clip: float = 2.0, grid: int = 8, blend: float = 1.0) -> Image.Image:
    rgb, a = _split_alpha(img)
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    boosted = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    lab[..., 0] = np.clip(lab[..., 0] * (1 - blend) + boosted * blend, 0, 255).astype(np.uint8)
    return _merge_alpha(Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), a)


def sharpen(img: Image.Image, amount: float = 50) -> Image.Image:
    if amount <= 0:
        return img
    rgb, a = _split_alpha(img)
    r = max(0.7, max(rgb.size) / 1600)
    return _merge_alpha(rgb.filter(ImageFilter.UnsharpMask(radius=r, percent=int(amount * 2.2), threshold=2)), a)


def denoise(img: Image.Image, strength: float = 30) -> Image.Image:
    if strength <= 0:
        return img
    rgb, a = _split_alpha(img)
    h = 2 + strength / 100 * 10
    big = rgb.width * rgb.height > 3_000_000  # smaller search window keeps large photos to a few seconds
    arr = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(np.asarray(rgb)), None, h, h,
                                          5 if big else 7, 13 if big else 21)
    return _merge_alpha(Image.fromarray(arr), a)


def low_light(img: Image.Image, amount: float = 50) -> Image.Image:
    if amount <= 0:
        return img
    t = amount / 100
    rgb, a = _split_alpha(img)
    x = np.power(np.asarray(rgb, np.float32) / 255.0, 1 / (1 + 1.2 * t))
    out = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    out = clahe(out, clip=1.5 + 2 * t, blend=0.5 * t)
    return _merge_alpha(out, a)


def auto_enhance(img: Image.Image, strength: float = 0.7) -> Image.Image:
    """One-click fix: gentle white balance, levels stretch, local contrast, colour and sharpness."""
    if strength <= 0:
        return img
    rgb, a = _split_alpha(img)
    x = np.asarray(rgb, np.float32)
    means = x[::4, ::4].reshape(-1, 3).mean(0)
    gains = np.clip(means.mean() / (means + 1e-6), 0.85, 1.15)
    x = np.clip(x * (1 + (gains - 1) * strength * 0.6), 0, 255)
    lum = x[::4, ::4] @ LUMA
    lo, hi = np.percentile(lum, (0.5, 99.5))
    stretched = np.clip((x - lo) / max(hi - lo, 1.0) * 255, 0, 255)
    x = x * (1 - strength) + stretched * strength
    out = Image.fromarray(x.astype(np.uint8))
    out = clahe(out, clip=1.5 + 1.5 * strength, blend=0.55 * strength)
    out = adjust(out, vibrance=28 * strength, saturation=6 * strength)
    return _merge_alpha(sharpen(out, 22 * strength), a)


def upscale(img: Image.Image, factor: float = 2.0, detail: float = 0.5) -> Image.Image:
    if factor <= 1:
        return img
    rgb, a = _split_alpha(img)
    w, h = rgb.size
    nw, nh = round(w * factor), round(h * factor)
    big = cv2.resize(np.asarray(rgb), (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    out = Image.fromarray(big)
    if detail > 0:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.0 + 0.5 * factor, percent=int(40 + 110 * detail), threshold=2))
    return _merge_alpha(out, a.resize((nw, nh), LANCZOS) if a is not None else None)


def enhance_pipeline(img: Image.Image, p: dict, preview: bool = False) -> Image.Image:
    out = img
    out = low_light(out, p.get("low_light", 0))
    out = denoise(out, p.get("denoise", 0))
    out = auto_enhance(out, p.get("auto", 0) / 100)
    out = adjust(out, exposure=p.get("exposure", 0), contrast=p.get("contrast", 0), highlights=p.get("highlights", 0),
                 shadows=p.get("shadows", 0), vibrance=p.get("vibrance", 0), saturation=p.get("saturation", 0),
                 warmth=p.get("warmth", 0), clarity=p.get("clarity", 0), sharpness=p.get("sharpen", 0))
    if not preview:
        out = upscale(out, p.get("upscale", 1), p.get("detail", 0.5))
    return out


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------
def rembg_available() -> bool:
    return importlib.util.find_spec("rembg") is not None


def remove_bg_ai(img: Image.Image, session=None, matting: bool = False) -> Image.Image:
    from rembg import remove  # optional dependency

    kw = {}
    if matting:
        kw = dict(alpha_matting=True, alpha_matting_foreground_threshold=240,
                  alpha_matting_background_threshold=10, alpha_matting_erode_size=8)
    out = remove(flatten(img) if img.mode != "RGB" else img, session=session, **kw)
    return out.convert("RGBA")


def _clean_mask(fg: np.ndarray, min_frac: float = 0.02, hole_frac: float = 0.002) -> np.ndarray:
    """Remove specks, keep the main subject(s) and fill small holes."""
    h, w = fg.shape
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, ar in enumerate(areas) if ar >= max(areas.max() * 0.15, min_frac * h * w * 0.25)]
        fg = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
    inv = 255 - fg
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, ar = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and ar < hole_frac * h * w:
            fg[labels == i] = 255
    return fg


def remove_bg_grabcut(img: Image.Image, iters: int = 6, rect_frac=None, feather: float = 1.0,
                      max_side: int = 900) -> Image.Image:
    """Built-in cut-out (OpenCV GrabCut). Best on subjects against a fairly plain background.

    rect_frac: optional (x0, y0, x1, y1) fractions - the subject lies inside this box.
    """
    rgb, a0 = _split_alpha(img)
    W, H = rgb.size
    s = min(1.0, max_side / max(W, H))
    small = rgb.resize((max(1, round(W * s)), max(1, round(H * s))), LANCZOS) if s < 1 else rgb
    bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    if rect_frac is not None:
        x0, y0, x1, y1 = rect_frac
        r = (int(x0 * w), int(y0 * h), max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h)))
        cv2.grabCut(bgr, mask, r, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    else:
        lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        b = max(3, int(min(h, w) * 0.03))
        border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                 lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        plain = float(np.mean(np.std(border, axis=0))) < 16  # is the backdrop roughly one colour?
        if not plain:  # busy background: fall back to a centred subject box
            m = 0.04
            r = (int(w * m), int(h * m), int(w * (1 - 2 * m)), int(h * (1 - 2 * m)))
            cv2.grabCut(bgr, mask, r, bgd, fgd, iters + 2, cv2.GC_INIT_WITH_RECT)
            return _finish_grabcut(mask, rgb, a0, feather)
        dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
        d8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(float(thr), 12.0)
        mask[d8 > thr] = cv2.GC_PR_FGD
        sure = cv2.erode((d8 > max(thr * 1.7, thr + 25)).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
        mask[sure > 0] = cv2.GC_FGD
        frame = np.zeros((h, w), bool)
        frame[:b, :] = frame[-b:, :] = frame[:, :b] = frame[:, -b:] = True
        mask[frame & (d8 <= thr)] = cv2.GC_BGD
        if not (mask == cv2.GC_FGD).any():  # nothing obvious: assume subject is centred
            mask[int(h * .25):int(h * .75), int(w * .25):int(w * .75)] = cv2.GC_PR_FGD
        cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)

    return _finish_grabcut(mask, rgb, a0, feather)


def _finish_grabcut(mask, rgb, a0, feather):
    W, H = rgb.size
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = _clean_mask(fg)
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_CUBIC)
    guide = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    try:  # edge-aware refinement when opencv-contrib is present
        fg = cv2.ximgproc.guidedFilter(guide, fg, max(4, int(max(W, H) * 0.004)), 40)
    except Exception:
        fg = cv2.GaussianBlur(fg, (0, 0), max(0.8, max(W, H) * 0.0012))
    if feather > 0:
        fg = cv2.GaussianBlur(fg, (0, 0), feather * max(0.6, max(W, H) / 1500))
    fg = np.clip((fg.astype(np.float32) - 60) * (255 / 135), 0, 255).astype(np.uint8)  # re-sharpen soft edge
    out = rgb.convert("RGBA")
    alpha = Image.fromarray(fg)
    out.putalpha(ImageChops.multiply(alpha, a0) if a0 is not None else alpha)
    return out


def corner_color(img: Image.Image) -> tuple[int, int, int]:
    """Median colour of the four corners - a good guess for a plain backdrop."""
    rgb = flatten(img)
    w, h = rgb.size
    p = max(2, int(min(w, h) * 0.03))
    arr = np.asarray(rgb)
    patches = [arr[:p, :p], arr[:p, -p:], arr[-p:, :p], arr[-p:, -p:]]
    med = np.median(np.concatenate([q.reshape(-1, 3) for q in patches]), axis=0)
    return tuple(int(v) for v in med)  # type: ignore[return-value]


def remove_color(img: Image.Image, color, tolerance: float = 30, softness: float = 20) -> Image.Image:
    """Make pixels close to `color` transparent (colour-key). Great for white / solid backdrops."""
    rgb, a0 = _split_alpha(img)
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - target, axis=2)
    alpha = (np.clip((dist - tolerance) / max(softness, 1e-3), 0, 1) * 255).astype(np.uint8)
    al = Image.fromarray(alpha)
    out = rgb.convert("RGBA")
    out.putalpha(ImageChops.multiply(al, a0) if a0 is not None else al)
    return out


def refine_edges(rgba: Image.Image, shrink: float = 0, feather: float = 0) -> Image.Image:
    """shrink / feather are in 'pixels at 1000px wide' so previews match the export."""
    if not shrink and not feather:
        return rgba
    k = max(max(rgba.size), 1) / 1000
    a = np.asarray(rgba.getchannel("A"))
    if shrink > 0:
        r = max(1, round(shrink * k))
        a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather * k)
    out = rgba.copy()
    out.putalpha(Image.fromarray(a))
    return out


def gradient(size: tuple[int, int], c1, c2, angle: float = 90) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(angle)
    t = xx * math.cos(rad) + yy * math.sin(rad)
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    arr = np.array(c1, np.float32) * (1 - t)[..., None] + np.array(c2, np.float32) * t[..., None]
    return Image.fromarray(arr.astype(np.uint8))


def shadow_under(img: Image.Image, dx: float = 0.0, dy: float = 0.03, blur: float = 0.025, opacity: float = 0.4) -> Image.Image:
    """Soft drop shadow (offsets/blur as fractions of the longest side) composited beneath an RGBA image."""
    img = img.convert("RGBA")
    L = max(img.size)
    alpha = Image.new("L", img.size, 0)
    alpha.paste(img.getchannel("A").point(lambda v: int(v * opacity)), (int(dx * L), int(dy * L)))
    alpha = alpha.filter(ImageFilter.GaussianBlur(max(1, blur * L)))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    layer.putalpha(alpha)
    return Image.alpha_composite(layer, img)


def build_cutout_output(fg: Image.Image, original: Image.Image | None, *, bg_mode: str = "Transparent",
                        color=(255, 255, 255), color2=(210, 220, 235), angle: float = 90, bg_image=None,
                        blur: float = 30, shrink: float = 0, feather: float = 0, trim: bool = False,
                        pad_pct: float = 4, shadow: bool = False) -> Image.Image:
    """Edge refine -> trim -> shadow -> background. Works on the RGBA cut-out from any engine."""
    fg = refine_edges(fg.convert("RGBA"), shrink, feather)
    orig = original.convert("RGBA").resize(fg.size, LANCZOS) if original is not None else None
    if trim:
        bbox = fg.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
        if bbox:
            pad = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * pad_pct / 100)
            box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(fg.width, bbox[2] + pad), min(fg.height, bbox[3] + pad))
            fg = fg.crop(box)
            orig = orig.crop(box) if orig is not None else None
    if shadow:
        fg = shadow_under(fg)
    W, H = fg.size
    if bg_mode == "Solid colour":
        bg = Image.new("RGBA", (W, H), (*color, 255))
    elif bg_mode == "Gradient":
        bg = gradient((W, H), color, color2, angle).convert("RGBA")
    elif bg_mode == "Image" and bg_image is not None:
        bg = ImageOps.fit(bg_image.convert("RGBA"), (W, H), LANCZOS)
    elif bg_mode == "Blurred original" and orig is not None:
        bg = orig.filter(ImageFilter.GaussianBlur(max(1, max(W, H) * blur / 1000)))
    else:
        return fg
    return Image.alpha_composite(bg, fg)


# ---------------------------------------------------------------------------
# Text, watermark, memes
# ---------------------------------------------------------------------------
POSITIONS = ["Top left", "Top center", "Top right", "Middle left", "Center", "Middle right",
             "Bottom left", "Bottom center", "Bottom right"]

_FONTS_REGULAR = ["arial.ttf", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"]
_FONTS_BOLD = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"]
_FONTS_IMPACT = ["impact.ttf", "Impact.ttf", "Anton-Regular.ttf"]


def get_font(size: int, bold: bool = False, impact: bool = False):
    size = max(8, int(size))
    names = (_FONTS_IMPACT if impact else []) + (_FONTS_BOLD if (bold or impact) else _FONTS_REGULAR) + _FONTS_REGULAR
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fsize(font) -> int:
    return int(getattr(font, "size", 16))


def _place_xy(pos: str, canvas, item, margin: int) -> tuple[int, int]:
    cw, ch = canvas
    iw, ih = item
    x = margin if "left" in pos.lower() else cw - iw - margin if "right" in pos.lower() else (cw - iw) // 2
    y = margin if pos.startswith("Top") else ch - ih - margin if pos.startswith("Bottom") else (ch - ih) // 2
    return x, y


def _overlay_at(base: Image.Image, layer: Image.Image, xy) -> Image.Image:
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(layer, xy)
    return Image.alpha_composite(base, canvas)


def _tile(canvas_size, item: Image.Image, gap: int) -> Image.Image:
    cw, ch = canvas_size
    iw, ih = item.size
    sx, sy = iw + gap, ih + gap
    ox, oy = sx // 2, sy // 2
    big = Image.new("RGBA", (cw + 2 * sx, ch + 2 * sy), (0, 0, 0, 0))
    row, y = 0, 0
    while y < big.height:
        x = ox if row % 2 else 0
        while x < big.width:
            big.alpha_composite(item, (x, y))
            x += sx
        y += sy
        row += 1
    return big.crop((ox, oy, ox + cw, oy + ch))


def _text_layer(text: str, font, color, opacity: float, shadow: bool) -> Image.Image:
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = d0.multiline_textbbox((0, 0), text, font=font, spacing=4)
    pad = max(2, int(_fsize(font) * 0.3))
    layer = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * opacity)
    if shadow:
        o = max(1, _fsize(font) // 18)
        d.multiline_text((pad - l + o, pad - t + o), text, font=font, fill=(0, 0, 0, int(a * 0.55)), spacing=4)
    d.multiline_text((pad - l, pad - t), text, font=font, fill=(*color, a), spacing=4)
    return layer


def _apply_layer(base: Image.Image, layer: Image.Image, position: str, angle: float, tile: bool,
                 margin_pct: float) -> Image.Image:
    if angle:
        layer = layer.rotate(-angle, expand=True, resample=BICUBIC)
    if tile:
        return Image.alpha_composite(base, _tile(base.size, layer, gap=int(min(base.size) * 0.12)))
    margin = int(min(base.size) * margin_pct / 100)
    return _overlay_at(base, layer, _place_xy(position, base.size, layer.size, margin))


def add_text_watermark(img: Image.Image, text: str, size_pct: float = 5, opacity: float = 0.55, color=(255, 255, 255),
                       position: str = "Bottom right", angle: float = 0, tile: bool = False, margin_pct: float = 2,
                       bold: bool = True, shadow: bool = True) -> Image.Image:
    if not text.strip():
        return img
    had = img.mode == "RGBA"
    base = img.convert("RGBA")
    font = get_font(base.width * size_pct / 100, bold=bold)
    layer = _text_layer(text, font, color, opacity, shadow)
    return _restore(_apply_layer(base, layer, position, angle, tile, margin_pct), had)


def add_image_watermark(img: Image.Image, logo: Image.Image, size_pct: float = 20, opacity: float = 0.6,
                        position: str = "Bottom right", angle: float = 0, tile: bool = False,
                        margin_pct: float = 2) -> Image.Image:
    had = img.mode == "RGBA"
    base = img.convert("RGBA")
    lw = max(4, int(base.width * size_pct / 100))
    logo = logo.convert("RGBA")
    logo = logo.resize((lw, max(1, round(logo.height * lw / logo.width))), LANCZOS)
    logo.putalpha(logo.getchannel("A").point(lambda v: int(v * opacity)))
    return _restore(_apply_layer(base, logo, position, angle, tile, margin_pct), had)


def _wrap(draw, text: str, font, max_w: float) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = (cur + " " + word).strip()
            if not cur or draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return [ln for ln in lines if ln != ""] or [""]


def draw_meme(img: Image.Image, top: str = "", bottom: str = "", size_pct: float = 9, color=(255, 255, 255),
              stroke=(0, 0, 0), uppercase: bool = True) -> Image.Image:
    had = img.mode == "RGBA"
    base = img.convert("RGBA")
    W, H = base.size
    d = ImageDraw.Draw(base)
    max_w = W * 0.94
    for text, is_top in ((top, True), (bottom, False)):
        if not text.strip():
            continue
        text = text.upper() if uppercase else text
        fs = int(W * size_pct / 100)
        while True:
            font = get_font(fs, impact=True)
            lines = _wrap(d, text, font, max_w)
            widest = max(d.textlength(ln, font=font) for ln in lines)
            if (len(lines) <= 3 and widest <= max_w and len(lines) * fs * 1.1 <= H * 0.34) or fs <= 12:
                break
            fs = int(fs * 0.92)
        lh = int(fs * 1.1)
        sw = max(2, fs // 11)
        y = int(H * 0.03) if is_top else H - int(H * 0.03) - lh * len(lines)
        for ln in lines:
            d.text((W / 2, y), ln, font=font, fill=color, stroke_width=sw, stroke_fill=stroke, anchor="ma")
            y += lh
    return _restore(base, had)


def draw_free_text(img: Image.Image, text: str, x_pct: float = 50, y_pct: float = 50, size_pct: float = 8,
                   color=(255, 255, 255), align: str = "center", bold: bool = True, stroke: int = 0,
                   stroke_color=(0, 0, 0), box: bool = False, box_color=(0, 0, 0), box_opacity: float = 0.5,
                   opacity: float = 1.0) -> Image.Image:
    if not text.strip():
        return img
    had = img.mode == "RGBA"
    base = img.convert("RGBA")
    W, H = base.size
    font = get_font(W * size_pct / 100, bold=bold)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    body = "\n".join(_wrap(d, text, font, W * 0.9))
    sw = int(stroke * max(1, _fsize(font) / 40))
    l, t, r, b = d.multiline_textbbox((0, 0), body, font=font, spacing=6, align=align, stroke_width=sw)
    tw, th = r - l, b - t
    cx, cy = W * x_pct / 100, H * y_pct / 100
    x0, y0 = cx - tw / 2, cy - th / 2
    if box:
        p = _fsize(font) * 0.45
        d.rounded_rectangle((x0 - p, y0 - p, x0 + tw + p, y0 + th + p), radius=int(p * 0.6),
                            fill=(*box_color, int(255 * box_opacity)))
    d.multiline_text((x0 - l, y0 - t), body, font=font, fill=(*color, int(255 * opacity)), spacing=6, align=align,
                     stroke_width=sw, stroke_fill=(*stroke_color, int(255 * opacity)))
    return _restore(Image.alpha_composite(base, layer), had)


# ---------------------------------------------------------------------------
# Privacy blur, borders, collage
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def detect_faces(img: Image.Image) -> list[tuple[float, float, float, float]]:
    """Frontal/profile Haar detection. Returns boxes as fractions of the image, slightly expanded."""
    small = downsize(flatten(img), 900)
    gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY))
    sw, sh = small.size
    raw: list[tuple[int, int, int, int]] = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml"):
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if casc.empty():
            continue
        found = casc.detectMultiScale(gray, 1.1, 5, minSize=(max(24, sw // 40),) * 2)
        raw += [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in found]
    merged: list[tuple[int, int, int, int]] = []
    for b in sorted(raw, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, m) < 0.3 for m in merged):
            merged.append(b)
    out = []
    for x0, y0, x1, y1 in merged:
        px, py = (x1 - x0) * 0.15, (y1 - y0) * 0.2
        out.append((max(0, (x0 - px) / sw), max(0, (y0 - py) / sh), min(1, (x1 + px) / sw), min(1, (y1 + py) / sh)))
    return out


def obscure(img: Image.Image, boxes, mode: str = "Blur", strength: float = 60, ellipse: bool = False) -> Image.Image:
    """Blur / pixelate / black-out regions given as fractional boxes (x0, y0, x1, y1)."""
    out = img.copy()
    W, H = out.size
    for x0, y0, x1, y1 in boxes:
        box = (max(0, int(x0 * W)), max(0, int(y0 * H)), min(W, int(x1 * W)), min(H, int(y1 * H)))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        region = out.crop(box)
        rw, rh = region.size
        s = min(rw, rh)
        if mode == "Pixelate":
            ps = max(2, int(s * (0.04 + strength / 100 * 0.22)))
            new = region.resize((max(1, rw // ps), max(1, rh // ps)), Image.Resampling.BILINEAR).resize((rw, rh), Image.Resampling.NEAREST)
        elif mode == "Black box":
            new = Image.new(region.mode, region.size, (0, 0, 0, 255) if region.mode == "RGBA" else (0, 0, 0))
        else:
            new = region.filter(ImageFilter.GaussianBlur(max(2, s * (0.05 + strength / 100 * 0.35))))
        mask = None
        if ellipse:
            mask = Image.new("L", (rw, rh), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, rw - 1, rh - 1), fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(s * 0.03))
        out.paste(new, box[:2], mask)
    return out


def add_border(img: Image.Image, style: str = "Solid", width_pct: float = 4, color=(255, 255, 255),
               radius_pct: float = 0, shadow: bool = False) -> Image.Image:
    base = img.convert("RGBA")
    W, H = base.size
    bw = max(1, int(min(W, H) * width_pct / 100))
    bottom = int(bw * 3.6) if style == "Polaroid" else bw
    canvas = Image.new("RGBA", (W + 2 * bw, H + bw + bottom), (*color, 255))
    canvas.paste(base, (bw, bw))
    if radius_pct:
        canvas = round_corners(canvas, radius_pct / 100)
    if shadow:
        pad = int(min(canvas.size) * 0.07)
        big = Image.new("RGBA", (canvas.width + 2 * pad, canvas.height + 2 * pad), (0, 0, 0, 0))
        big.paste(canvas, (pad, pad))
        canvas = shadow_under(big, 0, 0.012, 0.014, 0.45)
    return canvas


def make_collage(images: Sequence[Image.Image], cols: int = 3, cell_w: int = 600, aspect: tuple[int, int] = (1, 1),
                 gap_pct: float = 2, bg=(255, 255, 255), radius_pct: float = 0, fit: str = "fill") -> Image.Image:
    n = len(images)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cell_h = round(cell_w * aspect[1] / aspect[0])
    gap = int(cell_w * gap_pct / 100)
    W, H = cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), tuple(bg[:3]))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        tile = ImageOps.fit(im.convert("RGBA"), (cell_w, cell_h), LANCZOS) if fit == "fill" else \
            ImageOps.contain(im.convert("RGBA"), (cell_w, cell_h), LANCZOS)
        if radius_pct:
            tile = round_corners(tile, radius_pct / 100)
        in_row = cols if r < rows - 1 else n - r * cols
        x0 = gap + (cols - in_row) * (cell_w + gap) // 2 + c * (cell_w + gap)
        canvas.paste(tile, (x0 + (cell_w - tile.width) // 2, gap + r * (cell_h + gap) + (cell_h - tile.height) // 2), tile)
    return canvas


# ---------------------------------------------------------------------------
# PDF, metadata, palette
# ---------------------------------------------------------------------------
PAGE_SIZES = {"A4": (1240, 1754), "A5": (874, 1240), "Letter": (1275, 1650), "Legal": (1275, 2100)}


def images_to_pdf(images: Sequence[Image.Image], page: str = "A4", orientation: str = "Auto", margin_pct: float = 5,
                  quality: int = 90, dpi: int = 150) -> bytes:
    pages: list[Image.Image] = []
    for im in images:
        rgb = flatten(im)
        if page == "Fit to image":
            pages.append(rgb)
            continue
        pw, ph = PAGE_SIZES[page]
        land = orientation == "Landscape" or (orientation == "Auto" and rgb.width > rgb.height)
        if land:
            pw, ph = ph, pw
        m = int(min(pw, ph) * margin_pct / 100)
        fitted = ImageOps.contain(rgb, (pw - 2 * m, ph - 2 * m), LANCZOS)
        sheet = Image.new("RGB", (pw, ph), (255, 255, 255))
        sheet.paste(fitted, ((pw - fitted.width) // 2, (ph - fitted.height) // 2))
        pages.append(sheet)
    try:
        import img2pdf  # lossless JPEG embedding, exact page sizes

        blobs = []
        for p in pages:
            b = io.BytesIO()
            p.save(b, "JPEG", quality=quality, dpi=(dpi, dpi), optimize=True)
            blobs.append(b.getvalue())
        return img2pdf.convert(blobs)
    except ImportError:
        buf = io.BytesIO()
        pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=float(dpi))
        return buf.getvalue()


def pdf_to_images(data: bytes, dpi: int = 150, max_pages: int = 60) -> list[Image.Image]:
    import fitz  # PyMuPDF (optional dependency)

    doc = fitz.open(stream=data, filetype="pdf")
    out = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        out.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return out


def pymupdf_available() -> bool:
    return importlib.util.find_spec("fitz") is not None


def read_metadata(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    basic = {"Format": img.format, "Mode": img.mode, "Dimensions": f"{img.width} x {img.height} px",
             "Megapixels": f"{img.width * img.height / 1e6:.1f} MP", "File size": fmt_bytes(len(data))}
    if img.info.get("dpi"):
        basic["DPI"] = " x ".join(str(round(float(v))) for v in img.info["dpi"][:2])
    exif_out: dict = {}
    gps = False
    try:
        exif = img.getexif()
        merged = dict(exif.items())
        for ifd_id in (0x8769, 0xA005):
            try:
                merged.update(exif.get_ifd(ifd_id))
            except Exception:
                pass
        gps = bool(exif.get_ifd(0x8825))
        for tag, val in merged.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            if isinstance(val, bytes):
                val = val[:40].decode("utf-8", "ignore") if len(val) < 80 else f"<{len(val)} bytes>"
            exif_out[name] = str(val)[:120]
    except Exception:
        pass
    return {"basic": basic, "exif": exif_out, "has_gps": gps}


def dominant_colors(img: Image.Image, k: int = 6) -> list[tuple[str, float]]:
    small = flatten(downsize(img, 220))
    q = small.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    colors = sorted(q.getcolors() or [], reverse=True)
    total = sum(c for c, _ in colors) or 1
    return [("#%02x%02x%02x" % tuple(pal[i * 3:i * 3 + 3]), c / total * 100) for c, i in colors]


def text_on(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return "#1f1d1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"
