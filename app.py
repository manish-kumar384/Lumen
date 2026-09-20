"""Lumen - Photo Studio. A calm, professional, all-in-one image toolkit built with Streamlit.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import imgtools as T

st.set_page_config(page_title="Lumen · Photo Studio", page_icon=":material/photo_library:",
                   layout="wide", initial_sidebar_state="expanded")

# ==========================================================================
# 1. THEME
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
:root{
  --bg:#FAF8F5; --surface:#FFFFFF; --surface2:#F3F0EA; --line:#E8E3DA; --line2:#D9D2C6;
  --text:#25231F; --muted:#7B756B; --accent:#33473F; --accent-hover:#26362F; --accent-soft:#E6EDE9;
}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li, .stApp td, .stApp th {
  font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
}
.stApp{background:var(--bg); color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1200px; padding-top:2rem; padding-bottom:4rem;}
h1,h2,h3{font-family:'Fraunces',Georgia,serif !important; font-weight:600; letter-spacing:-0.01em; color:var(--text);}
h5{font-size:.78rem !important; text-transform:uppercase; letter-spacing:.09em; color:var(--muted) !important; font-weight:600 !important; margin:.2rem 0 .5rem 0 !important;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--surface2); border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start; background:transparent; border:none; box-shadow:none;
  color:var(--text); font-weight:500; padding:.42rem .7rem; border-radius:10px; min-height:0;}
[data-testid="stSidebar"] .stButton>button:hover{background:#E9E5DC; color:var(--text);}
[data-testid="stSidebar"] .stButton>button[kind="primary"], [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{
  background:var(--accent-soft); color:var(--accent); font-weight:600;}
.brand{font-family:'Fraunces',Georgia,serif; font-size:1.55rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .1rem .3rem;}
.brand span{color:var(--accent);}
.brand-sub{color:var(--muted); font-size:.8rem; margin:0 0 1rem .3rem;}
.nav-cat{color:var(--muted); font-size:.68rem; letter-spacing:.11em; text-transform:uppercase; font-weight:600; margin:1rem 0 .25rem .5rem;}

/* buttons */
.stButton>button, .stDownloadButton>button{border-radius:10px; font-weight:500; border:1px solid var(--line2); background:var(--surface); color:var(--text); transition:all .15s ease;}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--accent); color:var(--accent);}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"],
button[data-testid="stBaseButton-primary"]{background:var(--accent); border:1px solid var(--accent); color:#fff; font-weight:600;}
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover{background:var(--accent-hover); border-color:var(--accent-hover); color:#fff;}

/* containers, uploader, metrics, tabs */
[class*="st-key-card_"]{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:.4rem .3rem; transition:box-shadow .2s ease, border-color .2s ease;}
[class*="st-key-card_"]:hover{border-color:var(--line2); box-shadow:0 6px 24px rgba(60,50,30,.07);}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;}
[data-testid="stFileUploaderDropzone"]{background:var(--surface); border:1.5px dashed #CFC8BB; border-radius:16px; padding:1.6rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent); background:#FDFCFA;}
[data-testid="stMetric"]{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem;}
.stTabs [data-baseweb="tab-list"]{gap:6px; border-bottom:1px solid var(--line);}
[data-testid="stExpander"]{border:1px solid var(--line); border-radius:12px; background:var(--surface);}
img{border-radius:10px;}

/* custom blocks */
.hero{padding:2.2rem 0 1.4rem 0;}
.hero h1{font-size:2.9rem; line-height:1.1; margin:0 0 .6rem 0;}
.hero p{color:var(--muted); font-size:1.08rem; max-width:620px; margin:0;}
.page-head{margin:.2rem 0 1.4rem 0;}
.page-head h1{font-size:2.1rem; margin:0 0 .3rem 0;}
.page-head p{color:var(--muted); margin:0; font-size:1rem;}
.card-title{font-weight:600; font-size:1.02rem; margin:.1rem 0 .15rem 0;}
.card-desc{color:var(--muted); font-size:.88rem; line-height:1.45; min-height:2.6rem; margin-bottom:.4rem;}
.cat-title{font-family:'Fraunces',Georgia,serif; font-size:1.25rem; font-weight:600; margin:1.8rem 0 .7rem 0;}
.chips{display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 1rem 0;}
.chip{background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:.3rem .8rem; font-size:.84rem; color:var(--text);}
.chip b{font-weight:600;}
.chip.good{background:var(--accent-soft); border-color:#CBD8D1; color:var(--accent);}
.empty{text-align:center; color:var(--muted); padding:2.2rem 1rem 1rem 1rem; font-size:.95rem;}
.empty .big{font-family:'Fraunces',Georgia,serif; font-size:1.3rem; color:var(--text); margin-bottom:.3rem;}
.swatch{border-radius:14px; height:88px; display:flex; align-items:flex-end; padding:.55rem .7rem; font-size:.82rem; font-weight:600; border:1px solid rgba(0,0,0,.06);}
.foot{color:var(--muted); font-size:.8rem; text-align:center; margin-top:3rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# 2. SHARED HELPERS
# ==========================================================================
def go(name: str) -> None:
    st.session_state["page"] = name


def page_head(title: str, desc: str) -> None:
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{desc}</p></div>', unsafe_allow_html=True)


def chips(items: list[tuple[str, str]], good_last: bool = False) -> None:
    html = "".join(
        f'<span class="chip{" good" if good_last and i == len(items) - 1 else ""}">{k} <b>{v}</b></span>'
        for i, (k, v) in enumerate(items))
    st.markdown(f'<div class="chips">{html}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="empty"><div class="big">{title}</div>{text}</div>', unsafe_allow_html=True)


def upload(key: str, multiple: bool = False, types=None, label: str = "Drop your image here, or browse"):
    files = st.file_uploader(label, type=types or T.INPUT_TYPES, accept_multiple_files=multiple,
                             key=f"{key}_up", label_visibility="collapsed")
    if not multiple:
        return [files] if files else []
    return list(files or [])


def get_image(f) -> Image.Image:
    cache = st.session_state.setdefault("_imgs", {})
    if f.file_id not in cache:
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        cache[f.file_id] = T.load_image(f.getvalue())
    return cache[f.file_id]


def get_preview(f, side: int) -> Image.Image:
    cache = st.session_state.setdefault("_prevs", {})
    k = (f.file_id, side)
    if k not in cache:
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
        cache[k] = T.downsize(get_image(f), side)
    return cache[k]


def show(img: Image.Image, caption: str | None = None) -> None:
    st.image(T.on_checker(img) if T.has_alpha(img) else img, caption=caption, width="stretch")


def sig(*parts) -> str:
    def clean(x):
        if isinstance(x, dict):
            return sorted((k, clean(v)) for k, v in x.items() if not isinstance(v, Image.Image))
        return x
    return hashlib.md5(repr([clean(p) for p in parts]).encode()).hexdigest()


def sl(label, key, lo, hi, default=0, step=1, help=None):
    """Slider whose value lives in session_state so presets / reset buttons can change it."""
    st.session_state.setdefault(key, default)
    return st.slider(label, lo, hi, key=key, step=step, help=help)


def label_for(choice: str, fname: str) -> str:
    if choice != "Same as original":
        return choice
    return T.EXT_TO_LABEL.get(Path(fname).suffix.lower().lstrip("."), "PNG")


def output_controls(key: str, formats=("Same as original", "PNG", "JPG", "WEBP")):
    st.markdown("##### Output")
    fmt = st.selectbox("Format", formats, key=f"{key}_fmt")
    q = 92
    if fmt in ("JPG", "WEBP", "Same as original"):
        q = st.slider("Quality", 40, 100, 92, key=f"{key}_q", help="Applies to JPG and WEBP. 90+ is visually lossless.")
    return fmt, q


def auto_result(key: str, signature: str, build, spinner: str = "Working…"):
    """Compute once per unique settings; reruns triggered by unrelated widgets reuse the result."""
    store = st.session_state.setdefault("_auto", {})
    hit = store.get(key)
    if hit and hit["sig"] == signature:
        return hit["value"]
    with st.spinner(spinner):
        value = build()
    store[key] = {"sig": signature, "value": value}
    return value


def downloads(files: list[tuple[str, bytes, str]], key: str, zip_name: str = "lumen_images.zip") -> None:
    """files = [(name, bytes, mime)]. One file -> direct download, several -> ZIP."""
    if len(files) == 1:
        name, data, mime = files[0]
        st.download_button(f"Download  ·  {name}  ({T.fmt_bytes(len(data))})", data, name, mime, key=f"{key}_dl",
                           type="primary", width="stretch", on_click="ignore", icon=":material/download:")
    elif files:
        z = T.make_zip([(n, d) for n, d, _ in files])
        st.download_button(f"Download all as ZIP  ·  {len(files)} files ({T.fmt_bytes(len(z))})", z, zip_name,
                           "application/zip", key=f"{key}_dl", type="primary", width="stretch",
                           on_click="ignore", icon=":material/folder_zip:")


def export_block(key: str, signature: str, build, label: str = "Apply & prepare download") -> None:
    """Two-step export for heavier edits: process at full resolution on click, then offer the download."""
    store = st.session_state.setdefault("_exports", {})
    res = store.get(key)
    valid = bool(res) and res["sig"] == signature
    if not valid:
        if res:
            st.caption("Settings changed since the last export - prepare it again.")
        slot = st.empty()
        if slot.button(label, key=f"{key}_go", type="primary", width="stretch", icon=":material/auto_awesome:"):
            slot.empty()
            with st.spinner("Processing at full resolution…"):
                try:
                    store[key] = {"sig": signature, "files": build()}
                except Exception as e:  # keep the app alive, tell the user
                    store.pop(key, None)
                    st.error(f"Something went wrong: {e}")
                    return
            res, valid = store[key], True
    if valid:
        downloads(res["files"], key)


def default_preview(key: str, src: Image.Image, out: Image.Image, params: dict) -> None:
    same = src.size == out.size
    if same:
        mode = st.segmented_control("View", ["After", "Before", "Split"], default="After", key=f"{key}_view",
                                    label_visibility="collapsed") or "After"
    else:
        mode = "After"
    if mode == "Before":
        show(src)
    elif mode == "Split":
        pos = st.slider("Divider", 5, 95, 50, key=f"{key}_split", label_visibility="collapsed")
        st.image(T.split_compare(src, out, pos / 100), width="stretch")
        st.caption("Left: original · Right: edited")
    else:
        show(out)


def edit_workspace(key: str, *, controls, process, multiple: bool = False, preview_side: int = 1400, renderer=None,
                   formats=("Same as original", "PNG", "JPG", "WEBP"), suffix: str = "_edited",
                   empty_title: str = "Add a photo to get started",
                   empty_text: str = "JPG, PNG, WEBP, GIF, BMP or TIFF. Your files stay in this session.",
                   export_label: str = "Apply & prepare download") -> None:
    """Generic editor: upload -> controls (left) + live preview (right) -> full-resolution export."""
    files = upload(key, multiple)
    if not files:
        empty_state(empty_title, empty_text)
        return
    f = files[0]
    if len(files) > 1:
        chosen = st.selectbox(f"Previewing 1 of {len(files)} images (settings apply to all)",
                              [x.name for x in files], key=f"{key}_which")
        f = next(x for x in files if x.name == chosen)
    try:
        img = get_image(f)
    except Exception as e:
        st.error(f"Couldn't open {f.name}: {e}")
        return

    left, right = st.columns([5, 8], gap="large")
    with left:
        with st.container(border=True):
            params = controls(img)
        with st.container(border=True):
            fmt, q = output_controls(key, formats)

        def build():
            outs = []
            for x in files:
                res = process(get_image(x), params, preview=False)
                lab = label_for(fmt, x.name)
                outs.append((T.out_name(x.name, suffix, lab), T.encode(res, lab, q), T.FORMATS[lab][2]))
            return outs

        export_block(key, sig(key, params, fmt, q, [x.file_id for x in files]), build, export_label)
    with right:
        src = get_preview(f, preview_side)
        with st.spinner("Updating preview…"):
            out = process(src, params, preview=True)
        (renderer or default_preview)(key, src, out, params)
        if T.has_alpha(out) and label_for(fmt, f.name) == "JPG":
            st.caption("JPG has no transparency - transparent areas will become white.")
        w, h = img.size
        st.caption(f"Original {w} × {h} px · {T.fmt_bytes(len(f.getvalue()))}")


# ==========================================================================
# 3. TOOLS  -  OPTIMISE
# ==========================================================================
def tool_compress():
    files = upload("compress", multiple=True)
    if not files:
        empty_state("Drop one or many images", "Make them lighter without visible loss. Batch downloads arrive as a ZIP.")
        return
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        st.markdown("##### Compression")
        mode = st.radio("Mode", ["Recommended", "Custom quality", "Target file size"], key="cp_mode",
                        help="Recommended keeps quality high and trims the file to the sweet spot.")
        quality, target_kb, allow_resize = 80, 200, True
        if mode == "Custom quality":
            quality = st.slider("Quality", 10, 100, 75, key="cp_q")
        elif mode == "Target file size":
            c1, c2 = st.columns([2, 1])
            val = c1.number_input("Max size", 1, 100000, 200, key="cp_kb")
            unit = c2.selectbox("Unit", ["KB", "MB"], key="cp_unit")
            target_kb = val * (1024 if unit == "MB" else 1)
            allow_resize = st.checkbox("Allow reducing dimensions if needed", True, key="cp_allow",
                                       help="If quality alone can't reach the target, the image is scaled down.")
        fmt = st.selectbox("Output format", ["Keep original", "JPG", "WEBP", "PNG"], key="cp_fmt",
                           help="WEBP is typically 25-35% smaller than JPG at the same look.")
        max_side = st.number_input("Limit longest side (px, 0 = off)", 0, 20000, 0, step=100, key="cp_side")
        reduce_png = st.checkbox("Reduce PNG colours (much smaller)", True, key="cp_png")

    params = dict(mode=mode, quality=quality, kb=target_kb, allow=allow_resize, fmt=fmt, side=max_side, png=reduce_png)

    def build():
        rows = []
        for f in files:
            img = get_image(f)
            if max_side:
                img = T.downsize(img, int(max_side))
            lab = label_for("Same as original" if fmt == "Keep original" else fmt, f.name)
            info = {}
            if mode == "Target file size":
                data, info = T.encode_to_target(img, lab, int(target_kb * 1024), allow_resize=allow_resize)
            elif lab == "PNG":
                data = T._png_bytes(T.quantize(img, 256)) if reduce_png else T.encode(img, "PNG")
            else:
                data = T.encode(img, lab, quality)
            orig = len(f.getvalue())
            rows.append(dict(name=T.out_name(f.name, "_compressed", lab), data=data, mime=T.FORMATS[lab][2],
                             orig=orig, new=len(data), info=info, src=f.name))
        return rows

    rows = auto_result("compress", sig(params, [f.file_id for f in files]), build, "Compressing…")
    tot_o, tot_n = sum(r["orig"] for r in rows), sum(r["new"] for r in rows)
    with right:
        saved = (1 - tot_n / tot_o) * 100 if tot_o else 0
        chips([("Before", T.fmt_bytes(tot_o)), ("After", T.fmt_bytes(tot_n)),
               ("Saved", f"{saved:.0f}%" if saved > 0 else "already optimal")], good_last=saved > 0)
        met = [r for r in rows if r["info"] and not r["info"].get("met")]
        if met:
            st.warning(f"{len(met)} file(s) couldn't reach the target. Allow resizing or raise the limit.")
        downloads([(r["name"], r["data"], r["mime"]) for r in rows], "compress", "compressed_images.zip")
        if len(rows) > 1:
            st.dataframe(pd.DataFrame([{"File": r["src"], "Before": T.fmt_bytes(r["orig"]), "After": T.fmt_bytes(r["new"]),
                                        "Saved": f"{(1 - r['new'] / r['orig']) * 100:.0f}%"} for r in rows]),
                         hide_index=True, width="stretch")
            pick = st.selectbox("Compare", [r["src"] for r in rows], key="cp_cmp")
        else:
            pick = rows[0]["src"]
        r = next(x for x in rows if x["src"] == pick)
        f = next(x for x in files if x.name == pick)
        c1, c2 = st.columns(2)
        c1.image(T.downsize(get_image(f), 900), caption=f"Original · {T.fmt_bytes(r['orig'])}", width="stretch")
        c2.image(T.downsize(T.load_image(r["data"]), 900), caption=f"Compressed · {T.fmt_bytes(r['new'])}", width="stretch")
        info = r["info"]
        if info and info.get("scale", 1) < 1:
            st.caption(f"Scaled to {info['size'][0]} × {info['size'][1]} px to hit the target.")


SOCIAL_PRESETS = {
    "Instagram post · 1080 × 1080": (1080, 1080), "Instagram portrait · 1080 × 1350": (1080, 1350),
    "Instagram story / Reel · 1080 × 1920": (1080, 1920), "Facebook cover · 820 × 312": (820, 312),
    "X / Twitter post · 1600 × 900": (1600, 900), "LinkedIn banner · 1584 × 396": (1584, 396),
    "YouTube thumbnail · 1280 × 720": (1280, 720), "Pinterest pin · 1000 × 1500": (1000, 1500),
    "WhatsApp profile · 500 × 500": (500, 500), "Passport 35 × 45 mm @300dpi · 413 × 531": (413, 531),
    "HD · 1280 × 720": (1280, 720), "Full HD · 1920 × 1080": (1920, 1080), "4K UHD · 3840 × 2160": (3840, 2160),
    "Favicon · 512 × 512": (512, 512),
}
FIT_MODES = {"Stretch": "stretch", "Fit inside (keep ratio)": "fit", "Fill & crop": "fill", "Pad with colour": "pad"}


def tool_resize():
    files = upload("resize", multiple=True)
    if not files:
        empty_state("Resize by pixels, percentage, preset or file size", "Batch-friendly. Set the quality or a maximum file size.")
        return
    first = files[0]
    fimg = get_image(first)
    left, right = st.columns([5, 8], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("##### Dimensions")
            how = st.segmented_control("Resize by", ["Pixels", "Percentage", "Preset"], default="Pixels", key="rs_how",
                                       label_visibility="collapsed") or "Pixels"
            spec = dict(how=how)
            if how == "Pixels":
                axis = st.radio("Keep aspect ratio by", ["Width", "Height", "Exact size"], horizontal=True, key="rs_axis")
                spec["axis"] = axis
                fid = first.file_id
                if axis in ("Width", "Exact size"):
                    spec["w"] = st.number_input("Width (px)", 1, 20000, fimg.width, key=f"rs_w_{fid}")
                if axis in ("Height", "Exact size"):
                    spec["h"] = st.number_input("Height (px)", 1, 20000, fimg.height, key=f"rs_h_{fid}")
                if axis == "Exact size":
                    spec["fit"] = FIT_MODES[st.radio("If the ratio differs", list(FIT_MODES), key="rs_fit")]
            elif how == "Percentage":
                spec["pct"] = st.slider("Scale (%)", 5, 400, 50, key="rs_pct")
            else:
                name = st.selectbox("Preset", list(SOCIAL_PRESETS), key="rs_preset")
                spec["w"], spec["h"] = SOCIAL_PRESETS[name]
                spec["fit"] = FIT_MODES[st.radio("Fit", list(FIT_MODES)[1:], index=1, key="rs_pfit")]
            if spec.get("fit") == "pad":
                spec["bg"] = T.hex_to_rgb(st.color_picker("Padding colour", "#FFFFFF", key="rs_bg"))
        with st.container(border=True):
            st.markdown("##### Quality & file size")
            fmt = st.selectbox("Format", ["Keep original", "JPG", "PNG", "WEBP"], key="rs_fmt")
            lab_probe = label_for("Same as original" if fmt == "Keep original" else fmt, first.name)
            by = st.radio("Control", ["Quality", "Maximum file size"], horizontal=True, key="rs_by")
            quality, kb = 90, 300
            if by == "Quality":
                quality = st.slider("Quality", 10, 100, 90, key="rs_q", disabled=lab_probe not in T.LOSSY,
                                    help="Only JPG / WEBP have adjustable quality.")
            else:
                c1, c2 = st.columns([2, 1])
                v = c1.number_input("Max size", 1, 100000, 300, key="rs_kb")
                u = c2.selectbox("Unit", ["KB", "MB"], key="rs_unit")
                kb = v * (1024 if u == "MB" else 1)
            dpi = st.selectbox("Print resolution (DPI metadata)", ["Keep", 72, 96, 150, 300], key="rs_dpi")
    params = dict(spec=spec, fmt=fmt, by=by, q=quality, kb=kb, dpi=dpi)

    def resized(img):
        s = spec
        if s["how"] == "Percentage":
            return T.resize_to(img, img.width * s["pct"] / 100, img.height * s["pct"] / 100)
        if s["how"] == "Pixels" and s["axis"] != "Exact size":
            w, h = T.fit_dims(img.size, s.get("w") if s["axis"] == "Width" else None, s.get("h") if s["axis"] == "Height" else None)
            return T.resize_to(img, w, h)
        return T.resize_to(img, s["w"], s["h"], s.get("fit", "stretch"), s.get("bg", (255, 255, 255)))

    def build():
        rows = []
        for f in files:
            img = resized(get_image(f))
            lab = label_for("Same as original" if fmt == "Keep original" else fmt, f.name)
            kw = {} if dpi == "Keep" else {"dpi": int(dpi)}
            if by == "Maximum file size":
                data, _ = T.encode_to_target(img, lab, int(kb * 1024), allow_resize=False, **kw)
            else:
                data = T.encode(img, lab, quality, **kw)
            rows.append(dict(name=T.out_name(f.name, f"_{img.width}x{img.height}", lab), data=data,
                             mime=T.FORMATS[lab][2], size=img.size, src=f.name, orig=len(f.getvalue())))
        return rows

    rows = auto_result("resize", sig(params, [f.file_id for f in files]), build, "Resizing…")
    with right:
        r0 = rows[0]
        chips([("Original", f"{fimg.width} × {fimg.height}"), ("New", f"{r0['size'][0]} × {r0['size'][1]}"),
               ("File", f"{T.fmt_bytes(r0['orig'])} → {T.fmt_bytes(len(r0['data']))}")], good_last=True)
        if r0["size"][0] > fimg.width:
            st.info("You're enlarging the image, which can look soft. Try Enhance → Upscale for sharper results.")
        if by == "Maximum file size" and len(r0["data"]) > kb * 1024:
            st.warning("Couldn't reach that size at these dimensions. Lower the dimensions or raise the limit.")
        downloads([(r["name"], r["data"], r["mime"]) for r in rows], "resize", "resized_images.zip")
        pick = st.selectbox("Preview", [r["src"] for r in rows], key="rs_prev") if len(rows) > 1 else rows[0]["src"]
        r = next(x for x in rows if x["src"] == pick)
        show(T.downsize(T.load_image(r["data"]), 1200), f"{r['size'][0]} × {r['size'][1]} px · {T.fmt_bytes(len(r['data']))}")
        if len(rows) > 1:
            st.dataframe(pd.DataFrame([{"File": r["src"], "Size": f"{r['size'][0]} × {r['size'][1]}",
                                        "File size": T.fmt_bytes(len(r["data"]))} for r in rows]), hide_index=True, width="stretch")


RATIOS = {"1 : 1 (square)": (1, 1), "4 : 3": (4, 3), "3 : 2": (3, 2), "16 : 9": (16, 9), "5 : 4": (5, 4), "4 : 5 (portrait)": (4, 5),
          "3 : 4 (portrait)": (3, 4), "2 : 3 (portrait)": (2, 3), "9 : 16 (story)": (9, 16), "3 : 1 (banner)": (3, 1), "Custom": None}


def _crop_box(size, p):
    if p["mode"] == "Free":
        return T.frac_box(size, p["fx"], p["fy"])
    return T.aspect_box(size, p["ratio"], p["zoom"], p["cx"], p["cy"])


def tool_crop():
    def controls(img):
        st.markdown("##### Crop")
        mode = st.segmented_control("Mode", ["Aspect ratio", "Free"], default="Aspect ratio", key="cr_mode",
                                    label_visibility="collapsed") or "Aspect ratio"
        p = dict(mode="Free" if mode == "Free" else "Ratio")
        if mode == "Free":
            fx = st.slider("Horizontal", 0.0, 1.0, (0.0, 1.0), 0.005, key="cr_fx", format="%.3f")
            fy = st.slider("Vertical", 0.0, 1.0, (0.0, 1.0), 0.005, key="cr_fy", format="%.3f")
            p.update(fx=fx, fy=fy)
        else:
            name = st.selectbox("Aspect ratio", list(RATIOS), key="cr_ratio")
            ratio = RATIOS[name]
            if ratio is None:
                c1, c2 = st.columns(2)
                ratio = (c1.number_input("W", 1, 100, 4, key="cr_cw"), c2.number_input("H", 1, 100, 3, key="cr_ch"))
            p.update(ratio=ratio, zoom=st.slider("Zoom", 1.0, 4.0, 1.0, 0.05, key="cr_zoom"),
                     cx=st.slider("Position ← →", 0.0, 1.0, 0.5, 0.01, key="cr_cx"),
                     cy=st.slider("Position ↑ ↓", 0.0, 1.0, 0.5, 0.01, key="cr_cy"))
        p["shape"] = st.radio("Shape", ["Rectangle", "Rounded corners", "Circle"], horizontal=True, key="cr_shape")
        if p["shape"] == "Rounded corners":
            p["radius"] = st.slider("Corner radius", 2, 50, 15, key="cr_rad")
        if p["shape"] != "Rectangle":
            st.caption("Rounded and circular crops are saved with transparency (best as PNG or WEBP).")
        return p

    def process(img, p, preview=False):
        out = img.crop(_crop_box(img.size, p))
        if p["shape"] == "Circle":
            return T.round_corners(out, circle=True)
        if p["shape"] == "Rounded corners":
            return T.round_corners(out, p["radius"] / 100)
        return out

    def renderer(key, src, out, p):
        t1, t2 = st.tabs(["Selection", "Result"])
        box = _crop_box(src.size, p)
        with t1:
            st.image(T.draw_crop_overlay(src, box), width="stretch")
        with t2:
            show(out)
        st.caption(f"Selection covers {(box[2] - box[0]) / src.width * 100:.0f}% of the width and "
                   f"{(box[3] - box[1]) / src.height * 100:.0f}% of the height.")

    edit_workspace("crop", controls=controls, process=process, renderer=renderer, suffix="_cropped",
                   empty_title="Add a photo to crop", empty_text="Use an aspect ratio for social posts, or free-form for precision.")


def tool_rotate():
    def controls(img):
        st.markdown("##### Rotate & flip")
        quick = st.segmented_control("Rotate", ["0°", "90°", "180°", "270°"], default="0°", key="rt_q") or "0°"
        fine = st.slider("Straighten", -45.0, 45.0, 0.0, 0.5, key="rt_fine", help="Fine angle to level a tilted horizon.")
        c1, c2 = st.columns(2)
        fh = c1.checkbox("Flip horizontal", key="rt_fh")
        fv = c2.checkbox("Flip vertical", key="rt_fv")
        fill = st.selectbox("Empty corners", ["Transparent", "White", "Black"], key="rt_fill")
        return dict(angle=int(quick[:-1]) + fine, fh=fh, fv=fv, fill=fill)

    def process(img, p, preview=False):
        return T.rotate_flip(img, p["angle"], p["fh"], p["fv"], True, p["fill"])

    edit_workspace("rotate", controls=controls, process=process, multiple=True, suffix="_rotated",
                   empty_title="Add photos to rotate or flip", empty_text="Fix sideways shots or level a tilted horizon. Batch supported.")


# ==========================================================================
# 4. TOOLS  -  EDIT
# ==========================================================================
PE_KEYS = ["pe_exposure", "pe_brightness", "pe_contrast", "pe_highlights", "pe_shadows", "pe_saturation", "pe_vibrance",
           "pe_warmth", "pe_tint", "pe_sharpness", "pe_clarity", "pe_vignette", "pe_grain", "pe_fade"]


def _pe_reset():
    for k in PE_KEYS:
        st.session_state[k] = 0
    st.session_state["pe_filter"], st.session_state["pe_intensity"] = "None", 100


def _pe_set_filter(name):
    st.session_state["pe_filter"] = name


def tool_photo_editor():
    def controls(img):
        st.session_state.setdefault("pe_filter", "None")
        t1, t2, t3 = st.tabs(["Adjust", "Effects", "Filters"])
        p = {}
        with t1:
            for label, key in [("Exposure", "exposure"), ("Brightness", "brightness"), ("Contrast", "contrast"),
                               ("Highlights", "highlights"), ("Shadows", "shadows"), ("Saturation", "saturation"),
                               ("Vibrance", "vibrance"), ("Warmth", "warmth"), ("Tint", "tint"), ("Sharpness", "sharpness")]:
                p[key] = sl(label, f"pe_{key}", -100, 100)
        with t2:
            p["clarity"] = sl("Clarity", "pe_clarity", -100, 100, help="Local contrast: punchier texture.")
            p["fade"] = sl("Fade", "pe_fade", 0, 100)
            p["vignette"] = sl("Vignette", "pe_vignette", -100, 100, help="Negative values brighten the edges.")
            p["grain"] = sl("Film grain", "pe_grain", 0, 100)
        with t3:
            thumb = T.downsize(img, 150)
            cols = st.columns(3)
            for i, name in enumerate(T.FILTERS):
                with cols[i % 3]:
                    st.image(T.on_checker(T.apply_filter(thumb, name, 1.0)) if T.has_alpha(thumb) else T.apply_filter(thumb, name, 1.0), width="stretch")
                    st.button(name, key=f"pe_f_{name}", on_click=_pe_set_filter, args=(name,), width="stretch",
                              type="primary" if st.session_state["pe_filter"] == name else "secondary")
            p["filter"] = st.session_state["pe_filter"]
            p["intensity"] = sl("Filter strength", "pe_intensity", 0, 100, 100)
        st.button("Reset all", on_click=_pe_reset, key="pe_reset", icon=":material/restart_alt:")
        return p

    def process(img, p, preview=False):
        adj = {k: v for k, v in p.items() if k not in ("filter", "intensity")}
        out = T.adjust(img, **adj)
        return T.apply_filter(out, p["filter"], p["intensity"] / 100)

    edit_workspace("photo", controls=controls, process=process, multiple=True, suffix="_edited",
                   empty_title="Add photos to edit", empty_text="Light, colour, effects and filters, with a live before / after view.")


ENH_PRESETS = {
    "Balanced": dict(auto=60, exposure=0, contrast=8, highlights=-10, shadows=15, vibrance=15, warmth=0, low_light=0, denoise=10, sharpen=25, clarity=15),
    "Portrait": dict(auto=40, exposure=5, contrast=0, highlights=-15, shadows=12, vibrance=10, warmth=8, low_light=0, denoise=25, sharpen=18, clarity=-8),
    "Landscape": dict(auto=55, exposure=0, contrast=15, highlights=-20, shadows=20, vibrance=32, warmth=0, low_light=0, denoise=5, sharpen=35, clarity=35),
    "Low light": dict(auto=50, exposure=10, contrast=5, highlights=-10, shadows=30, vibrance=15, warmth=0, low_light=60, denoise=45, sharpen=25, clarity=10),
    "Document / text": dict(auto=30, exposure=10, contrast=35, highlights=-5, shadows=25, vibrance=0, warmth=0, low_light=0, denoise=10, sharpen=60, clarity=30),
    "Custom": None,
}
ENH_SLIDERS = [("Auto-enhance", "auto", 0, 100), ("Exposure", "exposure", -100, 100), ("Contrast", "contrast", -100, 100),
               ("Highlights", "highlights", -100, 100), ("Shadows", "shadows", -100, 100), ("Vibrance", "vibrance", -100, 100),
               ("Warmth", "warmth", -100, 100), ("Low-light boost", "low_light", 0, 100), ("Denoise", "denoise", 0, 100),
               ("Sharpen", "sharpen", 0, 100), ("Clarity", "clarity", -100, 100)]


def _enh_apply_preset():
    vals = ENH_PRESETS.get(st.session_state["enh_preset"])
    if vals:
        for k, v in vals.items():
            st.session_state[f"enh_{k}"] = v


def tool_enhance():
    def controls(img):
        st.markdown("##### Enhance")
        st.session_state.setdefault("enh_preset", "Balanced")
        if "enh_auto" not in st.session_state:
            _enh_apply_preset()
        st.selectbox("Preset", list(ENH_PRESETS), key="enh_preset", on_change=_enh_apply_preset,
                     help="Pick a starting point, then fine-tune any slider.")
        p = {}
        with st.expander("Light & colour", expanded=True):
            for label, k, lo, hi in ENH_SLIDERS[:7]:
                p[k] = sl(label, f"enh_{k}", lo, hi)
        with st.expander("Detail & noise", expanded=True):
            for label, k, lo, hi in ENH_SLIDERS[7:]:
                p[k] = sl(label, f"enh_{k}", lo, hi)
        with st.expander("Upscale (AI-free, detail-preserving)"):
            p["upscale"] = st.select_slider("Enlarge", [1, 2, 3, 4], 1, format_func=lambda x: "Off" if x == 1 else f"{x}×", key="enh_up")
            p["detail"] = st.slider("Detail recovery", 0.0, 1.0, 0.5, 0.05, key="enh_detail")
            if p["upscale"] > 1:
                w, h = img.size
                mp = w * h * p["upscale"] ** 2 / 1e6
                st.caption(f"Output: {w * p['upscale']} × {h * p['upscale']} px ({mp:.0f} MP)")
                if mp > 80:
                    st.warning("Very large output - this may be slow or run out of memory.")
        return p

    def process(img, p, preview=False):
        return T.enhance_pipeline(img, p, preview=preview)

    def renderer(key, src, out, p):
        default_preview(key, src, out, p)
        if p["upscale"] > 1:
            st.caption("Upscaling is applied on export - the preview shows all other adjustments.")
        if p["denoise"] > 0:
            st.caption("Denoising large photos can take several seconds on export.")

    edit_workspace("enhance", controls=controls, process=process, multiple=True, renderer=renderer, preview_side=1100,
                   suffix="_enhanced", empty_title="Add photos to enhance",
                   empty_text="One-click auto-fix, low-light rescue, denoise, sharpen and upscale.")


@st.cache_resource(show_spinner=False)
def _rembg_session(model: str):
    from rembg import new_session
    return new_session(model)


def tool_remove_bg():
    files = upload("bg", multiple=True)
    if not files:
        empty_state("Add photos to remove the background", "Cut-outs with transparent, coloured, gradient or custom backgrounds.")
        return
    ai_ok = T.rembg_available()
    left, right = st.columns([5, 8], gap="large")
    results = st.session_state.setdefault("_bg_results", {})

    with left:
        with st.container(border=True):
            st.markdown("##### 1 · Remove")
            engines = (["AI · high accuracy"] if ai_ok else []) + ["Smart cut-out", "Colour key"]
            engine = st.radio("Method", engines, key="bg_engine",
                              help="AI works on any photo. Smart cut-out suits plain backdrops. Colour key removes one solid colour.")
            opts = {}
            if engine.startswith("AI"):
                opts["model"] = st.selectbox("Model", ["isnet-general-use", "u2net", "u2net_human_seg", "silueta", "u2netp"],
                                             key="bg_model", help="isnet-general-use is the best all-rounder. Choose u2net_human_seg for people.")
                opts["matting"] = st.checkbox("Refine soft edges (hair, fur) - slower", False, key="bg_matting")
            elif engine == "Smart cut-out":
                opts["iters"] = st.slider("Precision", 3, 10, 6, key="bg_iters")
                opts["feather"] = st.slider("Edge softness", 0.0, 3.0, 1.0, 0.1, key="bg_feather")
                if st.checkbox("Mark the subject area", False, key="bg_rect_on", help="Drag the box edges to enclose the subject."):
                    fx = st.slider("Left → Right", 0.0, 1.0, (0.1, 0.9), 0.01, key="bg_rx")
                    fy = st.slider("Top → Bottom", 0.0, 1.0, (0.05, 0.95), 0.01, key="bg_ry")
                    opts["rect"] = (fx[0], fy[0], fx[1], fy[1])
                if not ai_ok:
                    st.caption("Works best on plain backgrounds. For busy scenes, install the optional AI engine: `pip install rembg`.")
            else:
                first = get_image(files[0])
                st.session_state.setdefault("bg_key", "#%02x%02x%02x" % T.corner_color(first))
                st.color_picker("Colour to remove", key="bg_key")
                st.button("Use corner colour", on_click=lambda: st.session_state.update(bg_key="#%02x%02x%02x" % T.corner_color(get_image(files[0]))),
                          key="bg_corner")
                opts["tol"] = st.slider("Tolerance", 0, 80, 22, key="bg_tol")
                opts["soft"] = st.slider("Softness", 0, 40, 14, key="bg_soft")
                opts["color"] = T.hex_to_rgb(st.session_state["bg_key"])
            if st.button(f"Remove background{'s' if len(files) > 1 else ''}", type="primary", width="stretch", key="bg_run",
                         icon=":material/content_cut:"):
                bar = st.progress(0.0, text="Starting…")
                for i, f in enumerate(files):
                    bar.progress(i / len(files), text=f"Processing {f.name}")
                    try:
                        img = get_image(f)
                        if engine.startswith("AI"):
                            out = T.remove_bg_ai(img, _rembg_session(opts["model"]), opts["matting"])
                        elif engine == "Smart cut-out":
                            out = T.remove_bg_grabcut(img, opts["iters"], opts.get("rect"), opts["feather"])
                        else:
                            out = T.remove_color(img, opts["color"], opts["tol"], opts["soft"])
                        results[f.file_id] = out
                    except Exception as e:
                        st.error(f"{f.name}: {e}" + ("  (the AI model downloads on first use and needs an internet connection)" if engine.startswith("AI") else ""))
                bar.empty()

        done = [f for f in files if f.file_id in results]
        with st.container(border=True):
            st.markdown("##### 2 · Background")
            bo = dict(bg_mode=st.selectbox("Background", ["Transparent", "Solid colour", "Gradient", "Blurred original", "Image"], key="bg_mode"))
            if bo["bg_mode"] in ("Solid colour", "Gradient"):
                c1, c2 = st.columns(2)
                bo["color"] = T.hex_to_rgb(c1.color_picker("Colour", "#FFFFFF", key="bg_c1"))
                if bo["bg_mode"] == "Gradient":
                    bo["color2"] = T.hex_to_rgb(c2.color_picker("To", "#D6E2F0", key="bg_c2"))
                    bo["angle"] = st.slider("Angle", 0, 180, 90, key="bg_ang")
            elif bo["bg_mode"] == "Blurred original":
                bo["blur"] = st.slider("Blur", 5, 80, 30, key="bg_blur")
            elif bo["bg_mode"] == "Image":
                up = st.file_uploader("Background image", type=T.INPUT_TYPES, key="bg_img")
                if up:
                    bo["bg_image"] = get_image(up)
            bo["trim"] = st.checkbox("Trim to subject", False, key="bg_trim")
            if bo["trim"]:
                bo["pad_pct"] = st.slider("Padding (%)", 0, 30, 6, key="bg_pad")
            bo["shadow"] = st.checkbox("Soft drop shadow", False, key="bg_shadow")
            bo["shrink"] = st.slider("Shrink edge (removes halo)", 0.0, 6.0, 0.0, 0.5, key="bg_shrink")
            bo["feather"] = st.slider("Feather edge", 0.0, 6.0, 0.0, 0.5, key="bg_feath")
        with st.container(border=True):
            st.markdown("##### 3 · Export")
            fmt = st.selectbox("Format", ["PNG", "WEBP", "JPG"], key="bg_fmt")
            if fmt == "JPG" and bo["bg_mode"] == "Transparent":
                st.caption("JPG can't be transparent - the background will be white.")

    with right:
        if not done:
            empty_state("Your cut-out will appear here", "Choose a method on the left, then press Remove background.")
            return
        f = done[0]
        if len(done) > 1:
            chosen = st.selectbox("Preview", [x.name for x in done], key="bg_which")
            f = next(x for x in done if x.name == chosen)
        orig = get_image(f)

        def compose(cut, original):
            return T.build_cutout_output(cut, original, **bo)

        prev_cut = results[f.file_id]
        scale = min(1.0, 1100 / max(prev_cut.size))
        pc = prev_cut.resize((round(prev_cut.width * scale), round(prev_cut.height * scale)), T.LANCZOS) if scale < 1 else prev_cut
        po = T.downsize(orig, 1100).resize(pc.size, T.LANCZOS)
        view = st.segmented_control("View", ["Result", "Original", "Side by side"], default="Result", key="bg_view",
                                    label_visibility="collapsed") or "Result"
        result_prev = compose(pc, po)
        if view == "Original":
            show(po)
        elif view == "Side by side":
            c1, c2 = st.columns(2)
            with c1:
                show(po, "Original")
            with c2:
                show(result_prev, "Result")
        else:
            show(result_prev)

        def build():
            outs = []
            for x in done:
                res = compose(results[x.file_id], get_image(x))
                outs.append((T.out_name(x.name, "_nobg", fmt), T.encode(res, fmt, 95), T.FORMATS[fmt][2]))
            return outs

        outs = auto_result("bg_export", sig(fmt, bo, [x.file_id for x in done], [id(results[x.file_id]) for x in done],
                                            bo.get("bg_image").size if bo.get("bg_image") else None), build, "Preparing files…")
        downloads(outs, "bg", "background_removed.zip")
        if len(done) < len(files):
            st.caption(f"{len(files) - len(done)} more image(s) not processed yet - press Remove background to include them.")


def tool_watermark():
    def controls(img):
        st.markdown("##### Watermark")
        kind = st.segmented_control("Type", ["Text", "Logo / image"], default="Text", key="wm_kind", label_visibility="collapsed") or "Text"
        p = dict(kind=kind)
        if kind == "Text":
            p["text"] = st.text_input("Text", "© Your Name", key="wm_text")
            p["size"] = st.slider("Size", 1.0, 30.0, 6.0, 0.5, key="wm_size", help="% of image width")
            p["color"] = T.hex_to_rgb(st.color_picker("Colour", "#FFFFFF", key="wm_col"))
            p["bold"] = st.checkbox("Bold", True, key="wm_bold")
            p["shadow"] = st.checkbox("Soft shadow (better legibility)", True, key="wm_shadow")
        else:
            up = st.file_uploader("Logo (PNG with transparency works best)", type=T.INPUT_TYPES, key="wm_logo")
            p["logo_id"] = up.file_id if up else None
            p["logo"] = get_image(up) if up else None
            p["size"] = st.slider("Size", 3.0, 80.0, 20.0, 1.0, key="wm_lsize", help="% of image width")
            if not up:
                st.info("Upload a logo to preview it here.")
        p["opacity"] = st.slider("Opacity", 5, 100, 60, key="wm_op") / 100
        p["tile"] = st.checkbox("Repeat across the image", False, key="wm_tile")
        if not p["tile"]:
            p["pos"] = st.select_slider("Position", T.POSITIONS, "Bottom right", key="wm_pos")
            p["margin"] = st.slider("Margin (%)", 0.0, 15.0, 2.5, 0.5, key="wm_margin")
        p["angle"] = st.slider("Rotation", -90, 90, -30 if p["tile"] else 0, key="wm_angle")
        return p

    def process(img, p, preview=False):
        kw = dict(position=p.get("pos", "Center"), angle=p["angle"], tile=p["tile"], margin_pct=p.get("margin", 2))
        if p["kind"] == "Text":
            return T.add_text_watermark(img, p["text"], p["size"], p["opacity"], p["color"], bold=p["bold"], shadow=p["shadow"], **kw)
        if p["logo"] is None:
            return img
        return T.add_image_watermark(img, p["logo"], p["size"], p["opacity"], **kw)

    edit_workspace("wm", controls=controls, process=process, multiple=True, suffix="_watermarked",
                   empty_title="Add photos to watermark", empty_text="Protect your work with text or a logo. Apply to many photos at once.")


def tool_text_meme():
    def controls(img):
        st.markdown("##### Text")
        kind = st.segmented_control("Style", ["Meme", "Free text"], default="Meme", key="tx_kind", label_visibility="collapsed") or "Meme"
        p = dict(kind=kind)
        if kind == "Meme":
            p["top"] = st.text_input("Top text", "TOP TEXT", key="tx_top")
            p["bottom"] = st.text_input("Bottom text", "BOTTOM TEXT", key="tx_bot")
            p["size"] = st.slider("Size", 3.0, 16.0, 9.0, 0.5, key="tx_msize")
            p["upper"] = st.checkbox("UPPERCASE", True, key="tx_upper")
            c1, c2 = st.columns(2)
            p["color"] = T.hex_to_rgb(c1.color_picker("Fill", "#FFFFFF", key="tx_mcol"))
            p["stroke"] = T.hex_to_rgb(c2.color_picker("Outline", "#000000", key="tx_mstroke"))
        else:
            p["text"] = st.text_area("Text", "Your text here", key="tx_text", height=90)
            p["size"] = st.slider("Size", 2.0, 30.0, 8.0, 0.5, key="tx_fsize")
            p["x"] = st.slider("Horizontal position (%)", 0, 100, 50, key="tx_x")
            p["y"] = st.slider("Vertical position (%)", 0, 100, 50, key="tx_y")
            p["align"] = st.radio("Align", ["left", "center", "right"], index=1, horizontal=True, key="tx_align")
            p["color"] = T.hex_to_rgb(st.color_picker("Colour", "#FFFFFF", key="tx_fcol"))
            p["bold"] = st.checkbox("Bold", True, key="tx_bold")
            p["stroke"] = st.slider("Outline", 0, 6, 0, key="tx_fstroke")
            p["box"] = st.checkbox("Background box", False, key="tx_box")
            if p["box"]:
                p["box_color"] = T.hex_to_rgb(st.color_picker("Box colour", "#000000", key="tx_bcol"))
                p["box_op"] = st.slider("Box opacity", 10, 100, 55, key="tx_bop") / 100
        return p

    def process(img, p, preview=False):
        if p["kind"] == "Meme":
            return T.draw_meme(img, p["top"], p["bottom"], p["size"], p["color"], p["stroke"], p["upper"])
        return T.draw_free_text(img, p["text"], p["x"], p["y"], p["size"], p["color"], p["align"], p["bold"], p["stroke"],
                                box=p.get("box", False), box_color=p.get("box_color", (0, 0, 0)), box_opacity=p.get("box_op", .5))

    edit_workspace("text", controls=controls, process=process, suffix="_text",
                   empty_title="Add a photo for your caption", empty_text="Classic meme captions or freely placed text with optional background box.")


def tool_blur():
    def controls(img):
        st.markdown("##### Privacy blur")
        p = {}
        p["faces"] = st.checkbox("Detect & obscure faces automatically", True, key="bl_faces",
                                 help="Works best on front-facing faces. Use a manual region for anything it misses.")
        p["manual"] = st.checkbox("Also obscure a manual region", False, key="bl_manual")
        if p["manual"]:
            p["fx"] = st.slider("Left → Right", 0.0, 1.0, (0.3, 0.7), 0.01, key="bl_fx")
            p["fy"] = st.slider("Top → Bottom", 0.0, 1.0, (0.3, 0.7), 0.01, key="bl_fy")
            p["ellipse"] = st.checkbox("Oval shape", False, key="bl_oval")
        p["mode"] = st.radio("Effect", ["Blur", "Pixelate", "Black box"], horizontal=True, key="bl_mode")
        p["strength"] = st.slider("Strength", 10, 100, 65, key="bl_str", disabled=p["mode"] == "Black box")
        if p["faces"]:
            n = len(T.detect_faces(T.downsize(img, 900)))
            st.caption(f"{n} face{'s' if n != 1 else ''} detected." if n else "No faces detected - try a manual region.")
        return p

    def process(img, p, preview=False):
        out = img
        if p["faces"]:
            out = T.obscure(out, T.detect_faces(out), p["mode"], p["strength"], ellipse=True)
        if p["manual"]:
            fx, fy = p["fx"], p["fy"]
            out = T.obscure(out, [(fx[0], fy[0], fx[1], fy[1])], p["mode"], p["strength"], ellipse=p.get("ellipse", False))
        return out

    edit_workspace("blur", controls=controls, process=process, multiple=True, suffix="_blurred",
                   empty_title="Add photos to blur faces or details", empty_text="Hide faces, licence plates or documents in seconds.")


def tool_borders():
    def controls(img):
        st.markdown("##### Frame")
        p = dict(style=st.radio("Style", ["Solid", "Polaroid"], horizontal=True, key="bd_style"))
        p["width"] = st.slider("Border width (%)", 0.0, 15.0, 4.0, 0.5, key="bd_w")
        p["color"] = T.hex_to_rgb(st.color_picker("Border colour", "#FFFFFF", key="bd_col"))
        p["radius"] = st.slider("Rounded corners (%)", 0, 40, 0, key="bd_r")
        p["shadow"] = st.checkbox("Soft shadow", False, key="bd_sh")
        if p["radius"] or p["shadow"]:
            st.caption("Rounded corners and shadows use transparency (PNG or WEBP recommended).")
        return p

    def process(img, p, preview=False):
        return T.add_border(img, p["style"], p["width"], p["color"], p["radius"], p["shadow"])

    edit_workspace("border", controls=controls, process=process, multiple=True, suffix="_framed", formats=("PNG", "WEBP", "JPG"),
                   empty_title="Add photos to frame", empty_text="Clean borders, polaroid frames, rounded corners and drop shadows.")


# ==========================================================================
# 5. TOOLS  -  CREATE / CONVERT / INSPECT
# ==========================================================================
def tool_collage():
    files = upload("collage", multiple=True)
    if len(files) < 2:
        empty_state("Add two or more photos", "Arranged in the order you upload them.")
        return
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        st.markdown("##### Layout")
        cols = st.slider("Columns", 1, min(6, len(files)), min(3, len(files)), key="co_cols")
        asp = st.selectbox("Cell shape", ["Square 1:1", "Landscape 4:3", "Portrait 3:4", "Wide 16:9"], key="co_asp")
        aspect = {"Square 1:1": (1, 1), "Landscape 4:3": (4, 3), "Portrait 3:4": (3, 4), "Wide 16:9": (16, 9)}[asp]
        gap = st.slider("Spacing (%)", 0.0, 10.0, 2.0, 0.5, key="co_gap")
        rad = st.slider("Rounded corners (%)", 0, 30, 4, key="co_rad")
        fit = "fill" if st.radio("Photos", ["Fill cell (crop)", "Fit inside"], horizontal=True, key="co_fit").startswith("Fill") else "fit"
        bg = T.hex_to_rgb(st.color_picker("Background", "#FAF8F5", key="co_bg"))
        width = st.select_slider("Cell width (px)", [300, 450, 600, 800, 1000], 600, key="co_w")
        fmt = st.selectbox("Format", ["JPG", "PNG", "WEBP"], key="co_fmt")
    params = dict(cols=cols, aspect=aspect, gap=gap, rad=rad, fit=fit, bg=bg, width=width, fmt=fmt)

    def build():
        imgs = [get_image(f) for f in files]
        sheet = T.make_collage(imgs, cols, width, aspect, gap, bg, rad, fit)
        return sheet, T.encode(sheet, fmt, 92)

    sheet, data = auto_result("collage", sig(params, [f.file_id for f in files]), build, "Composing…")
    with right:
        chips([("Canvas", f"{sheet.width} × {sheet.height}"), ("File", T.fmt_bytes(len(data)))])
        downloads([(f"collage.{T.FORMATS[fmt][1]}", data, T.FORMATS[fmt][2])], "collage")
        st.image(T.downsize(sheet, 1400), width="stretch")


def tool_convert():
    files = upload("convert", multiple=True)
    if not files:
        empty_state("Convert between image formats", "JPG, PNG, WEBP, GIF, BMP, TIFF, ICO and PDF. Batch supported.")
        return
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        st.markdown("##### Convert to")
        fmt = st.radio("Format", list(T.FORMATS), horizontal=True, key="cv_fmt")
        q = 92
        if fmt in T.LOSSY:
            q = st.slider("Quality", 40, 100, 92, key="cv_q")
        bg = (255, 255, 255)
        if fmt in ("JPG", "BMP", "GIF", "PDF"):
            bg = T.hex_to_rgb(st.color_picker("Background for transparent areas", "#FFFFFF", key="cv_bg"))
        ico = None
        if fmt == "ICO":
            ico = st.multiselect("Icon sizes (px)", [16, 32, 48, 64, 128, 256], [16, 32, 48, 256], key="cv_ico") or [32]
            st.caption("Great for favicons. The image is centred on a square canvas.")
    params = dict(fmt=fmt, q=q, bg=bg, ico=ico)

    def build():
        return [dict(name=T.out_name(f.name, "", fmt), data=T.encode(get_image(f), fmt, q, bg=bg, ico_sizes=ico),
                     mime=T.FORMATS[fmt][2], src=f.name, orig=len(f.getvalue())) for f in files]

    rows = auto_result("convert", sig(params, [f.file_id for f in files]), build, "Converting…")
    with right:
        chips([("Files", str(len(rows))), ("Total", f"{T.fmt_bytes(sum(r['orig'] for r in rows))} → {T.fmt_bytes(sum(len(r['data']) for r in rows))}")])
        downloads([(r["name"], r["data"], r["mime"]) for r in rows], "convert", f"converted_{fmt.lower()}.zip")
        if fmt != "PDF":
            show(T.downsize(T.load_image(rows[0]["data"]) if fmt != "ICO" else get_image(files[0]), 1000), f"Preview · {rows[0]['name']}")
        if len(rows) > 1:
            st.dataframe(pd.DataFrame([{"File": r["src"], "→": r["name"], "Size": T.fmt_bytes(len(r["data"]))} for r in rows]),
                         hide_index=True, width="stretch")


def tool_img2pdf():
    files = upload("i2p", multiple=True)
    if not files:
        empty_state("Turn images into a single PDF", "Photos, scans and screenshots, in the order you upload them.")
        return
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        st.markdown("##### PDF settings")
        order = st.checkbox("Sort by file name", False, key="ip_sort")
        page = st.selectbox("Page size", ["A4", "Letter", "A5", "Legal", "Fit to image"], key="ip_page")
        orient = st.radio("Orientation", ["Auto", "Portrait", "Landscape"], horizontal=True, key="ip_orient", disabled=page == "Fit to image")
        margin = st.slider("Margin (%)", 0, 15, 5, key="ip_margin", disabled=page == "Fit to image")
        quality = st.slider("Image quality", 50, 100, 90, key="ip_q")
    use = sorted(files, key=lambda f: f.name.lower()) if order else files
    params = dict(page=page, orient=orient, margin=margin, q=quality)

    def build():
        return T.images_to_pdf([get_image(f) for f in use], page, orient, margin, quality)

    data = auto_result("i2p", sig(params, [f.file_id for f in use]), build, "Building PDF…")
    with right:
        chips([("Pages", str(len(use))), ("File", T.fmt_bytes(len(data)))])
        downloads([("images.pdf", data, "application/pdf")], "i2p")
        st.markdown("##### Page order")
        cols = st.columns(min(4, len(use)))
        for i, f in enumerate(use[:12]):
            cols[i % len(cols)].image(T.downsize(get_image(f), 300), caption=f"{i + 1}. {f.name[:22]}", width="stretch")


def tool_pdf2img():
    up = st.file_uploader("Drop a PDF here", type=["pdf"], key="p2i_up", label_visibility="collapsed")
    if not up:
        empty_state("Turn PDF pages into images", "Each page becomes a PNG or JPG, delivered as a ZIP.")
        return
    if not T.pymupdf_available():
        st.warning("PDF rendering needs the optional PyMuPDF package. Install it with `pip install pymupdf`, then restart the app.")
        return
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        st.markdown("##### Output")
        dpi = st.select_slider("Resolution (DPI)", [72, 100, 150, 200, 300], 150, key="p2i_dpi")
        fmt = st.radio("Format", ["PNG", "JPG", "WEBP"], horizontal=True, key="p2i_fmt")
        q = st.slider("Quality", 50, 100, 90, key="p2i_q", disabled=fmt == "PNG")
    params = dict(dpi=dpi, fmt=fmt, q=q)

    def build():
        pages = T.pdf_to_images(up.getvalue(), dpi)
        stem = Path(up.name).stem
        return pages, [(f"{stem}_page{i + 1}.{T.FORMATS[fmt][1]}", T.encode(p, fmt, q), T.FORMATS[fmt][2]) for i, p in enumerate(pages)]

    try:
        pages, outs = auto_result("p2i", sig(params, up.file_id), build, "Rendering pages…")
    except Exception as e:
        st.error(f"Couldn't read that PDF: {e}")
        return
    with right:
        chips([("Pages", str(len(pages))), ("Page size", f"{pages[0].width} × {pages[0].height}")])
        downloads(outs, "p2i", f"{Path(up.name).stem}_pages.zip")
        cols = st.columns(3)
        for i, p in enumerate(pages[:9]):
            cols[i % 3].image(T.downsize(p, 500), caption=f"Page {i + 1}", width="stretch")
        if len(pages) > 9:
            st.caption(f"Showing the first 9 of {len(pages)} pages. All pages are in the download.")


def tool_metadata():
    files = upload("meta")
    if not files:
        empty_state("Inspect a photo", "See its details and EXIF data, remove hidden metadata, or pull a colour palette.")
        return
    f = files[0]
    raw = f.getvalue()
    img = get_image(f)
    t1, t2, t3 = st.tabs(["Details & privacy", "Colour palette", "Preview"])
    with t1:
        info = T.read_metadata(raw)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("##### File")
            st.dataframe(pd.DataFrame(list(info["basic"].items()), columns=["Property", "Value"]), hide_index=True, width="stretch")
        with c2:
            st.markdown("##### Camera & EXIF")
            if info["exif"]:
                st.dataframe(pd.DataFrame(list(info["exif"].items()), columns=["Tag", "Value"]), hide_index=True, width="stretch", height=260)
            else:
                st.caption("No EXIF metadata found.")
        if info["has_gps"]:
            st.warning("This photo contains GPS location data. Anyone you share it with could see where it was taken.")
        lab = label_for("Same as original", f.name)
        if lab == "PNG" or lab not in T.LOSSY:
            clean = T.encode(img, lab)
        else:
            clean = T.encode(img, lab, 95)
        downloads([(T.out_name(f.name, "_clean", lab), clean, T.FORMATS[lab][2])], "meta")
        st.caption("The clean copy has all metadata removed (location, camera, timestamps). Lossy formats are re-saved at high quality.")
    with t2:
        k = st.slider("Number of colours", 3, 10, 6, key="mt_k")
        cols = T.dominant_colors(img, k)
        grid = st.columns(3)
        for i, (hx, pct) in enumerate(cols):
            grid[i % 3].markdown(f'<div class="swatch" style="background:{hx};color:{T.text_on(hx)}">{hx.upper()} · {pct:.0f}%</div>', unsafe_allow_html=True)
        st.code(", ".join(h.upper() for h, _ in cols), language=None)
    with t3:
        show(T.downsize(img, 1200))


# ==========================================================================
# 6. REGISTRY, SHELL, HOME
# ==========================================================================
TOOLS = {
    "Compress": dict(icon="compress", cat="Optimise", fn=tool_compress, blurb="Shrink file size while keeping the picture crisp. Aim for a target size."),
    "Resize": dict(icon="aspect_ratio", cat="Optimise", fn=tool_resize, blurb="Set exact dimensions, a percentage, a social preset, quality or a maximum file size."),
    "Crop": dict(icon="crop", cat="Optimise", fn=tool_crop, blurb="Crop by aspect ratio or freely. Circle and rounded shapes included."),
    "Rotate & Flip": dict(icon="rotate_right", cat="Optimise", fn=tool_rotate, blurb="Rotate, flip and straighten tilted horizons."),
    "Photo Editor": dict(icon="tune", cat="Edit", fn=tool_photo_editor, blurb="Light, colour, effects and 15 filters with a live before / after."),
    "Enhance": dict(icon="auto_fix_high", cat="Edit", fn=tool_enhance, blurb="One-click auto-fix, low-light rescue, denoise, sharpen and upscale."),
    "Remove Background": dict(icon="content_cut", cat="Edit", fn=tool_remove_bg, blurb="Cut out the subject, then go transparent, solid, gradient or blurred."),
    "Watermark": dict(icon="branding_watermark", cat="Edit", fn=tool_watermark, blurb="Stamp text or a logo across one photo or a whole batch."),
    "Text & Meme": dict(icon="text_fields", cat="Edit", fn=tool_text_meme, blurb="Add captions, classic meme text or styled text with a background."),
    "Blur & Pixelate": dict(icon="blur_on", cat="Edit", fn=tool_blur, blurb="Hide faces and sensitive details automatically or by region."),
    "Borders & Corners": dict(icon="rounded_corner", cat="Edit", fn=tool_borders, blurb="Add frames, polaroid borders, rounded corners and shadows."),
    "Collage": dict(icon="grid_view", cat="Create", fn=tool_collage, blurb="Combine photos into a tidy grid with custom spacing and corners."),
    "Convert": dict(icon="sync_alt", cat="Convert", fn=tool_convert, blurb="JPG, PNG, WEBP, GIF, BMP, TIFF, ICO and PDF. Batch friendly."),
    "Images to PDF": dict(icon="picture_as_pdf", cat="Convert", fn=tool_img2pdf, blurb="Merge photos or scans into one PDF with page size control."),
    "PDF to Images": dict(icon="pageview", cat="Convert", fn=tool_pdf2img, blurb="Turn every PDF page into a crisp PNG, JPG or WEBP."),
    "Metadata & Palette": dict(icon="palette", cat="Inspect", fn=tool_metadata, blurb="Read EXIF, strip location data and extract a colour palette."),
}
CATEGORIES = ["Optimise", "Edit", "Create", "Convert", "Inspect"]


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand">Lumen<span>.</span></div><div class="brand-sub">Photo studio</div>', unsafe_allow_html=True)
        cur = st.session_state.get("page", "Home")
        st.button("Home", key="nav_Home", on_click=go, args=("Home",), width="stretch", icon=":material/home:",
                  type="primary" if cur == "Home" else "secondary")
        for cat in CATEGORIES:
            st.markdown(f'<div class="nav-cat">{cat}</div>', unsafe_allow_html=True)
            for name, t in TOOLS.items():
                if t["cat"] == cat:
                    st.button(name, key=f"nav_{name}", on_click=go, args=(name,), width="stretch",
                              icon=f":material/{t['icon']}:", type="primary" if cur == name else "secondary")


def home() -> None:
    st.markdown('<div class="hero"><h1>Every photo tool you need,<br>beautifully simple.</h1>'
                '<p>Compress, resize, enhance, cut out backgrounds and more. Sixteen focused tools in one calm workspace.</p></div>',
                unsafe_allow_html=True)
    q = st.text_input("Find a tool", placeholder="Search tools, e.g. “background”, “PDF”, “resize”…", label_visibility="collapsed", key="home_q").strip().lower()
    shown = 0
    for cat in CATEGORIES:
        items = [(n, t) for n, t in TOOLS.items() if t["cat"] == cat and (not q or q in n.lower() or q in t["blurb"].lower())]
        if not items:
            continue
        shown += len(items)
        st.markdown(f'<div class="cat-title">{cat}</div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, (name, t) in enumerate(items):
            with cols[i % 3]:
                with st.container(border=True, key=f"card_{name.replace(' ', '_').replace('&', 'and')}"):
                    st.markdown(f'<div class="card-title">:material/{t["icon"]}: &nbsp;{name}</div><div class="card-desc">{t["blurb"]}</div>',
                                unsafe_allow_html=True)
                    st.button("Open", key=f"open_{name}", on_click=go, args=(name,), width="stretch")
    if not shown:
        empty_state("No matching tools", "Try a different word.")


def main() -> None:
    sidebar()
    page = st.session_state.get("page", "Home")
    if page == "Home" or page not in TOOLS:
        home()
    else:
        t = TOOLS[page]
        st.button("All tools", key="back", on_click=go, args=("Home",), icon=":material/arrow_back:")
        page_head(page, t["blurb"])
        t["fn"]()
    st.markdown('<div class="foot">Images are processed in memory during your session and are not stored by this app.</div>',
                unsafe_allow_html=True)


main()
