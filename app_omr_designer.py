# -*- coding: utf-8 -*-
"""
OMR Grid & Mark Designer — Mouse Drag Only + Block Summary JSON
- ผู้ใช้ "ลากสี่เหลี่ยม" บนภาพเฉลย เพื่อกำหนดกรอบ (Rect) ของ Digit Zone / Answer Block
- โหมด Transform: ย้าย/ย่อ-ขยายกรอบเดิมได้
- พรีวิวเส้นกริดตาม rows/cols และโหมด (uniform/weighted/auto)
- เพิ่ม Digit Zone / Answer Block จาก "กรอบล่าสุด" ที่ลาก
- Import/Export layout.json (ใช้ format เดียวกันกับกระดาษนิสิต)
- ประมวลผล → CSV เฉลย/นิสิต (ช่องว่างถ้าไม่ตอบ) + เทียบคะแนน และดาวน์โหลด JSON สรุปคะแนน:
    {
      "digit": {...},              # string ของแต่ละ digit zone
      "answer_student": {...},     # คำตอบนิสิตรายข้อ (ไม่ตอบ = "")
      "answer_key": {...},         # เฉลยรายข้อ
      "blocks": [...],             # สรุปคะแนนรายบล็อก
      "sum_answer_block": <int>,   # คะแนนรวม
      "score_percent": <float>
    }

รัน: streamlit run app_omr_designer.py
"""

import json, time, warnings, hashlib, base64, io
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import streamlit as st

warnings.filterwarnings("ignore")

# ===== Optional Canvas =====
USE_CANVAS = True
try:
    from streamlit_drawable_canvas import st_canvas as _st_canvas  # เราจะห่อกับ safe_st_canvas อีกชั้น
except Exception:
    USE_CANVAS = False

# ===================== GLOBAL DEFAULTS =====================
ABS_MIN_BLACK_DEFAULT  = 35
MIN_FILL_RATIO_DEFAULT = 0.22
TOP2_DELTA_RATIO_DEFAULT = 0.10

OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== COMPAT HELPERS =====================
def show_image(img, caption=None, width=None):
    """รองรับทั้ง Streamlit เก่า/ใหม่ (บางรุ่นไม่มี use_container_width)"""
    try:
        import inspect
        if "use_container_width" in inspect.signature(st.image).parameters:
            return st.image(img, caption=caption, use_container_width=True)
        else:
            raise TypeError
    except Exception:
        return st.image(img, caption=caption, width=width)

def pil_to_data_url(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def safe_st_canvas(bg_pil, **kwargs):
    """
    streamlit-drawable-canvas มี 2 สาย:
    - รุ่นปกติ: ใช้ background_image=pil
    - รุ่น -fix: ใช้ background_image_url=data_url (รุ่นใหม่ของ Streamlit)
    ห่อไว้ให้เรียกได้ทั้งสองแบบ
    """
    try:
        return _st_canvas(
            background_image=None,
            background_image_url=pil_to_data_url(bg_pil),
            **kwargs
        )
    except TypeError:
        # รุ่นเดิมไม่มี background_image_url
        return _st_canvas(
            background_image=bg_pil,
            **kwargs
        )

# ===================== OMR CORE UTILS =====================
def pillow_read_to_bgr(file_or_path):
    pil = Image.open(file_or_path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def pct_to_px(rect_pct, W, H):
    x1p,y1p,x2p,y2p = rect_pct
    x1 = int(round(x1p*W)); y1 = int(round(y1p*H))
    x2 = int(round(x2p*W)); y2 = int(round(y2p*H))
    x1 = max(0,min(x1,W-2)); y1 = max(0,min(y1,H-2))
    x2 = max(1,min(x2,W-1)); y2 = max(1,min(y2,H-1))
    if x2<=x1+2: x2=x1+3
    if y2<=y1+2: y2=y1+3
    return [x1,y1,x2,y2]

def _smooth1d(a,k=15):
    k = max(3, int(k//2*2+1))
    ker = np.ones(k, np.float32)/k
    return np.convolve(a.astype(np.float32), ker, mode="same")

def _borders_from_proj(proj, nsplit, search_ratio):
    L=len(proj); sp=_smooth1d(proj, max(9, L//200))
    borders=[0]
    for i in range(1, nsplit+1):
        x0=int(round(i*L/(nsplit+1)))
        win=max(2,int(L*search_ratio))
        a=max(1,x0-win); b=min(L-2,x0+win)
        x=a+int(np.argmin(sp[a:b+1])); borders.append(x)
    borders.append(L)
    for j in range(1,len(borders)):
        borders[j]=max(borders[j], borders[j-1]+1)
    return borders

def uniform_grid(crop_gray, rows, cols):
    H, W = crop_gray.shape[:2]
    bx = [int(round(i * W / cols)) for i in range(cols + 1)]
    by = [int(round(i * H / rows)) for i in range(rows + 1)]
    boxes = []
    for r in range(rows):
        for c in range(cols):
            boxes.append((bx[c], by[r], bx[c+1], by[r+1]))
    return boxes, bx, by, None

def weighted_edges(n, total_len, weights):
    if not weights or len(weights) != n: weights = [1.0]*n
    s = float(sum(weights)) if sum(weights) > 0 else float(n)
    edges = [0]; acc = 0.0
    for i in range(n):
        acc += weights[i] / s
        edges.append(int(round(acc * total_len)))
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i-1]+1)
    edges[-1] = total_len
    return edges

def weighted_grid(crop_gray, rows, cols, row_weights=None, col_weights=None):
    H, W = crop_gray.shape[:2]
    bx = weighted_edges(cols, W, col_weights)
    by = weighted_edges(rows, H, row_weights)
    boxes=[]
    for r in range(rows):
        for c in range(cols):
            boxes.append((bx[c],by[r],bx[c+1],by[r+1]))
    return boxes, bx, by, None

def auto_grid(crop_gray, rows, cols):
    g=cv2.GaussianBlur(crop_gray,(3,3),0)
    binv=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    binv=cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    H,W=binv.shape
    bx=_borders_from_proj(binv.sum(axis=0), cols-1, 0.06)
    by=_borders_from_proj(binv.sum(axis=1), rows-1, 0.05)
    if len(bx)!=cols+1: bx=[int(round(i*W/cols)) for i in range(cols+1)]
    if len(by)!=rows+1: by=[int(round(i*H/rows)) for i in range(rows+1)]
    boxes=[]
    for r in range(rows):
        for c in range(cols):
            boxes.append((bx[c],by[r],bx[c+1],by[r+1]))
    return boxes, bx, by, binv

def build_grid(crop_gray, rows, cols, mode="uniform", row_weights=None, col_weights=None):
    mode = (mode or "uniform").lower()
    if mode == "uniform":  return uniform_grid(crop_gray, rows, cols)
    if mode == "weighted": return weighted_grid(crop_gray, rows, cols, row_weights, col_weights)
    return auto_grid(crop_gray, rows, cols)

def _apply_inner_pad(cell, pad_x=0.0, pad_y=0.0):
    H, W = cell.shape[:2]
    dx = int(round(pad_x * W)); dy = int(round(pad_y * H))
    x1 = min(max(0, dx), W-2); y1 = min(max(0, dy), H-2)
    x2 = max(x1+1, W-dx);      y2 = max(y1+1, H-dy)
    return cell[y1:y2, x1:x2]

def cell_count(cell_gray, pad_x=0.0, pad_y=0.0):
    if pad_x>0 or pad_y>0:
        cell_gray = _apply_inner_pad(cell_gray, pad_x, pad_y)
    th = cv2.threshold(cell_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return int(cv2.countNonZero(th)), th.size

def pick_choice(counts, abs_min=None, min_ratio=None, top2_delta_ratio=None):
    abs_min = ABS_MIN_BLACK_DEFAULT if abs_min is None else abs_min
    min_ratio = MIN_FILL_RATIO_DEFAULT if min_ratio is None else min_ratio
    top2_delta_ratio = TOP2_DELTA_RATIO_DEFAULT if top2_delta_ratio is None else top2_delta_ratio
    arr = np.array(counts, dtype=np.float32)
    k = int(np.argmax(arr))
    if arr[k] < abs_min: return None
    if arr[k] < min_ratio * (arr.mean()+1e-6): return None
    if arr.size >= 2:
        top2 = np.partition(arr, -2)[-2:]
        if (top2[-1]-top2[-2]) < top2_delta_ratio * max(1.0, top2[-1]): return None
    return k

def read_digit_zone(gray, rect_px, rows, cols, grid_mode="uniform",
                    row_weights=None, col_weights=None):
    x1,y1,x2,y2 = rect_px
    crop = gray[y1:y2, x1:x2]
    boxes,_,_,_ = build_grid(crop, rows, cols, grid_mode, row_weights, col_weights)
    digits=[]
    for c in range(cols):
        counts=[]
        for r in range(rows):
            bx1,by1,bx2,by2 = boxes[r*cols + c]
            cell = crop[by1:by2, bx1:bx2]
            cnt,_ = cell_count(cell, 0.08, 0.12)
            counts.append(cnt)
        d = pick_choice(counts)
        digits.append(None if d is None else d)
    return digits, boxes

def read_answer_block(gray, rect_px, rows, cols, q_start,
                      col_labels="ABCDE", grid_mode="uniform",
                      row_weights=None, col_weights=None,
                      cell_pad=(0.12,0.18), col_pad_x=None,
                      abs_min=None, min_ratio=None, top2_delta_ratio=None):
    x1,y1,x2,y2 = rect_px
    crop = gray[y1:y2, x1:x2]
    boxes,_,_,_ = build_grid(crop, rows, cols, grid_mode, row_weights, col_weights)
    answers={}
    labels = (col_labels if col_labels else "ABCDE")
    for r in range(rows):
        counts=[]
        for c in range(cols):
            bx1,by1,bx2,by2 = boxes[r*cols + c]
            cell = crop[by1:by2, bx1:bx2]
            px = (col_pad_x[c] if (col_pad_x and c < len(col_pad_x)) else cell_pad[0])
            py = cell_pad[1]
            cnt,_ = cell_count(cell, px, py)
            counts.append(cnt)

        k = pick_choice(counts, abs_min, min_ratio, top2_delta_ratio)
        # ไม่ตอบ -> ค่าว่าง "", ไม่ fallback เป็น "E"
        picked = "" if k is None else (labels[k] if k < len(labels) else "")
        qno = q_start + r
        if qno >= 1:
            answers[qno] = picked
    return answers, boxes

# ============== GRID PREVIEW ==============
def build_grid_for_preview(pil_img, rect_pct, rows, cols,
                           grid_mode="uniform", row_weights=None, col_weights=None,
                           rect_color=(0,255,255), grid_color=(0,255,0)):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    x1,y1,x2,y2 = pct_to_px(rect_pct, W, H)
    cv2.rectangle(img, (x1,y1), (x2,y2), rect_color, 2)
    crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    boxes, bx, by, _ = build_grid(crop_gray, int(rows), int(cols), grid_mode, row_weights, col_weights)
    for xx in bx[1:-1]:
        cv2.line(img, (x1+xx, y1), (x1+xx, y2), grid_color, 1)
    for yy in by[1:-1]:
        cv2.line(img, (x1, y1+yy), (x2, y1+yy), grid_color, 1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ===================== APP =====================
st.set_page_config(page_title="OMR Grid & Mark Designer", layout="wide")
st.title("🖱️ OMR Grid & Mark Designer — Mouse Drag Only")

with st.sidebar:
    st.header("1) อัปโหลดไฟล์")
    key_file = st.file_uploader("กระดาษเฉลย (รูปภาพ)", type=["png","jpg","jpeg","bmp","tif","tiff"], key="up_key")
    stu_file = st.file_uploader("กระดาษนิสิต (รูปภาพ)", type=["png","jpg","jpeg","bmp","tif","tiff"], key="up_stu")

    st.markdown("---")
    st.header("2) Layout JSON")
    layout_file = st.file_uploader("Import layout.json", type=["json"], key="up_layout_json")
    export_btn = st.button("💾 Export layout.json", key="btn_export_json")

    st.markdown("---")
    st.header("3) เกณฑ์ OMR (global)")
    ABS_MIN_BLACK_DEFAULT  = st.number_input("abs_min_black", 0, 9999, 35, key="g_absmin")
    MIN_FILL_RATIO_DEFAULT = st.slider("min_fill_ratio", 0.0, 1.0, 0.22, 0.01, key="g_minratio")
    TOP2_DELTA_RATIO_DEFAULT = st.slider("top2_delta_ratio", 0.0, 1.0, 0.10, 0.01, key="g_top2")

    st.markdown("---")
    auto_compare = st.checkbox("เทียบอัตโนมัติเมื่อมีไฟล์นิสิต", value=True, key="auto_compare_on")

# Session state
def _init_state():
    d = st.session_state
    d.setdefault("digit_zones", [])
    d.setdefault("answer_blocks", [])
    d.setdefault("preview_rows", 25)
    d.setdefault("preview_cols", 5)
    d.setdefault("preview_grid_mode", "uniform")
    d.setdefault("preview_roww", "")
    d.setdefault("preview_colw", "")
    d.setdefault("last_rect_pct", None)
    d.setdefault("canvas_w", 1000)
    d.setdefault("canvas_key_ver", 0)
    d.setdefault("last_run_hash", "")
_init_state()

# Import layout
if layout_file is not None:
    try:
        data = json.loads(layout_file.read().decode("utf-8"))
        st.session_state.digit_zones   = data.get("ZONES_UI", [])
        st.session_state.answer_blocks = data.get("ANSWER_BLOCKS_UI", [])
        st.success("นำเข้า layout สำเร็จ")
    except Exception as e:
        st.error(f"อ่าน layout ไม่ได้: {e}")

# ======= DESIGN AREA =======
st.markdown("## พื้นที่ออกแบบ (ลากกรอบ + พรีวิวกริด)")

bg_img = None
if key_file is not None:
    bg_img_bgr = pillow_read_to_bgr(key_file)
    bg_img = Image.fromarray(cv2.cvtColor(bg_img_bgr, cv2.COLOR_BGR2RGB))
else:
    st.info("อัปโหลดภาพเฉลยเพื่อเริ่มออกแบบ")

canvas_w = st.slider("ความกว้างผืนผ้าใบ (แสดงผล)", 600, 1400, int(st.session_state.canvas_w), 50, key="canvas_w")
if bg_img is not None:
    ratio = canvas_w / bg_img.width
    canvas_h = int(bg_img.height * ratio)
else:
    canvas_h = 600

st.markdown("#### Grid Preview Controls")
pv_cols = st.columns(3)
with pv_cols[0]:
    st.session_state.preview_rows = st.number_input("Rows", 1, 400, int(st.session_state.preview_rows), key="pv_rows")
with pv_cols[1]:
    st.session_state.preview_cols = st.number_input("Cols", 1, 20,  int(st.session_state.preview_cols), key="pv_cols")
with pv_cols[2]:
    st.session_state.preview_grid_mode = st.selectbox("Grid mode", ["uniform","weighted","auto"],
        index=["uniform","weighted","auto"].index(st.session_state.preview_grid_mode), key="pv_gridmode")

if st.session_state.preview_grid_mode == "weighted":
    w_cols = st.columns(2)
    with w_cols[0]:
        st.session_state.preview_roww = st.text_input("row_weights (คั่น ,)", st.session_state.preview_roww, key="pv_roww")
    with w_cols[1]:
        st.session_state.preview_colw = st.text_input("col_weights (คั่น ,)", st.session_state.preview_colw, key="pv_colw")
else:
    st.session_state.preview_roww = ""
    st.session_state.preview_colw = ""

def _clamp_rect(r):
    x1,y1,x2,y2 = r
    x1 = max(0.0, min(0.99, float(x1)))
    y1 = max(0.0, min(0.99, float(y1)))
    x2 = max(x1+0.001, min(1.0, float(x2)))
    y2 = max(y1+0.001, min(1.0, float(y2)))
    return [x1,y1,x2,y2]

def _rect_from_canvas_obj(obj, imgW, imgH, canvasW, canvasH):
    left, top, width, height = obj.get("left"), obj.get("top"), obj.get("width"), obj.get("height")
    if left is None or top is None or width is None or height is None: return None
    sx = imgW / canvasW; sy = imgH / canvasH
    x1 = left*sx; y1 = top*sy; x2 = (left+width)*sx; y2=(top+height)*sy
    x1p = max(0.0, min(1.0, x1/imgW)); y1p = max(0.0, min(1.0, y1/imgH))
    x2p = max(0.0, min(1.0, x2/imgW)); y2p = max(0.0, min(1.0, y2/imgH))
    if x2p <= x1p + 0.002: x2p = x1p + 0.003
    if y2p <= y1p + 0.002: y2p = y1p + 0.003
    return [x1p,y1p,x2p,y2p]

def _set_rect(r):
    st.session_state.last_rect_pct = _clamp_rect(r)

# ===== Canvas =====
st.markdown("#### Canvas")
if USE_CANVAS and bg_img is not None:
    mode = st.radio("โหมด", ["วาดสี่เหลี่ยม (rect)", "แปลง/ย้าย (transform)"], horizontal=True, key="canvas_mode")
    draw_mode = "rect" if mode.startswith("วาด") else "transform"

    canvas_key = f"canvas_{st.session_state.canvas_key_ver}"
    canvas_result = safe_st_canvas(
        bg_img,
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_width=2,
        stroke_color="#ff0000",
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=draw_mode,
        key=canvas_key,
    )

    # อัปเดตกรอบล่าสุดอัตโนมัติ
    if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
        objs = canvas_result.json_data.get("objects", [])
        rect = next((o for o in reversed(objs) if o.get("type")=="rect"), None)
        if rect:
            _set_rect(_rect_from_canvas_obj(rect, bg_img.width, bg_img.height, canvas_w, canvas_h))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("รีเซ็ต Canvas (ล้างกรอบ)", key="btn_clear_canvas"):
            st.session_state.canvas_key_ver += 1
            st.session_state.last_rect_pct = None
            st.experimental_rerun()
    with c2:
        if st.button("รีเซ็ตกรอบพรีวิว", key="btn_reset_rect"):
            st.session_state.last_rect_pct = [0.05, 0.20, 0.30, 0.45]
else:
    if not USE_CANVAS:
        st.error("ไม่พบคอมโพเนนต์ Canvas (ติดตั้ง: streamlit-drawable-canvas==0.9.3 หรือ streamlit-drawable-canvas-fix)")

# ===== Preview & read-only coords =====
st.markdown("#### พรีวิวกริด + พิกัดกรอบ (อ่านอย่างเดียว)")
col_img, col_info = st.columns([6,2])

with col_img:
    if bg_img is None:
        st.info("อัปโหลดภาพเฉลยก่อน")
    else:
        if st.session_state.last_rect_pct is None:
            st.warning("ยังไม่มีกรอบจากการลาก — ลากสี่เหลี่ยมบน Canvas ด้านบนก่อน")
        else:
            roww = [float(x) for x in st.session_state.preview_roww.split(",") if x.strip()] if st.session_state.preview_roww else None
            colw = [float(x) for x in st.session_state.preview_colw.split(",") if x.strip()] if st.session_state.preview_colw else None
            preview_img = build_grid_for_preview(
                bg_img, st.session_state.last_rect_pct,
                st.session_state.preview_rows, st.session_state.preview_cols,
                grid_mode=st.session_state.preview_grid_mode,
                row_weights=roww, col_weights=colw
            )
            show_image(preview_img, caption=f"พรีวิว {st.session_state.preview_rows}x{st.session_state.preview_cols} ({st.session_state.preview_grid_mode})", width=canvas_w)

with col_info:
    st.markdown("**พิกัด (สัดส่วน 0..1)**")
    if st.session_state.last_rect_pct:
        x1,y1,x2,y2 = st.session_state.last_rect_pct
        st.code(f"x1={x1:.4f}\ny1={y1:.4f}\nx2={x2:.4f}\ny2={y2:.4f}", language="text")
    else:
        st.code("x1=?\ny1=?\nx2=?\ny2=?", language="text")

# ===== Add blocks from current rect =====
st.subheader("เพิ่มบล็อกจากกรอบที่ลาก")
cols_add = st.columns(2)

with cols_add[0]:
    with st.expander("➕ เพิ่ม Digit Zone", expanded=True):
        dz_name = st.text_input("ชื่อโซน", key="dz_name")
        dz_rows = st.number_input("rows (แถว 0-9)", 1, 50, min(50, int(st.session_state.preview_rows or 10)), key="dz_rows")
        dz_cols = st.number_input("cols (จำนวนหลัก)", 1, 50, min(50, int(st.session_state.preview_cols or 6)), key="dz_cols")
        dz_grid = st.selectbox("grid mode", ["uniform","weighted","auto"],
                               index=["uniform","weighted","auto"].index(st.session_state.preview_grid_mode),
                               key="dz_grid")
        dz_roww = st.text_input("row_weights (weighted)", value=st.session_state.preview_roww, key="dz_roww")
        dz_colw = st.text_input("col_weights (weighted)", value=st.session_state.preview_colw, key="dz_colw")

        if st.button("เพิ่มเป็น Digit Zone", key="btn_add_dz"):
            if st.session_state.last_rect_pct is None:
                st.warning("ยังไม่มีกรอบจากการลาก")
            else:
                row_weights = [float(x) for x in dz_roww.split(",") if x.strip()] if dz_roww.strip() else []
                col_weights = [float(x) for x in dz_colw.split(",") if x.strip()] if dz_colw.strip() else []
                st.session_state.digit_zones.append({
                    "name": dz_name or f"digit_zone_{len(st.session_state.digit_zones)+1}",
                    "rect_pct": st.session_state.last_rect_pct,
                    "rows": int(dz_rows),
                    "cols": int(dz_cols),
                    "grid": dz_grid,
                    "row_weights": row_weights,
                    "col_weights": col_weights,
                    "visible": True
                })
                st.success("เพิ่ม Digit Zone แล้ว")

with cols_add[1]:
    with st.expander("➕ เพิ่ม Answer Block", expanded=True):
        ab_name = st.text_input("ชื่อบล็อก", key="ab_name")
        ab_rows = st.number_input("rows (จำนวนข้อ)", 1, 400, int(st.session_state.preview_rows or 25), key="ab_rows")
        ab_cols = st.number_input("cols (ตัวเลือก/แถว)", 1, 10, int(st.session_state.preview_cols or 5), key="ab_cols")
        ab_qstart = st.number_input("เริ่มข้อที่ (q_start)", 1, 10000, 1, key="ab_qstart")
        ab_labels = st.text_input("ป้ายตัวเลือก", value="ABCDE", key="ab_labels")
        ab_grid = st.selectbox("grid mode ", ["uniform","weighted","auto"],
                               index=["uniform","weighted","auto"].index(st.session_state.preview_grid_mode),
                               key="ab_grid")
        ab_roww = st.text_input("row_weights (weighted)", value=st.session_state.preview_roww, key="ab_roww")
        ab_colw = st.text_input("col_weights (weighted)", value=st.session_state.preview_colw, key="ab_colw")
        st.markdown("**Padding**")
        ab_padx = st.slider("cell_pad[0] (แนวนอน)", 0.0, 0.5, 0.12, 0.01, key="ab_padx")
        ab_pady = st.slider("cell_pad[1] (แนวตั้ง)", 0.0, 0.5, 0.18, 0.01, key="ab_pady")
        ab_colpad = st.text_input("col_pad_x (เช่น 0.12,0.12,0.12,0.12,0.26)", value="", key="ab_colpad")
        st.markdown("**เกณฑ์จำแนก** (ว่าง=ใช้ global)")
        ab_absmin   = st.text_input("abs_min_black", value="", key="ab_absmin")
        ab_minratio = st.text_input("min_fill_ratio", value="", key="ab_minratio")
        ab_top2     = st.text_input("top2_delta_ratio", value="", key="ab_top2")

        if st.button("เพิ่มเป็น Answer Block", key="btn_add_ab"):
            if st.session_state.last_rect_pct is None:
                st.warning("ยังไม่มีกรอบจากการลาก")
            else:
                row_weights = [float(x) for x in ab_roww.split(",") if x.strip()] if ab_roww.strip() else []
                col_weights = [float(x) for x in ab_colw.split(",") if x.strip()] if ab_colw.strip() else []
                col_pad_x = [float(x) for x in ab_colpad.split(",") if x.strip()] if ab_colpad.strip() else []
                absmin   = None if ab_absmin.strip()=="" else float(ab_absmin.strip())
                minratio = None if ab_minratio.strip()=="" else float(ab_minratio.strip())
                top2     = None if ab_top2.strip()=="" else float(ab_top2.strip())
                st.session_state.answer_blocks.append({
                    "name": ab_name or f"ans_block_{len(st.session_state.answer_blocks)+1}",
                    "rect_pct": st.session_state.last_rect_pct,
                    "rows": int(ab_rows),
                    "cols": int(ab_cols),
                    "q_start": int(ab_qstart),
                    "col_labels": ab_labels or "ABCDE",
                    "grid": ab_grid,
                    "row_weights": row_weights,
                    "col_weights": col_weights,
                    "cell_pad": [ab_padx, ab_pady],
                    "col_pad_x": col_pad_x,
                    "abs_min_black": absmin,
                    "min_fill_ratio": minratio,
                    "top2_delta_ratio": top2,
                    "visible": True
                })
                st.success("เพิ่ม Answer Block แล้ว")

# ===== Layout summary =====
st.markdown("### Layout ปัจจุบัน")
lcol, rcol = st.columns(2)
with lcol:
    st.markdown("**Digit Zones**")
    if not st.session_state.digit_zones:
        st.info("ยังไม่มี Digit Zone")
    else:
        for i, dz in enumerate(st.session_state.digit_zones):
            c1,c2,c3 = st.columns([6,2,2])
            c1.write(f"{i+1}. `{dz['name']}` rows={dz['rows']} cols={dz['cols']} grid={dz['grid']}")
            st.session_state.digit_zones[i]["visible"] = c2.checkbox("แสดง", value=dz.get("visible", True), key=f"dz_vis_{i}")
            if c3.button("ลบ", key=f"dz_del_{i}"):
                st.session_state.digit_zones.pop(i); st.experimental_rerun()
with rcol:
    st.markdown("**Answer Blocks**")
    if not st.session_state.answer_blocks:
        st.info("ยังไม่มี Answer Block")
    else:
        for i, ab in enumerate(st.session_state.answer_blocks):
            c1,c2,c3 = st.columns([6,2,2])
            c1.write(f"{i+1}. `{ab['name']}` q_start={ab['q_start']} rows={ab['rows']} cols={ab['cols']} labels={ab['col_labels']}")
            st.session_state.answer_blocks[i]["visible"] = c2.checkbox("แสดง", value=ab.get("visible", True), key=f"ab_vis_{i}")
            if c3.button("ลบ", key=f"ab_del_{i}"):
                st.session_state.answer_blocks.pop(i); st.experimental_rerun()

# ===== Export layout =====
export_payload = {
    "ZONES_UI": st.session_state.digit_zones,
    "ANSWER_BLOCKS_UI": st.session_state.answer_blocks,
    "global_defaults": {
        "abs_min_black": ABS_MIN_BLACK_DEFAULT,
        "min_fill_ratio": MIN_FILL_RATIO_DEFAULT,
        "top2_delta_ratio": TOP2_DELTA_RATIO_DEFAULT,
    }
}
if export_btn:
    b = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ ดาวน์โหลด layout.json", data=b, file_name="layout.json", mime="application/json", key="dl_layout_json")

# ===== Processing =====
st.markdown("---")
st.header("ประมวลผล OMR → CSV + คะแนน")

col_proc1, col_proc2 = st.columns(2)
with col_proc1:
    do_run = st.button("🚀 ประมวลผล (อ่านเฉลย + นิสิต)", key="btn_run")
with col_proc2:
    st.caption("ระบบจะใช้เลย์เอาท์เดียวกันกับทั้งสองภาพ")

def read_all(img_file, layout):
    img_bgr = pillow_read_to_bgr(img_file)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = {"answers": {}, "digit": {}}

    # Digit zones
    for dz in layout["ZONES_UI"]:
        rect_px = pct_to_px(dz["rect_pct"], W, H)
        x1,y1,x2,y2 = rect_px
        digits,_ = read_digit_zone(
            gray, rect_px, int(dz["rows"]), int(dz["cols"]),
            dz.get("grid","uniform"), dz.get("row_weights") or None, dz.get("col_weights") or None
        )
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        # เก็บทั้งดิบ (list) และ string (digit[name])
        result[dz["name"]] = digits
        s = "".join(str(d) if d is not None else "" for d in digits)
        result["digit"][dz["name"]] = s

    # Answer blocks
    for ab in layout["ANSWER_BLOCKS_UI"]:
        rect_px = pct_to_px(ab["rect_pct"], W, H)
        x1,y1,x2,y2 = rect_px
        abs_min = ab.get("abs_min_black"); min_ratio = ab.get("min_fill_ratio"); top2 = ab.get("top2_delta_ratio")
        if abs_min  is None: abs_min  = ABS_MIN_BLACK_DEFAULT
        if min_ratio is None: min_ratio = MIN_FILL_RATIO_DEFAULT
        if top2    is None:   top2     = TOP2_DELTA_RATIO_DEFAULT

        ans, boxes = read_answer_block(
            gray, rect_px, int(ab["rows"]), int(ab["cols"]), int(ab["q_start"]),
            col_labels=ab.get("col_labels","ABCDE"), grid_mode=ab.get("grid","uniform"),
            row_weights=ab.get("row_weights") or None, col_weights=ab.get("col_weights") or None,
            cell_pad=tuple(ab.get("cell_pad",[0.12,0.18])), col_pad_x=ab.get("col_pad_x") or None,
            abs_min=abs_min, min_ratio=min_ratio, top2_delta_ratio=top2
        )
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
        for (bx1,by1,bx2,by2) in boxes:
            cv2.rectangle(vis,(x1+bx1,y1+by1),(x1+bx2,y1+by2),(0,0,255),1)
        result["answers"].update(ans)

    return result, vis

def run_omr_and_export(img_file, layout, who, canvas_w_hint=900):
    res, vis = read_all(img_file, layout)
    ts = time.strftime("%Y%m%d-%H%M%S")
    overlay_path = OUT_DIR / f"{who}_overlay_{ts}.jpg"
    meta_path    = OUT_DIR / f"{who}_meta_{ts}.json"
    csv_path     = OUT_DIR / f"{who}_answers_{ts}.csv"

    # บันทึกภาพ overlay
    cv2.imwrite(str(overlay_path), vis)

    # บันทึก JSON meta (รวม digit string แล้ว)
    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # บันทึก CSV (ไม่ตอบ = ช่องว่าง)
    import csv
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["question","answer"])
        for q in sorted(res["answers"].keys()):
            a = res["answers"][q] or ""   # ช่องว่างถ้าไม่ตอบ
            w.writerow([q, a])

    # แสดงตัวอย่าง overlay
    show_image(str(overlay_path), caption=f"Overlay ({who})", width=canvas_w_hint)

    # ปุ่มโหลด
    with open(csv_path,"rb") as f:
        st.download_button(f"⬇️ ดาวน์โหลด CSV ({who})", f.read(), file_name=Path(csv_path).name, mime="text/csv", key=f"dl_{who}_csv_{Path(csv_path).name}")
    with open(meta_path,"rb") as f:
        st.download_button(f"⬇️ ดาวน์โหลด JSON meta ({who})", f.read(), file_name=Path(meta_path).name, mime="application/json", key=f"dl_{who}_json_{Path(meta_path).name}")

    return res, overlay_path, meta_path, csv_path

def build_block_score_summary(layout_answer_blocks, key_ans: dict, stu_ans: dict):
    """คืนค่า (blocks_summary, sum_correct, total_all)"""
    blocks_summary = []
    sum_correct = 0
    total_all = 0
    for ab in layout_answer_blocks:
        name = ab.get("name","block")
        q_start = int(ab.get("q_start",1))
        rows = int(ab.get("rows",0))
        q_range = list(range(q_start, q_start + rows))

        correct = 0
        for q in q_range:
            k = key_ans.get(q, "")
            s = stu_ans.get(q, "")
            if s != "" and s == k:
                correct += 1

        blocks_summary.append({
            "name": name,
            "q_start": q_start,
            "rows": rows,
            "sum_correct": correct,
            "total": rows
        })
        sum_correct += correct
        total_all += rows
    return blocks_summary, sum_correct, total_all

def do_compare_and_show(key_file, stu_file, layout_payload, canvas_w_hint=900):
    st.info("กำลังอ่านกระดาษ 'เฉลย' ...")
    key_result, *_ = run_omr_and_export(key_file, layout_payload, who="key", canvas_w_hint=canvas_w_hint)
    st.success("อ่านเฉลยเสร็จ")

    st.info("กำลังอ่านกระดาษ 'นิสิต' ...")
    stu_result, *_ = run_omr_and_export(stu_file, layout_payload, who="student", canvas_w_hint=canvas_w_hint)
    st.success("อ่านนิสิตเสร็จ")

    # คะแนนรวม
    key_ans = key_result["answers"]; stu_ans = stu_result["answers"]
    blocks_summary, sum_correct, total_all = build_block_score_summary(layout_payload["ANSWER_BLOCKS_UI"], key_ans, stu_ans)
    score_pct = (100.0*sum_correct/total_all) if total_all>0 else 0.0

    st.markdown("### ✅ สรุปคะแนนนิสิตเทียบเฉลย (รวมทุกบล็อก)")
    c1,c2,c3 = st.columns(3)
    c1.metric("ถูก (รวม)", sum_correct)
    c2.metric("จำนวนข้อทั้งหมด", total_all)
    c3.metric("คะแนน (%)", f"{score_pct:.2f}")

    # JSON สรุปคะแนน
    summary_json = {
        "digit": stu_result.get("digit", {}),
        "answer_student": stu_ans,
        "answer_key": key_ans,
        "blocks": blocks_summary,
        "sum_answer_block": sum_correct,
        "score_percent": round(score_pct, 2)
    }
    summary_path = OUT_DIR / f"compare_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    with open(summary_path, "rb") as f:
        st.download_button("⬇️ ดาวน์โหลด JSON สรุปคะแนน (นิสิต)", f.read(), file_name=summary_path.name, mime="application/json", key=f"dl_compare_json_{summary_path.name}")

# Manual run
if do_run:
    if key_file is None or stu_file is None:
        st.error("กรุณาอัปโหลดรูปทั้ง 'เฉลย' และ 'นิสิต' ก่อน")
    elif not st.session_state.answer_blocks:
        st.error("ต้องเพิ่ม Answer Block อย่างน้อย 1 รายการ")
    else:
        layout_payload = {"ZONES_UI": st.session_state.digit_zones, "ANSWER_BLOCKS_UI": st.session_state.answer_blocks}
        do_compare_and_show(key_file, stu_file, layout_payload, canvas_w_hint=canvas_w)

# Auto compare on change (ถ้าเปิด)
def _auto_hash():
    m = hashlib.md5()
    m.update(b"1" if key_file else b"0")
    m.update(b"1" if stu_file else b"0")
    m.update(json.dumps(st.session_state.answer_blocks, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return m.hexdigest()

if st.session_state.get("auto_compare_on", True):
    h = _auto_hash()
    if h != st.session_state.last_run_hash:
        st.session_state.last_run_hash = h
        if (key_file is not None) and (stu_file is not None) and st.session_state.answer_blocks:
            layout_payload = {"ZONES_UI": st.session_state.digit_zones, "ANSWER_BLOCKS_UI": st.session_state.answer_blocks}
            do_compare_and_show(key_file, stu_file, layout_payload, canvas_w_hint=canvas_w)

st.markdown("---")
st.markdown("""
**Tips**
- ลากกรอบบน Canvas ได้เลย (โหมด Transform ช่วยเลื่อน/ย่อ/ขยายกรอบเดิม)
- ตั้ง Rows/Cols + โหมดกริด แล้วดูพรีวิว
- เพิ่มเป็น Digit Zone / Answer Block จากกรอบที่ลาก
- Export layout.json เพื่อใช้กับรูปนิสิต (ฟอร์แมตเดียวกัน)
- ไม่ตอบ = ค่าว่าง ("") ทั้งใน CSV และ JSON
""")
