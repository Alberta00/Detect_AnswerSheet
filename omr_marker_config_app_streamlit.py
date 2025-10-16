# app.py — OMR Configurator (let users place marks/sections themselves)
# Usage:
#   1) pip install streamlit opencv-python pillow numpy streamlit-drawable-canvas
#   2) streamlit run app.py
# What it does:
#   • Upload an answer sheet image
#   • Draw rectangles for ZONES (subject_code, student_id, etc.) and ANSWER_BLOCKS (answers)
#   • For each rectangle, set rows/cols and (for answers) start question + choices
#   • Exports a config.json the same shape your Detect.py expects
#   • Preview an overlay (grid + boxes) to verify alignment before saving

import json
import io
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --------------------- Helpers ---------------------
@dataclass
class ZoneDef:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    rows: int
    cols: int

@dataclass
class AnswerBlockDef:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    rows: int
    cols: int
    q_start: int
    choices: int


def clamp01(a: float) -> float:
    return float(max(0.0, min(1.0, a)))


def rect_abs_to_pct(rect_px, W, H):
    x1, y1, x2, y2 = rect_px
    x1p = clamp01(x1 / W); y1p = clamp01(y1 / H)
    x2p = clamp01(x2 / W); y2p = clamp01(y2 / H)
    # ensure ordering
    if x2p < x1p:
        x1p, x2p = x2p, x1p
    if y2p < y1p:
        y1p, y2p = y2p, y1p
    return x1p, y1p, x2p, y2p


def pct_to_px(rect_pct, W, H):
    x1p, y1p, x2p, y2p = rect_pct
    x1 = int(round(x1p * W)); y1 = int(round(y1p * H))
    x2 = int(round(x2p * W)); y2 = int(round(y2p * H))
    x1 = max(0, min(x1, W-2)); y1 = max(0, min(y1, H-2))
    x2 = max(1, min(x2, W-1)); y2 = max(1, min(y2, H-1))
    if x2 <= x1 + 2: x2 = x1 + 3
    if y2 <= y1 + 2: y2 = y1 + 3
    return [x1, y1, x2, y2]


def smooth1d(a, k=15):
    k = max(3, int(k//2*2+1))
    ker = np.ones(k, np.float32)/k
    return np.convolve(a.astype(np.float32), ker, mode="same")


def borders_from_proj(proj, nsplit, search_ratio):
    L = len(proj); sp = smooth1d(proj, max(9, L//200))
    borders = [0]
    for i in range(1, nsplit+1):
        x0 = int(round(i * L / (nsplit+1)))
        win = max(2, int(L * search_ratio))
        a = max(1, x0 - win); b = min(L-2, x0 + win)
        x = a + int(np.argmin(sp[a:b+1])); borders.append(x)
    borders.append(L)
    for j in range(1, len(borders)):
        borders[j] = max(borders[j], borders[j-1] + 1)
    return borders


def auto_grid(crop_gray, rows, cols):
    g = cv2.GaussianBlur(crop_gray, (3,3), 0)
    binv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    H, W = binv.shape
    bx = borders_from_proj(binv.sum(axis=0), cols-1, 0.10)
    by = borders_from_proj(binv.sum(axis=1), rows-1, 0.08)
    if len(bx) != cols+1: bx = [int(round(i*W/cols)) for i in range(cols+1)]
    if len(by) != rows+1: by = [int(round(i*H/rows)) for i in range(rows+1)]
    boxes = []
    for r in range(rows):
        for c in range(cols):
            boxes.append((bx[c], by[r], bx[c+1], by[r+1]))
    return boxes, bx, by, binv


def draw_overlay(base_rgb, defs_zones, defs_answers):
    vis = base_rgb.copy()
    H, W = vis.shape[:2]
    for z in defs_zones:
        rect = pct_to_px((z.x1, z.y1, z.x2, z.y2), W, H)
        x1,y1,x2,y2 = rect
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0, 200, 0), 2)
        crop = cv2.cvtColor(vis[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        boxes, *_ = auto_grid(crop, z.rows, z.cols)
        for (cx1,cy1,cx2,cy2) in boxes:
            cv2.rectangle(vis, (x1+cx1,y1+cy1), (x1+cx2, y1+cy2), (255, 0, 0), 1)
        cv2.putText(vis, z.name, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv2.LINE_AA)
    for a in defs_answers:
        rect = pct_to_px((a.x1, a.y1, a.x2, a.y2), W, H)
        x1,y1,x2,y2 = rect
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255, 220, 0), 2)
        crop = cv2.cvtColor(vis[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        boxes, *_ = auto_grid(crop, a.rows, a.cols)
        for (cx1,cy1,cx2,cy2) in boxes:
            cv2.rectangle(vis, (x1+cx1,y1+cy1), (x1+cx2, y1+cy2), (255, 0, 0), 1)
        cv2.putText(vis, f"{a.name} [{a.q_start}:{a.q_start+a.rows-1}]", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1, cv2.LINE_AA)
    return vis


# --------------------- UI ---------------------
st.set_page_config(page_title="OMR Configurator", layout="wide")
st.title("🖊️ OMR Configurator — วางกรอบด้วยตัวเอง")

with st.sidebar:
    st.markdown("### 1) อัปโหลดภาพกระดาษคำตอบ")
    up = st.file_uploader("ภาพคำตอบ (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=False)
    dpi = st.number_input("ย่อภาพแสดงผล (px กว้าง)", 600, 2200, 1400, step=100)
    st.markdown("---")
    st.markdown("### 2) โหมดการวาด")
    mode = st.radio("คุณกำลังวางกรอบสำหรับ", ["ZONES (ตัวเลขแนวตั้ง)", "ANSWER_BLOCKS (คำตอบ)"])
    draw_color = "#FF0000" if mode.startswith("ANSWER") else "#00AA00"
    stroke = st.slider("ความหนาเส้น", 1, 6, 2)
    st.markdown("---")
    st.markdown("### 3) รายการกรอบที่บันทึกไว้")

# โหลดภาพ
if up is None:
    st.info("⬆️ กรุณาอัปโหลดภาพก่อน")
    st.stop()

image = Image.open(up).convert("RGB")
W0, H0 = image.size
scale = dpi / float(W0)
H_disp = int(H0 * scale)
img_disp = image.resize((dpi, H_disp), Image.BILINEAR)
img_disp_np = np.array(img_disp)

# Canvas for drawing
st.markdown("### วาดกรอบ (ลาก Mouse สร้างสี่เหลี่ยม) แล้วกดปุ่ม 'เพิ่มกรอบ' ด้านล่าง")
canvas_res = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # transparent fill
    stroke_width=stroke,
    stroke_color=draw_color,
    background_image=img_disp,
    height=H_disp,
    width=dpi,
    drawing_mode="rect",
    key="canvas",
)

# Session state for collected rects
if "zones" not in st.session_state:
    st.session_state.zones = []  # list[ZoneDef]
if "answers" not in st.session_state:
    st.session_state.answers = []  # list[AnswerBlockDef]

# Read last rectangle drawn from canvas json
last_rect = None
if canvas_res and canvas_res.json_data is not None:
    objs = canvas_res.json_data.get("objects", [])
    if objs:
        # pick the latest object
        o = objs[-1]
        if o.get("type") == "rect":
            left = o.get("left", 0)
            top = o.get("top", 0)
            width = o.get("width", 0)
            height = o.get("height", 0)
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(dpi-1, int(left + width))
            y2 = min(H_disp-1, int(top + height))
            # convert to original image percentage
            # first convert canvas px -> original px
            Wc, Hc = dpi, H_disp
            x1p, y1p, x2p, y2p = rect_abs_to_pct((x1 * (W0/Wc), y1 * (H0/Hc), x2 * (W0/Wc), y2 * (H0/Hc)), W0, H0)
            last_rect = (x1p, y1p, x2p, y2p)

col1, col2 = st.columns(2)
with col1:
    st.subheader("เพิ่มกรอบจากสี่เหลี่ยมล่าสุดที่วาด")
    if last_rect is None:
        st.warning("ยังไม่มีสี่เหลี่ยมบน canvas — ลากเพื่อสร้างก่อน")
    else:
        st.success("จับกรอบล่าสุดได้แล้ว พร้อมเพิ่มข้อมูล")

    if mode.startswith("ZONES"):
        z_name = st.text_input("ชื่อโซน (เช่น subject_code / student_id)", value="subject_code")
        z_rows = st.number_input("rows (จำนวนแถว)", 1, 200, 10)
        z_cols = st.number_input("cols (จำนวนคอลัมน์)", 1, 50, 6)
        if st.button("➕ เพิ่มโซน (ZONES)"):
            x1,y1,x2,y2 = last_rect
            st.session_state.zones.append(ZoneDef(z_name, x1,y1,x2,y2, int(z_rows), int(z_cols)))
            st.experimental_rerun()
    else:
        a_name = st.text_input("ชื่อบล็อก (เช่น ans_1_25)", value="ans_1_25")
        a_rows = st.number_input("rows (จำนวนข้อในบล็อก)", 1, 400, 25)
        a_cols = st.number_input("cols (ตัวเลือกต่อข้อ)", 1, 10, 5)
        a_qstart = st.number_input("หมายเลขข้อเริ่มต้น (q_start)", 1, 10000, 1)
        a_choices = st.number_input("จำนวนตัวเลือก (choices)", 1, 10, 5)
        if st.button("➕ เพิ่มบล็อกคำตอบ (ANSWER_BLOCKS)"):
            x1,y1,x2,y2 = last_rect
            st.session_state.answers.append(AnswerBlockDef(a_name, x1,y1,x2,y2, int(a_rows), int(a_cols), int(a_qstart), int(a_choices)))
            st.experimental_rerun()

with col2:
    st.subheader("รายการกรอบ")
    if st.session_state.zones:
        st.write("**ZONES**")
        for i, z in enumerate(st.session_state.zones):
            st.caption(f"{i+1}. {z.name} — rows={z.rows}, cols={z.cols}, rect_pct=({z.x1:.3f},{z.y1:.3f},{z.x2:.3f},{z.y2:.3f})")
    if st.session_state.answers:
        st.write("**ANSWER_BLOCKS**")
        for i, a in enumerate(st.session_state.answers):
            st.caption(f"{i+1}. {a.name} — rows={a.rows}, cols={a.cols}, q_start={a.q_start}, choices={a.choices}, rect_pct=({a.x1:.3f},{a.y1:.3f},{a.x2:.3f},{a.y2:.3f})")

st.markdown("---")

# Preview overlay
st.subheader("🔍 Preview Overlay (ตรวจตำแหน่งกริด)")
base_rgb = np.array(image)
preview = draw_overlay(base_rgb, st.session_state.zones, st.session_state.answers)
st.image(preview, caption="Overlay Preview", use_column_width=True)

# Export config.json (compatible with Detect.py)
st.markdown("### 💾 Export as config.json")
config = {
    "ZONES": {
        z.name: {
            "rect_pct": [z.x1, z.y1, z.x2, z.y2],
            "rows": z.rows,
            "cols": z.cols,
        } for z in st.session_state.zones
    },
    "ANSWER_BLOCKS": [
        [a.name, [a.x1, a.y1, a.x2, a.y2], a.rows, a.cols, a.q_start, a.choices]
        for a in st.session_state.answers
    ],
}

conf_bytes = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button("⬇️ ดาวน์โหลด config.json", data=conf_bytes, file_name="config.json", mime="application/json")

# Optional: Save to working folder
save_local = st.checkbox("บันทึกไฟล์ config.json ลงโฟลเดอร์นี้ด้วย")
if save_local:
    Path("config.json").write_bytes(conf_bytes)
    st.success("บันทึกแล้ว: config.json")

st.info("**วิธีใช้กับ Detect.py:** เพิ่มโค้ดโหลด config.json แล้วแทนค่า ZONES/ANSWER_BLOCKS ก่อนประมวลผล")
