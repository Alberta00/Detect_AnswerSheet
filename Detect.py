# Detect.py — OMR with adjustable grids + per-block anti-false-E
import cv2, numpy as np, json, csv
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ================= PATH / OUTPUT =================
BASE = Path(__file__).resolve().parent
IMG_PATH = str(BASE / "R1103.jpg")       # <-- เปลี่ยนชื่อภาพที่นี่
OUT_DIR  = BASE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================ GLOBAL CONFIG =================
ABS_MIN_BLACK  = 35     # ขั้นต่ำพิกเซลดำต่อช่อง (ค่าเริ่มต้น)
MIN_FILL_RATIO = 0.22   # (ดำสุด) ต้องมากกว่า mean ของคอลัมน์ * ค่านี้
TOP2_DELTA_RATIO = 0.10 # ช่องดำสุดต้องทิ้งห่างอันดับสอง >= 10%

# ================ ZONES (ตัวเลขแนวตั้ง) =================
ZONES = {
    "subject_code": {
        "rect_pct": (0.0485, 0.22, 0.22, 0.42),  # 6 หลัก x 10 แถว
        "rows": 10, "cols": 6
    },
    "student_id": {
        "rect_pct": (0.2455, 0.22, 0.47, 0.422), # 8 หลัก x 10 แถว
        "rows": 10, "cols": 8
    },
}

"""
ANSWER_BLOCKS tuple:
(name, rect_pct, rows, cols, q_start, choices, row_shift, col_labels, options_dict)

options_dict (ไม่บังคับ):
{
  "grid": "uniform"|"weighted"|"auto",
  "row_weights": [...],   # ถ้า grid=weighted
  "col_weights": [...],   # ถ้า grid=weighted
  "cell_pad": (px, py),   # padding ภายในช่อง (สัดส่วน 0..1 ของกว้าง/สูง) เช่น (0.12, 0.18)
  "col_pad_x": [..]*cols, # padding แนวนอนรายคอลัมน์ (override cell_pad[0])
  "abs_min_black": int,   # override เกณฑ์ขั้นต่ำดำ
  "min_fill_ratio": float,# override สัดส่วนเทียบ mean
  "top2_delta_ratio": float # override เกณฑ์ทิ้งห่างอันดับสอง
}
"""

ANSWER_BLOCKS = [
    # ซ้าย
    ("ans_1_25",  (0.077, 0.439, 0.22, 0.925), 25, 5,  1, 5,  0, "ABCDE",
        {"grid":"uniform",
         "cell_pad": (0.10, 0.16),
         "abs_min_black": 35, "min_fill_ratio": 0.24, "top2_delta_ratio": 0.12}
    ),
    ("ans_26_50", (0.275, 0.439, 0.42, 0.925), 25, 5, 26, 5,  0, "ABCDE",
        {"grid":"uniform",
         "cell_pad": (0.10, 0.16),
         "abs_min_black": 35, "min_fill_ratio": 0.24, "top2_delta_ratio": 0.12}
    ),

    # กลาง 51–90  (เพิ่ม padding คอลัมน์ E เพื่อกัน false positive)
    ("ans_51_90",  (0.500, 0.145, 0.640, 0.924), 40, 5,  51, 5,  0, "ABCDE",
        {"grid":"uniform",
         "cell_pad": (0.12, 0.18),
         "col_pad_x": [0.12, 0.12, 0.12, 0.12, 0.26],  # E บีบมากขึ้น
         "abs_min_black": 45, "min_fill_ratio": 0.28, "top2_delta_ratio": 0.18}
    ),

    # ขวา 91–170
    ("ans_91_130", (0.670, 0.145, 0.810, 0.923), 40, 5,  91, 5,  0, "ABCDE",
        {"grid":"uniform",
         "cell_pad": (0.12, 0.18),
         "col_pad_x": [0.12, 0.12, 0.12, 0.12, 0.26],
         "abs_min_black": 45, "min_fill_ratio": 0.28, "top2_delta_ratio": 0.18}
    ),
    ("ans_131_170",(0.840, 0.145, 0.978, 0.923), 40, 5, 131, 5,  0, "ABCDE",
        {"grid":"uniform",
         "cell_pad": (0.12, 0.18),
         "col_pad_x": [0.12, 0.12, 0.12, 0.12, 0.26],
         "abs_min_black": 45, "min_fill_ratio": 0.28, "top2_delta_ratio": 0.18}
    ),
]

# ================ UTILS =================
def pillow_read(path):
    pil = Image.open(path).convert("RGB")
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

# --------- GRID BUILDERS ----------
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
    if weights is None or len(weights) != n:
        weights = [1.0]*n
    s = float(sum(weights)) if sum(weights) > 0 else float(n)
    edges = [0]
    acc = 0.0
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
    if mode == "uniform":
        return uniform_grid(crop_gray, rows, cols)
    elif mode == "weighted":
        return weighted_grid(crop_gray, rows, cols, row_weights, col_weights)
    else:
        return auto_grid(crop_gray, rows, cols)

def draw_boxes(vis, boxes, offset, color=(0,0,255), thick=1):
    ox,oy = offset
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(vis, (ox+x1,oy+y1), (ox+x2,oy+y2), color, thick)

# ====== COUNT WITH INNER PADDING ======
def _apply_inner_pad(cell, pad_x=0.0, pad_y=0.0):
    H, W = cell.shape[:2]
    dx = int(round(pad_x * W))
    dy = int(round(pad_y * H))
    x1 = min(max(0, dx), W-2); y1 = min(max(0, dy), H-2)
    x2 = max(x1+1, W-dx);      y2 = max(y1+1, H-dy)
    return cell[y1:y2, x1:x2]

def cell_count(cell_gray, pad_x=0.0, pad_y=0.0):
    if pad_x>0 or pad_y>0:
        cell_gray = _apply_inner_pad(cell_gray, pad_x, pad_y)
    th = cv2.threshold(cell_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # ตัดเส้นขอบบาง ๆ ออก
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return int(cv2.countNonZero(th)), th.size

def pick_choice(counts, abs_min=None, min_ratio=None, top2_delta_ratio=None):
    abs_min = ABS_MIN_BLACK if abs_min is None else abs_min
    min_ratio = MIN_FILL_RATIO if min_ratio is None else min_ratio
    top2_delta_ratio = TOP2_DELTA_RATIO if top2_delta_ratio is None else top2_delta_ratio

    arr = np.array(counts, dtype=np.float32)
    k = int(np.argmax(arr))
    if arr[k] < abs_min: return None
    if arr[k] < min_ratio * (arr.mean()+1e-6): return None
    if arr.size >= 2:
        top2 = np.partition(arr, -2)[-2:]
        if (top2[-1] - top2[-2]) < top2_delta_ratio * max(1.0, top2[-1]):
            return None
    return k

# --------- READERS ----------
def read_digit_zone(gray, rect, rows, cols, grid_mode="uniform",
                    row_weights=None, col_weights=None):
    x1,y1,x2,y2 = rect
    crop = gray[y1:y2, x1:x2]
    boxes, _, _, _ = build_grid(crop, rows, cols, grid_mode, row_weights, col_weights)

    digits=[]
    for c in range(cols):
        counts=[]
        for r in range(rows):
            bx1,by1,bx2,by2 = boxes[r*cols + c]
            cell = crop[by1:by2, bx1:bx2]
            cnt, _ = cell_count(cell, 0.08, 0.12)  # padding เล็กน้อยสำหรับตัวเลข
            counts.append(cnt)
        d = pick_choice(counts)
        digits.append(None if d is None else d)
    return digits, boxes

def read_answer_block(gray, rect, rows, cols, q_start, choices=5, row_shift=0,
                      col_labels="ABCDE", grid_mode="uniform", row_weights=None, col_weights=None,
                      cell_pad=(0.12,0.18), col_pad_x=None,
                      abs_min=None, min_ratio=None, top2_delta_ratio=None):
    x1,y1,x2,y2 = rect
    crop = gray[y1:y2, x1:x2]
    boxes, _, _, _ = build_grid(crop, rows, cols, grid_mode, row_weights, col_weights)

    answers = {}
    for r in range(rows):
        counts=[]
        for c in range(cols):
            bx1,by1,bx2,by2 = boxes[r*cols + c]
            cell = crop[by1:by2, bx1:bx2]

            # padding: รายคอลัมน์ > ทั่วไป
            px = (col_pad_x[c] if (col_pad_x and c < len(col_pad_x)) else cell_pad[0])
            py = cell_pad[1]
            cnt, _ = cell_count(cell, px, py)
            counts.append(cnt)

        k = pick_choice(counts, abs_min, min_ratio, top2_delta_ratio)
        picked = "" if k is None else (col_labels[k] if k < len(col_labels) else "ABCDE"[k])
        qno = q_start + r + row_shift
        if qno >= 1:
            answers[qno] = picked

    return answers, boxes

# ================ MAIN =================
def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        img = pillow_read(IMG_PATH)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = {"subject_code": None, "student_id": None, "answers": {}}

    # -------- subject_code & student_id --------
    for name, cfg in ZONES.items():
        rect = pct_to_px(cfg["rect_pct"], W, H)
        x1,y1,x2,y2 = rect
        grid_mode   = cfg.get("grid", "uniform")
        row_w       = cfg.get("row_weights")
        col_w       = cfg.get("col_weights")

        digits, boxes = read_digit_zone(gray, rect, cfg["rows"], cfg["cols"],
                                        grid_mode, row_w, col_w)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        draw_boxes(vis, boxes, (x1,y1), (0,0,255))
        result[name] = digits

    # -------- answers --------
    for blk in ANSWER_BLOCKS:
        name, rect_pct, rows, cols, q_start, choices, *extra = blk
        row_shift   = extra[0] if len(extra) >= 1 else 0
        col_labels  = extra[1] if len(extra) >= 2 else "ABCDE"
        opts        = extra[2] if len(extra) >= 3 and isinstance(extra[2], dict) else {}

        rect = pct_to_px(rect_pct, W, H)
        x1,y1,x2,y2 = rect

        grid_mode   = opts.get("grid", "uniform")
        row_w       = opts.get("row_weights")
        col_w       = opts.get("col_weights")
        cell_pad    = opts.get("cell_pad", (0.12, 0.18))
        col_pad_x   = opts.get("col_pad_x")  # อาจเป็น None
        abs_min     = opts.get("abs_min_black", ABS_MIN_BLACK)
        min_ratio   = opts.get("min_fill_ratio", MIN_FILL_RATIO)
        top2_delta  = opts.get("top2_delta_ratio", TOP2_DELTA_RATIO)

        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
        ans, boxes = read_answer_block(
            gray, rect, rows, cols, q_start, choices, row_shift, col_labels,
            grid_mode, row_w, col_w,
            cell_pad=cell_pad, col_pad_x=col_pad_x,
            abs_min=abs_min, min_ratio=min_ratio, top2_delta_ratio=top2_delta
        )
        draw_boxes(vis, boxes, (x1,y1), (0,0,255))
        result["answers"].update(ans)

    # -------- Save overlay + meta + answers.csv --------
    cv2.imwrite(str(OUT_DIR/"overlay.jpg"), vis)
    with open(OUT_DIR/"meta.json","w",encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR/"answers.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["question","answer"])
        for q in sorted(result["answers"].keys()):
            w.writerow([q, result["answers"][q]])

    print("✅ เสร็จแล้ว")
    print(" - overlay :", OUT_DIR/"overlay.jpg")
    print(" - meta    :", OUT_DIR/"meta.json")
    print(" - answers :", OUT_DIR/"answers.csv")

if __name__ == "__main__":
    main()
