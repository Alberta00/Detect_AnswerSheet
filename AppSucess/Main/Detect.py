# Detect.py — OMR Checker + Summary (FULL)
# ✅ รองรับบล็อก StuID / SubID โดยตรง (และ alias เดิม StudentID/SubjectCode/ไทย)
# ✅ StudentID = 8 หลัก, SubjectCode = 6 หลัก (มี auto-swap ถ้าสลับกัน)
# ✅ Auto-align (ORB+Homography), Overlay, DB บันทึกผล, Summary UI
# ✅ Subject Title เป็น combobox แบบพิมพ์ได้ + จำประวัติ
# ✅ พรีวิวคงที่, ซูมด้วย Ctrl+/Ctrl- เท่านั้น (ปิดซูมเม้าส์)
# ✅ แถบซ้ายแสดงเฉพาะ "คะแนน | ไฟล์" (ไม่โชว์ StudentID/SubjectCode)

import sys, json, csv, os, re, cv2, sqlite3
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox, QMessageBox,
    QComboBox, QScrollArea, QSlider, QListWidget, QListWidgetItem,
    QSplitter, QTextEdit
)
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence

# -------- PDF (PyMuPDF)
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.db")

# ======================== DB ========================
def _ensure_columns():
    con = sqlite3.connect(DB_FILE); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        subject_code TEXT,
        student_id TEXT,
        score INTEGER,
        total INTEGER,
        subject_title TEXT,
        answers_json TEXT,
        overlay_path TEXT
    )""")
    con.commit(); con.close()

def insert_result(filename, subject_code, student_id, score, total, subject_title, answers_dict, overlay_path):
    _ensure_columns()
    con = sqlite3.connect(DB_FILE); cur = con.cursor()
    cur.execute("""INSERT INTO results(filename, subject_code, student_id, score, total, subject_title, answers_json, overlay_path)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (filename, subject_code, student_id, score, total, subject_title,
                 json.dumps(answers_dict, ensure_ascii=False), overlay_path))
    con.commit(); con.close()

def fetch_subject_titles():
    _ensure_columns()
    con = sqlite3.connect(DB_FILE); cur = con.cursor()
    cur.execute("SELECT DISTINCT subject_title FROM results WHERE subject_title IS NOT NULL AND subject_title!=''")
    rows = [r[0] for r in cur.fetchall()]
    con.close()
    return sorted(rows)

def fetch_results_by_subject_title(subject_title, sort_by_student=True):
    _ensure_columns()
    con = sqlite3.connect(DB_FILE); cur = con.cursor()
    order = "student_id ASC" if sort_by_student else "score DESC"
    if subject_title and subject_title != "(ทั้งหมด)":
        cur.execute(f"""SELECT id, filename, subject_code, student_id, score, total, answers_json, overlay_path
                        FROM results WHERE subject_title=? ORDER BY {order}""", (subject_title,))
    else:
        cur.execute(f"""SELECT id, filename, subject_code, student_id, score, total, answers_json, overlay_path
                        FROM results ORDER BY {order}""")
    rows = cur.fetchall(); con.close()
    return rows

# ====================== Utils =======================
def to_qpixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def orb_align_to_template(input_bgr, template_bgr):
    """
    Align input sheet to template preview using ORB+Homography.
    Return (warped_bgr, ok:bool).
    """
    try:
        gray1 = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(3000)
        k1, d1 = orb.detectAndCompute(gray1, None)
        k2, d2 = orb.detectAndCompute(gray2, None)
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
            return input_bgr, False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)[:80]
        src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return input_bgr, False
        h, w = template_bgr.shape[:2]
        warped = cv2.warpPerspective(input_bgr, H, (w, h))
        return warped, True
    except Exception:
        return input_bgr, False

def read_pdf_first_page(path):
    if not HAS_PDF:
        raise RuntimeError("ยังไม่ได้ติดตั้ง PyMuPDF (fitz)")
    doc = fitz.open(path)
    if len(doc) == 0:
        raise RuntimeError("PDF ว่าง")
    page = doc[0]
    pix = page.get_pixmap()
    out_dir = os.path.join(os.path.dirname(path), "_pdf_frames")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(path)}_page1.png")
    pix.save(out_path)
    return out_path

# ---------- canonical aliases ----------
_STUDENT_ALIASES = [
    "studentid","student_id","student","sid","stdid","stuid","id",
    "เลขประจำตัว","เลขประจำตัวสอบ","รหัสนักเรียน","เลขที่นั่ง","เลขที่สอบ"
]
_SUBJECT_ALIASES = [
    "subjectcode","subject_code","subject","code","subid",
    "รหัสวิชา","รหัสรายวิชา","วิชา","วิชาเรียน"
]

def _canon(s: str) -> str:
    if not s: return ""
    s = s.strip().lower()
    s = re.sub(r"[\s_–\-:|()]+", "", s)
    return s

def _digits_only(s: str) -> str:
    return s if (s and s.isdigit()) else ""

def pick_student_and_subject(decoded_blocks: dict):
    """
    รองรับ StuID/SubID โดยตรง + alias เดิม StudentID/SubjectCode + ภาษาไทย
    พร้อมตรวจความยาว (StuID=8, SubID=6)
    """
    student_id = ""
    subject_code = ""

    # ✅ ตรงชื่อก่อน (StuID/SubID)
    for k, v in decoded_blocks.items():
        ck = k.strip().lower()
        if ck == "stuid" and v.isdigit():
            student_id = v
        if ck == "subid" and v.isdigit():
            subject_code = v

    # ✅ รองรับชื่อเดิม (StudentID/SubjectCode)
    if not student_id:
        for k, v in decoded_blocks.items():
            if "studentid" in k.lower() and v.isdigit():
                student_id = v
    if not subject_code:
        for k, v in decoded_blocks.items():
            if "subjectcode" in k.lower() and v.isdigit():
                subject_code = v

    # ✅ ตรวจความยาว + auto swap
    if len(student_id) == 6 and len(subject_code) == 8:
        student_id, subject_code = subject_code, student_id

    if len(student_id) != 8:
        student_id = ""
    if len(subject_code) != 6:
        subject_code = ""

    return student_id, subject_code

# --------- OMR core: decode blocks & answers ----------
def analyze_from_grids(gray, grids):
    """
    Non-Answer blocks (e.g., StuID/SubID/StudentID/SubjectCode/ไทย):
      - แบ่งกลุ่มตาม 'แถว' ของเลข 0..9 แล้วเลือกคอลัมน์ที่ดำสุด → map A..J -> 0..9 → ต่อเป็นสตริง
    Answer block:
      - เลือกตัวเลือกที่ดำสุด; ถ้าต่างจากค่าเฉลี่ยน้อยไป → NULL (ไม่มั่นใจ)
    """
    results = {}
    for cell in grids:
        name = cell["name"]; block = cell["block"]
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        roi = gray[y:y+h, x:x+w]
        mean_intensity = float(np.mean(roi)) if roi.size > 0 else 255.0
        results.setdefault(block, {})[name] = mean_intensity

    letter_to_digit = {chr(ord("A")+i): str(i) for i in range(10)}
    decoded_blocks, answers = {}, {}

    # decode non-answer blocks (ตัวเลข)
    for block, cells in results.items():
        if str(block).lower() == "answer": continue
        grouped = {}
        for name, val in cells.items():
            m1 = re.match(r"(\d+)([A-Z])", name)  # 1A
            m2 = re.match(r"([A-Z])(\d+)", name)  # A1
            if m1:
                row, col = int(m1.group(1)), m1.group(2)
            elif m2:
                row, col = int(m2.group(2)), m2.group(1)
            else:
                continue
            grouped.setdefault(row, {})[col] = val
        digits = []
        for r, cols in sorted(grouped.items()):
            if not cols:
                digits.append("")
                continue
            selected = min(cols, key=cols.get)  # darkest col
            digits.append(letter_to_digit.get(selected, ""))
        decoded_blocks[str(block)] = "".join(digits).strip()

    # decode Answer block
    if "Answer" in results:
        grouped = {}
        for name, val in results["Answer"].items():
            m = re.match(r"Answer(\d+)([A-Z])", name)
            if not m: continue
            q, choice = int(m.group(1)), m.group(2)
            grouped.setdefault(q, {})[choice] = val
        for q in sorted(grouped.keys()):
            values = grouped[q]
            if not values:
                answers[q] = "NULL"; continue
            min_choice = min(values, key=values.get)
            min_val = values[min_choice]
            mean_val = np.mean(list(values.values()))
            answers[q] = "NULL" if (mean_val - min_val < 25) else min_choice

    return decoded_blocks, answers

def draw_overlay(base_bgr, grids, answers, answer_key, limit_q):
    img = base_bgr.copy()
    for cell in grids:
        x,y,w,h = cell["x"],cell["y"],cell["w"],cell["h"]
        block = str(cell.get("block","")); name = cell["name"]
        if block.lower()=="answer":
            m = re.match(r"Answer(\d+)([A-Z])", name)
            if m:
                q, choice = int(m.group(1)), m.group(2)
                if q > limit_q:
                    color = (180,180,180)         # not counted
                else:
                    user_choice = answers.get(q,"NULL")
                    key_choice = answer_key.get(q,None)
                    if user_choice == "NULL": color = (0,215,255)      # unsure
                    else: color = (0,200,0) if (user_choice==choice and choice==key_choice) else (0,0,220)
            else:
                color = (180,180,180)
        else:
            color = (255,160,60)  # non-answer blocks (StuID/SubID etc.)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,name,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
    return img

# ===================== Zoom Label ====================
class ZoomLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.base_pix=None
        self.scale=1.0
        self.setAlignment(Qt.AlignCenter)

    def set_base_pixmap(self, pix):
        self.base_pix=pix
        self.scale=1.0
        self._apply()

    def set_scale(self, s):
        self.scale = max(0.1, min(4.0, s))
        self._apply()

    def wheelEvent(self, e):  # disable mouse-wheel zoom
        e.ignore()

    def _apply(self):
        if self.base_pix is None: return
        scaled = self.base_pix.scaled(self.base_pix.size()*self.scale, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())  # works inside QScrollArea

# =================== Summary Window ==================
class SummaryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 สรุปผลคะแนน")
        self.resize(1100, 740)

        # --- Top controls ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Subject:"))
        self.cmb_subject = QComboBox()
        self.cmb_subject.addItem("(ทั้งหมด)")
        for s in fetch_subject_titles(): self.cmb_subject.addItem(s)
        top.addWidget(self.cmb_subject)

        top.addWidget(QLabel("Sort by:"))
        self.cmb_sort = QComboBox()
        self.cmb_sort.addItems(["student_id (asc)", "score (desc)"])
        top.addWidget(self.cmb_sort)

        btn_reload = QPushButton("Reload")
        btn_reload.clicked.connect(self.reload)
        top.addWidget(btn_reload)

        self.lbl_most_wrong = QLabel("Most wrong: -")
        self.lbl_most_wrong.setStyleSheet("font-weight: 600; color:#c33;")
        top.addWidget(self.lbl_most_wrong)
        top.addStretch(1)

        # --- Body splitters ---
        body_h = QSplitter(Qt.Horizontal); body_h.setHandleWidth(10); body_h.setChildrenCollapsible(False)

        self.list = QListWidget()
        self.list.setMinimumWidth(260); self.list.setMaximumWidth(300)
        body_h.addWidget(self.list)

        right_v = QSplitter(Qt.Vertical); right_v.setHandleWidth(8); right_v.setChildrenCollapsible(False)

        self.preview_label = ZoomLabel()
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        right_v.addWidget(self.preview_scroll)

        self.txt_detail = QTextEdit(); self.txt_detail.setReadOnly(True)
        right_v.addWidget(self.txt_detail)

        right_v.setSizes([520, 130])  # preview:detail ≈ 4:1
        body_h.addWidget(right_v)
        body_h.setStretchFactor(0,1); body_h.setStretchFactor(1,3); body_h.setSizes([280, 780])

        # --- Bottom nav ---
        nav = QHBoxLayout(); nav.addStretch(1)
        btn_back = QPushButton("🔙 กลับหน้าหลัก"); btn_back.clicked.connect(self.back_to_main)
        nav.addWidget(btn_back)

        root = QVBoxLayout(); root.setContentsMargins(10,8,10,10); root.setSpacing(8)
        root.addLayout(top); root.addWidget(body_h, 1); root.addLayout(nav)
        w = QWidget(); w.setLayout(root); self.setCentralWidget(w)

        self.list.currentItemChanged.connect(self._on_select_item)
        self.reload()

        # keyboard zoom
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self.preview_label.set_scale(self.preview_label.scale*1.1))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self.preview_label.set_scale(self.preview_label.scale*1.1))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.preview_label.set_scale(self.preview_label.scale/1.1))

    def _set_preview_pix(self, bgr, keep_zoom=True):
        pix = to_qpixmap(bgr)
        if keep_zoom and self.preview_label.base_pix is not None:
            cur = self.preview_label.scale
            self.preview_label.set_base_pixmap(pix)
            self.preview_label.set_scale(cur)
        else:
            self.preview_label.set_base_pixmap(pix)

    def reload(self):
        subj = self.cmb_subject.currentText()
        sort_student = (self.cmb_sort.currentText().startswith("student_id"))
        rows = fetch_results_by_subject_title(subj, sort_by_student=sort_student)

        # คำนวณข้อที่ผิดเยอะสุด (นับ NULL เป็นผิด)
        wrong_counts = {}
        for _,_,_,_,_,total,answers_json,_ in rows:
            try:
                answers = json.loads(answers_json) if answers_json else {}
                for q, ch in answers.items():
                    q=int(q)
                    if q>total: continue
                    if ch == "NULL":
                        wrong_counts[q] = wrong_counts.get(q, 0) + 1
            except Exception:
                pass
        if wrong_counts:
            most_q = max(wrong_counts, key=lambda k: wrong_counts[k])
            self.lbl_most_wrong.setText(f"Most wrong: Q{most_q}  ({wrong_counts[most_q]} คน)")
        else:
            self.lbl_most_wrong.setText("Most wrong: -")

        # เติมรายการ
        self.list.clear()
        for rid, filename, subject_code, student_id, score, total, answers_json, overlay_path in rows:
            item = QListWidgetItem(f"{score:>2}/{total} | {filename}")
            item.setData(Qt.UserRole, (rid, filename, subject_code, student_id, score, total, answers_json, overlay_path))
            self.list.addItem(item)

    def _on_select_item(self, curr, prev):
        if not curr: return
        rid, filename, subject_code, student_id, score, total, answers_json, overlay_path = curr.data(Qt.UserRole)
        if overlay_path and os.path.exists(overlay_path):
            bgr = cv2.imread(overlay_path)
            if bgr is not None:
                self._set_preview_pix(bgr, keep_zoom=True)

        lines = [f"Student: {student_id or '-'}",
                 f"File: {filename}",
                 f"Score: {score}/{total}",
                 f"SubjectCode: {subject_code or '-'}", ""]
        try:
            ans = json.loads(answers_json) if answers_json else {}
            for q in range(1, int(total)+1):
                ch = ans.get(str(q)) or ans.get(q) or "NULL"
                lines.append(f"Q{q}: {'-' if ch=='NULL' else ch}")
        except Exception:
            lines.append("(no answers_json)")
        self.txt_detail.setPlainText("\n".join(lines))

    def back_to_main(self):
        path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(path):
            QProcess.startDetached(sys.executable, [path])
            self.close()

# ===================== Checker Window =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Auto Checker ")
        self.resize(1200, 860)


        self.images = []
        self.grids = None
        self.answer_key = {}
        self.answer_key_img = None
        self.template_preview_bgr = None
        self.template_loaded = False
        self.answer_loaded = False

        # --- Subject Title combobox (editable dropdown with history) ---
        subj_row = QHBoxLayout()
        subj_row.addWidget(QLabel("วิชา (Subject Title):"))
        self.cmb_subject_title = QComboBox()
        self.cmb_subject_title.setEditable(True)   # input + dropdown
        self._reload_subject_dropdown()
        subj_row.addWidget(self.cmb_subject_title)
        btn_summary = QPushButton("📊 สรุปผล")
        btn_summary.clicked.connect(self.open_summary)
        subj_row.addWidget(btn_summary)

        # --- Template chooser ---
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("📄 Template:"))
        self.grid_combo = QComboBox()
        self.grid_combo.currentIndexChanged.connect(self.change_template)
        top_layout.addWidget(self.grid_combo)
        self.lbl_subject_show = QLabel("")
        self.lbl_subject_show.setStyleSheet("color:#888; font-style:italic;")
        top_layout.addWidget(self.lbl_subject_show, stretch=1, alignment=Qt.AlignRight)

        def _sync_subject_label():
            t = self.cmb_subject_title.currentText().strip()
            self.lbl_subject_show.setText(f"วิชา: {t}" if t else "")
        self.cmb_subject_title.editTextChanged.connect(lambda _=None: _sync_subject_label())
        self.cmb_subject_title.currentIndexChanged.connect(lambda _=None: _sync_subject_label())

        # --- Splitter (left list | right panel) ---
        self.split = QSplitter()
        self.split.setHandleWidth(10); self.split.setChildrenCollapsible(False)

        self.left_list = QListWidget()
        self.left_list.setMinimumWidth(300); self.left_list.setMaximumWidth(320)
        self.left_list.addItem("— ยังไม่มีรายการ ตรวจแล้วจะแสดงที่นี่ —")
        self.left_list.currentItemChanged.connect(self._select_run_item)
        self.split.addWidget(self.left_list)

        right = QWidget(); rlay = QVBoxLayout(right)

        # Fixed-size preview area with scroll
        self.image_label = ZoomLabel()
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        rlay.addWidget(self.image_scroll, stretch=5)

        self.detail_box = QTextEdit(); self.detail_box.setReadOnly(True)
        self.detail_box.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
        rlay.addWidget(self.detail_box, stretch=1)

        self.split.addWidget(right)
        self.split.setStretchFactor(0,0); self.split.setStretchFactor(1,1); self.split.setSizes([300, 900])

        # --- Status ---
        self.lbl_status = QLabel("📂 สถานะ: ยังไม่ได้โหลด Template")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #ddd; font-size: 14px; padding: 6px; background:#333;")

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.btn_load_answer_image = QPushButton("🧾 โหลดเฉลย (รูป/PDF หน้าแรก)")
        self.btn_load_answer_image.clicked.connect(self.load_answer_image)
        self.btn_add_images = QPushButton("🖼️ เพิ่มภาพนักเรียน")
        self.btn_add_images.clicked.connect(self.add_images)
        self.btn_add_images.setEnabled(False)
        self.btn_run = QPushButton("🚀 เริ่มตรวจทั้งหมด")
        self.btn_run.clicked.connect(self.run_check)
        self.btn_run.setEnabled(False)
        button_layout.addWidget(self.btn_load_answer_image)
        button_layout.addWidget(self.btn_add_images)
        button_layout.addWidget(self.btn_run)

        # --- Options ---
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("จำนวนข้อที่ต้องตรวจ:"))
        self.spin_q = QSpinBox(); self.spin_q.setRange(1,200); self.spin_q.setValue(30)
        opt_layout.addWidget(self.spin_q)
        opt_layout.addSpacing(20)
        opt_layout.addWidget(QLabel("🔍 Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(10,400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(lambda v: self.image_label.set_scale(v/100.0))
        opt_layout.addWidget(self.zoom_slider)

        # --- Back ---
        nav = QHBoxLayout(); nav.addStretch(1)
        btn_back = QPushButton("🔙 กลับหน้าเมนูหลัก"); btn_back.clicked.connect(self.back_to_main)
        nav.addWidget(btn_back)

        # --- Root ---
        root = QVBoxLayout()
        root.addLayout(subj_row)
        root.addLayout(top_layout)
        root.addWidget(self.split, stretch=2)
        root.addWidget(self.lbl_status)
        root.addLayout(button_layout)
        root.addLayout(opt_layout)
        root.addLayout(nav)
        cw = QWidget(); cw.setLayout(root); self.setCentralWidget(cw)

        # keyboard zoom
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self.image_label.set_scale(self.image_label.scale*1.1))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self.image_label.set_scale(self.image_label.scale*1.1))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.image_label.set_scale(self.image_label.scale/1.1))

        self.load_grid_list()
        _sync_subject_label()
        _ensure_columns()

    # ---------- Subject dropdown helpers ----------
    def _reload_subject_dropdown(self):
        self.cmb_subject_title.blockSignals(True)
        self.cmb_subject_title.clear()
        titles = fetch_subject_titles()
        self.cmb_subject_title.addItems(titles)
        self.cmb_subject_title.setCurrentIndex(-1)
        self.cmb_subject_title.blockSignals(False)

    # ---------- helper: set pixmap with optional zoom persistence ----------
    def _set_pix(self, bgr, keep_zoom=True):
        if bgr is None: return
        pix = to_qpixmap(bgr)
        if keep_zoom and self.image_label.base_pix is not None:
            cur = self.image_label.scale
            self.image_label.set_base_pixmap(pix)
            self.image_label.set_scale(cur)
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(round(cur*100)))
            self.zoom_slider.blockSignals(False)
        else:
            self.image_label.set_base_pixmap(pix)
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(100)
            self.zoom_slider.blockSignals(False)

    # ---------------- Template ----------------
    def load_grid_list(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "grids")
        os.makedirs(folder, exist_ok=True)
        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        self.grid_combo.clear()
        if not json_files:
            self.grid_combo.addItem("(ไม่มี Template ในโฟลเดอร์ grids)")
        else:
            self.grid_combo.addItems(json_files)
            self.load_template(os.path.join(folder, json_files[0]))

    def change_template(self):
        selected = self.grid_combo.currentText()
        if not selected or selected.startswith("("): return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_template(os.path.join(base_dir, "grids", selected))

    def load_template(self, file):
        if not os.path.exists(file):
            QMessageBox.warning(self, "Error", f"❌ หาไฟล์ Template ไม่เจอ:\n{file}"); return
        try:
            with open(file,"r",encoding="utf-8") as f: content = json.load(f)
        except Exception as e:
            QMessageBox.warning(self,"Error",f"อ่าน Template ไม่ได้:\n{e}"); return
        if isinstance(content, dict) and "grids" in content:
            self.grids = content["grids"]; img_path = content.get("image_path", None)
            self.template_loaded = True
            if img_path:
                if not os.path.isabs(img_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    if "grids" in img_path.lower(): img_path = os.path.join(base_dir, img_path)
                    else: img_path = os.path.join(os.path.dirname(file), img_path)
                img_path = os.path.normpath(img_path)
            if img_path and os.path.exists(img_path):
                prev = cv2.imread(img_path)
                if prev is not None: self.template_preview_bgr = prev.copy()
                self._set_pix(prev, keep_zoom=False)   # reset zoom on new template
                self.lbl_status.setText(f"✅ โหลด Template แล้ว: {os.path.basename(file)}")
            else:
                QMessageBox.warning(self,"Image Missing",f"⚠️ Template โหลดได้แต่ไม่พบภาพ Preview:\n{img_path}")
                self.lbl_status.setText("⚠️ Template ไม่มีภาพ Preview หรือหาไฟล์ไม่เจอ")
        else:
            QMessageBox.warning(self,"Error","Template ไม่ถูกต้อง")

    # ---------------- Answer (single sheet) ----------------
    def load_answer_image(self):
        if not self.template_loaded:
            QMessageBox.warning(self,"Error","กรุณาโหลด Template ก่อน"); return
        file, _ = QFileDialog.getOpenFileName(self,"เลือกเฉลย (รูปภาพ/PDF หน้าแรก)","", "Images/PDF (*.jpg *.png *.pdf)")
        if not file: return
        if file.lower().endswith(".pdf"): file = read_pdf_first_page(file)
        self.answer_key_img = file
        base_bgr = cv2.imread(file)
        if base_bgr is None:
            QMessageBox.warning(self,"Error","อ่านภาพเฉลยไม่ได้"); return
        work = base_bgr
        if self.template_preview_bgr is not None:
            aligned, ok = orb_align_to_template(base_bgr, self.template_preview_bgr)
            if ok: work = aligned
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        _, ans = analyze_from_grids(gray, self.grids)
        self.answer_key = ans; self.answer_loaded = True
        self.lbl_status.setText(f"✅ โหลดเฉลยสำเร็จ ({len(self.answer_key)} ข้อ)")
        self.btn_add_images.setEnabled(True); self.btn_run.setEnabled(True)
        self._set_pix(base_bgr, keep_zoom=False)  # first load → reset zoom

    # ---------------- Students ----------------
    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self,"เลือกภาพนักเรียน","", "Images (*.jpg *.png *.jpeg *.bmp)")
        if not files: return
        self.images = files.copy()
        self.lbl_status.setText(f"📸 เลือกภาพใหม่แล้ว {len(self.images)} ไฟล์")
        self.left_list.clear()
        for f in self.images:
            base = os.path.basename(f)
            item = QListWidgetItem(f"{base}")  # show filename only
            item.setData(Qt.UserRole, {"file": base, "path": f})
            self.left_list.addItem(item)

    # ---------------- Run check ----------------
    def run_check(self):
        if not self.answer_loaded:
            QMessageBox.warning(self,"Error","ยังไม่มีเฉลย"); return
        if not self.images:
            QMessageBox.warning(self,"Error","ยังไม่มีภาพนักเรียน"); return

        limit_q = self.spin_q.value()
        subject_title = self.cmb_subject_title.currentText().strip()

        all_results = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "out"); os.makedirs(out_dir, exist_ok=True)

        # collect non-answer block names for CSV header
        block_names = sorted({cell["block"] for cell in self.grids if str(cell["block"]).lower()!="answer"})

        self.left_list.clear()

        for img_path in self.images:
            bgr = cv2.imread(img_path)
            if bgr is None: continue

            work = bgr
            if self.template_preview_bgr is not None:
                aligned, ok = orb_align_to_template(bgr, self.template_preview_bgr)
                if ok: work = aligned

            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            decoded_blocks, answers = analyze_from_grids(gray, self.grids)

            # ---- robust pick (StuID/SubID first, then aliases, then length/swap) ----
            student_id, subject_code = pick_student_and_subject(decoded_blocks)

            # ---- Score ----
            score = sum(1 for q, correct in self.answer_key.items() if q<=limit_q and answers.get(q)==correct)

            # ---- Overlay & preview ----
            overlay = draw_overlay(work, self.grids, answers, self.answer_key, limit_q)
            out_name = os.path.splitext(os.path.basename(img_path))[0]
            ov_path = os.path.join(out_dir, f"{out_name}_overlay.jpg")
            cv2.imwrite(ov_path, overlay)
            self._set_pix(overlay, keep_zoom=True)

            # ---- Persist & UI list ----
            data = {"file": os.path.basename(img_path), "blocks": decoded_blocks, "answers": answers,
                    "score": score, "total": limit_q, "overlay": ov_path,
                    "student_id": student_id, "subject_code": subject_code}
            all_results.append(data)

            insert_result(data["file"], subject_code, student_id, score, limit_q, subject_title, answers, ov_path)

            # LEFT LIST: show only "score | filename"
            item = QListWidgetItem(f"{score}/{limit_q}  |  {data['file']}")
            item.setData(Qt.UserRole, data)
            self.left_list.addItem(item)

        # ---------- exports ----------
        with open("results.json","w",encoding="utf-8") as f:
            json.dump(all_results,f,indent=2,ensure_ascii=False)

        with open("scores.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); header=["ไฟล์"]+block_names+["คะแนน","เต็ม"]; w.writerow(header)
            for r in all_results:
                row=[r["file"]] + [r["blocks"].get(b,"") for b in block_names] + [r["score"], r["total"]]
                w.writerow(row)

        with open("answers.csv","w",newline="",encoding="utf-8-sig") as f2:
         w2 = csv.writer(f2)
         header = ["ไฟล์"] + [f"Q{i}" for i in range(1, limit_q+1)]
         w2.writerow(header)

         # เขียนคำตอบของแต่ละนักเรียน
         for r in all_results:
             ans_row = [r["file"]] + [r["answers"].get(i, "NULL") for i in range(1, limit_q+1)]
         w2.writerow(ans_row)

          # 🟢 เพิ่มบรรทัดสุดท้ายเป็นเฉลย (AnswerKey)
         key_row = ["เฉลย"] + [self.answer_key.get(i, "-") for i in range(1, limit_q+1)]
         w2.writerow(key_row)


         # refresh subject dropdown if new title appears
         self._reload_subject_dropdown()
         if subject_title:
            idx = self.cmb_subject_title.findText(subject_title, Qt.MatchFixedString)
            if idx >= 0:
                self.cmb_subject_title.setCurrentIndex(idx)
            else:
                self.cmb_subject_title.setEditText(subject_title)

         QMessageBox.information(self,"✅ เสร็จสิ้น", f"ตรวจแล้ว {len(all_results)} ไฟล์")

    # ---------------- Sidebar select ----------------
    def _select_run_item(self, curr, prev):
        if not curr: return
        data = curr.data(Qt.UserRole)
        if not isinstance(data, dict): return
        if data.get("overlay") and os.path.exists(data["overlay"]):
            bgr = cv2.imread(data["overlay"])
            if bgr is not None:
                self._set_pix(bgr, keep_zoom=True)

        # detail panel — show StudentID/SubjectCode correctly here
        lines = [f"File: {data['file']}",
                 f"StudentID: {data.get('student_id') or data['blocks'].get('StuID') or data['blocks'].get('StudentID','-')}",
                 f"SubjectCode: {data.get('subject_code') or data['blocks'].get('SubID') or data['blocks'].get('SubjectCode','-')}",
                 f"Score: {data['score']}/{data['total']}", ""]
        for q in range(1, data["total"]+1):
            ch = data["answers"].get(q, "NULL")
            lines.append(f"Q{q}: {'-' if ch=='NULL' else ch}")
        self.detail_box.setPlainText("\n".join(lines))

    # ---------------- Navigation ----------------
    def open_summary(self):
        self.sum_win = SummaryWindow(); self.sum_win.show()

    def back_to_main(self):
        path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(path):
            QProcess.startDetached(sys.executable, [path])
            self.close()

# ========================== run =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
