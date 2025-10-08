
import sys, json, traceback
from pathlib import Path
import importlib.util

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QMessageBox, QProgressBar, QSizePolicy, QToolBar, QStyle, QStatusBar,
    QSplitter, QComboBox
)
from PySide6.QtGui import QIcon, QAction, QPalette, QColor, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QSize

import cv2
import numpy as np

APP_TITLE = "OMR Desktop (PySide6) — v4"

# ---------- Themes ----------
def apply_palette(app: QApplication, theme: str):
    app.setStyle("Fusion")
    p = QPalette()
    if theme == "Light":
        p.setColor(QPalette.Window, QColor("#f5f5f7"))
        p.setColor(QPalette.WindowText, Qt.black)
        p.setColor(QPalette.Base, QColor("#ffffff"))
        p.setColor(QPalette.AlternateBase, QColor("#ededed"))
        p.setColor(QPalette.ToolTipBase, Qt.black)
        p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.black)
        p.setColor(QPalette.Button, QColor("#e5e7eb"))
        p.setColor(QPalette.ButtonText, Qt.black)
        p.setColor(QPalette.Highlight, QColor("#2563eb"))
        p.setColor(QPalette.HighlightedText, Qt.white)
    elif theme == "Emerald":
        p.setColor(QPalette.Window, QColor("#0b1410"))
        p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor("#0a0f0d"))
        p.setColor(QPalette.AlternateBase, QColor("#0f1a15"))
        p.setColor(QPalette.ToolTipBase, Qt.white)
        p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.white)
        p.setColor(QPalette.Button, QColor("#065f46"))
        p.setColor(QPalette.ButtonText, Qt.white)
        p.setColor(QPalette.Highlight, QColor("#10b981"))
        p.setColor(QPalette.HighlightedText, Qt.black)
    elif theme == "Rose":
        p.setColor(QPalette.Window, QColor("#1a0f13"))
        p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor("#140b0f"))
        p.setColor(QPalette.AlternateBase, QColor("#1e1217"))
        p.setColor(QPalette.ToolTipBase, Qt.white); p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.white)
        p.setColor(QPalette.Button, QColor("#9f1239"))
        p.setColor(QPalette.ButtonText, Qt.white)
        p.setColor(QPalette.Highlight, QColor("#f43f5e"))
        p.setColor(QPalette.HighlightedText, Qt.black)
    else:  # Dark (default)
        p.setColor(QPalette.Window, QColor("#18181b"))
        p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor("#111113"))
        p.setColor(QPalette.AlternateBase, QColor("#202023"))
        p.setColor(QPalette.ToolTipBase, Qt.white); p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.white)
        p.setColor(QPalette.Button, QColor("#2d61ff"))
        p.setColor(QPalette.ButtonText, Qt.white)
        p.setColor(QPalette.Highlight, QColor("#4a80ff"))
        p.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(p)

APP_QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Noto Sans', Arial; font-size: 11pt; color: #f5f5f7; }
QMainWindow { background-color: #18181b; }
QGroupBox { 
  border: 1px solid #2a2a2e;
  border-radius: 12px;
  margin-top: 10px;
  padding: 10px;
  background: rgba(255,255,255,0.03);
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 2px 6px; color: #c9c9cf; }
QPushButton {
  background: #2d61ff; border: none; border-radius: 10px; padding: 10px 16px;
}
QPushButton:hover { background: #4b78ff; }
QPushButton:disabled { background: #4a4a4f; color: #bbb; }
QToolBar { background: transparent; border: none; padding: 6px; }
QStatusBar { background: #111; border-top: 1px solid #2a2a2e; }
QTableWidget { background: #141418; gridline-color: #2a2a2e; }
QHeaderView::section { background: #25252a; padding: 6px; border: none; }
QSpinBox, QDoubleSpinBox, QLabel { background: transparent; }
#Chip { border-radius: 16px; background: #2a2a33; padding: 6px 12px; margin-right: 8px; }
"""

# ---------- Detect.py loader ----------
def import_detect():
    here = Path(__file__).resolve().parent
    base = Path(getattr(sys, "_MEIPASS", here))
    candidates = [
        here.parent / "Detect_AnswerSheet" / "Detect.py",
        here / "Detect_AnswerSheet" / "Detect.py",
        base / "Detect_AnswerSheet" / "Detect.py",
        Path.cwd() / "Detect_AnswerSheet" / "Detect.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("detect_module", str(p))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["detect_module"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("ไม่พบ Detect.py ในโฟลเดอร์ Detect_AnswerSheet")

# ---------- PDF helper ----------
def pdf_first_page_to_png_bytes(pdf_path: Path):
    import fitz
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("PDF ไม่มีหน้า")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")

# ---------- Worker for single run ----------
class DetectWorker(QThread):
    success = Signal(dict)
    failure = Signal(str)

    def __init__(self, file_path: Path, abs_min_black: int, min_fill_ratio: float, tag: str):
        super().__init__()
        self.file_path = Path(file_path)
        self.abs_min_black = abs_min_black
        self.min_fill_ratio = min_fill_ratio
        self.tag = tag  # "student" or "key"

    def run(self):
        try:
            detect = import_detect()
            if hasattr(detect, "ABS_MIN_BLACK"):
                detect.ABS_MIN_BLACK = int(self.abs_min_black)
            if hasattr(detect, "MIN_FILL_RATIO"):
                detect.MIN_FILL_RATIO = float(self.min_fill_ratio)

            # prepare input path
            if self.file_path.suffix.lower() == ".pdf":
                png_bytes = pdf_first_page_to_png_bytes(self.file_path)
                tmp = Path.cwd() / f"_omr_input_{self.tag}.png"
                tmp.write_bytes(png_bytes)
                detect.IMG_PATH = str(tmp)
            else:
                detect.IMG_PATH = str(self.file_path)

            detect.OUT_DIR.mkdir(parents=True, exist_ok=True)
            if not hasattr(detect, "main"):
                raise RuntimeError("Detect.py ไม่มีฟังก์ชัน main()")
            detect.main()

            overlay = detect.OUT_DIR / "overlay.jpg"
            meta_json = detect.OUT_DIR / "meta.json"
            answers_csv = detect.OUT_DIR / "answers.csv"
            result = {
                "overlay": str(overlay) if overlay.exists() else "",
                "meta_json": str(meta_json) if meta_json.exists() else "",
                "answers_csv": str(answers_csv) if answers_csv.exists() else "",
                "tag": self.tag,
            }
            self.success.emit(result)
        except Exception as e:
            self.failure.emit(f"[{self.tag}] {e}\n\n{traceback.format_exc()}")

# ---------- Helpers for compare & draw ----------
def extract_subject_student(meta: dict):
    subj = meta.get("subject_code") or meta.get("subject") or meta.get("subjectCode")
    stud = meta.get("student_id") or meta.get("studentId") or meta.get("sid")
    return subj, stud

def extract_answers(meta: dict):
    return meta.get("answers", {})

def try_get_cells(meta: dict):
    if "cells" in meta:
        return meta["cells"]
    if "bubbles" in meta:
        return meta["bubbles"]
    if "answer_cells" in meta:
        return meta["answer_cells"]
    if "grid" in meta and isinstance(meta["grid"], dict) and "cells" in meta["grid"]:
        return meta["grid"]["cells"]
    return None

def draw_marked_overlay(student_overlay_path: str, meta_student: dict, correct_map: dict, circle_scale_diam: float=0.5) -> str:
    cells = try_get_cells(meta_student)
    if not cells:
        return student_overlay_path  # no-op

    img = cv2.imread(student_overlay_path)
    if img is None:
        return student_overlay_path

    for q, verdict in correct_map.items():
        cell = cells.get(str(q)) if isinstance(cells, dict) else None
        if cell is None and isinstance(q, int):
            cell = cells.get(int(q))
        if not cell:
            continue
        cx = cell.get("cx"); cy = cell.get("cy")
        w = cell.get("w") or cell.get("width"); h = cell.get("h") or cell.get("height")
        if cx is None or cy is None:
            x = cell.get("x"); y = cell.get("y")
            if x is not None and y is not None and w and h:
                cx, cy = x + w/2, y + h/2
        if cx is None or cy is None or not w or not h:
            continue
        radius = int(0.5 * circle_scale_diam * min(float(w), float(h)))  # 0.25 of cell if diam=0.5
        color = (0,255,0) if verdict else (0,0,255)
        cv2.circle(img, (int(cx), int(cy)), max(6, radius), color, thickness=3, lineType=cv2.LINE_AA)

    out_path = str(Path(student_overlay_path).with_name("overlay_marked.jpg"))
    cv2.imwrite(out_path, img)
    return out_path

# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        icon = Path(__file__).resolve().parent / "app.ico"
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.setMinimumSize(QSize(1200, 820))

        self.input_student = None
        self.input_key = None
        self.meta_student = None
        self.meta_key = None
        self.overlay_student = ""
        self.overlay_marked = ""

        # Toolbar
        tb = QToolBar("Tools"); tb.setIconSize(QSize(20,20)); self.addToolBar(tb)
        act_open_student = QAction("กระดาษนิสิต", self)
        act_open_student.triggered.connect(self.choose_student)
        tb.addAction(act_open_student)

        act_open_key = QAction("กระดาษเฉลย", self)
        act_open_key.triggered.connect(self.choose_key)
        tb.addAction(act_open_key)

        act_run_student = QAction("อ่านนิสิต", self); act_run_student.triggered.connect(self.run_student)
        act_run_compare = QAction("อ่าน+เทียบ", self); act_run_compare.triggered.connect(self.run_compare)
        tb.addSeparator()
        tb.addAction(act_run_student); tb.addAction(act_run_compare)

        tb.addSeparator()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark","Light","Emerald","Rose"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_change)
        tb.addWidget(QLabel("ธีม: "))
        tb.addWidget(self.theme_combo)

        tb.addSeparator()
        tip_btn = QAction("Tip", self); tip_btn.triggered.connect(self.show_tips)
        tb.addAction(tip_btn)

        # Layout
        central = QWidget(); self.setCentralWidget(central)
        lay = QVBoxLayout(central); lay.setContentsMargins(12,12,12,12)

        # Pattern line
        self.pattern_label = QLabel("รหัสวิชา : 036032   เลขประจำตัวสอบ : 1234567")
        self.pattern_label.setStyleSheet("font-weight:700; font-size:16pt;")
        lay.addWidget(self.pattern_label)

        # Row: inputs & settings
        top = QGroupBox("อินพุต & ตั้งค่า"); lay.addWidget(top)
        tl = QHBoxLayout(top)
        self.lbl_student = QLabel("นิสิต: ยังไม่เลือกไฟล์")
        btn_student = QPushButton("เลือกกระดาษนิสิต"); btn_student.clicked.connect(self.choose_student)
        self.lbl_key = QLabel("เฉลย: ยังไม่เลือกไฟล์")
        btn_key = QPushButton("เลือกกระดาษเฉลย"); btn_key.clicked.connect(self.choose_key)

        self.abs_spin = QSpinBox(); self.abs_spin.setRange(0,10000); self.abs_spin.setValue(35); self.abs_spin.setPrefix("ABS ")
        self.fill_spin = QDoubleSpinBox(); self.fill_spin.setRange(0.0,1.0); self.fill_spin.setSingleStep(0.01); self.fill_spin.setDecimals(2); self.fill_spin.setValue(0.22); self.fill_spin.setPrefix("FILL ")
        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)

        tl.addWidget(self.lbl_student,1); tl.addWidget(btn_student)
        tl.addWidget(self.lbl_key,1); tl.addWidget(btn_key)
        tl.addWidget(self.abs_spin); tl.addWidget(self.fill_spin); tl.addWidget(self.progress)

        # Image preview
        self.image = QLabel("ยังไม่มีภาพ"); self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(480)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.image, 1)

        # Answers table
        ans_box = QGroupBox("ผลลัพธ์คำตอบ (นิสิต)"); lay.addWidget(ans_box, 1)
        v = QVBoxLayout(ans_box)
        self.table = QTableWidget(0,3); self.table.setHorizontalHeaderLabels(["ข้อ","นิสิต","เฉลย"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)

        self.setStatusBar(QStatusBar())

    # ---------- UI actions ----------
    def on_theme_change(self, name: str):
        apply_palette(QApplication.instance(), name)

    def show_tips(self):
        QMessageBox.information(self, "Tips",
            "ABS (ABS_MIN_BLACK):\n"
            "- เกณฑ์จำนวนพิกเซลสีเข้มอย่างต่ำที่ต้องมีในช่องหนึ่ง ๆ เพื่อถือว่า 'ฝนแล้ว'\n"
            "- ถ้ากระดาษ/สแกนจาง ให้ลดค่า ABS ลง\n\n"
            "FILL (MIN_FILL_RATIO):\n"
            "- สัดส่วนความเข้มเทียบกับค่าเฉลี่ยของแถว/คอลัมน์\n"
            "- ช่วยกันกรณีบางช่องเข้มกว่าช่องอื่น\n"
            "- ถ้าอ่านพลาดบ่อย ลองลด/เพิ่มเล็กน้อย (เช่น 0.18–0.30)"
        )

    def choose_student(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกกระดาษคำตอบนิสิต", "", "Documents (*.pdf *.png *.jpg *.jpeg)")
        if path:
            self.input_student = Path(path)
            self.lbl_student.setText(f"นิสิต: {self.input_student.name}")

    def choose_key(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกกระดาษเฉลยคำตอบ", "", "Documents (*.pdf *.png *.jpg *.jpeg)")
        if path:
            self.input_key = Path(path)
            self.lbl_key.setText(f"เฉลย: {self.input_key.name}")

    # ---------- Run pipelines ----------
    def run_student(self):
        if not self.input_student:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกกระดาษนิสิตก่อน")
            return
        self.progress.setValue(10)
        self.run_single(self.input_student, tag="student")

    def run_compare(self):
        if not self.input_student or not self.input_key:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกทั้งกระดาษนิสิตและกระดาษเฉลย")
            return
        self.progress.setValue(10)
        self.run_single(self.input_key, tag="key", then=lambda: self.run_single(self.input_student, tag="student", then=self.compare_now))

    def run_single(self, path: Path, tag: str, then=None):
        w = DetectWorker(path, int(self.abs_spin.value()), float(self.fill_spin.value()), tag=tag)
        self.worker = w
        w.success.connect(lambda res: self.on_detect_success(res, then=then))
        w.failure.connect(self.on_detect_failure)
        w.start()

    def on_detect_success(self, res: dict, then=None):
        meta_path = res.get("meta_json",""); overlay = res.get("overlay",""); tag = res.get("tag","")
        if tag == "student":
            self.overlay_student = overlay
        if meta_path and Path(meta_path).exists():
            data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            if tag == "student":
                self.meta_student = data
            else:
                self.meta_key = data

        if tag == "student" and overlay and Path(overlay).exists():
            pm = QPixmap(overlay)
            self.image.setPixmap(pm.scaled(self.image.width(), self.image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.progress.setValue(60 if tag=="key" else 80)
        if then:
            then()

    def on_detect_failure(self, msg: str):
        self.progress.setValue(0)
        QMessageBox.critical(self, "รันไม่สำเร็จ", msg)

    def compare_now(self):
        if not self.meta_student or not self.meta_key:
            QMessageBox.warning(self, "ขาดข้อมูล", "ไม่พบ meta.json ทั้งสองฝั่ง")
            return
        stud_ans = extract_answers(self.meta_student); key_ans = extract_answers(self.meta_key)
        subj, stud = extract_subject_student(self.meta_student)
        subj_txt = subj if subj is not None else "-"
        stud_txt = stud if stud is not None else "-"
        self.pattern_label.setText(f"รหัสวิชา : {subj_txt}   เลขประจำตัวสอบ : {stud_txt}")

        all_q = sorted({*map(str, stud_ans.keys()), *map(str, key_ans.keys())}, key=lambda x: int(x) if x.isdigit() else x)
        correct_map = {}
        self.table.setRowCount(0)
        for q in all_q:
            sa = stud_ans.get(q) or stud_ans.get(int(q)) if isinstance(stud_ans, dict) else None
            ka = key_ans.get(q) or key_ans.get(int(q)) if isinstance(key_ans, dict) else None
            ok = (sa == ka) and (sa is not None) and (ka is not None)
            correct_map[q] = ok
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(q)))
            self.table.setItem(r, 1, QTableWidgetItem("" if sa is None else str(sa)))
            self.table.setItem(r, 2, QTableWidgetItem("" if ka is None else str(ka)))

        if self.overlay_student and Path(self.overlay_student).exists():
            out_path = draw_marked_overlay(self.overlay_student, self.meta_student, correct_map, circle_scale_diam=0.5)
            self.overlay_marked = out_path
            pm = QPixmap(out_path)
            if not pm.isNull():
                self.image.setPixmap(pm.scaled(self.image.width(), self.image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.statusBar().showMessage("ไม่สามารถแสดง overlay ที่ทำเครื่องหมายได้", 3000)
        else:
            self.statusBar().showMessage("ไม่พบ overlay.jpg ของนิสิต", 3000)

        self.progress.setValue(100)

# ---------- Entry ----------
def main():
    app = QApplication(sys.argv)
    apply_palette(app, "Dark")
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
