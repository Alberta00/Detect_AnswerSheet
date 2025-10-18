import sys, json, csv, os, re, cv2, numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox, QMessageBox,
    QComboBox, QScrollArea, QSlider
)
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QPixmap, QImage, QFont


# ===================== ฟังก์ชันตรวจจากภาพ =====================
def analyze_image(image_path, grids, total_questions):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบภาพ: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = {}
    for cell in grids:
        name = cell["name"]
        block = cell["block"]
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        roi = gray[y:y+h, x:x+w]
        mean_intensity = np.mean(roi)
        if block not in results:
            results[block] = {}
        results[block][name] = mean_intensity

    letter_to_digit = {chr(ord("A") + i): str(i) for i in range(10)}
    decoded_blocks, answers = {}, {}

    # ตรวจ Block ที่ไม่ใช่ Answer
    for block, cells in results.items():
        if block.lower() == "answer":
            continue
        grouped = {}
        for name, val in cells.items():
            m1 = re.match(r"(\d+)([A-Z])", name)
            m2 = re.match(r"([A-Z])(\d+)", name)
            if m1:
                row, col = int(m1.group(1)), m1.group(2)
            elif m2:
                row, col = int(m2.group(2)), m2.group(1)
            else:
                continue
            grouped.setdefault(row, {})[col] = val

        decoded = ""
        for r, cols in sorted(grouped.items()):
            selected = min(cols, key=cols.get)
            digit = letter_to_digit.get(selected, "?")
            decoded += digit
        decoded_blocks[block] = decoded

    # ✅ ตรวจ Answer — อ่านครบทุกข้อ แต่ใส่ NULL ถ้าไม่ฝน
    if "Answer" in results:
        grouped = {}
        for name, val in results["Answer"].items():
            m = re.match(r"Answer(\d+)([A-Z])", name)
            if not m:
                continue
            q, choice = int(m.group(1)), m.group(2)
            grouped.setdefault(q, {})[choice] = val

        for q in sorted(grouped.keys()):
            if q not in grouped or not grouped[q]:
                answers[q] = "NULL"
                continue
            values = grouped[q]
            min_choice = min(values, key=values.get)
            min_val = values[min_choice]
            mean_val = np.mean(list(values.values()))
            if mean_val - min_val < 25:
                answers[q] = "NULL"
            else:
                answers[q] = min_choice

    return {"file": os.path.basename(image_path), "blocks": decoded_blocks, "answers": answers}


# ===================== หน้าตรวจหลัก =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Auto Checker — ตรวจข้อสอบอัตโนมัติ ✅")
        self.resize(1000, 750)

        self.images = []
        self.selected_image_path = None
        self.grids = None
        self.answer_key = {}
        self.answer_key_img = None
        self.preview_pix = None
        self.template_loaded = False
        self.answer_loaded = False

        # ---------- Layout หลัก ----------
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # ---------- Template ----------
        top_layout = QHBoxLayout()
        lbl_temp = QLabel("📄 Template:")
        lbl_temp.setFont(QFont("Segoe UI", 11))
        self.grid_combo = QComboBox()
        self.grid_combo.currentIndexChanged.connect(self.change_template)
        combo_style = """
        QComboBox {
            background-color: #222;
            color: white;
            padding: 6px;
            border: 1px solid #555;
            border-radius: 4px;
        }
        QComboBox:hover {
            border: 1px solid #7B68EE;
        }
        QComboBox QAbstractItemView {
            background-color: #333;
            selection-background-color: #7B68EE;
            color: white;
        }
        """
        self.grid_combo.setStyleSheet(combo_style)
        top_layout.addWidget(lbl_temp)
        top_layout.addWidget(self.grid_combo)
        main_layout.addLayout(top_layout)

        # ---------- ภาพ ----------
        self.image_label = QLabel("🔲 ยังไม่มีภาพ Template แสดง")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #aaa; background: #fafafa;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        main_layout.addWidget(scroll, stretch=2)

        # ---------- แถบเลือกภาพนักเรียน ----------
        self.student_combo = QComboBox()
        self.student_combo.setStyleSheet(combo_style)
        self.student_combo.currentIndexChanged.connect(self.show_student_preview)
        self.student_combo.hide()  # ซ่อนไว้ก่อน
        main_layout.addWidget(self.student_combo)

        # ---------- สถานะ ----------
        self.lbl_status = QLabel("📂 สถานะ: ยังไม่ได้โหลด Template")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #ddd; font-size: 14px; padding: 6px;")
        main_layout.addWidget(self.lbl_status)

        # ---------- ปุ่ม ----------
        button_layout = QHBoxLayout()
        self.btn_load_answer_image = QPushButton("🧾 โหลดภาพเฉลย")
        self.btn_load_answer_image.clicked.connect(self.load_answer_image)
        self.btn_load_answer_image.setStyleSheet(self._btn_style("#7B68EE", "#6A5ACD", "#483D8B"))

        self.btn_add_images = QPushButton("🖼️ เพิ่มภาพนักเรียน")
        self.btn_add_images.clicked.connect(self.add_images)
        self.btn_add_images.setEnabled(False)
        self.btn_add_images.setStyleSheet(self._btn_style("#4682B4", "#5A9BD4", "#36648B"))

        self.btn_run = QPushButton("🚀 เริ่มตรวจทั้งหมด")
        self.btn_run.clicked.connect(self.run_check)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(self._btn_style("#228B22", "#32CD32", "#006400"))

        button_layout.addWidget(self.btn_load_answer_image)
        button_layout.addWidget(self.btn_add_images)
        button_layout.addWidget(self.btn_run)
        main_layout.addLayout(button_layout)

        # ---------- จำนวนข้อ ----------
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("จำนวนข้อที่ต้องตรวจ:"))
        self.spin_q = QSpinBox()
        self.spin_q.setRange(1, 200)
        self.spin_q.setValue(30)
        hlayout.addWidget(self.spin_q)
        main_layout.addLayout(hlayout)

        # ---------- Zoom ----------
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("🔍 Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.apply_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        main_layout.addLayout(zoom_layout)

        # ---------- กลับ ----------
        btn_back = QPushButton("🔙 กลับหน้าเมนูหลัก")
        btn_back.clicked.connect(self.back_to_main)
        btn_back.setStyleSheet("background: #DC143C; color: white; font-weight: bold; padding: 6px 10px; border-radius: 6px;")
        main_layout.addWidget(btn_back)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.load_grid_list(auto_load=True)

    # ---------- ปุ่มสไตล์ ----------
    def _btn_style(self, color, hover, press):
        return f"""
        QPushButton {{
            background-color: {color};
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: bold;
            transition: all 0.2s;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {press};
            transform: scale(0.97);
        }}
        QPushButton:disabled {{
            background-color: #555;
            color: #999;
        }}
        """

    # ---------- โหลด Template ----------
    def load_grid_list(self, auto_load=False):
        # ใช้ path ของไฟล์โปรแกรมเป็นฐานเสมอ
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "grids")

        os.makedirs(folder, exist_ok=True)
        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

        self.grid_combo.clear()
        if not json_files:
            self.grid_combo.addItem("(ไม่มี Template ในโฟลเดอร์ grids)")
            return

        self.grid_combo.addItems(json_files)

        if auto_load:
            first_path = os.path.join(folder, json_files[0])
            self.load_template(first_path)


    def change_template(self):
        selected = self.grid_combo.currentText()
        if not selected or selected.startswith("("):
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "grids", selected)
        self.load_template(file_path)


    def load_template(self, file):
        # ✅ ตรวจสอบและแปลง path ให้เป็น absolute เสมอ
        if not os.path.isabs(file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file = os.path.join(base_dir, file)
        file = os.path.normpath(file)

        if not os.path.exists(file):
            QMessageBox.warning(self, "Error", f"❌ หาไฟล์ Template ไม่เจอ:\n{file}")
            return

        try:
            with open(file, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"อ่าน Template ไม่ได้:\n{e}")
            return

        # ✅ โหลดข้อมูล grids และรูปภาพ
        if isinstance(content, dict) and "grids" in content:
            self.grids = content["grids"]
            img_path = content.get("image_path", None)
            self.template_loaded = True

            # 🔍 ถ้า image_path เป็น relative (ไม่มี :\ หรือ /home) → ให้ต่อ path กับที่อยู่ JSON
            if img_path:
                # ✅ ถ้า path จาก JSON มีคำว่า "grids" อยู่แล้ว → แสดงว่าเป็น path สมบูรณ์ภายในโปรเจกต์
                if not os.path.isabs(img_path):
                    if "grids" in img_path.lower():
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        img_path = os.path.join(base_dir, img_path)
                    else:
                        img_path = os.path.join(os.path.dirname(file), img_path)
                img_path = os.path.normpath(img_path)

            # ✅ ตรวจสอบว่ารูปมีจริงไหม
            if img_path and os.path.exists(img_path):
                self.show_grid_preview(img_path, draw_boxes=True)
                self.lbl_status.setText(f"✅ โหลด Template แล้ว: {os.path.basename(file)}")
            else:
                QMessageBox.warning(
                    self, "Image Missing",
                    f"⚠️ Template โหลดได้แต่ไม่พบภาพ Preview:\n{img_path}"
                )
                self.lbl_status.setText("⚠️ Template ไม่มีภาพ Preview หรือหาไฟล์ไม่เจอ")
        else:
            QMessageBox.warning(self, "Error", "Template ไม่ถูกต้อง")


    # ---------- แสดง Grid ----------
    def show_grid_preview(self, image_path, draw_boxes=True):
        img = cv2.imread(image_path)
        if img is None:
            return
        draw = img.copy()
        if draw_boxes and self.grids:
            for cell in self.grids:
                x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
                block = str(cell.get("block", "")).lower()
                color = (0, 255, 0, 80) if block == "answer" else (255, 128, 0, 80)
                cv2.rectangle(draw, (x, y), (x + w, y + h), color[:3], 2)
        rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.preview_pix = QPixmap.fromImage(qimg)
        self.apply_zoom()

    # ---------- โหลดเฉลย ----------
    def load_answer_image(self):
        if not self.template_loaded:
            QMessageBox.warning(self, "Error", "กรุณาโหลด Template ก่อน")
            return
        file, _ = QFileDialog.getOpenFileName(self, "เลือกภาพเฉลย", "", "Images (*.jpg *.png)")
        if not file:
            return
        data = analyze_image(file, self.grids, 999)
        self.answer_key_img = file
        self.answer_key = data["answers"]
        self.answer_loaded = True
        self.lbl_status.setText(f"✅ โหลดเฉลยสำเร็จ ({len(self.answer_key)} ข้อ)")
        self.btn_add_images.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.show_grid_preview(file, draw_boxes=True)

    # ---------- เพิ่มภาพ ----------
    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "เลือกภาพนักเรียน", "", "Images (*.jpg *.png)")
        if not files:
            return
        self.images = files.copy()
        self.lbl_status.setText(f"📸 เลือกภาพใหม่แล้ว {len(self.images)} ไฟล์")

        # ✅ แสดงรายชื่อใน dropdown
        self.student_combo.clear()
        self.student_combo.addItems([os.path.basename(f) for f in files])
        self.student_combo.show()
        self.student_combo.setCurrentIndex(0)
        self.selected_image_path = self.images[0]
        self.show_student_preview()

    # ---------- แสดงภาพนักเรียน ----------
    def show_student_preview(self):
        if not self.images or not self.template_loaded:
            return
        index = self.student_combo.currentIndex()
        if index < 0 or index >= len(self.images):
            return
        img_path = self.images[index]
        self.selected_image_path = img_path
        self.show_grid_preview(img_path, draw_boxes=True)
        self.lbl_status.setText(f"👁️ ดูภาพ: {os.path.basename(img_path)}")

    # ---------- ตรวจ ----------
    def run_check(self):
        if not self.answer_loaded:
            QMessageBox.warning(self, "Error", "ยังไม่มีเฉลย")
            return
        if not self.images:
            QMessageBox.warning(self, "Error", "ยังไม่มีภาพนักเรียน")
            return

        limit_q = self.spin_q.value()
        all_results = []

        for img_path in self.images:
            data = analyze_image(img_path, self.grids, limit_q)
            score = sum(
                1 for q, correct in self.answer_key.items()
                if q <= limit_q and data["answers"].get(q) == correct
            )
            data["score"] = score
            all_results.append(data)

        # ✅ เก็บชื่อ block ที่ไม่ใช่ Answer (เช่น StudentID, SubjectCode)
        block_names = sorted(
            {cell["block"] for cell in self.grids if str(cell["block"]).lower() != "answer"}
        )

        # ✅ results.json (เก็บผลละเอียด)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # ✅ scores.csv — แสดง block ทั้งหมดที่ user มี + คะแนน + เต็ม
        with open("scores.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            header = ["ไฟล์"] + block_names + ["คะแนน", "เต็ม"]
            writer.writerow(header)

            for r in all_results:
                row = [r["file"]]
                for b in block_names:
                    row.append(r["blocks"].get(b, ""))
                row += [r["score"], limit_q]
                writer.writerow(row)

        # ✅ answers.csv — รายข้อ
        with open("answers.csv", "w", newline="", encoding="utf-8-sig") as f2:
            writer2 = csv.writer(f2)
            header = ["ไฟล์"] + [f"Q{i}" for i in range(1, limit_q + 1)]
            writer2.writerow(header)
            for r in all_results:
                ans_row = [r["file"]] + [
                    r["answers"].get(i, "NULL") for i in range(1, limit_q + 1)
                ]
                writer2.writerow(ans_row)

        QMessageBox.information(
            self,
            "✅ เสร็จสิ้น",
            f"ตรวจแล้ว {len(all_results)} ไฟล์\n"
            f"• results.json\n• scores.csv\n• answers.csv\n\n"
            f"บันทึกผลเรียบร้อยแล้ว!"
        )


    # ---------- ซูม ----------
    def apply_zoom(self):
        if self.preview_pix is None:
            return
        scale = self.zoom_slider.value() / 100.0
        scaled = self.preview_pix.scaled(
            int(self.preview_pix.width() * scale),
            int(self.preview_pix.height() * scale),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def back_to_main(self):
        path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(path):
            QProcess.startDetached(sys.executable, [path])
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
