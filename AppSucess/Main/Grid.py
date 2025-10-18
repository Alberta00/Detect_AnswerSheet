import sys, json, os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout,
    QWidget, QToolBar, QInputDialog, QMessageBox, QPushButton,
    QScrollArea, QStatusBar, QSizePolicy
)
from PySide6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QColor, QIcon
from PySide6.QtCore import Qt, QRect, QPoint, QProcess


# ===================== วิดเจ็ตวาดกริด =====================
class GridLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.pix = None
        self.scale = 1.0
        self.rects = []  # (QRect, name, block)
        self.mode = "select"  # select | add_block | add_answer
        self.start = None
        self.temp_rect = None
        self.dragging = False
        self.drag_block = None
        self.answer_count = 0
        self.block_counts = {}
        self.undo_stack = []
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background: #1e1e1e;")

    # ---------- Undo ----------
    def save_state_for_undo(self):
        snapshot = [(QRect(r), n, b) for (r, n, b) in self.rects]
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo_last_action(self):
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "ไม่มีการกระทำให้ย้อนกลับ")
            return
        last_state = self.undo_stack.pop()
        self.rects = [(QRect(r), n, b) for (r, n, b) in last_state]
        self.update()

    def setImage(self, cv_img):
        """โหลดภาพและปรับพอดีกับพื้นที่ทำงานเต็มจอ"""
        if cv_img is None:
            return

        # ขนาดพื้นที่ปัจจุบันของ scroll area
        parent = self.parent()
        if parent and hasattr(parent, "width"):
            area_w = parent.width()
            area_h = parent.height()
        else:
            area_w, area_h = 1920, 1080

        h, w, ch = cv_img.shape
        scale = min(area_w / w, area_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.scale = scale
        bytes_per_line = ch * new_w
        qimg = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format_BGR888)
        self.pix = QPixmap.fromImage(qimg)
        self.setPixmap(self.pix)

        # ✅ ปรับขนาดภาพและจัดให้อยู่ตรงกลาง
        self.setFixedSize(new_w, new_h)
        self.setAlignment(Qt.AlignCenter)
        self.update()


    # ---------- โหมดเครื่องมือ ----------
    def setMode(self, mode):
        self.mode = mode
        if mode == "select":
            self.setCursor(Qt.OpenHandCursor)
        elif mode in ("add_block", "add_answer"):
            self.setCursor(Qt.CrossCursor)
        self.update()

    # ---------- เริ่มลาก / เริ่มวาด ----------
    def mousePressEvent(self, event):
        if not self.pix:
            return
        if event.button() == Qt.LeftButton:
            if self.mode == "select":
                # รวมกรอบใหญ่ตามชื่อ block (ลากไปเป็นกลุ่ม)
                block_boxes = {}
                for rect, name, block in self.rects:
                    display_rect = QRect(
                        int(rect.x() * self.scale),
                        int(rect.y() * self.scale),
                        int(rect.width() * self.scale),
                        int(rect.height() * self.scale)
                    )
                    if block not in block_boxes:
                        block_boxes[block] = display_rect
                    else:
                        block_boxes[block] = block_boxes[block].united(display_rect)
                for block, big_rect in block_boxes.items():
                    if big_rect.contains(event.pos()):
                        self.save_state_for_undo()
                        self.dragging = True
                        self.drag_block = block
                        self.drag_start = event.pos()
                        return
                self.dragging = False
                self.drag_block = None
            else:
                self.start = event.pos()
                self.temp_rect = QRect(self.start, self.start)

    # ---------- ขณะลาก / ขณะวาด ----------
    def mouseMoveEvent(self, event):
        if self.dragging and self.drag_block:
            dx = (event.pos().x() - self.drag_start.x()) / self.scale
            dy = (event.pos().y() - self.drag_start.y()) / self.scale
            self.drag_start = event.pos()
            moved_rects = []
            for (r, n, b) in self.rects:
                if b == self.drag_block:
                    moved = QRect(int(r.x() + dx), int(r.y() + dy), r.width(), r.height())
                    moved_rects.append((moved, n, b))
                else:
                    moved_rects.append((r, n, b))
            self.rects = moved_rects
            self.update()
        elif self.start and self.mode in ("add_block", "add_answer"):
            self.temp_rect = QRect(self.start, event.pos())
            self.update()

    # ---------- ปล่อยเมาส์ ----------
    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.drag_block = None
            return

        if event.button() == Qt.LeftButton and self.temp_rect:
            rect = self.temp_rect.normalized()
            self.save_state_for_undo()

            # ---------- เพิ่ม Answer ----------
            if self.mode == "add_answer":
                rows, ok1 = QInputDialog.getInt(self, "จำนวนข้อ", "จำนวนแถว (rows):", 25, 1, 300)
                if not ok1:
                    self.temp_rect = None
                    self.start = None
                    self.update()
                    return

                cols, ok2 = QInputDialog.getInt(self, "จำนวนตัวเลือก", "จำนวนคอลัมน์ (choices):", 5, 1, 10)
                if not ok2:
                    self.temp_rect = None
                    self.start = None
                    self.update()
                    return

                # ตั้งชื่อบล็อกเป็น Answer_1, Answer_2, ...
                block_index = len([1 for _, _, b in self.rects if str(b).startswith("Answer_")]) + 1
                block_name = f"Answer_{block_index}"
                letters = [chr(ord('A') + i) for i in range(cols)]

                cell_w, cell_h = rect.width() / cols, rect.height() / rows
                for r in range(rows):
                    self.answer_count += 1
                    for c in range(cols):
                        name = f"Answer{self.answer_count}{letters[c]}"
                        cell = QRect(int(rect.x() + c * cell_w),
                                     int(rect.y() + r * cell_h),
                                     int(cell_w),
                                     int(cell_h))
                        scaled_rect = QRect(int(cell.x() / self.scale),
                                            int(cell.y() / self.scale),
                                            int(cell.width() / self.scale),
                                            int(cell.height() / self.scale))
                        self.rects.append((scaled_rect, name, block_name))
                QMessageBox.information(self, "Answer", f"สร้าง {block_name} ({rows}×{cols}) สำเร็จ")

            # ---------- เพิ่ม Block ----------
            elif self.mode == "add_block":
                block, okB = QInputDialog.getText(self, "ชื่อบล็อก", "ชื่อบล็อก (เช่น StudentID, SubID):")
                if not okB or not block.strip():
                    self.temp_rect = None
                    self.start = None
                    self.update()
                    return
                block = block.strip()

                rows, ok1 = QInputDialog.getInt(self, "จำนวนแถว", "Rows :", 1, 1, 300)
                if not ok1:
                    self.temp_rect = None; self.start = None; self.update(); return
                cols, ok2 = QInputDialog.getInt(self, "จำนวนคอลัมน์", "Columns :", 1, 1, 26)
                if not ok2:
                    self.temp_rect = None; self.start = None; self.update(); return

                direction, ok3 = QInputDialog.getItem(
                    self, "ทิศทางการนับ", "เลือกรูปแบบ:",
                    ["แนวนอน (→) — 1A, 1B, 1C...",
                     "แนวตั้ง (↓) — 1A, 2A, 3A..."], 0, False)
                if not ok3:
                    self.temp_rect = None; self.start = None; self.update(); return
                is_horizontal = direction.startswith("แนวนอน")

                cell_w, cell_h = rect.width() / cols, rect.height() / rows
                letters = [chr(ord("A") + i) for i in range(cols)]
                start_idx = self.block_counts.get(block, 0)

                if is_horizontal:
                    for r in range(rows):
                        row_index = start_idx + r + 1
                        for c in range(cols):
                            name = f"{row_index}{letters[c]}"
                            x, y = rect.x() + c * cell_w, rect.y() + r * cell_h
                            scaled_rect = QRect(int(x / self.scale), int(y / self.scale),
                                                int(cell_w / self.scale), int(cell_h / self.scale))
                            self.rects.append((scaled_rect, name, block))
                else:
                    letters_row = [chr(ord("A") + i) for i in range(rows)]
                    for r in range(rows):
                        row_letter = letters_row[r]
                        for c in range(cols):
                            col_index = start_idx + c + 1
                            name = f"{col_index}{row_letter}"
                            x, y = rect.x() + c * cell_w, rect.y() + r * cell_h
                            scaled_rect = QRect(int(x / self.scale), int(y / self.scale),
                                                int(cell_w / self.scale), int(cell_h / self.scale))
                            self.rects.append((scaled_rect, name, block))

                self.block_counts[block] = start_idx + rows
                QMessageBox.information(self, "Block", f"สร้างบล็อก {block} สำเร็จ ({rows}×{cols})")

            self.temp_rect = None
            self.start = None
            self.update()

    # ---------- วาดทั้งหมด ----------
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pix:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for rect, name, block in self.rects:
            is_answer = str(block).lower().startswith("answer")
            color = QColor(0, 200, 100, 200) if is_answer else QColor(60, 160, 255, 200)
            pen = QPen(color, 2)
            painter.setPen(pen)
            drect = QRect(int(rect.x() * self.scale), int(rect.y() * self.scale),
                          int(rect.width() * self.scale), int(rect.height() * self.scale))
            painter.drawRect(drect)
            # ชื่ออยู่ด้านบนกรอบ
            font_h = 14
            text_rect = QRect(drect.x(), drect.y() - font_h - 2, drect.width(), font_h + 2)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, f"{name}")

        if self.temp_rect:
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.temp_rect)


# ===================== หน้าต่างหลัก =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Designer v2 — Drag, Group & Save Preview")
        self.resize(1200, 820)
        self.setStyleSheet("""
            QMainWindow { background: #111; color: #ddd; }
            QLabel { color: #ddd; }
            QToolBar { background: #181818; border: none; padding: 4px; }
            QToolButton { color: #ddd; padding: 6px 10px; border-radius: 6px; }
            QToolButton:hover { background: #242424; }
            QToolButton:pressed { background: #2e2e2e; }
            QPushButton {
                background: #2a2a2a; color: #eee; border: 1px solid #3a3a3a;
                padding: 8px 12px; border-radius: 8px;
            }
            QPushButton:hover { background: #363636; }
            QPushButton:pressed { background: #404040; }
            QScrollArea { background: #151515; border: 1px solid #222;}
            QStatusBar { background: #181818; color: #aaa; }
        """)

        self.label = GridLabel()
        self.original_image = None

        # พื้นที่แสดงภาพแบบอยู่กึ่งกลางเสมอ
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)

        # พื้นที่แสดงภาพแบบเต็มจอ
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("background-color: #101010; border: none;")

        self.label = GridLabel()
        self.scroll.setWidget(self.label)

        # ✅ ใช้ layout แบบ Stretch เต็มหน้าจอ
                # ✅ ใช้ layout แบบ Stretch เต็มหน้าจอ
        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.scroll, stretch=1)

        # ปุ่ม "กลับหน้าเมนูหลัก" เชื่อมกับ back_to_main โดยตรง
        btn_back = QPushButton("🔙 กลับหน้าเมนูหลัก")
        btn_back.clicked.connect(self.back_to_main)
        btn_back.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                color: white;
                padding: 8px 14px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #505050; }
            QPushButton:pressed { background: #666; }
        """)
        vbox.addWidget(btn_back, alignment=Qt.AlignRight, stretch=0)
        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("พร้อมใช้งาน")

        # Toolbar
        tb = QToolBar("Tools")
        self.addToolBar(tb)

        act_open = QAction("📂 เปิดภาพ", self)
        act_open.triggered.connect(self.load_image)
        tb.addAction(act_open)

        act_select = QAction("🖐️ เลือก/ย้าย", self)
        act_select.triggered.connect(lambda: self.set_mode_and_status("select"))
        tb.addAction(act_select)

        act_block = QAction("🧩 เพิ่ม Block", self)
        act_block.triggered.connect(lambda: self.set_mode_and_status("add_block"))
        tb.addAction(act_block)

        act_answer = QAction("✅ เพิ่ม Answer", self)
        act_answer.triggered.connect(lambda: self.set_mode_and_status("add_answer"))
        tb.addAction(act_answer)

        act_undo = QAction("↩️ Undo (Ctrl+Z)", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self.label.undo_last_action)
        tb.addAction(act_undo)

        tb.addSeparator()

        act_save = QAction("💾 Save", self)
        act_save.triggered.connect(self.save_grids)
        tb.addAction(act_save)

    def set_mode_and_status(self, mode):
        self.label.setMode(mode)
        tip = {"select": "โหมดเลือก/ย้าย",
               "add_block": "โหมดเพิ่ม Block",
               "add_answer": "โหมดเพิ่ม Answer"}[mode]
        self.status.showMessage(tip, 3000)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "เลือกภาพ", "", "Images (*.jpg *.png)")
        if not file:
            return
        img = cv2.imread(file)
        if img is None:
            QMessageBox.warning(self, "Error", "ไม่สามารถเปิดภาพได้")
            return
        self.original_image = img.copy()
        self.label.setImage(img)
        self.status.showMessage(f"เปิดภาพ: {file}", 5000)
        QMessageBox.information(self, "โหลดภาพแล้ว", f"เปิดภาพ: {file}")

    def save_grids(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "กรุณาเปิดภาพก่อนบันทึก")
            return

        # reset counter เมื่อเซฟใหม่ทุกครั้ง (เฉพาะไว้ใช้ตั้งชื่อช่อง AnswerX… ต่อเนื่องภายในเซสชัน)
        self.label.answer_count = 0

        folder = "grids"
        os.makedirs(folder, exist_ok=True)
        filename, _ = QFileDialog.getSaveFileName(
            self, "บันทึก Template", os.path.join(folder, "new_grid.json"),
            "JSON (*.json)"
        )
        if not filename:
            return

        # ลบไฟล์เก่าก่อนถ้ามี
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                QMessageBox.warning(self, "ลบไฟล์เก่าไม่สำเร็จ", f"ไม่สามารถลบไฟล์เก่าได้:\n{e}")
                return

        # เตรียมข้อมูลช่องทั้งหมด (Answer_* → Answer)
        data = []
        for r, n, b in self.label.rects:
            block_name = "Answer" if str(b).lower().startswith("answer_") else b
            data.append({
                "block": block_name,
                "name": n,
                "x": r.x(), "y": r.y(),
                "w": r.width(), "h": r.height()
            })

        # สร้างภาพ Preview
        preview_path = os.path.splitext(filename)[0] + "_preview.jpg"
        try:
            img = self.original_image.copy()
            # วาดกรอบทั้งหมดบนภาพต้นฉบับ (พิกัดเป็นสเกล 1:1)
            for cell in data:
                x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
                block = cell["block"]
                color = (0, 200, 100) if block.lower() == "answer" else (255, 160, 60)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, cell["name"], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            if os.path.exists(preview_path):
                os.remove(preview_path)
            cv2.imwrite(preview_path, img)

            # ย่อภาพแสดงไม่ให้เต็มจอ
            disp = img.copy()
            max_w, max_h = 1000, 700
            ih, iw = disp.shape[:2]
            s = min(max_w / iw, max_h / ih, 1.0)
            if s < 1.0:
                disp = cv2.resize(disp, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)

            cv2.imshow("Template Preview", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"เกิดข้อผิดพลาดในการสร้าง Preview:\n{e}")

        # สร้าง JSON (รวม path รูปพรีวิว)
        meta = {
            "template_path": filename,
            "image_path": preview_path,
            "grids": data
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.status.showMessage(f"บันทึก {os.path.basename(filename)} สำเร็จ", 5000)
        QMessageBox.information(
            self, "บันทึกแล้ว ✅",
            f"Template ถูกบันทึกพร้อมภาพพรีวิว!\n\n"
            f"📄 {os.path.basename(filename)}\n"
            f"🖼️ {os.path.basename(preview_path)}"
        )

    def back_to_main(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(current_dir, "main.py")

        if not os.path.exists(main_path):
            QMessageBox.warning(self, "ไม่พบไฟล์", f"❌ หา main.py ไม่เจอในโฟลเดอร์:\n{current_dir}")
            return

        try:
            # ✅ ใช้วิธี universal — เปิดไฟล์ Python ด้วย interpreter เดิม หรือเปิดด้วย os.startfile
            import subprocess

            if sys.executable and os.path.exists(sys.executable):
                subprocess.Popen([sys.executable, main_path], shell=False)
            else:
                # สำหรับ .exe build หรือบางกรณีไม่มี sys.executable
                os.startfile(main_path)

            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"⚠️ ไม่สามารถเปิด main.py ได้:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
