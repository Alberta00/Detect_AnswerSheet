import sys, json, os
import cv2
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout,
    QWidget, QToolBar, QInputDialog, QMessageBox, QPushButton,
    QScrollArea, QStatusBar, QSlider, QHBoxLayout
)
from PySide6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QColor, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QRect

class GridLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.base_qpix = None
        self.scale = 1.0
        self.rects = []           # (QRect, name, block) base coords
        self.mode = "select"
        self.start = None
        self.temp_rect = None
        self.dragging = False
        self.drag_block = None
        self.answer_count = 0
        self.block_counts = {}
        self.undo_stack = []
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #1e1e1e; color:#ddd;")

    # ---- zoom (ไม่มีเม้าส์สกอลล์) ----
    def set_base_image(self, cv_img):
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch*w, QImage.Format_BGR888)
        self.base_qpix = QPixmap.fromImage(qimg)
        self.scale = 1.0
        self._apply()

    def set_scale(self, s):
        self.scale = max(0.1, min(4.0, s))
        self._apply()

    # ปิดการซูมด้วยเม้าส์
    def wheelEvent(self, e):
        e.ignore()

    def _apply(self):
        if self.base_qpix is None: return
        scaled = self.base_qpix.scaled(self.base_qpix.size()*self.scale, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # ขนาด QLabel จะตามรูป (ด้านในอยู่ใน QScrollArea ทำให้ "ช่องแสดงผล" คงที่)
        self.setPixmap(scaled)
        self.resize(scaled.size())
        self.update()

    # ---- undo ----
    def save_state_for_undo(self):
        self.undo_stack.append([(QRect(r), n, b) for (r, n, b) in self.rects])
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo_last_action(self):
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "ไม่มีการกระทำให้ย้อนกลับ"); return
        last = self.undo_stack.pop()
        self.rects = [(QRect(r), n, b) for (r, n, b) in last]
        self.update()

    def setMode(self, mode):
        self.mode = mode
        self.setCursor(Qt.OpenHandCursor if mode=="select" else Qt.CrossCursor)

    # ---- mouse helpers ----
    def img_to_base(self, p):
        return int(p.x()/self.scale), int(p.y()/self.scale)

    def mousePressEvent(self, e):
        if not self.base_qpix: return
        if e.button() == Qt.LeftButton:
            if self.mode == "select":
                self.save_state_for_undo()
                self.dragging, self.drag_block = True, self._block_at_point(e.pos())
                self.drag_prev = e.pos()
            else:
                self.start = e.pos()
                self.temp_rect = QRect(self.start, self.start)

    def mouseMoveEvent(self, e):
        if self.dragging and self.drag_block:
            dx = (e.pos().x()-self.drag_prev.x())/self.scale
            dy = (e.pos().y()-self.drag_prev.y())/self.scale
            self.drag_prev = e.pos()
            nr = []
            for r, n, b in self.rects:
                if b == self.drag_block:
                    nr.append((QRect(int(r.x()+dx), int(r.y()+dy), r.width(), r.height()), n, b))
                else:
                    nr.append((r, n, b))
            self.rects = nr; self.update()
        elif self.start and self.mode in ("add_block","add_answer"):
            self.temp_rect = QRect(self.start, e.pos()); self.update()

    def mouseReleaseEvent(self, e):
        if self.dragging:
            self.dragging = False; self.drag_block = None; return
        if e.button() == Qt.LeftButton and self.temp_rect:
            rect = self.temp_rect.normalized()
            self.save_state_for_undo()
            x0,y0 = self.img_to_base(rect.topLeft())
            x1,y1 = self.img_to_base(rect.bottomRight())
            bw,bh = max(1,x1-x0), max(1,y1-y0)

            if self.mode == "add_answer":
                rows, ok1 = QInputDialog.getInt(self, "จำนวนข้อ","จำนวนแถว (rows):",25,1,300)
                if not ok1: self.start=None; self.temp_rect=None; self.update(); return
                cols, ok2 = QInputDialog.getInt(self, "จำนวนตัวเลือก","จำนวนคอลัมน์ (choices):",5,1,10)
                if not ok2: self.start=None; self.temp_rect=None; self.update(); return
                block_index = len([1 for _,_,b in self.rects if str(b).startswith("Answer_")])+1
                block_name = f"Answer_{block_index}"
                letters = [chr(ord('A')+i) for i in range(cols)]
                cell_w,cell_h = bw/cols, bh/rows
                for r in range(rows):
                    self.answer_count += 1
                    for c in range(cols):
                        name = f"Answer{self.answer_count}{letters[c]}"
                        self.rects.append((QRect(int(x0+c*cell_w), int(y0+r*cell_h), int(cell_w), int(cell_h)), name, block_name))
                QMessageBox.information(self,"Answer",f"สร้าง {block_name} ({rows}×{cols}) สำเร็จ")

            elif self.mode == "add_block":
                block, okB = QInputDialog.getText(self, "ชื่อบล็อก","ชื่อบล็อก (เช่น StudentID, SubjectCode):")
                if not okB or not block.strip(): self.start=None; self.temp_rect=None; self.update(); return
                block = block.strip()
                rows, ok1 = QInputDialog.getInt(self,"จำนวนแถว","Rows :",1,1,300)
                if not ok1: self.start=None; self.temp_rect=None; self.update(); return
                cols, ok2 = QInputDialog.getInt(self,"จำนวนคอลัมน์","Columns :",1,1,26)
                if not ok2: self.start=None; self.temp_rect=None; self.update(); return
                direction, ok3 = QInputDialog.getItem(self,"ทิศทางการนับ","เลือกรูปแบบ:",["แนวนอน (→) — 1A, 1B, 1C...","แนวตั้ง (↓) — 1A, 2A, 3A..."],0,False)
                if not ok3: self.start=None; self.temp_rect=None; self.update(); return
                is_horizontal = direction.startswith("แนวนอน")
                cell_w,cell_h = bw/cols, bh/rows
                letters = [chr(ord("A")+i) for i in range(cols)]
                start_idx = self.block_counts.get(block,0)
                if is_horizontal:
                    for r in range(rows):
                        row_index = start_idx+r+1
                        for c in range(cols):
                            name=f"{row_index}{letters[c]}"
                            self.rects.append((QRect(int(x0+c*cell_w),int(y0+r*cell_h),int(cell_w),int(cell_h)), name, block))
                else:
                    letters_row=[chr(ord("A")+i) for i in range(rows)]
                    for r in range(rows):
                        row_letter=letters_row[r]
                        for c in range(cols):
                            col_index=start_idx+c+1
                            name=f"{col_index}{row_letter}"
                            self.rects.append((QRect(int(x0+c*cell_w),int(y0+r*cell_h),int(cell_w),int(cell_h)), name, block))
                self.block_counts[block]=start_idx+rows
                QMessageBox.information(self,"Block",f"สร้างบล็อก {block} สำเร็จ ({rows}×{cols})")
            self.start=None; self.temp_rect=None; self.update()

    def _block_at_point(self, pos):
        bx, by = self.img_to_base(pos)
        for r,n,b in self.rects:
            if QRect(r).contains(bx,by): return b
        return None

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.base_qpix is None: return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        for rect,name,block in self.rects:
            color = QColor(0,200,100,200) if str(block).lower().startswith("answer") else QColor(60,160,255,200)
            p.setPen(QPen(color,2))
            drect = QRect(int(rect.x()*self.scale), int(rect.y()*self.scale), int(rect.width()*self.scale), int(rect.height()*self.scale))
            p.drawRect(drect)
            text_rect = QRect(drect.x(), drect.y()-16, drect.width(), 16)
            p.drawText(text_rect, Qt.AlignLeft|Qt.AlignVCenter, name)
        if self.temp_rect:
            p.setPen(QPen(Qt.red,2,Qt.DashLine)); p.drawRect(self.temp_rect)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Designer — ซูมด้วยสไลเดอร์/Ctrl เท่านั้น")
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

        # ช่องพรีวิว “คงที่” ด้วย QScrollArea
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)  # viewport คงที่
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("background-color: #101010; border: none;")
        self.scroll.setWidget(self.label)

        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(8,8,8,8)
        vbox.setSpacing(8)
        vbox.addWidget(self.scroll, stretch=1)

        zlayout = QHBoxLayout()
        zlayout.addWidget(QLabel("🔍 Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(10,400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(lambda v: self.label.set_scale(v/100.0))
        zlayout.addWidget(self.zoom_slider)
        vbox.addLayout(zlayout)

        btn_back = QPushButton("🔙 กลับหน้าเมนูหลัก")
        btn_back.clicked.connect(self.back_to_main)
        vbox.addWidget(btn_back, alignment=Qt.AlignRight, stretch=0)
        self.setCentralWidget(central)

        self.status = QStatusBar(); self.setStatusBar(self.status); self.status.showMessage("พร้อมใช้งาน")

        tb = QToolBar("Tools"); self.addToolBar(tb)
        act_open = QAction("📂 เปิดภาพ", self); act_open.triggered.connect(self.load_image); tb.addAction(act_open)
        act_select = QAction("🖐️ เลือก/ย้าย", self); act_select.triggered.connect(lambda: self.set_mode_and_status("select")); tb.addAction(act_select)
        act_block  = QAction("🧩 เพิ่ม Block", self); act_block.triggered.connect(lambda: self.set_mode_and_status("add_block")); tb.addAction(act_block)
        act_answer = QAction("✅ เพิ่ม Answer", self); act_answer.triggered.connect(lambda: self.set_mode_and_status("add_answer")); tb.addAction(act_answer)
        act_undo = QAction("↩️ Undo (Ctrl+Z)", self); act_undo.setShortcut("Ctrl+Z"); act_undo.triggered.connect(self.label.undo_last_action); tb.addAction(act_undo)
        tb.addSeparator()
        act_save = QAction("💾 Save", self); act_save.triggered.connect(self.save_grids); tb.addAction(act_save)

        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self._step_zoom(True))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self._step_zoom(True))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self._step_zoom(False))

    def _step_zoom(self, inc):
        v = self.zoom_slider.value()
        self.zoom_slider.setValue(min(400, v+10) if inc else max(10, v-10))

    def set_mode_and_status(self, mode):
        self.label.setMode(mode)
        tip = {"select":"โหมดเลือก/ย้าย","add_block":"โหมดเพิ่ม Block","add_answer":"โหมดเพิ่ม Answer"}[mode]
        self.status.showMessage(tip, 3000)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "เลือกภาพ", "", "Images (*.jpg *.png)")
        if not file: return
        img = cv2.imread(file)
        if img is None:
            QMessageBox.warning(self, "Error", "ไม่สามารถเปิดภาพได้"); return
        self.original_image = img.copy()
        self.label.set_base_image(img)
        self.zoom_slider.setValue(100)
        self.status.showMessage(f"เปิดภาพ: {file}", 5000)
        QMessageBox.information(self, "โหลดภาพแล้ว", f"เปิดภาพ: {file}")

    def _validate_template(self, data, img_shape):
        H, W = img_shape[:2]
        names = set()
        rects = []
        for c in data:
            x,y,w,h = c["x"],c["y"],c["w"],c["h"]
            if x<0 or y<0 or w<=0 or h<=0 or x+w>W or y+h>H:
                return False, f"กรอบ {c['name']} อยู่นอกขอบภาพหรือขนาดไม่ถูกต้อง"
            if c["name"] in names:
                return False, f"พบชื่อซ้ำ: {c['name']}"
            names.add(c["name"]); rects.append((x,y,w,h,c["name"]))
        for i in range(len(rects)):
            x1,y1,w1,h1,n1 = rects[i]; r1=(x1,y1,x1+w1,y1+h1)
            for j in range(i+1,len(rects)):
                x2,y2,w2,h2,n2 = rects[j]; r2=(x2,y2,x2+w2,y2+h2)
                ix1,iy1=max(r1[0],r2[0]),max(r1[1],r2[1])
                ix2,iy2=min(r1[2],r2[2]),min(r1[3],r2[3])
                if ix2>ix1 and iy2>iy1:
                    return False, f"กรอบ {n1} ทับกับ {n2}"
        return True,"OK"

    def save_grids(self):
        if self.original_image is None:
            QMessageBox.warning(self,"Error","กรุณาเปิดภาพก่อนบันทึก"); return
        data=[]
        for r,n,b in self.label.rects:
            data.append({"block":"Answer" if str(b).lower().startswith("answer_") else b,
                         "name":n, "x":r.x(), "y":r.y(), "w":r.width(), "h":r.height()})
        ok,msg = self._validate_template(data, self.original_image.shape)
        if not ok:
            QMessageBox.warning(self, "Template ไม่ผ่านการตรวจสอบ", msg); return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "grids"); os.makedirs(folder, exist_ok=True)
        filename, _ = QFileDialog.getSaveFileName(self,"บันทึก Template",
                                                  os.path.join(folder,"new_grid.json"),"JSON (*.json)")
        if not filename: return

        preview_path = os.path.splitext(filename)[0] + "_preview.jpg"
        try:
            img = self.original_image.copy()
            for c in data:
                x,y,w,h=c["x"],c["y"],c["w"],c["h"]
                color = (0,200,100) if c["block"].lower()=="answer" else (255,160,60)
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,c["name"],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
            cv2.imwrite(preview_path, img)
        except Exception as e:
            QMessageBox.warning(self,"Preview Error",f"เกิดข้อผิดพลาดในการสร้าง Preview:\n{e}")

        rel_template = os.path.relpath(filename, start=base_dir)
        rel_preview  = os.path.relpath(preview_path, start=base_dir)
        meta = {"template_path": rel_template, "image_path": rel_preview, "grids": data}
        with open(filename,"w",encoding="utf-8") as f: json.dump(meta,f,indent=2,ensure_ascii=False)
        QMessageBox.information(self,"บันทึกแล้ว ✅","บันทึก Template และ Preview เรียบร้อย")

    def back_to_main(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(current_dir, "main.py")
        try:
            if sys.executable and os.path.exists(sys.executable):
                import subprocess; subprocess.Popen([sys.executable, main_path], shell=False)
            else:
                os.startfile(main_path)
            self.close()
        except Exception as e:
            QMessageBox.critical(self,"Error",f"ไม่สามารถเปิด main.py ได้:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
