# main.py — หน้าหลักของระบบ OMR
import os, sys, subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt, QProcess, QUrl
from PySide6.QtGui import QDesktopServices

APP_DIR = Path(__file__).resolve().parent

# ---- ตัวช่วยเปิดไฟล์/โฟลเดอร์อย่างปลอดภัย ----
def open_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR System — Main Menu")
        self.resize(720, 420)

        title = QLabel("OMR Answer Sheet — Main Menu")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700; padding: 8px;")

        # ปุ่มหลัก
        btn_checker = QPushButton("📝 ตรวจข้อสอบ (Open Checker)")
        btn_checker.setMinimumHeight(44)
        btn_checker.clicked.connect(self.open_checker)

        btn_summary = QPushButton("📊 สรุปผลคะแนน (Open Summary)")
        btn_summary.setMinimumHeight(44)
        btn_summary.clicked.connect(self.open_summary)

        btn_grid = QPushButton("🧩 จัดการ Grid / Template")
        btn_grid.setMinimumHeight(44)
        btn_grid.clicked.connect(self.open_grid_editor)

        btn_out = QPushButton("📂 เปิดโฟลเดอร์ผลลัพธ์ (out/)")
        btn_out.setMinimumHeight(40)
        btn_out.clicked.connect(lambda: open_dir(APP_DIR / "out"))

        btn_exit = QPushButton("❌ ออกจากโปรแกรม")
        btn_exit.setMinimumHeight(40)
        btn_exit.clicked.connect(self.close)

        # เค้าโครง
        col = QVBoxLayout()
        col.addWidget(title)
        col.addSpacing(6)
        col.addWidget(btn_checker)
        col.addWidget(btn_summary)
        col.addWidget(btn_grid)

        col.addSpacing(10)
        row = QHBoxLayout()
        row.addWidget(btn_out)
        row.addWidget(btn_exit)
        col.addLayout(row)

        wrap = QWidget()
        wrap.setLayout(col)
        wrap.setStyleSheet("""
    QMainWindow, QWidget { background: white; color: black; }
    QLabel { color: #111; }
    QTextEdit { background: #f9f9f9; color: #111; border: 1px solid #ccc; }
    QPushButton {
        background: #f2f2f2; color: #111; border: 1px solid #d0d0d0;
        padding: 8px 12px; border-radius: 10px; font-weight: 600;
    }
    QPushButton:hover { background: #e9e9e9; }
    QPushButton:pressed { background: #dddddd; }
""")
        self.setCentralWidget(wrap)

        # กัน GC: เก็บรีเฟอเรนซ์หน้าลูก
        self._child_windows = []

    # ---------- เปิดหน้าตรวจ ----------
    def open_checker(self):
        """
        พยายาม import Detect.MainWindow ถ้าไม่สำเร็จให้รัน Detect.py เป็นโปรเซสใหม่
        """
        try:
            import Detect  # ต้องอยู่โฟลเดอร์เดียวกัน
            win = Detect.MainWindow()
            win.show()
            self._child_windows.append(win)
        except Exception as e:
            # fallback: เปิดไฟล์ Detect.py เป็นโปรเซสใหม่ของ Python
            det_path = APP_DIR / "Detect.py"
            if not det_path.exists():
                QMessageBox.critical(self, "ไม่พบไฟล์", f"หา Detect.py ไม่เจอที่\n{det_path}")
                return
            QProcess.startDetached(sys.executable, [str(det_path)])

    # ---------- เปิดหน้าสรุป ----------
    def open_summary(self):
        try:
            import Detect
            win = Detect.SummaryWindow()
            win.show()
            self._child_windows.append(win)
        except Exception as e:
            det_path = APP_DIR / "Detect.py"
            if not det_path.exists():
                QMessageBox.critical(self, "ไม่พบไฟล์", f"หา Detect.py ไม่เจอที่\n{det_path}")
                return
            # ให้ Detect.py เปิด Summary เองไม่ได้ จึงรัน Detect แล้วผู้ใช้กดเปิดสรุปจากในแอพได้
            QProcess.startDetached(sys.executable, [str(det_path)])

    # ---------- เปิด Grid/Template Editor ----------
    def open_grid_editor(self):
        """
        เพื่อเลี่ยงการพึ่งพาคลาส/ชื่อคลาสใน Grid.py (ซึ่งอาจต่างจากที่เราคาด)
        ใช้วิธีเปิด Grid.py เป็นโปรเซส Python แยก
        """
        grid_path = APP_DIR / "Grid.py"
        if not grid_path.exists():
            QMessageBox.critical(self, "ไม่พบไฟล์", f"หา Grid.py ไม่เจอที่\n{grid_path}")
            return
        QProcess.startDetached(sys.executable, [str(grid_path)])

def main():
    app = QApplication(sys.argv)
    w = MainMenu()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
