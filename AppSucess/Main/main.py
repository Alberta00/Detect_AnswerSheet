import sys
import subprocess
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 OMR System — Main Menu")
        self.resize(500, 400)
        self.setStyleSheet("""
            QWidget { background-color: #f0f3f8; font-family: 'Segoe UI'; }
            QPushButton {
                background-color: #4a90e2; color: white;
                border-radius: 12px; padding: 12px; font-size: 16px;
            }
            QPushButton:hover { background-color: #357ABD; }
            QLabel { font-size: 18px; font-weight: bold; color: #333; }
        """)

        layout = QVBoxLayout()
        label = QLabel("📋 ระบบตรวจข้อสอบ OMR")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(label)

        layout.addSpacing(30)

        btn1 = QPushButton("🧩 สร้างกริด (OMR Designer)")
        btn1.clicked.connect(self.open_designer)
        layout.addWidget(btn1)

        btn2 = QPushButton("✅ ตรวจข้อสอบ (OMR Checker)")
        btn2.clicked.connect(self.open_checker)
        layout.addWidget(btn2)

        layout.addSpacing(20)
        footer = QLabel("KU Sriracha")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(footer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ---------- เปิดโปรแกรม Designer ----------
    def open_designer(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "Grid.py")
            subprocess.Popen([sys.executable, file_path])
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ไม่สามารถเปิด Grid.py ได้\n{e}")

    # ---------- เปิดโปรแกรม Checker ----------
    def open_checker(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "Detect.py")
            subprocess.Popen([sys.executable, file_path])
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ไม่สามารถเปิด Detect.py ได้\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainMenu()
    w.show()
    sys.exit(app.exec())
