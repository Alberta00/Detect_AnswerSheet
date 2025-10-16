from PIL import Image
import cv2
import numpy as np

path = r"C:\Users\USER\Desktop\a\Detect_AnswerSheet\R1102.jpg"

# ทดสอบว่า Pillow อ่านได้ไหม
try:
    pil_img = Image.open(path)
    pil_img.show()  # จะเปิด preview ถ้าเปิดได้ = ไฟล์โอเค
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    print("✅ โหลดด้วย PIL สำเร็จ:", img.shape)
except Exception as e:
    print("❌ อ่านด้วย PIL ไม่ได้:", e)
