import cv2
import numpy as np

# โหลดภาพ
img = cv2.imread("C:/Users/DELL/Desktop/webapp554/Detect_AnswerSheet/R1101.jpg")


# 1. แปลงเป็น grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. หามุมเอียง (deskew)
#    - แปลงเป็นขาวดำก่อน
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#    - หาตำแหน่ง pixel ที่เป็นตัวหนังสือ/เส้น
coords = np.column_stack(np.where(thresh > 0))

#    - ใช้ minAreaRect เพื่อหามุมเอียง
angle = cv2.minAreaRect(coords)[-1]

# ปรับมุมให้ถูกต้อง
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

print(f"Detected angle = {angle:.2f} degrees")

# 3. หมุนกลับให้ตรง
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
deskewed = cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# 4. บันทึกผลลัพธ์
cv2.imwrite("output_gray_deskew.jpg", deskewed)
