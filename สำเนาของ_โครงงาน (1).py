import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์เปอร์เซ็นต์ความสะอาดของมือ")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # โหลดภาพมือจริง
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # สร้าง mask มือจำลอง
    hand_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(hand_mask, (256, 420), (140, 160), 0, 0, 360, 255, thickness=cv2.FILLED)

    def draw_finger(x, y_base, length, width=40):
        segment = length // 3
        for i in range(3):
            y_top = y_base - segment * (i + 1)
            y_bottom = y_base - segment * i
            cv2.rectangle(hand_mask, (x - width//2, y_top), (x + width//2, y_bottom), 255, thickness=cv2.FILLED)
        cv2.ellipse(hand_mask, (x, y_base - length - 16), (width//2, 16), 0, 0, 360, 255, thickness=cv2.FILLED)

    fingers = [
        {"x": 110, "length": 240},
        {"x": 170, "length": 260},
        {"x": 256, "length": 280},
        {"x": 342, "length": 260},
    ]
    for f in fingers:
        draw_finger(f["x"], 420, f["length"])

    thumb_pts = np.array([[380, 440], [490, 340], [510, 360], [400, 470]], np.int32)
    cv2.fillPoly(hand_mask, [thumb_pts], 255)

    # ปรับขนาด mask ให้ตรงกับภาพจริง
    hand_mask_resized = cv2.resize(hand_mask, (image.shape[1], image.shape[0]))

    # สร้าง mask เรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    uv_mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)
    glow_mask = cv2.bitwise_and(uv_mask, hand_mask_resized)

    # วิเคราะห์ภาพรวม
    hand_area = cv2.countNonZero(hand_mask_resized)
    glow_area = cv2.countNonZero(glow_mask)
    clean_percent = 100 - ((glow_area / hand_area) * 100 if hand_area > 0 else 0)

    # แสดงภาพไฮไลต์จุดเรืองแสง
    highlight = cv2.bitwise_and(image, image, mask=glow_mask)
    highlight[np.where(glow_mask == 0)] = [0, 0, 0]

    st.image(highlight, caption="📍 จุดที่พบสารเรืองแสงบนมือ", use_column_width=True)
    st.markdown(f"🧼 **ความสะอาดของมือโดยรวม: {clean_percent:.2f}%**")

    if clean_percent >= 95:
        st.success("✅ มือสะอาดดีมาก ไม่จำเป็นต้องล้างเพิ่ม")
    elif clean_percent >= 80:
        st.warning("🟠 มือค่อนข้างสะอาด แต่ควรล้างบางจุดเพิ่มเติม")
    else:
        st.error("🔴 มือยังมีสารตกค้างหลายจุด ควรล้างให้สะอาดขึ้น")

else:
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA เพื่อเริ่มวิเคราะห์")
