import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์จุดเรืองแสงและคำแนะนำการล้างมือ")

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

    # สร้างโซนต่าง ๆ บนมือ (ตำแหน่งจำลอง)
    zones = {
        "นิ้วมือ": cv2.rectangle(np.zeros_like(hand_mask_resized), (100, 0), (400, 200), 255, -1),
        "เล็บ": cv2.rectangle(np.zeros_like(hand_mask_resized), (100, 0), (400, 50), 255, -1),
        "ฝ่ามือ": cv2.rectangle(np.zeros_like(hand_mask_resized), (150, 250), (350, 400), 255, -1),
        "ข้อมือ": cv2.rectangle(np.zeros_like(hand_mask_resized), (150, 400), (350, 512), 255, -1),
        "ซอกนิ้ว": cv2.rectangle(np.zeros_like(hand_mask_resized), (180, 100), (320, 200), 255, -1),
        "หลังมือ": cv2.rectangle(np.zeros_like(hand_mask_resized), (100, 200), (400, 250), 255, -1),
    }

    # วิเคราะห์แต่ละโซน
    results = {}
    for name, zone_mask in zones.items():
        zone_mask = cv2.resize(zone_mask, (image.shape[1], image.shape[0]))
        zone_area = cv2.countNonZero(zone_mask)
        zone_glow = cv2.countNonZero(cv2.bitwise_and(glow_mask, zone_mask))
        percent = (zone_glow / zone_area) * 100 if zone_area > 0 else 0
        results[name] = percent

    # สร้างภาพไฮไลต์จุดเรืองแสง
    highlight = cv2.bitwise_and(image, image, mask=glow_mask)
    highlight[np.where(glow_mask == 0)] = [0, 0, 0]

    st.image(highlight, caption="📍 บริเวณที่ควรล้างมือเพิ่มเติม", use_column_width=True)

    # แสดงผลการวิเคราะห์
    st.markdown("### 📊 ผลการวิเคราะห์แต่ละบริเวณ")
    for name, percent in results.items():
        st.markdown(f"- **{name}**: {percent:.1f}%")

    # สรุปคำแนะนำ
    st.markdown("### 🧼 คำแนะนำในการล้างมือเพิ่มเติม")
    for name, percent in results.items():
        if percent > 10:
            st.markdown(f"🔴 ควรล้างบริเวณ **{name}** ให้สะอาดขึ้น (พบสารเรืองแสง {percent:.1f}%)")
        elif percent > 5:
            st.markdown(f"🟠 ล้างบริเวณ **{name}** เพิ่มเติมเล็กน้อย ({percent:.1f}%)")
        else:
            st.markdown(f"🟢 บริเวณ **{name}** สะอาดดี ({percent:.1f}%)")

else:
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA เพื่อเริ่มวิเคราะห์")
