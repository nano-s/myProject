import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์จุดเรืองแสงและคำแนะนำการล้างมือ")

# อัปโหลดภาพมือจริงและภาพเส้นมือ
uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])
outline_file = st.file_uploader("✏️ อัปโหลดภาพเส้นมือ (มือ = ขาว, พื้นหลัง = ดำ)", type=["jpg", "png", "jpeg"])

if uploaded_file and outline_file:
    # โหลดภาพมือจริง
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # โหลดภาพเส้นมือ
    outline_pil = Image.open(outline_file).convert('L')
    hand_outline = np.array(outline_pil)
    _, hand_mask = cv2.threshold(hand_outline, 127, 255, cv2.THRESH_BINARY)
    hand_mask_resized = cv2.resize(hand_mask, (image.shape[1], image.shape[0]))

    # สร้าง mask เรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    uv_mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)
    glow_mask = cv2.bitwise_and(uv_mask, hand_mask_resized)

    # สร้างโซนต่าง ๆ บนมือ (จำลองตำแหน่ง)
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

    # แสดงภาพมือจริง
    st.image(pil_image, caption="ภาพมือจริงที่อัปโหลด", use_column_width=True)

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
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA และภาพเส้นมือเพื่อเริ่มวิเคราะห์")
