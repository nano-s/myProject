import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์จุดเรืองแสงจากภาพมือจริง")

# อัปโหลดภาพมือจริงและภาพเส้นมือ
uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])
outline_file = st.file_uploader("✏️ อัปโหลดภาพเส้นมือ (มือ = ขาว, พื้นหลัง = ดำ)", type=["jpg", "png", "jpeg"])

if uploaded_file and outline_file:
    # โหลดภาพมือจริง
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # กำหนดช่วงสีของสารเรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    uv_mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    # โหลดภาพเส้นมือ
    outline_pil = Image.open(outline_file).convert('L')
    hand_outline = np.array(outline_pil)
    _, hand_mask = cv2.threshold(hand_outline, 127, 255, cv2.THRESH_BINARY)

    # ปรับขนาดให้ตรงกับภาพจริง
    hand_mask_resized = cv2.resize(hand_mask, (uv_mask.shape[1], uv_mask.shape[0]))

    # วิเคราะห์
    hand_area = cv2.countNonZero(hand_mask_resized)
    glow_mask = cv2.bitwise_and(uv_mask, hand_mask_resized)
    glow_area = cv2.countNonZero(glow_mask)
    percent = (glow_area / hand_area) * 100 if hand_area > 0 else 0

    # สร้างภาพไฮไลต์จุดเรืองแสงบนมือจริง
    highlight = cv2.bitwise_and(image, image, mask=glow_mask)
    highlight[np.where(glow_mask == 0)] = [0, 0, 0]  # พื้นหลังดำ

    # แสดงเฉพาะภาพมือจริงที่มีจุดเรืองแสง
    st.image(highlight, caption="📍 บริเวณที่ควรล้างมือเพิ่มเติม", use_column_width=True)
    st.markdown(f"🔍 พบจุดเรืองแสงประมาณ **{percent:.2f}%** ของพื้นที่รูปมือ")

    if percent > 5:
        st.warning("🧼 ควรล้างมือเพิ่มเติม โดยเฉพาะบริเวณที่เห็นจุดเรืองแสงในภาพด้านบน")
    else:
        st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

elif uploaded_file and not outline_file:
    st.info("กรุณาอัปโหลดภาพเส้นมือ (มือ = ขาว, พื้นหลัง = ดำ) เพื่อใช้เป็น mask ในการวิเคราะห์")

else:
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA และภาพเส้นมือเพื่อเริ่มวิเคราะห์")
