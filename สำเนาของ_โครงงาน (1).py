import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์สารเรืองแสงจากภาพมือจริง")

# อัปโหลดภาพมือจริงภายใต้แสง UVA
uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # โหลดภาพมือจริง
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # กำหนดช่วงสีของสารเรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    uv_mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    # สร้าง mask รูปมือด้วยโค้ด (ขนาด 512x512)
    hand_mask = np.zeros((512, 512), dtype=np.uint8)

    # วาดฝ่ามือ
    cv2.ellipse(hand_mask, (256, 350), (100, 120), 0, 0, 360, 255, thickness=cv2.FILLED)

    # วาดนิ้วทั้ง 5
    for i, x in enumerate([180, 220, 256, 292, 330]):
        cv2.rectangle(hand_mask, (x - 10, 150), (x + 10, 350), 255, thickness=cv2.FILLED)

    # ปรับขนาดให้ตรงกับภาพจริง
    hand_mask_resized = cv2.resize(hand_mask, (uv_mask.shape[1], uv_mask.shape[0]))

    # วิเคราะห์เฉพาะพื้นที่ที่เป็นรูปมือ
    hand_area = cv2.countNonZero(hand_mask_resized)
    glow_area = cv2.countNonZero(cv2.bitwise_and(uv_mask, hand_mask_resized))
    percent = (glow_area / hand_area) * 100 if hand_area > 0 else 0

    # แสดงภาพ
    st.image(pil_image, caption="ภาพที่อัปโหลด", use_column_width=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("ภาพต้นฉบับ")
    ax[0].axis('off')

    ax[1].imshow(uv_mask, cmap='gray')
    ax[1].set_title("บริเวณสารเรืองแสง")
    ax[1].axis('off')

    ax[2].imshow(hand_mask_resized, cmap='gray')
    ax[2].set_title("รูปมือที่ใช้วิเคราะห์")
    ax[2].axis('off')

    st.pyplot(fig)
    plt.close(fig)

    # แสดงผลการวิเคราะห์
    st.markdown(f"🔍 พบสารเรืองแสงประมาณ **{percent:.2f}%** ของพื้นที่รูปมือ")

    if percent > 5:
        st.warning("🧼 ควรล้างมือให้สะอาดขึ้น โดยเฉพาะบริเวณที่ยังมีสารตกค้าง")
    else:
        st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

else:
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA เพื่อเริ่มวิเคราะห์")
