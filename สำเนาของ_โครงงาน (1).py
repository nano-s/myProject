import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="UV Analyzer", layout="centered")
st.title("🧼 วิเคราะห์สารเรืองแสงจากรูปมือจริง")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # กำหนดช่วงสีของสารเรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    # หารูปทรงมือจากสารเรืองแสง
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, contours, -1, 255, thickness=cv2.FILLED)

    # คำนวณเฉพาะพื้นที่รูปมือ
    fluorescent_area = cv2.countNonZero(cv2.bitwise_and(mask, hand_mask))
    hand_area = cv2.countNonZero(hand_mask)
    percentage = (fluorescent_area / hand_area) * 100 if hand_area > 0 else 0

    # แสดงภาพ
    st.image(pil_image, caption="ภาพที่อัปโหลด", use_column_width=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("ภาพต้นฉบับ")
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("สารเรืองแสง")
    ax[1].axis('off')
    ax[2].imshow(hand_mask, cmap='gray')
    ax[2].set_title("รูปมือที่ตรวจพบ")
    ax[2].axis('off')
    st.pyplot(fig)

    # แสดงผล
    st.markdown(f"🔍 พบสารเรืองแสงประมาณ **{percentage:.2f}%** ของพื้นที่รูปมือ")

    if percentage > 5:
        st.warning("🧼 ควรล้างมือให้สะอาดขึ้น โดยเฉพาะบริเวณที่ยังมีสารตกค้าง")
    else:
        st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

    # ดาวน์โหลดผล
    result_text = f"พบสารเรืองแสงประมาณ {percentage:.2f}% ของพื้นที่รูปมือ\n"
    if percentage > 5:
        result_text += "ควรล้างมือให้สะอาดขึ้น โดยเฉพาะบริเวณที่ยังมีสารตกค้าง"
    else:
        result_text += "มือสะอาดดี ไม่พบสารเรืองแสงเพิ่มเติม"

    st.download_button("📥 ดาวน์โหลดผลลัพธ์", result_text, file_name="uv_result.txt")

else:
    st.info("กรุณาอัปโหลดภาพเพื่อเริ่มวิเคราะห์")
