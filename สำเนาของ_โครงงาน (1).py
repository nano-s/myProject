import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์สารเรืองแสงตามรูปมือเส้น")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # กำหนดช่วงสีของสารเรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    # สร้าง template mask รูปมือเส้น (ขนาด 512x768)
    hand_template = np.zeros((512, 768), dtype=np.uint8)
    cv2.rectangle(hand_template, (200, 100), (570, 450), 255, thickness=cv2.FILLED)  # ฝ่ามือ
    cv2.rectangle(hand_template, (250, 30), (280, 100), 255, thickness=cv2.FILLED)   # นิ้ว 1
    cv2.rectangle(hand_template, (310, 20), (340, 100), 255, thickness=cv2.FILLED)   # นิ้ว 2
    cv2.rectangle(hand_template, (370, 30), (400, 100), 255, thickness=cv2.FILLED)   # นิ้ว 3
    cv2.rectangle(hand_template, (430, 40), (460, 100), 255, thickness=cv2.FILLED)   # นิ้ว 4
    cv2.rectangle(hand_template, (490, 60), (520, 100), 255, thickness=cv2.FILLED)   # นิ้ว 5

    # ปรับขนาด template ให้ตรงกับภาพจริง
    hand_mask = cv2.resize(hand_template, (mask.shape[1], mask.shape[0]))

    # วิเคราะห์เฉพาะบริเวณที่ตรงกับรูปมือเส้น
    hand_area = cv2.countNonZero(hand_mask)
    fluorescent_area = cv2.countNonZero(cv2.bitwise_and(mask, hand_mask))
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
    ax[2].set_title("รูปมือที่ใช้วิเคราะห์")
    ax[2].axis('off')
    st.pyplot(fig)

    # แสดงผล
    st.markdown(f"🔍 พบสารเรืองแสงประมาณ **{percentage:.2f}%** ของพื้นที่รูปมือ")

    if percentage > 5:
        st.warning("🧼 ควรล้างมือให้สะอาดขึ้น โดยเฉพาะบริเวณที่ยังมีสารตกค้าง")
    else:
        st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

else:
    st.info("กรุณาวางมือให้ตรงกับตำแหน่งในภาพ แล้วอัปโหลดภาพเพื่อเริ่มวิเคราะห์")
