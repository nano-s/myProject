import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="UV Hand Analyzer", layout="centered")
st.title("🖐️ วิเคราะห์สารเรืองแสงจากรูปมือภายใต้แสง UVA")

# อัปโหลดภาพมือจริง
uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # โหลดภาพมือจริง
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 🔹 กำหนดช่วงสีของสารเรืองแสง (ปรับได้)
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])
    uv_mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    # 🔹 โหลด “ภาพมือเส้น” ที่ล็อกไว้
    hand_outline = cv2.imread("hand_outline.jpg", cv2.IMREAD_GRAYSCALE)
    if hand_outline is None:
        st.error("❌ ไม่พบไฟล์ hand_outline.jpg ในโฟลเดอร์! กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับโค้ด")
    else:
        # สร้าง mask
        _, hand_mask = cv2.threshold(hand_outline, 127, 255, cv2.THRESH_BINARY)

        # ปรับขนาดให้ตรงกับภาพจริง
        hand_mask_resized = cv2.resize(hand_mask, (uv_mask.shape[1], uv_mask.shape[0]))

        # วิเคราะห์เฉพาะบริเวณรูปมือ
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
        ax[1].set_title("สารเรืองแสง")
        ax[1].axis('off')

        ax[2].imshow(hand_mask_resized, cmap='gray')
        ax[2].set_title("รูปมือที่ล็อกไว้")
        ax[2].axis('off')

        st.pyplot(fig)
        plt.close(fig)

        # แสดงผลวิเคราะห์
        st.markdown(f"🔍 พบสารเรืองแสงประมาณ **{percent:.2f}%** ของพื้นที่รูปมือ")

        if percent > 5:
            st.warning("🧼 ควรล้างมือให้สะอาดขึ้น โดยเฉพาะบริเวณที่ยังมีสารตกค้าง")
        else:
            st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

else:
    st.info("กรุณาอัปโหลดภาพมือภายใต้แสง UVA เพื่อเริ่มวิเคราะห์")
