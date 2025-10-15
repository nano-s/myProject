import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="UV Analyzer", layout="centered")
st.title("🧼 วิเคราะห์สารเรืองแสงจากภาพมือ")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือภายใต้แสง UVA", type=["jpg", "png", "jpeg"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image = np.array(pil_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # ✅ กำหนดขอบเขต HSV ของสารเรืองแสง
    lower_fluorescent = np.array([105, 80, 160])
    upper_fluorescent = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_fluorescent, upper_fluorescent)

    st.write(f"hsv type: {type(hsv)}, shape: {hsv.shape}")
    st.write(f"lower: {lower_fluorescent}, upper: {upper_fluorescent}")

    # ✅ คำนวณเปอร์เซ็นต์พื้นที่ fluoresence ทั้งหมด
    fluorescent_area = cv2.countNonZero(mask)
    total_area = image.shape[0] * image.shape[1]
    percentage = (fluorescent_area / total_area) * 100

    # ✅ วิเคราะห์ตามโซน
    height, width = mask.shape
    zones = {
        "ฝ่ามือ": mask[int(height*0.35):int(height*0.65), int(width*0.25):int(width*0.75)],
        "นิ้วมือ": mask[0:int(height*0.35), int(width*0.2):int(width*0.8)],
        "หลังมือ": mask[int(height*0.35):int(height*0.65), 0:int(width*0.25)],
        "ข้อมือ": mask[int(height*0.65):, int(width*0.25):int(width*0.75)],
        "นิ้วหัวแม่มือ": mask[int(height*0.2):int(height*0.5), int(width*0.8):],
        "หลังนิ้วหัวแม่มือ": mask[int(height*0.2):int(height*0.5), 0:int(width*0.2)],
        "ระหว่างนิ้ว": mask[int(height*0.2):int(height*0.35), int(width*0.3):int(width*0.7)]
    }

    recommendations = []
    zone_results = {}

    for zone_name, zone_mask in zones.items():
        zone_area = zone_mask.shape[0] * zone_mask.shape[1]
        zone_fluorescent = cv2.countNonZero(zone_mask)
        zone_percent = (zone_fluorescent / zone_area) * 100 if zone_area > 0 else 0
        zone_results[zone_name] = zone_percent
        if zone_percent > 5:
            recommendations.append(f"- ควรล้างบริเวณ **{zone_name}** ให้สะอาดขึ้น ({zone_percent:.1f}%)")

    # ✅ แสดงภาพ
    st.image(pil_image, caption="ภาพที่อัปโหลด", use_column_width=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("ภาพต้นฉบับ")
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("บริเวณที่มีสารเรืองแสง")
    ax[1].axis('off')
    st.pyplot(fig)

    # ✅ แสดงผลวิเคราะห์
    st.markdown(f"🔍 พบสารเรืองแสงประมาณ **{percentage:.2f}%** ของพื้นที่ภาพ")

    if recommendations:
        st.markdown("🧼 **คำแนะนำในการล้างมือเพิ่มเติม:**")
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("✅ มือสะอาดดี ไม่พบสารเรืองแสงในระดับที่ต้องล้างเพิ่ม")

    # ✅ ปุ่มดาวน์โหลดผลลัพธ์
    result_text = f"พบสารเรืองแสงประมาณ {percentage:.2f}%\n"
    if recommendations:
        result_text += "\nคำแนะนำ:\n" + "\n".join(recommendations)
    else:
        result_text += "\nมือสะอาดดี ไม่พบสารเรืองแสงเพิ่มเติม"

    st.download_button("📥 ดาวน์โหลดผลลัพธ์", result_text, file_name="uv_result.txt")
