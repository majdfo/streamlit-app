import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل النموذج المدرب
@st.cache_resource
def load_model():
    model = YOLO('path_to_your_model.pt')  # ضع هنا المسار الصحيح للنموذج المدرب
    return model

model = load_model()

# واجهة المستخدم
st.title("Driver Distraction Detection")

st.write("""
This web application detects driver distractions like phone use, seatbelt violations, and smoking.
Upload an image or video, and the model will process and show predictions.
""")

# تحميل الصورة أو الفيديو
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "png", "mp4", "mov"])

if uploaded_file is not None:
    # إذا كانت الصورة
    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        # تحويل الصورة إلى تنسيق مناسب وتوقع النتائج
        results = model.predict(uploaded_file, conf=0.25, iou=0.7)

        # عرض النتائج
        st.write(f"Predictions: {results.names}")
        st.write(f"Confidence: {results.xywh[0, 4]}")  # عرض الثقة

        # عرض الصورة مع الإطارات المحددة
        st.image(results.imgs[0], caption="Predicted Image with Bounding Boxes", use_column_width=True)
    
    # إذا كان الفيديو
    elif uploaded_file.type.startswith("video"):
        # هنا يمكن إضافة تحليل الفيديو، لكن يحتاج لتعديل معالجة الفيديو.
        st.video(uploaded_file)
