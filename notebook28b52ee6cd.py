# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tempfile
import os

# إعداد واجهة المستخدم
st.set_page_config(page_title="🦜 نظام التحليل الذكي للمسابقات", layout="wide")
st.title("🧠 نظام فهم وتحليل البيانات الذكي")

# تحميل الملف
uploaded_file = st.file_uploader("رفع ملف البيانات (Excel أو CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    # معالجة الملف
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(tmp_file_path)
        else:
            df = pd.read_csv(tmp_file_path)
        
        os.unlink(tmp_file_path)  # حذف الملف المؤقت
        
        # تحويل البيانات إلى نص مفهوم
        data_description = "\n".join([
            f"العمود '{col}' يحتوي على: {df[col].dropna().unique()[:5]}... (إجمالي {len(df[col].unique())} قيمة فريدة)" 
            for col in df.columns
        ])
        
        sample_data = "\n".join([
            "أمثلة على الصفوف:",
            *[f"{i}: {row.to_dict()}" for i, row in df.head(3).iterrows()]
        ])
        
        full_context = f"""
        هيكلة البيانات:
        {data_description}
        
        {sample_data}
        
        ملاحظات:
        - الأرقام تم تقريبها للتبسيط
        - التاريخ قد يكون بتنسيق مختلف
        """
        
        # عرض عينة من البيانات
        with st.expander("عرض عينة من البيانات"):
            st.dataframe(df.head(3))
            st.write("وصف البيانات:", full_context)
        
        # نظام الأسئلة الذكية
        st.divider()
        question = st.text_input("اطرح أي سؤال عن البيانات:")
        
        if question:
            # نموذج الذكاء الاصطناعي
            llm = OpenAI(temperature=0.7, max_tokens=500)
            
            # تصميم Prompt ذكي
            prompt_template = PromptTemplate(
                input_variables=["question", "data_context"],
                template="""
                أنت خبير في تحليل البيانات. لديك البيانات التالية:
                {data_context}
                
                السؤال: {question}
                
                المطلوب:
                1. فهم طبيعة السؤال
                2. تحليل البيانات المتاحة
                3. الإجابة بطريقة واضحة ومفهومة
                4. إذا كان السؤال يحتاج حساباً، أجر العملية الحسابية
                5. إذا كان السؤال استنتاجياً، قدم الاستنتاج المنطقي
                
                الإجابة:
                """
            )
            
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = chain.run(question=question, data_context=full_context)
            
            st.subheader("التحليل الذكي:")
            st.write(response)
            
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# إضافة تعليمات
with st.sidebar:
    st.header("تعليمات الاستخدام")
    st.markdown("""
    **كيف تستخدم النظام:**
    1. ارفع ملف Excel أو CSV
    2. اكتب أي سؤال عن البيانات
    3. النظام سيفهم السؤال ويحلله
    
    **أمثلة لأسئلة ذكية:**
    - "ما العلاقة بين العمود X والعمود Y؟"
    - "ما هي الفئة الأكثر ظهوراً في العمود Z؟"
    - "إذا كان لدينا شرط كذا، ما هي النتيجة المتوقعة؟"
    - "ما هو المتوسط الحسابي للأعمار؟"
    - "أعطني تحليلاً عن توزيع المنتجات"
    """)
