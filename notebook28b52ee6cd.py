# -*- coding: utf-8 -*-
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# حل مشكلة chromadb
try:
    from langchain.vectorstores import Chroma
except ImportError:
    st.warning("جارٍ تثبيت حزم إضافية... (قد يستغرق دقائق)")
    os.system("pip install chromadb sentence-transformers")
    from langchain.vectorstores import Chroma

# واجهة المستخدم
st.set_page_config(page_title="نظام مساعدة المسابقة", layout="wide")
st.title("🧠 نظام الإجابة عن أسئلة المسابقة")

# 1. تحميل الملفات
uploaded_file = st.file_uploader(
    "رفع وثيقة المسابقة (PDF, DOCX, TXT)", 
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=False
)

# 2. معالجة الملفات
def process_file(uploaded_file):
    try:
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file)
        elif uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_file)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file, encoding='utf-8')
        
        documents = loader.load()
        os.remove(temp_file)  # حذف الملف المؤقت
        
        return documents
    except Exception as e:
        st.error(f"خطأ في معالجة الملف: {str(e)}")
        return None

# 3. نظام الأسئلة والإجابات
if uploaded_file:
    documents = process_file(uploaded_file)
    
    if documents:
        st.success(f"تم تحميل {len(documents)} صفحة بنجاح!")
        
        # إنشاء فهرس للبحث
        try:
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(),
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            ).from_loaders([loader])
            
            retriever = index.vectorstore.as_retriever()
            
            # واجهة الأسئلة
            st.divider()
            question = st.text_input("اطرح سؤالاً عن وثيقة المسابقة:")
            
            if question:
                from langchain.chains import RetrievalQA
                qa = RetrievalQA.from_chain_type(
                    llm=llm,  # تأكد من تعريف llm الخاص بك
                    chain_type="stuff",
                    retriever=retriever
                )
                result = qa({"query": question})
                st.subheader("الإجابة:")
                st.write(result["result"])
                
        except Exception as e:
            st.error(f"خطأ في معالجة الأسئلة: {str(e)}")

# 4. إعدادات إضافية
with st.sidebar:
    st.header("الإعدادات")
    st.info("""
    **تعليمات الاستخدام:**
    1. ارفع وثيقة المسابقة
    2. اكتب سؤالك في مربع النص
    3. اضغط Enter للحصول على الإجابة
    """)
