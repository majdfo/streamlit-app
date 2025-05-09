# -*- coding: utf-8 -*-
import streamlit as st
import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

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
    "رفع وثيقة المسابقة (PDF, DOCX, TXT, XLSX)", 
    type=['pdf', 'docx', 'txt', 'xlsx'],
    accept_multiple_files=False
)

# 2. معالجة الملفات (بما في ذلك Excel)
def process_file(uploaded_file):
    try:
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file)
            documents = loader.load()
        elif uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_file)
            documents = loader.load()
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file, encoding='utf-8')
            documents = loader.load()
        elif uploaded_file.name.endswith('.xlsx'):
            # معالجة ملفات Excel
            df = pd.read_excel(temp_file)
            documents = []
            for index, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(Document(page_content=content, metadata={"source": "excel"}))
            loader = None  # لا يوجد loader لملفات Excel
        else:
            st.error("نوع الملف غير مدعوم")
            return None, None
        
        os.remove(temp_file)  # حذف الملف المؤقت
        return documents, loader
    
    except Exception as e:
        st.error(f"خطأ في معالجة الملف: {str(e)}")
        return None, None

# 3. نظام الأسئلة والإجابات
if uploaded_file:
    documents, loader = process_file(uploaded_file)
    
    if documents:  # التأكد من وجود المستندات
        st.success(f"تم تحميل البيانات بنجاح! (عدد السجلات: {len(documents)})")
        
        # إنشاء فهرس للبحث
        try:
            if loader:  # إذا كان هناك loader (للملفات غير Excel)
                index = VectorstoreIndexCreator(
                    embedding=HuggingFaceEmbeddings(),
                    text_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                ).from_loaders([loader])
                retriever = index.vectorstore.as_retriever()
            else:  # لملفات Excel
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                embeddings = HuggingFaceEmbeddings()
                vectorstore = Chroma.from_documents(texts, embeddings)
                retriever = vectorstore.as_retriever()
            
            # واجهة الأسئلة
            st.divider()
            question = st.text_input("اطرح سؤالاً عن بيانات المسابقة:")
            
            if question:
                from langchain.chains import RetrievalQA
                from langchain.llms import OpenAI
                
                # استبدل هذا بالنموذج الذي تريد استخدامه (WatsonX أو غيره)
                llm = OpenAI(temperature=0)
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
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
    1. ارفع ملف المسابقة (PDF, Word, Excel, Text)
    2. اكتب سؤالك في مربع النص
    3. اضغط Enter للحصول على الإجابة
    """)
    st.warning("ملاحظة: لملفات Excel، يتم معالجة جميع الصفوف كبيانات نصية")
