# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import tempfile
from langchain.llms import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# إعدادات التطبيق
st.set_page_config(page_title="🧠 المحلل الشامل للبيانات العربية", layout="wide")
st.title("📊 نظام فهم وتحليل البيانات الذكي")

# إعداد مفتاح API
with st.sidebar:
    st.subheader("الإعدادات الأساسية")
    api_key = st.text_input("أدخل مفتاح OpenAI API", type="password", key="api_key")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        st.success("تم الإعداد بنجاح!")
    
    st.markdown("---")
    st.subheader("تحميل البيانات")
    uploaded_file = st.file_uploader(
        "رفع الملف (Excel, PDF, Word, CSV, TXT)",
        type=['xlsx', 'xls', 'csv', 'pdf', 'docx', 'txt']
    )

# معالجة جميع أنواع الملفات
def process_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        if file.name.endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(tmp_path)
        elif file.name.endswith('.csv'):
            loader = CSVLoader(tmp_path, encoding='utf-8')
        elif file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(tmp_path, encoding='utf-8')
        else:
            st.error("نوع الملف غير مدعوم")
            return None
        
        data = loader.load()
        return data
    
    except Exception as e:
        st.error(f"خطأ في معالجة الملف: {str(e)}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# تحويل البيانات إلى وثائق
def create_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۔", " ", ""]
    )
    return text_splitter.split_documents(data)

# واجهة التحليل
if uploaded_file and 'api_key' in st.session_state:
    data = process_file(uploaded_file)
    
    if data:
        st.success(f"تم تحميل {len(data)} مستند/صف بنجاح!")
        
        # إنشاء فهرس البحث
        documents = create_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="uer/sbert-base-arabic-light")
        db = FAISS.from_documents(documents, embeddings)
        st.session_state.db = db
        
        # عرض عينة من البيانات
        with st.expander("عرض عينة من البيانات"):
            if uploaded_file.name.endswith(('.xlsx', '.xls', '.csv')):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.dataframe(df.head(3))
                except:
                    st.write(documents[:1])
            else:
                st.write(documents[:1])

# نظام الأسئلة الذكية
if 'db' in st.session_state and 'api_key' in st.session_state:
    st.markdown("---")
    st.subheader("اطرح سؤالك")
    
    question = st.text_area("اكتب سؤالك هنا (يمكنك استخدام العامية)", height=100)
    
    if question:
        try:
            # البحث في البيانات
            docs = st.session_state.db.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # تحديد نوع السؤال
            is_general_question = any(word in question for word in [
                "شو", "ليش", "كيف", "لما", "علاج", "سبب", "رأيك"
            ])
            
            # تصميم Prompt ذكي
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                البيانات المتاحة:
                {context}
                
                السؤال: {question}
                
                المطلوب:
                1. فهم السؤال بدقة (حتى لو كان عامياً)
                2. التحقق من وجود إجابة في البيانات
                3. إذا لم توجد إجابة محددة، قدم إجابة عامة
                4. استخدم لغة عربية واضحة
                
                الإجابة:
                """
            )
            
            llm = OpenAI(temperature=0.6, max_tokens=800)
            qa_chain = LLMChain(llm=llm, prompt=prompt_template)
            response = qa_chain.run(context=context, question=question)
            
            st.markdown("---")
            st.subheader("النتيجة:")
            st.write(response)
            
            with st.expander("عرض البيانات المستخدمة في الإجابة"):
                st.write(docs)
                
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")

# إضافة أمثلة توضيحية
with st.expander("أمثلة على أسئلة يمكن طرحها"):
    st.markdown("""
    **لبيانات Excel/CSV:**
    - "شو أعلى راتب في الشركة؟"
    - "كم عدد الموظفين في قسم المبيعات؟"
    - "لما المبيعات نزلت في يناير؟"
    
    **للمستندات النصية:**
    - "شو أسباب المشاكل الإدارية؟"
    - "إيش الحلول المقترحة للبطالة؟"
    - "عندي مشكلة مع المدير، شو الحل؟"
    
    **أسئلة عامة:**
    - "شو أفضل طريقة لتربية الأطفال؟"
    - "كيف أتعامل مع ضغوط العمل؟"
    """)
