# -*- coding: utf-8 -*-
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# إعدادات واجهة المستخدم
st.set_page_config(page_title="مسابقة الذكاء الاصطناعي", layout="wide")
st.title("🧠 نظام الإجابة الذكية للمسابقة")

# 1. حل مشكلة بيانات الاعتماد - استخدام قيم افتراضية (يمكن تغييرها لاحقاً)
creds = {
    'apikey': 'your_api_key_here' if 'apikey' not in st.session_state else st.session_state.apikey,
    'url': 'https://us-south.ml.cloud.ibm.com'
}

project_id = 'your_project_id' if 'project_id' not in st.session_state else st.session_state.project_id

# 2. واجهة لإدخال البيانات السرية (بدون حفظ في الكود)
with st.sidebar:
    st.subheader("إعدادات الوصول لـ WatsonX")
    new_api = st.text_input("أدخل مفتاح API", type="password")
    new_project = st.text_input("أدخل معرف المشروع")
    
    if new_api:
        st.session_state.apikey = new_api
        creds['apikey'] = new_api
    if new_project:
        st.session_state.project_id = new_project
        project_id = new_project

# 3. تحميل ملف PDF من واجهة المستخدم
uploaded_file = st.file_uploader("رفع ملف PDF للمسابقة", type="pdf")
pdf_ready = False

if uploaded_file:
    # حفظ الملف مؤقتاً
    with open("competition_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_ready = True
    st.success("تم تحميل الملف بنجاح!")

# 4. نظام الدردشة الأساسي
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("اطرح سؤالاً عن المسابقة..."):
    try:
        # تهيئة النموذج اللغوي
        from watsonxlangchain import LangChainInterface
        llm = LangChainInterface(
            credentials=creds,
            model='meta-llama/llama-2-70b-chat',
            params={
                'decoding_method': 'sample',
                'max_new_tokens': 200,
                'temperature': 0.7  # زيادة الإبداع للإجابات
            },
            project_id=project_id
        )
        
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = llm(prompt)
        
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"خطأ في الاتصال: {str(e)}")

# 5. نظام الأسئلة عن ملف PDF
if pdf_ready:
    st.divider()
    st.subheader("أسئلة عن وثيقة المسابقة")
    
    try:
        @st.cache_resource
        def prepare_document():
            loader = PyPDFLoader("competition_file.pdf")
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(),
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )
            ).from_loaders([loader])
            return index.vectorstore.as_retriever()
        
        doc_question = st.text_input("اطرح سؤالاً عن محتوى الوثيقة")
        if doc_question:
            retriever = prepare_document()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            result = qa_chain({"query": doc_question})
            st.info("الإجابة من الوثيقة:")
            st.write(result["result"])
    except Exception as e:
        st.warning(f"حدث خطأ في معالجة الوثيقة: {str(e)}")
