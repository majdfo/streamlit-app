# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import torch

# إعدادات التطبيق
st.set_page_config(page_title="🧠 نظام NLP العربي المتقدم", layout="wide")
st.title("📖 محلل النصوص العربي الذكي (مستوى احترافي)")

# تحميل نماذج مسبقة التدريب
@st.cache_resource
def load_models():
    # نموذج تقسيم النصوص العربي
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    
    # نموذج فهم السياق العربي
    model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/MARBERT")
    
    # خط أنابيب معالجة اللغة العربية
    nlp_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # نموذج التضمين للبحث الدلالي
    embeddings = HuggingFaceEmbeddings(
        model_name="uer/sbert-base-arabic-light",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    return tokenizer, nlp_pipeline, embeddings

tokenizer, nlp_pipeline, embeddings = load_models()

# معالجة الملف النصي
def process_text_file(file_path):
    try:
        # تحميل الملف مع مراعاة الترميز العربي
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # تقسيم النص مع مراعاة الخصائص العربية
        arabic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=lambda x: len(tokenizer.encode(x)),
            separators=["\n\n", "\n", "۔", "۔ ", " ", ""]
        )
        
        return arabic_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"خطأ في معالجة الملف: {str(e)}")
        return None

# واجهة تحميل الملف
uploaded_file = st.file_uploader("رفع ملف نصي عربي (TXT)", type=['txt'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # معالجة الملف
    documents = process_text_file(tmp_path)
    os.unlink(tmp_path)
    
    if documents:
        st.success(f"تم تحميل {len(documents)} قسم نصي")
        
        # إنشاء فهرس بحث دلالي
        db = FAISS.from_documents(documents, embeddings)
        st.session_state.db = db
        
        # عرض تحليل أولي
        with st.expander("التحليل اللغوي الأولي"):
            sample_text = documents[0].page_content[:500]
            st.write("**العينة النصية:**", sample_text)
            
            # تحليل التوكنز
            tokens = tokenizer.tokenize(sample_text)
            st.write("**عدد التوكنز:**", len(tokens))
            st.write("**أول 20 توكن:**", tokens[:20])

# نظام الأسئلة المتقدم
if 'db' in st.session_state:
    st.markdown("---")
    st.subheader("نظام التحليل السياقي المتقدم")
    
    question = st.text_area("أدخل سؤالك التحليلي (بدقة عالية):", height=100)
    
    if question:
        try:
            # البحث الدلالي
            docs = st.session_state.db.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # توليد إجابة متقدمة
            response = nlp_pipeline(
                f"السؤال: {question}\nالنص: {context}",
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            st.markdown("---")
            st.subheader("النتيجة التحليلية:")
            st.write(response[0]['generated_text'])
            
            # عرض التحليل التفصيلي
            with st.expander("التفاصيل التقنية"):
                st.write("**السياق المستخدم:**", context)
                st.write("**إعدادات النموذج:**")
                st.json({
                    "model": "MARBERT",
                    "max_length": 512,
                    "num_beams": 5,
                    "early_stopping": True
                })
                
        except Exception as e:
            st.error(f"خطأ في التحليل: {str(e)}")

# قسم التحليل الإحصائي
if 'db' in st.session_state:
    st.markdown("---")
    st.subheader("الإحصائيات النصية")
    
    if st.button("تحليل إحصائي متقدم"):
        try:
            all_text = " ".join([doc.page_content for doc in documents])
            
            # تحليل التوكنز
            tokens = tokenizer.tokenize(all_text)
            vocab = set(tokens)
            
            # إحصائيات نصية
            stats = {
                "إجمالي الكلمات": len(all_text.split()),
                "إجمالي التوكنز": len(tokens),
                "حجم المفردات": len(vocab),
                "أكثر الكلمات تكرارا": dict(sorted(
                    {word: tokens.count(word) for word in vocab}.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:10])
            }
            
            st.write("**التحليل الإحصائي:**")
            st.json(stats)
            
        except Exception as e:
            st.error(f"خطأ في التحليل الإحصائي: {str(e)}")
