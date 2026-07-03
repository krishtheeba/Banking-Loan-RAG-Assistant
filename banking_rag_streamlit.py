# banking_rag_streamlit.py
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title='Banking RAG Demo',layout='wide')
st.title('🏦 Banking Loan RAG Assistant')

FILE='loan_policy.txt'
sample='''Home Loan Policy 2026

Applicants must be at least 21 years old.
Minimum annual income: RM 50,000
Maximum loan tenure: 35 years
Maximum Loan Amount: RM 2,000,000

Required Documents
Passport
Salary Slips (6 months)
Bank Statement
EPF Statement
Income Tax Return

Interest Rate
Floating Rate: Base Rate + 1.25%
Processing Time: 5-7 Working Days
'''
if not Path(FILE).exists():
    Path(FILE).write_text(sample,encoding='utf-8')

@st.cache_resource
def init():
    docs=TextLoader(FILE,encoding='utf-8').load()
    chunks=RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50).split_documents(docs)
    emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.from_documents(chunks,emb)
    retriever=db.as_retriever(search_kwargs={'k':3})
    llm=ChatOllama(model='gemma2:2b',temperature=0)
    prompt=ChatPromptTemplate.from_template('''You are a Banking Loan Assistant.
Use only the context.

Context:
{context}

Question:
{question}

Answer:''')
    return retriever,llm,prompt

retriever,llm,prompt=init()
if 'messages' not in st.session_state:
    st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])
q=st.chat_input('Ask a question...')
if q:
    st.session_state.messages.append({'role':'user','content':q})
    with st.chat_message('user'):
        st.markdown(q)
    docs=retriever.invoke(q)
    context='\n\n'.join(d.page_content for d in docs)
    msgs=prompt.invoke({'context':context,'question':q})
    ans=llm.invoke(msgs).content
    with st.chat_message('assistant'):
        st.markdown(ans)
        with st.expander('Retrieved Context'):
            st.code(context)
    st.session_state.messages.append({'role':'assistant','content':ans})
with st.sidebar:
    st.header('Run')
    st.code('pip install streamlit langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu sentence-transformers\nstreamlit run banking_rag_streamlit.py')
