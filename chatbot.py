
##############################################################################################################
#This file is the same that main.py, it only this file have as database the chromadb engine instead of  FAISS
##############################################################################################################
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from chromadb import ChromaDB
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

load_dotenv()
#Cargar el api_key de google gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def obtener_pdf_texto(pdfs):
    text=""
    for pdf in pdfs:
        pdf_file = PdfReader(pdf)
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

def obtener_chunks(texto):
    splitter = RecursiveCharacterTextSplitter(chunks_size=10000, overlap_size=1000)
    chunks = splitter.split(texto)
    return chunks

def obtener_vectordb(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = ChromaDB.from_texts(chunks, embeddings=embeddings)
    vector_db.save_local("vector_index")

def obtener_cadena_conversacion():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model=GoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model=model, prompt=prompt, chain_type="stuff")
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = ChromaDB.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=obtener_cadena_conversacion()

    response=chain(
        {
            "input_documents":docs, "question":user_question
        }, return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chatbot de preguntas y respuestas con tus PDFs", page_icon=":robot:")
    st.title("Chatbot")
    st.header("Chatbot usando el modelo Gemini-pro de Google y tus PDFs")
    user_question = st.text_input("Ingresa tu pregunta: ")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.write("Menu")
        pdfs = st.file_uploader("Cargar tus archivos pds", type="pdf", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Cargando PDFs..."):
                text = obtener_pdf_texto(pdfs)
                chunks = obtener_chunks(text)
                obtener_vectordb(chunks)
                st.success("Carga de PDFs exitosa")
                
if __name__ == "__main__":
    main()
