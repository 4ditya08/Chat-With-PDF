import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
OPENAI_API_KEY = "sk-proj-i9DWHH9cdSOq5tXsv0KdT3BlbkFJ9zhUAVevUhMd2SESjC1c"
# Load environment variables
load_dotenv()

# Sidebar contents
with st.sidebar:
    st.title('💬 Chat With PDF')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot.
    ''')
    add_vertical_space(2)
    st.markdown('''
    Made By:
    - [Aditya Bhusari](https://www.linkedin.com/in/aditya-bhusari-60267821b/)
    - [Aditya Kulkarni](https://www.linkedin.com/in/adityakulkarni03/)
    - [Anish Karkera](https://www.linkedin.com/in/anish-karkera-73b309221/)
    ''')

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings loaded from disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings generated and saved to disk')

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write(response)
                print(cb)

if __name__ == '__main__':
    main()
