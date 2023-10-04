import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    '''This function accepts the multiple PDF and extracts the texts of all the PDFs and stored in one variable called text'''
    text = ""
    for pdf in pdf_docs:
        pdfs = PdfReader(pdf)
        for pages in pdfs.pages:
            text += pages.extract_text()
    return text

def get_text_chunk(raw_text):
    '''This function accepts the texts extracted from the previous function called get_pdf_text and this will convert them into chunks
    Chunk size = This is the size that defines how much words to be in a single chunk. In our case it's going to be 1000. 
    In each chunks, we will be having 1000 words in a single chunk
    Chunk Overlap =  Let's take a case. We have a two paragraphs. In the first paragraph we have 600 words and in second paragraph we have 500 words. 
    The chunk size of 1000 will take first paragraph and second paragraph's 400 words. The rest 100 words will be left out in the 
    second paragraph. The next chunk should not start with the last 100 words of the second paragraph. Now we have chunk overlap as
    200 so it will take previous 200 words in the second paragraph.Apparently It will start from last 300th word of the second paragraph'''
    text_splitter = CharacterTextSplitter(separator = '\n', chunk_size = 1000, chunk_overlap = 200, length_function = len )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title = "Multiple PDF Chatbot", page_icon = ":books:")
    st.header("Chat with the multiple PDFs :books:")
    st.text_input("Ask a question about your documents:") 

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on proceess", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing Please Wait"):
                # get the pdf
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks =  get_text_chunk(raw_text)
                # st.write(text_chunks)


                # create vector store for the embeddings
                vector_store = get_vectorstore(text_chunks)
                




if __name__ == '__main__':
    main()
