import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    '''This function accepts the multiple PDF and extracts the texts of all the PDFs and stored in one variable called text'''
    text = ""
    for pdf in pdf_docs:
        pdfs = PdfReader(pdf)
        for pages in pdfs.pages:
            text += pages.extract_text()
    return text

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
                st.write(raw_text)

                # get the text chunks

                # create vector store for the embeddings




if __name__ == '__main__':
    main()
