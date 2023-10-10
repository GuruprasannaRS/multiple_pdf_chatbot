import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template


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
    # print('Chunks',chunks)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    # print('Embeddings', embeddings)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    # print('Vector-store', vectorstore)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(),memory = memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)





def main():
    load_dotenv()
    st.set_page_config(page_title = "Multiple PDF Chatbot", page_icon = ":books:")

    st.write(css, unsafe_allow_html = True)

    st.header("Chat with the multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:") 
    if user_question:
        handle_userinput(user_question)
    

    
    

    with st.sidebar:
        if  "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on proceess", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing Please Wait"):
                # get the pdf
                raw_text = get_pdf_text(pdf_docs)
                print("raw text is collected from the documents loaded")
                # st.write(raw_text)

                # get the text chunks
                text_chunks =  get_text_chunk(raw_text)
                print('The raw texts are converted into chunks')
                # st.write(text_chunks)


                # create vector store for the embeddings
                vector_store = get_vectorstore(text_chunks)
                print("The chunks are stored in the vector store")
                

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__ == '__main__':
    main()
