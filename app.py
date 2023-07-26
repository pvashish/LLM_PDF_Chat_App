#creation of app
import os
import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback




# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with 🍍 by [PineappleTales](https://pineappletales.medium.com/)')
    st.write('⚠️⚠️ The app is under development, you might encounter some error while using it ⚠️⚠️')


def main():
    st.header("Chat with PDF 💬")

    load_dotenv()

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    #st.write(pdf.name)

    # st.write(pdf)
    #the below block of code say if the pdf file is uploaded then only execute the below block of code
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() #reads each and every page and append it together
# RIGHT HERE: DATA LOADING AND EXTRACTION PART ENDS

        #the recursive charecter text splitter will:
        ##divide a token into 1000 chuns each
        ##then there will be a overlap of 200 tokens between the consequtive chunks
         
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        
        # now that we have chunks, we are going to compute the corresponding embeddings.
        # To do that we'll be using OPEN AI's text embeddings
        # Using FIAS vectorstore 
        
        store_name = pdf.name[:-4]
        
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write("Embeddings loaded from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user question/query
        query = st.text_input("Ask questions based on PDF file:")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
## Till here we created a knowledge base and stored it as a file. 
## And can accept questions from user

## Now we need to computer embeddings based on the questions and do a semantic search
# now that we returned the documents as a final step, we will be feeding that as a context to the LLM
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
