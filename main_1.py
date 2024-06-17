import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


#Bedrock client
bedrock = boto3.client(service_name = "bedrock-runtime", region_name = "us-east-1")

##create embeddings for text data using the model Amazon titan
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock)


# Load data and split
def get_documents():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000, 
                                        chunk_overlap=500)
    docs = text_spliter.split_documents(documents)
    return docs

#create a vector store using the FAISS (Facebook AI Similarity Search) library.
def get_vector_store(docs):
   vectorstore_faiss =  FAISS.from_documents(
        docs,
        bedrock_embedding
    )
   vectorstore_faiss.save_local("faiss_local")


# initialize a language model (LLM) using the Bedrock framework
def get_llm():
    llm = Bedrock(model_id = "mistral.mistral-7b-instruct-v0:2", client = bedrock)
    return llm


# format & structure the input to the LLM, ensuring that the LLM receives the necessary context and question 
# information in a consistent and understandable format

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#call with the user's query as input, retrieve the relevant documents from the vector store, generate a response using the LLM, and return the result.
def get_llm_response(llm, vectorstore_faiss, query):

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever= vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}),

        return_source_documents = True,
        chain_type_kwargs={"prompt": PROMPT})

    
    response = qa({"query": query})
    return response['result']


#create a user interface using the Streamlit library, allowing users to interact with the RAG (Retrieval-Augmented Generation) system. 
def main():
    st.set_page_config("RAG")
    st.header("End to end RAG using Bedrock")

    user_question = st.text_input("Ask a question from the PDF file")

    with st.sidebar:
        st.title("Update & create vectore store")

        if st.button("Store Vector"):
            with st.spinner("Processing.."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Done")

        if st.button("Send"):
            with st.spinner("Processing.."):
               faiss_index = FAISS.load_local("faiss_local", bedrock_embedding, allow_dangerous_deserialization=True) 
               llm = get_llm()
               st.write(get_llm_response(llm,faiss_index,  user_question))

            
if __name__ == "__main__":
    main()