import boto3
import streamlit as st

## We will be using Titan Embeddings Model To Generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding And Vectore Store

from langchain_community.vectorstores import FAISS

## LLM Models

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


## Data ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


# ## create the Anthropic Model
# def get_claude_llm():
#     llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock, model_kwargs={'maxTokens': 512})
#     return llm


## create the Meta Model
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context> 

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a Question from the PDF files")

    with st.sidebar:
        st.title("Update Or Create Vectore Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    # if st.button("Claude Output"):
    #     with st.spinner("Processing..."):
    #         faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
    #         llm = get_claude_llm()
    #
    #         st.write(get_response_llm(llm, faiss_index, user_question))

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))


if __name__ == '__main__':
    main()
