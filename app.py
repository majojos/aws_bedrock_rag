import json
import os
import sys
import boto3

## We will be using Titan Embeddings Model To Generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding And Vectore Store

from langchain.vectorstores import FAISS

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

    text_splitter = RecursiveCharacterTextSplitter(chuck_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    ## create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm


def get_llama2_llm():
    ## create the Anthropic Model
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm
