import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


print("Retrieving...")

openai_embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

query = "Who is Nagarjuna?"


# The PineconeVectorStore is initialized with the OpenAIEmbeddings instance and the index name.
# This setup binds the embedding method to the vector store.
# This prepares the vector store for querying.
# When a query is issued, the PineconeVectorStore uses the OpenAIEmbeddings instance to convert the query text into an embedding.
# This embedding is then used to find the most similar document embeddings stored in the Pinecone index.
# This initializes the Pinecone vector store and binds it to the OpenAIEmbeddings instance. This setup allows the
# vector store to use the embedding model for both storing document embeddings and converting queries into embeddings.
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=openai_embeddings
)

# query embedding and retrieval
# This pulls a predefined prompt template for retrieval-based question answering from the LangChain hub.
# This template defines how the input query and the retrieved documents should be formatted and combined
# before being passed to the language model.
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")


# Takes llm and prompt template as input
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# This does two main things 1. Retrieves documents from vector store, combines retrieved docs with input query and uses an llm to generate response
# retriever=vectorstore.as_retriever(): Converts the vector store into a retriever object that can embed queries and retrieve relevant documents.
retrieval_chain = create_retrieval_chain(
    # retriever is responsible for searching the vector store to find documents that are relevant to the input query. It converts the query into an embedding and performs a similarity search to retrieve the top matching documents.
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=combine_docs_chain
    # combine_docs_chain takes the retrieved documents and the original query, combines them using a specified prompt template, and generates a response using a language model.
)

#  executes the retrieval chain with the input query.
result = retrieval_chain.invoke(input={"input": query})
print(result["answer"])
# Retrieval Steps
# When a query is passed to the retrieval chain, it first retrieves relevant documents using the retriever.
# The retrieved documents are then combined with the query using the prompt template.
# The language model processes this combined input to generate the final response.


variable_prompt = """
Use the following piece of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to makeup an answer.
Use three sentences maximum and keep the answer as concise as possible

{context}

Question: {question}

Helpful Answer: """

# RAG Steps:
# We have our query, we embed it, and then we ask the vector store to get the similar embeddings for that query.
# Take those relevant documents shared by vector db, append it with our prompt with some instructions and share to llm for response.