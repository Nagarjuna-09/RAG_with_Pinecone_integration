# This file Ingests text document, breaks it down into embeddings and stores it in Pinecone
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


# Defining a custom text loader class that inherits from TextLoader.
# You can have redefined functions in langchain library like whatsapp chat import, YouTube transcript import etc.,
# class CustomTextLoader(TextLoader):
#     def lazy_load(self):
#         try:
#             with open(self.file_path, "r", encoding="utf-8") as f:
#                 text = f.read()
#             # Yield a Document object with the loaded text
#             yield Document(page_content=text)
#         except Exception as e:
#             raise RuntimeError(f"Error loading {self.file_path}") from e


if __name__ == "__main__":
    print("Ingesting...")
    # loader = CustomTextLoader("./mediumblog1.txt")  # Use the custom loader
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()  # Load the document using the custom loader (this returns a list of Document objects)

    print("Splitting...")
    # Creating a CharacterTextSplitter instance with a chunk size of 1000 characters and no overlap
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Splitting the document into chunks using the text splitter
    texts = text_splitter.split_documents(document) # list of documents
    # Printing out the number of chunks created
    print(f"Created {len(texts)} chunks")

    # Creating an instance of OpenAIEmbeddings with the API key retrieved from environment variables
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Print a message indicating the start of ingestion into Pinecone
    print("ingesting...")

    # PineconeVectorStore is a class used to interact with a Pinecone vector database, allowing you to store and retrieve embeddings efficiently.
    # The from_documents method is a class method that facilitates the process of embedding documents and storing them in a Pinecone index.

    # texts: This is a list of text documents that you want to embed and store in Pinecone.
    # embeddings: This is the OpenAIEmbeddings instance initialized earlier, which will be used to generate embeddings for each document in texts.
    # index_name: The name of the Pinecone index where the embeddings will be stored. This is retrieved from env variable using os.environ["INDEX_NAME"].

    #The PineconeVectorStore.from_documents method takes the list of text documents (texts), uses the
    # embeddings instance to generate embeddings, and stores these embeddings in a specified Pinecone index (index_name).
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
