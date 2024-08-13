import os
import pinecone as pnc
import nest_asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

# Initialize Huggingface API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Initialize Pinecone API key
pc = pnc.Pinecone(api_key="4129280c-3316-429c-abd0-5db6392a5544")

# Here it will check that, is there the index exists and if not then it will create
index_name = "quickstart"
if index_name not in pc.list_indexes().names():
    # Here it will create the index if it does not exist
    index = pc.create_index(
        name=index_name,
        dimension=1536,  # Embedding dimension from my Pinecone account
        metric='cosine',  # metric from my Pinecone account
        spec=pnc.ServerlessSpec(
            cloud='aws',  # cloud provider from my Pinecone account
            region='us-east-1'  # region from my Pinecone account
        )
    )
else:
    # Load the existing index
    index = pc.Index(index_name)

nest_asyncio.apply()

def setup_qa_system():
    # website's sitemap 
    loader = SitemapLoader(
    "https://romero.sparktechwp.com/sitemap.xml", 
    filter_urls=["https://romero.sparktechwp.com/"]
    )
    # Load documents
    docs = loader.load()

    # Here Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, length_function=len)
    docs_chunks = text_splitter.split_documents(docs)

    # Initialize Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize Chroma vector store
    docsearch = Chroma.from_documents(docs_chunks, embeddings)

    # Initialize the Hugging Face model for the LLM
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0})

    # Set up the RetrievalQA chain
    qa_with_sources = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    
    return qa_with_sources

def get_answer(qa_system, query):
    result = qa_system({"query": query})
    return result["result"], result["source_documents"]



