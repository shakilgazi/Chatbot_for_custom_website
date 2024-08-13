## Project Overview

This project implements a custom chatbot capable of answering questions based on the content of the website.

Step-by-Step Implementation

# 1. Data Collection

-Sitemap Parsing: 
I collected website data by parsing the sitemap using "LangChain’s" "SitemapLoader". This allowed me to automatically gather all relevant URLs and extract their content efficiently.

# 2. Text Splitting

-Document Chunking: 
To handle the large content efficiently, I used "LangChain’s" "RecursiveCharacterTextSplitter" to split the website content into smaller, manageable chunks. This ensured that the content could be processed effectively by the model without losing context.

# 3. Embedding Creation

-Hugging Face Model: 
I employed the "sentence-transformers/all-MiniLM-L6-v2" model from "Hugging Face" to generate embeddings for each chunk of text. These embeddings are numerical representations of the text that capture semantic meaning, making it easier to compare and retrieve relevant content later.

# 4. Vector Storage and Retrieval

-Pinecone Indexing: 
I used "Pinecone" to store these embeddings in a vector index, which allows for fast and efficient similarity searches. Pinecone was chosen for its scalability and real-time search capabilities.

# 5. Question-Answering System

-Retrieval and Response Generation: 
Using LangChain’s RetrievalQA chain, I integrated the "Hugging Face" "google/flan-t5-large" model to process user queries. The system retrieves the most relevant document chunks from "Pinecone" and generates accurate answers based on the retrieved content.


# 6. Streamlit Application

-Chatbot Interface: 
Developed a chatbot using Streamlit to visualize the question-answering system. The app provides an intuitive interface where users can ask questions related to the website's content and receive instant answers.

-Real-Time Interaction: 
The Streamlit app displays both the generated responses and the sources of information, ensuring transparency and context for each answer.




