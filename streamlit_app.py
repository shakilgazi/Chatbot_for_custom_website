# streamlit_app.py

import streamlit as st
from langchain_code import setup_qa_system, get_answer

# Set up the QA system
qa_system = setup_qa_system()

st.title("Chatbot Interface")

st.write("Ask me anything related to whole body cryotherapy therapy chamber:")

# Create an input box for the user's query
user_query = st.text_input("Your question:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Processing..."):
            # Get the answer from the LangChain QA system
            answer, sources = get_answer(qa_system, user_query)
            st.write("**Answer:**")
            st.write(answer)
            st.write("**Sources:**")
            for source in sources:
                st.write(f"- {source.metadata['source']}")





#streamlit run streamlit_app.py
