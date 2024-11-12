import streamlit as st
from langchain_helper import get_response

st.title("EVIS Virtual Assistant (gemini-1.5-flash Memo)")
# btn = st.button("UPDATE Knowledgebase")
# if btn:
#     pass

question = st.text_input("",placeholder="Ask anything about our products .. ")

if question:
    response = get_response(question)
    # response = qa_chain.invoke(question)
    st.header("Answer: ")
    #response = qa_chain.invoke(question)
    st.write(response)
