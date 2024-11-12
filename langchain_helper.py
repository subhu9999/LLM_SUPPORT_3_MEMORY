import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import KEYS

# import KEYS
# if "GOOGLE_API_KEY" not in os.environ:
#      os.environ["GOOGLE_API_KEY"] = KEYS.GOOGLE_API_KEY

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = KEYS.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = KEYS.LANGCHAIN_PROJECT
os.environ["GOOGLE_API_KEY"] = KEYS.GOOGLE_API_KEY



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)


instructor_embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
vectordb_filepath="faiss_index_final"


vectordb = FAISS.load_local(vectordb_filepath, embeddings=instructor_embeddings,allow_dangerous_deserialization=True)
system_prompt = (
        "You are a customer support representative for {company}. "
        "Your goal is to assist the customer with any issues they have, "
        "using the following context:\n\n{context}\n\n"
        "Please provide a helpful and polite response to the customer's current question: {input}. "
        "If you are unable to answer the question based on the provided context, "
        "kindly respond with: 'I'm sorry, I don't have the information right now. "
        "You can contact expert support team at evis@support.com.au for further assistance.'"
    )

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answering_chain=create_stuff_documents_chain(llm, chat_prompt)

retriever = vectordb.as_retriever()

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
     ]
)

history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history = []

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def get_response(input):
    return conversational_rag_chain.invoke(
        {"input": input, "company": RunnablePassthrough()},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
