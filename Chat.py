import streamlit as st
import logging
import time

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector


logging.basicConfig(level=logging.INFO)

#init chat history in session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

def prepare_chat(model, messages, retriever):
    try:
        #sets up the chat model and RAG prompt template
        llm = ChatOllama(model=model)

        rag_template = """Answer the question based on the context below, 
        and if the question can't be answered based on the context, say "I don't know"

        Context:
        {context}

        Question: 
        {question}
        """

        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        #get the response from ollama
        full_llm_response = ""
        animated_response = ""
        response_container = ""

        full_llm_response = rag_chain.invoke(messages)
        response_container = st.empty()

        #animate the response as if the chat bot was actively typing
        for r in full_llm_response:
            animated_response += r
            time.sleep(0.008)
            response_container.write(animated_response)
        
        return full_llm_response
    except Exception as e:
        #log and re-raise any errors that occur
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("Sample RAG app")    
    model = "llama3.1"

    embeddings = OllamaEmbeddings(model="llama3.1")
    CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
    COLLECTION_NAME = "NJIT-workshop"
    vectorstore = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True)
    retriever = vectorstore.as_retriever()
    logging.info("Connected to DB successfully")

    #prompt user for input and save to chat history
    if prompt := st.chat_input("Enter your prompt:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        #display the user's prompt
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        #generate a new response if the last message isnt from the asssitant (is from the user)
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Generating response..."):
                    try:
                        #prepare messages for the LLM and stream the response
                        response_message = prepare_chat(model, str(st.session_state.messages[-1]["content"]), retriever)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f":blue[Response duration: {duration:.2f} seconds]")
                        logging.info(f"Model: {model}, Response: {response_message}, Duration: {duration:.2f} seconds")

                    except Exception as e:
                        #handle errors and display any error message
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error has occured while generating the response.")
                        logging.error(f"Error: {str(e)}")

main()