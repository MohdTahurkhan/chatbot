import langchain_google_genai
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import itertools
import streamlit as st

# Directly declared API keys
API_KEYS = [
    "AIzaSyA7-lUzRsmTsocpAqsGs3-_F7-WYm8vIjE",
]

# Create a cycle iterator for the API keys
api_key_cycle = itertools.cycle(API_KEYS)

# Function to configure API with a specific key
def configure_api():
    api_key = next(api_key_cycle)
    genai.configure(api_key=api_key)
    return api_key

# Conversational chain using Google Gemini model
def get_conversational_chain():
    api_key = configure_api()
    prompt_template = """
    Answer the question as detailed as possible. If the answer is not in the context, provide a meaningful answer based on your knowledge.\n\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return model, prompt

# Function to handle user queries and provide responses
def user_input(user_question):
    api_key = configure_api()
    model, prompt = get_conversational_chain()

    # Use 'invoke()' and pass the question as 'input'
    response = model.invoke(input=user_question)
    
    # Access 'content' directly from the response object
    chatbot_reply = response.content if hasattr(response, 'content') else 'Sorry, I did not understand that.'

    return chatbot_reply

# Streamlit GUI
def streamlit_gui():
    st.title("Chatbot Interface")
    st.write("Ask me anything, and I will try to answer!")
    
    # Input box for user question
    user_question = st.text_input("You:", "")
    
    # Display response if a question is asked
    if user_question:
        with st.spinner('Thinking...'):
            response = user_input(user_question)
        st.write(f"Chatbot: {response}")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_gui()
