import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input
    
    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
            selected_groq_model = self.user_controls_input.get('selected_groq_model', 'llama3-70b-8192')
            if not groq_api_key:
                st.error("Error: Groq API key not provided")
                return None
            llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)
            return llm
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {e}")
            return None