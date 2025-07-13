import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

class GoogleLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input
    
    def get_llm_model(self):
        try:
            google_api_key = self.user_controls_input.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            selected_google_genai_model = self.user_controls_input.get('selected_google_genai_model')
            if not google_api_key:
                st.error("Error: Google API key not provided")
                return None
            llm = ChatGoogleGenerativeAI(api_key=google_api_key, model=selected_google_genai_model)
            return llm
        except Exception as e:
            st.error(f"Error initializing Google LLM: {e}")
            return None

