import os
import streamlit as st
from langchain_openai import ChatOpenAI

class OpenaiLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input
    
    def get_llm_model(self):
        try:
            openai_api_key = self.user_controls_input.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            selected_OPENAI_model = self.user_controls_input.get('selected_OPENAI_model', 'gpt-3.5-turbo')
            
            if not openai_api_key:
                st.error("Error: OpenAI API key not provided")
                return None
            
            llm = ChatOpenAI(api_key=openai_api_key, model=selected_OPENAI_model)
            return llm
        except Exception as e:
            st.error(f"Error initializing OpenAI LLM: {e}")
            return None