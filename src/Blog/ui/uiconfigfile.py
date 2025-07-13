from configparser import ConfigParser
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_file=os.path.join(os.path.dirname(__file__), 'uiconfigfile.ini')):
        self.config_file = config_file
        self.config = ConfigParser()
        self.config.read(config_file)
        
 
    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    def get_google_model_options(self):
        return self.config["DEFAULT"].get("Google_MODEL_OPTIONS").split(", ")
    
    def get_openai_model_options(self):
        return self.config["DEFAULT"].get("OPENAI_MODEL_OPTIONS").split(", ")

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")