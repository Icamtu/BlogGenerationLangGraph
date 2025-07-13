# src/langgraphagenticai/graph/graph_builder.py
from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver
from src.langgraphagenticai.graph.graph_builder_blog import BlogGraphBuilder
from src.langgraphagenticai.graph.graph_builder_basic import BasicChatbotGraphBuilder
from src.langgraphagenticai.graph.graph_bulider_tool import ChatbotWithToolGraphBuilder
from src.langgraphagenticai.graph.graph_builder_sdlc import SdlcGraphBuilder




class GraphBuilder:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.memory = MemorySaver()
        self.blog_builder = BlogGraphBuilder(self.llm, self.memory)
        self.basic_builder = BasicChatbotGraphBuilder(self.llm, self.memory)
        self.tool_builder = ChatbotWithToolGraphBuilder(self.llm, self.memory)
        self.sdlc_builder = SdlcGraphBuilder(self.llm, self.memory)

    def validate_and_standardize_structure(self, user_input: str) -> list:
        """
        Uses an LLM to interpret user input and generate a standardized list of blog section names.
        Ensures the user's specified structure is respected if provided.

        Args:
            user_input (str): The full user input from the Streamlit form (e.g., "Topic: AI\nStructure: Intro, Benefits, Summary").

        Returns:
            List[str]: A list of standardized section names (e.g., ["Intro", "Benefits", "Summary"]).
        """
        return self.blog_builder.validate_and_standardize_structure(user_input)

    def setup_graph(self, usecase: str):
        """
        Sets up the appropriate graph based on the selected use case.
        """
        if usecase == "Basic Chatbot":
            return self.basic_builder.build_graph()
        elif usecase == "Chatbot with Tool":
            return self.tool_builder.build_graph()
        elif usecase == "Blog Generation":
            return self.blog_builder.build_graph()
        elif usecase == "SDLC":
            return self.sdlc_builder.build_graph()
        else:
            raise ValueError(f"Unknown use case: {usecase}")