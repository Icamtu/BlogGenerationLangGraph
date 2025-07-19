# src/langgraphagenticai/graph/graph_builder.py
from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver
from src.Blog.graph.graph_builder_blog import BlogGraphBuilder




class GraphBuilder:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.memory = MemorySaver()
        self.blog_builder = BlogGraphBuilder(self.llm, self.memory)
        


    def setup_graph(self, usecase: str):
        """
        Sets up the appropriate graph based on the selected use case.
        """
        if usecase == "Blog Generation":
            return self.blog_builder.build_graph()
        
        else:
            raise ValueError(f"Unknown use case: {usecase}")