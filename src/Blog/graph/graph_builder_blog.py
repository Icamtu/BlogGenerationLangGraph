from langgraph.graph import StateGraph, START, END
from src.Blog.nodes.blog_generation_node import BlogGenerationNode
from src.Blog.state.state import BlogState as State
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import json
import logging
import functools
import time
from src.Blog.logging.logging_utils import logger, log_entry_exit


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")



from langchain_google_genai import ChatGoogleGenerativeAI
# Define the modelcls
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)




class ReviewFeedback(BaseModel):
    approved: bool = Field(description="Approval status: True for approved, False for rejected")
    comments: str = Field(description="Reviewer comments")

class BlogGraphBuilder:
    def __init__(self, llm, memory: MemorySaver=None):
        self.llm = llm
        self.memory = memory if memory is not None else MemorySaver()

    @log_entry_exit
    def build_graph(self):
        """Builds a graph for the Blog Generation use case."""
        try:
            if not self.llm:
                raise ValueError("LLM model not initialized")

            graph_builder = StateGraph(state_schema=State)
            blog_node = BlogGenerationNode(self.llm)

            # Add nodes
            graph_builder.add_node("user_input", blog_node.user_input)
            graph_builder.add_node("orchestrator", blog_node.orchestrator)
            graph_builder.add_node("llm_call", blog_node.llm_call)
            graph_builder.add_node("synthesizer", blog_node.synthesizer)
            graph_builder.add_node("feedback_collector", blog_node.feedback_collector)
            graph_builder.add_node("revise_blog", blog_node.revise_blog)
            graph_builder.add_node("file_generator", blog_node.file_generator)

            # Add edges
            graph_builder.add_edge(START, "user_input")
            graph_builder.add_edge("user_input", "orchestrator")
            graph_builder.add_conditional_edges("orchestrator", lambda state: blog_node.assign_workers(state), ["llm_call"])
            graph_builder.add_edge("llm_call", "synthesizer")
            graph_builder.add_edge("synthesizer", "feedback_collector")
            
            # CONDITIONAL EDGES
            graph_builder.add_conditional_edges(
                "feedback_collector",
                blog_node.route_feedback,
                {
                    "file_generator": "file_generator",
                    "revise_blog": "revise_blog"
                }
            )
            
            graph_builder.add_edge("revise_blog", "feedback_collector")
            graph_builder.add_edge("file_generator", END)

            compiled_graph = graph_builder.compile(interrupt_before=["feedback_collector"], checkpointer=self.memory)
            return compiled_graph

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise


Blog_builder_instance = BlogGraphBuilder(model)
agent = Blog_builder_instance.build_graph()