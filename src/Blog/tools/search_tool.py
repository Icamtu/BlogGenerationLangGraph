
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
import os
import streamlit as st

def get_tools(max_results=3):
    """
    Returns a list of tools with configurable max_results.
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY", st.session_state.get("TAVILY_API_KEY", ""))
        if not tavily_api_key:
            st.error("Error: Tavily API key not provided")
            return []
        tools = [TavilySearchResults(max_results=max_results, api_key=tavily_api_key)]
        return tools
    except Exception as e:
        st.error(f"Error initializing search tools: {e}")
        return []

def create_tool_nodes(tools):
    """
    Creates tool nodes based on the provided tools.
    """
    try:
        if not tools:
            st.error("Error: No tools provided")
            return None
        return ToolNode(tools=tools)
    except Exception as e:
        st.error(f"Error creating tool nodes: {e}")
        return None