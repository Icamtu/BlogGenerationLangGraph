import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json
from datetime import datetime
import base64
from src.Blog.ui.streamlit.display_result_blog import DisplayBlogResult
import functools
import time

from src.Blog.logging.logging_utils import logger, log_entry_exit

class DisplayResultStreamlit:
    def __init__(self, graph, with_message_history, config, usecase):
        self.graph = graph
        self.with_message_history = with_message_history
        self.config = config
        self.usecase = usecase
        self._initialize_session_state()
        self.session_history = self._get_session_history()

    def _initialize_session_state(self):
        """Initialize all session state variables."""
        defaults = {
            "current_session_id": None,
            "current_stage": "requirements",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
      

    def _get_session_history(self):
        from langchain_community.chat_message_histories import ChatMessageHistory
        store = {}
        session_id = self.config["configurable"]["session_id"]
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        if st.session_state.current_session_id != session_id:
            st.session_state.current_session_id = session_id
        return store[session_id]
    @log_entry_exit
    def display_chat_history(self):
        """Display the chat history from the session."""
        for message in self.session_history.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content, unsafe_allow_html=True)

    @log_entry_exit
    def process_user_input(self):
        """Process user input and display results based on the use case."""
        if self.usecase == "Blog Generation":
            blog_display = DisplayBlogResult(self.graph, self.config)
            blog_display.handle_blog_workflow()
        else:
            pass

    @log_entry_exit
    def _handle_chatbot_input(self):
        user_message = st.chat_input("Enter your message:")
        if user_message:
            self.session_history.add_user_message(user_message)
            with st.chat_message("user"):
                st.markdown(user_message, unsafe_allow_html=True)
            self._process_graph_stream(HumanMessage(content=user_message))

    @log_entry_exit
    def _process_graph_stream(self, input_message=None):
        with st.spinner("Processing..."):
            try:
                input_data = {"messages": [input_message]} if input_message else None
                for event in self.graph.stream(input_data, self.config):
                    logger.info(f"Graph event: {event}")
                    for node, state in event.items():
                        if "messages" in state and state["messages"]:
                            with st.chat_message("assistant"):
                                content = state["messages"][-1].content
                                st.markdown(content)
                            self.session_history.add_ai_message(content)
            except Exception as e:
                logger.error(f"Error in graph streaming: {e}")
                st.error(f"Error processing workflow: {e}")