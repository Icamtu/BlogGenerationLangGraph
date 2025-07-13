import streamlit as st
import uuid
import logging
import os
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.Blog.ui.streamlit.loadui import LoadStreamlitUI
from src.Blog.llms.groqllm import GroqLLM
from src.Blog.llms.geminillm import GoogleLLM
from src.Blog.llms.chatgptllm import OpenaiLLM
from src.Blog.graph.graph_builder import GraphBuilder
from src.Blog.ui.streamlit.display_result import DisplayResultStreamlit

logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'  # Format for log messages
)
logger = logging.getLogger(__name__)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def BlogApp():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    Initializes UI, configures LLM, sets up graph, and manages session state.
    """
    ui = LoadStreamlitUI()
    user_controls = ui.load_streamlit_ui()

    if not user_controls:
        st.error("Error: Failed to load user controls from the UI.")
        return

    selected_llm = user_controls.get("selected_llm")
    if not selected_llm:
        st.info("Please select an LLM in the sidebar to proceed.")
        return

    tavily_api_key = user_controls.get("TAVILY_API_KEY", st.session_state.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", "")))
    if not tavily_api_key and user_controls.get("selected_usecase") in ["Blog Generation", "Chatbot with Tool"]:
        st.warning("Tavily API key not found. Web search will be skipped.")
    else:
        st.session_state["TAVILY_API_KEY"] = tavily_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    if selected_llm == "Groq" and not user_controls.get("GROQ_API_KEY"):
        st.warning("Please enter your Groq API key in the sidebar.")
        return
    elif selected_llm == "Google" and not user_controls.get("GOOGLE_API_KEY"):
        st.warning("Please enter your Google API key in the sidebar.")
        return
    elif selected_llm == "OpenAI" and not user_controls.get("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Session state initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = None
    if "waiting_for_feedback" not in st.session_state:
        st.session_state.waiting_for_feedback = False
    if "blog_requirements_collected" not in st.session_state:
        st.session_state.blog_requirements_collected = False
    if "current_usecase" not in st.session_state:
        st.session_state.current_usecase = None

    config = {"configurable": {"session_id": st.session_state.session_id, "thread_id": st.session_state.thread_id, "recursion_limit": 10}}
    logger.info(f"Session ID: {st.session_state.session_id}, Thread ID: {st.session_state.thread_id}")

    # Load LLM
    try:
        if selected_llm == "Groq":
            llm_config = GroqLLM(user_controls_input=user_controls)
        elif selected_llm == "Google":
            llm_config = GoogleLLM(user_controls_input=user_controls)
        elif selected_llm == "OpenAI":
            llm_config = OpenaiLLM(user_controls_input=user_controls)
        else:
            st.error(f"Error: Unsupported LLM selected: '{selected_llm}'")
            return

        model = llm_config.get_llm_model()
        if not model:
            st.error("Error: LLM model could not be initialized.")
            return

        # Graph setup
        usecase = user_controls.get("selected_usecase")
        if not usecase:
            st.error("Error: No use case selected.")
            return

        if st.session_state.current_usecase != usecase:
            logger.info(f"Use case changed to: {usecase}. Resetting session state.")
            st.session_state.waiting_for_feedback = False
            st.session_state.blog_requirements_collected = False
            st.session_state.current_usecase = usecase
            get_session_history(st.session_state.session_id).clear()
            if "graph" in st.session_state:
                del st.session_state.graph
            if "with_message_history" in st.session_state:
                del st.session_state.with_message_history

        if "graph" not in st.session_state:
            graph_builder = GraphBuilder(model)
            graph = graph_builder.setup_graph(usecase)
            with_message_history = RunnableWithMessageHistory(
                graph,
                get_session_history,
                input_messages_key="messages",
                history_messages_key="messages"
            )
            st.session_state.graph = graph
            st.session_state.with_message_history = with_message_history

        # Display chat history and process input
        display = DisplayResultStreamlit(st.session_state.graph, st.session_state.with_message_history, config, usecase)
        display.display_chat_history()
        display.process_user_input()

    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        st.error(f"Failed to initialize application: {e}")

if __name__ == "__main__":
    BlogApp()