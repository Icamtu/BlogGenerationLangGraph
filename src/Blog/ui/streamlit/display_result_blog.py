import streamlit as st
from langchain_core.messages import HumanMessage
import logging
import markdown  
import json
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import functools
import time
from src.Blog.logging.logging_utils import logger, log_entry_exit
from src.Blog.ui.uiconfigfile import Config


class ReviewFeedback(BaseModel):
    approved: bool = Field(description="Approval status: True for approved, False for rejected")
    comments: str = Field(description="Reviewer comments")

class DisplayBlogResult:
    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        self.session_history = []
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            "current_stage": "requirements",  # Track the current stage
            "waiting_for_feedback": False,
            "blog_requirements_collected": False,
            "content_displayed": False,
            "graph_state": None,
            "feedback": "",
            "blog_content": None,
            "blog_generation_complete": False,
            "feedback_submitted": False,  # Track if feedback was submitted
            "processing_complete": False,  # Track if processing is complete
            "feedback_result": None,
            "generated_draft": None,
            "synthesizer_output_processed": False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def show_sidebar_progress(self):
        st.sidebar.markdown("## 🚀 Workflow Progress")

        stages = [
            ("requirements", "1️⃣ Collect Requirements"),
            ("processing", "2️⃣ Generating Blog"),
            ("feedback", "3️⃣ Feedback Review"),
            ("complete", "✅ Finalized"),
        ]

        current_stage = st.session_state.get("current_stage", "requirements")

        for key, label in stages:
            if key == current_stage:
                st.sidebar.success(f"**{label}**")
            elif stages.index((key, label)) < stages.index((current_stage, dict(stages)[current_stage])):
                st.sidebar.markdown(f"✅ {label}")
            else:
                st.sidebar.markdown(f"🔲 {label}")

    @log_entry_exit
    def collect_blog_requirements(self):
        """Collect blog requirements from the user."""
        self.show_sidebar_progress()
        st.markdown("## Stage 1: Blog Requirements")
        with st.expander("Stage 1: Blog Requirements", expanded=True):
            st.info("ℹ️ Fill in the details below to generate your blog post")

            with st.form("blog_requirements_form"):
                topic = st.text_input("Topic", value="", placeholder="Enter the blog topic here")

                objective_options = ["Informative", "Persuasive", "Storytelling", "Other"]
                objective = st.radio("Objective", objective_options)
                custom_objective = None
                if objective == "Other":
                    custom_objective = st.text_input("Specify Objective")

                audience_options = ["Beginners", "Experts", "General Audience", "Other"]
                target_audience = st.radio("Target Audience", audience_options)
                custom_audience = None
                if target_audience == "Other":
                    custom_audience = st.text_input("Specify Target Audience")

                tone_options = ["Formal", "Casual", "Technical", "Engaging", "Other"]
                tone_style = st.radio("Tone & Style", tone_options)
                custom_tone = None
                if tone_style == "Other":
                    custom_tone = st.text_input("Specify Tone & Style")

                word_count = st.number_input("Word Count", min_value=100, max_value=5000, value=100, step=100)
                structure = st.text_area("Structure", placeholder="e.g., Introduction, Key Points, Conclusion")

                submit_button = st.form_submit_button("Next")

                if submit_button:
                    # Handle custom inputs
                    if objective == "Other" and custom_objective:
                        objective = custom_objective
                    if target_audience == "Other" and custom_audience:
                        target_audience = custom_audience
                    if tone_style == "Other" and custom_tone:
                        tone_style = custom_tone

                    if not all([topic, objective, target_audience, tone_style]):
                        st.error("Please fill in all required fields.")
                        return None

                    # Create message and add to history
                    message = HumanMessage(content=f"Topic: {topic}\nObjective: {objective}\n"
                                                    f"Target Audience: {target_audience}\nTone & Style: {tone_style}\n"
                                                    f"Word Count: {word_count}\nStructure: {structure}\n"
                                                    f"feedback: {st.session_state.get('feedback')}")
                    self.session_history.append(message)
                    st.session_state.blog_requirements_collected = True
                    logger.info(f"\n\n--------------:Blog requirements collected:------------------\n{message.content}--------------------\n\n")
                    # Show summary
                    st.success("✅ Blog requirements submitted successfully!")

                    return message
        return None
    
    @log_entry_exit
    def _handle_approved_click(self):
        print("\n\n----approved button ON_CLICK call back executed----\n\n")
        logger.info("----approved button ON_CLICK call back executed----")
        st.session_state['feedback_result'] = ReviewFeedback(approved=True, comments=st.session_state.get('feedback'))
        st.session_state["feedback_submitted"] = True 
        print(f"\n\n----------exiting _handle_approved_click function{st.session_state['feedback_result']}---------------\n\n")
    
    @log_entry_exit
    def _handle_revised_click(self):
    
        print("\n\n----Revised button ON_CLICK call back executed----\n\n")
        logger.info("----Revised button ON_CLICK call back executed----")
        st.session_state['feedback_result'] = ReviewFeedback(approved=False, comments=st.session_state.get('feedback'))
        st.session_state["feedback_submitted"]=True
        print(f"\n\n----------feedback_submitted: {st.session_state['feedback_submitted']} & Exiting _handle_revised_click function with {st.session_state['feedback_result']}---------------\n\n")
   


    def process_graph_events(self, input_data=None):
        """Processes graph events, handling initial runs and resumes."""
        try:
            if not input_data:
                logger.warning("process_graph_events called with no input data.")
                return 

            logger.info(f"Starting graph processing/resuming with input keys: {list(input_data.keys())}")

            progress_bar = st.progress(0)
            last_node_output = None # To store the output of the last node before interrupt


            for i, event in enumerate(self.graph.stream(input_data, self.config)):
                logger.info(f"Graph event received: #{i+1}")
                event_key = list(event.keys())[0]
                logger.info(f"Processing node/event: {event_key}")

                # Update progress indicator
                progress_value = min((i + 1) * 0.1, 0.9) # Adjust progress calculation as needed
                progress_bar.progress(progress_value)

                # Store the state from the event right before a potential interrupt
                # This ensures we have the latest state if an interrupt occurs
                last_node_output = event.get(event_key)

                # Check for interrupt signal - LangGraph handles the actual state saving via checkpointer
                if event_key == "__interrupt__":
                    logger.info("Interrupt event received - transitioning to feedback stage")
                    st.session_state.waiting_for_feedback = True
                    st.session_state.current_stage = "feedback"
                    # Store the latest draft if available from the step before interrupt
                    if last_node_output and "initial_draft" in last_node_output:
                        st.session_state.generated_draft = last_node_output["initial_draft"]
                        st.session_state.content_displayed = True
                    st.rerun() # Rerun Streamlit to display feedback UI
                    return # Stop processing events after interrupt

                # Store the generated draft when the synthesizer node completes
                if event_key == "synthesizer" and last_node_output and "initial_draft" in last_node_output:
                    logger.info("Draft generated by synthesizer, storing for display.")
                    st.session_state.generated_draft = last_node_output["initial_draft"]
                    st.session_state.content_displayed = True # Flag that content is ready

                # Check for graph completion (reaching the end or a specific final node)
                if event_key == "file_generator": 
                    
                    logger.info("Graph processing complete.")
                    if last_node_output and "final_report" in last_node_output:
                        st.session_state["blog_content"] = last_node_output["final_report"]
                    st.session_state.current_stage = "complete"
                    progress_bar.progress(1.0)
                    st.rerun() 
                    return # Stop processing

            # If loop finishes without interrupt or explicit end node signal
            progress_bar.progress(1.0)
            logger.info("Graph stream finished.")
            # Potentially update state if needed, e.g., move to complete stage if not already handled
            if st.session_state.current_stage != "complete" and st.session_state.current_stage != "feedback":
                st.session_state.current_stage = "complete" # Assume completion if stream ends normally
                st.rerun()


        except Exception as e:
            logger.exception(f"Error in graph streaming: {e}")
            st.error(f"⚠️ Error processing workflow: {e}")
            
    @log_entry_exit
    def process_graph_events_with_checkpoint(self, input_data):
        try:
            logger.info(f"Starting graph resume with checkpoint. Input keys: {list(input_data.keys())}")
            if "__checkpoint__" in input_data:
                logger.info(f"Checkpoint keys: {list(input_data['__checkpoint__'].keys())}")
            progress_bar = st.progress(0)
            for i, event in enumerate(self.graph.stream(input_data, self.config)):
                logger.info(f"Graph event received (resuming): #{i+1}")
                logger.info(f"Processing node: {list(event.keys())[0]}")
                node = list(event.keys())[0]
                state = event[node]
                if node == "synthesizer" and "initial_draft" in state:
                    logger.info(f"Synthesizer draft: {state['initial_draft']}")
                progress_bar.progress(min(i * 0.1 + 0.1, 1.0))
                if "__interrupt__" in event:
                    logger.info("Interrupt detected during resume")
                    st.session_state.graph_state = event.get("__checkpoint__", input_data["__checkpoint__"])
                    return "interrupted"
                if "file_generator" in event:
                    logger.info(f"Final draft: {state.get('initial_draft', 'No draft generated')}")
                    return "completed"
            return "completed"
        except Exception as e:
            logger.exception(f"Error resuming graph with checkpoint: {e}")
            st.error(f"⚠️ Error resuming workflow: {e}")
            return "error"
            
    @log_entry_exit
    def process_feedback(self):
        """Process user feedback on the generated blog draft."""
        self.show_sidebar_progress()
        print("\n\n----blog_display process_feedback function entered----\n\n")
        logger.info("---blog_display process_feedback function entered ----\n\n")
        
        st.markdown("## Stage 3: Feedback")
        if st.session_state.get("generated_draft"):
            st.markdown("### Drafted Blog Content:")
            st.markdown(st.session_state["generated_draft"])
            
        with st.expander("Stage 3: Feedback", expanded=True):
            feedback_text = st.text_input(
                "Revision comments:",
                placeholder="Please explain what changes you would like to see.",
                key="revision_comments_area",
                value=st.session_state.get("revision_comments_area", "")
            )
            st.session_state["feedback"] = feedback_text  

            col1, col2 = st.columns(2)
            with col1:
                st.button("✅ Approve Content", on_click=self._handle_approved_click, key="blog_feedback_approve_button")
            with col2:
                st.button("Submit Revision Request", on_click=self._handle_revised_click, key="blog_feedback_revise_button")
        
        return st.session_state.get('feedback_result')
    
    @log_entry_exit
    def _download_blog_content(self, blog_content):
        """Create a download link for the blog content."""
        self.show_sidebar_progress()
        if blog_content:
            import base64 # Ensure import
            b64 = base64.b64encode(blog_content.encode()).decode()
            # Use a timestamp or unique ID in filename if needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blog_content_{timestamp}.md" # Use .md extension for markdown
            href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">⬇️ Download Blog Content (Markdown)</a>'
            st.markdown(href, unsafe_allow_html=True)

    def handle_blog_workflow(self):
        """Encapsulate the entire blog workflow logic for use in DisplayResultStreamlit."""
        # Stage 1: Collect Requirements
        if st.session_state.current_stage == "requirements":
            if not st.session_state.blog_requirements_collected:
                input_message = self.collect_blog_requirements()
                if input_message:
                    st.session_state.blog_requirements_collected = True
                    st.session_state.initial_input_message = input_message
                    st.session_state.current_stage = "processing"
                    st.rerun()

        # Stage 2: Initial Processing
        elif st.session_state.current_stage == "processing":
            logger.info("Entering processing stage.")
            initial_input = st.session_state.get('initial_input_message')
            if initial_input:
                input_data = {"messages": [initial_input]}
                logger.info(f"DEBUG: Calling process_graph_events with initial input: {input_data}")
                self.process_graph_events(input_data)
            else:
                logger.error("Processing stage reached but initial input message is missing.")
                st.error("Error: Initial requirements not found. Please start over.")
                st.session_state.current_stage = "requirements"
                st.rerun()

        # Stage 3: Handle Feedback
        elif st.session_state.current_stage == "feedback":
            logger.info(f"Entering feedback stage. Submitted: {st.session_state.get('feedback_submitted', False)}")
            if st.session_state.get("generated_draft"):
                if not st.session_state.get("feedback_ui_displayed", False):
                    st.session_state["feedback_ui_displayed"] = True
                feedback_result = self.process_feedback()
            else:
                st.warning("Waiting for draft to be generated before collecting feedback.")

            if st.session_state.get("feedback_submitted"):
                logger.info("Feedback form submitted.")
                feedback_result = st.session_state.get('feedback_result')
                st.session_state["feedback_submitted"] = False
                st.session_state["feedback_ui_displayed"] = False

                if feedback_result:
                    if feedback_result.approved:
                        logger.info("Feedback: Approved")
                        st.session_state["blog_content"] = st.session_state.get("generated_draft")
                        st.session_state["generated_draft"] = None
                        st.session_state.current_stage = "complete"
                        st.session_state['feedback_result'] = None
                        st.rerun()
                    else:
                        logger.info(f"Feedback: Revision requested - comments: {feedback_result.comments}")
                        st.session_state["feedback"] = feedback_result.comments
                        st.session_state.current_stage = "processing_feedback"
                        st.session_state['feedback_result'] = None
                        st.session_state["generated_draft"] = None
                        st.session_state["completed_sections"] = None
                        logger.info(f"{'='*20}\n:session state after revision request:\n {st.session_state}{'='*20}")
                        st.rerun()
                else:
                    logger.warning("Feedback submitted but no result found in session state.")

        # Stage 4: Process Feedback (Resume Graph)
        elif st.session_state.current_stage == "processing_feedback":
            logger.info("Entering processing_feedback stage.")
            feedback_comment = st.session_state.get("feedback")
            if feedback_comment is not None:
                feedback_message = HumanMessage(content=json.dumps({
                    "approved": False,
                    "comments": feedback_comment
                }))
                input_data = {"messages": [feedback_message]}
                logger.info(f"Resuming graph with feedback message: {feedback_message.content}")
                st.session_state["feedback"] = ""
                self.process_graph_events(input_data=input_data)
            else:
                logger.error("Processing feedback stage reached but feedback comments are missing.")
                st.error("Error: Feedback comments not found. Please provide feedback again.")
                st.session_state.current_stage = "feedback"
                st.rerun()

        # Stage 5: Completion
        elif st.session_state.current_stage == "complete":
            logger.info("Entering complete stage.")
            st.success("✅ Blog generation complete!")
            if st.session_state.get("blog_content"):
                st.markdown("### Final Blog Content:")
                st.markdown(st.session_state["blog_content"])
                self._download_blog_content(st.session_state["blog_content"])
            else:
                st.warning("Final blog content is not available.")