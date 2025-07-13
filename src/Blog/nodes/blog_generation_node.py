from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.langgraphagenticai.state.state import BlogState as State, Sections, Section  # Import from state.py
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
import json
from datetime import datetime
from typing import List

from src.langgraphagenticai.logging.logging_utils import logger, log_entry_exit

import functools
import time

class BlogGenerationNode:
    def __init__(self, model):
        """Initialize the BlogGenerationNode with an LLM."""
        self.llm = model
        self.planner = model.with_structured_output(Sections)

    @log_entry_exit
    def validate_and_standardize_structure(self, user_input: str) -> List[str]:
        """
        Uses an LLM to interpret user input and generate a standardized list of blog section names.
        Ensures the user's specified structure is respected if provided.

        Args:
            user_input (str): The full user input from the Streamlit form (e.g., "Topic: AI\nStructure: Intro, Benefits, Summary").

        Returns:
            List[str]: A list of standardized section names (e.g., ["Intro", "Benefits", "Summary"]).
        """
        # Default structure if all else fails
        default_structure = ["Introduction", "Main Content", "Conclusion"]

        # If input is empty or whitespace-only, return default
        if not user_input or not user_input.strip():
            logger.info("Empty or whitespace-only input; returning default structure")
            return default_structure

        # Extract the user's structure if provided
        user_structure = None
        for line in user_input.split("\n"):
            if line.lower().startswith("structure:"):
                user_structure = line.split(":", 1)[1].strip()
                break

        if not user_structure:
            logger.info("No structure provided; returning default structure")
            return default_structure

        # Define the prompt for the LLM
        system_prompt = (
            "You are an expert blog planner. Your task is to analyze the user's input and extract or infer a clear, concise structure "
            "for a blog post as a list of section names. The input may explicitly list sections (e.g., 'Structure: Intro, Benefits, Summary') "
            "or describe them implicitly (e.g., 'I want an intro, some benefits, and a conclusion'). "
            "If the user provides a 'Structure' field (e.g., 'Structure: Intro, Benefits, Summary'), you MUST use those exact section names "
            "without modification, except for capitalizing the first letter of each section. "
            "If no structure is provided or it's unclear, propose a logical default structure based on the topic or context. "
            "Return the result as a JSON object with a single key 'sections' containing the list of section names. "
            "Capitalize each section name and avoid adding unnecessary sections beyond what’s indicated."
        )

        # Prepare messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User input: {user_input}")
        ]

        try:
            # Invoke the LLM and expect a JSON response
            response = self.llm.invoke(messages)
            response_content = response.content if hasattr(response, "content") else str(response)
            logger.info(f"LLM response for structure: {response_content}")

            # Parse the JSON response
            result = json.loads(response_content)
            sections = result.get("sections", default_structure)

            # Validate and standardize the output
            if not isinstance(sections, list) or not sections:
                logger.warning("LLM returned invalid sections; using default structure")
                return default_structure

            # Clean up section names: strip whitespace, capitalize, remove empty strings
            cleaned_sections = [s.strip().capitalize() for s in sections if s.strip()]

            # If user provided a structure, enforce it
            if user_structure:
                user_sections = [s.strip().capitalize() for s in user_structure.split(",") if s.strip()]
                if len(cleaned_sections) == len(user_sections):
                    # Override LLM sections with user sections if lengths match
                    cleaned_sections = user_sections
                else:
                    logger.warning(f"LLM section count ({len(cleaned_sections)}) doesn't match user section count ({len(user_sections)}); using user structure")
                    cleaned_sections = user_sections

            return cleaned_sections if cleaned_sections else default_structure

        except Exception as e:
            logger.error(f"Error in LLM structure generation: {e}")
            return default_structure
    
    @log_entry_exit
    def user_input(self, state: State) -> dict:
        """Handle user input, distinguishing between initial requirements and feedback."""
        logger.info(f"Executing user_input with state: {state}")
        
        # Initialize requirements with existing state values to preserve them
        requirements = {
            "topic": state.get("topic", "No topic provided"),
            "objective": state.get("objective", "Informative"),
            "target_audience": state.get("target_audience", "General Audience"),
            "tone_style": state.get("tone_style", "Casual"),
            "word_count": state.get("word_count", 1000),
            "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
            "feedback": state.get("feedback", "No feedback provided yet."),
            # Always reset these values to ensure old content doesn't persist
            "initial_draft": "",  
            "completed_sections": []  
        }
        
        # Get the latest message
        user_message = state["messages"][-1].content if state["messages"] else ""
        if not user_message:
            logger.warning("No user message provided; returning existing requirements with reset content")
            return requirements

        # Flag to track if the message is feedback
        is_feedback = False
        
        try:
            # Check if the message is feedback (JSON format)
            feedback_data = json.loads(user_message)
            if isinstance(feedback_data, dict) and "approved" in feedback_data:
                # This is a feedback message, update only the feedback field
                requirements["feedback"] = feedback_data.get("comments", "No feedback provided.")
                is_feedback = True
                logger.info(f"Processed feedback message: {requirements['feedback']}")
                
                # For feedback, we definitely want to ensure content reset
                requirements["initial_draft"] = ""
                requirements["completed_sections"] = []
            else:
                # Treat as requirements input
                temp_requirements = {}
                for line in user_message.split("\n"):
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        temp_requirements[key.lower().replace(" & ", "_").replace(" ", "_")] = value
                
                # Update requirements only for provided fields
                requirements.update({
                    "topic": temp_requirements.get("topic", requirements["topic"]),
                    "objective": temp_requirements.get("objective", requirements["objective"]),
                    "target_audience": temp_requirements.get("target_audience", requirements["target_audience"]),
                    "tone_style": temp_requirements.get("tone_style", requirements["tone_style"]),
                    "word_count": int(temp_requirements.get("word_count", requirements["word_count"])),
                    "structure": temp_requirements.get("structure", requirements["structure"]),
                    "feedback": temp_requirements.get("feedback", requirements["feedback"]),
                    # Always reset content for new requirements
                    "initial_draft": "",  
                    "completed_sections": []  
                })
                logger.info(f"Processed requirements input: {requirements}")
        except Exception as e:
            logger.error(f"Unexpected error processing user message: {e}")
            # Return existing requirements to avoid crashing, but still clear content
            requirements["initial_draft"] = ""
            requirements["completed_sections"] = []
            return requirements

        structure_input = requirements["structure"] if is_feedback else user_message
        standardized_structure = self.validate_and_standardize_structure(structure_input)
        requirements["structure"] = ", ".join(standardized_structure)

        # Log the final state that will be returned
        logger.info(f"Final parsed requirements with reset content: {requirements}")
        logger.info(f"Completed sections (should be empty): {requirements['completed_sections']}")
        logger.info(f"Initial draft (should be empty): {requirements['initial_draft']}")
        
        return requirements

    
    @log_entry_exit        
    def orchestrator(self, state: State) -> dict:
        logger.info(f"Executing orchestrator with state: {state}")
        needs_revision = False

        logger.info(f"Orchestrator received completed_sections: {state.get('completed_sections', [])}")
        
        # Initialize default return values in case of early return or exception
        return_state = {
            "sections": [],
            "completed_sections": [],
            "initial_draft": ""
        }
        needs_revision=False

        if state.get("messages"):
            last_message_content = state["messages"][-1].content
            try:
                feedback_data = json.loads(last_message_content)
                if isinstance(feedback_data, dict) and feedback_data.get("approved") is False:
                    needs_revision = True
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.warning(f"Error checking last message for revision trigger: {e}")

        if needs_revision:
            logger.info("Orchestrator identified revision cycle: Clearing completed_sections.")
            # Don't modify state directly, include this in return dictionary instead
            return_state["completed_sections"] = []

        structure_list = [s.strip() for s in state["structure"].split(",")]
        section_count = len(structure_list)
        feedback = state.get("feedback", "No feedback provided yet.")

        prompt = (
            f"Create a detailed and structured plan for a blog report consisting of exactly {section_count} sections. "
            f"The content should be directly relevant to the topic: '{state['topic']}'. "
            f"The primary objective of the blog is to {state['objective']}, targeting an audience of {state['target_audience']}. "
            f"Please maintain a {state['tone_style']} tone throughout the writing. "
            f"Aim for a total word count of approximately {state['word_count']} words. "
            f"Follow this specific structure and section names: {', '.join(structure_list)}. "
            f"Incorporate {feedback} to enhance the quality of the content. "
            f"Please refrain from adding any extra sections or altering the section names unless {feedback} is provided."
        )

        try:
            report_sections = self.planner.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=f"Topic: {state['topic']} with feedback {feedback}")
            ])
            return_state["sections"] = report_sections.sections
            
        except Exception as e:
            logger.error(f"Error generating plan with LLM: {e}")
            # Keep the default empty values in return_state
        
        logger.info(f"Orchestrator returning: {return_state}")
        return return_state
    @log_entry_exit
    def llm_call(self, state: State) -> dict:
        """Worker writes a section of the report."""
        section = self.llm.invoke([
            SystemMessage(content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."),
            HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}")
        ])
        logger.info(f"\n{'='*20}:llm_call output:{'='*20}\nGenerated section: {section.content}\n{'='*20}\n")
        logger.info(f"\n---------------------state[completed_sections]:---------------------------- \n{state.get('completed_sections', [])}")

        return {"completed_sections": state.get("completed_sections", []) + [section.content]}
    
    @log_entry_exit
    def synthesizer(self, state: State) -> dict:
            """Synthesize full report from sections and clear the sections list."""
            # Safely get the list, defaulting to empty if it's None or missing
            completed_sections = state.get("completed_sections", []) 
            
            # Handle case where synthesizer might be called unexpectedly with no sections
            if not completed_sections:
                logger.warning("Synthesizer called but 'completed_sections' is empty or None.")
                # Return an empty draft and ensure the sections list is cleared in the state
                return {"initial_draft": "", "completed_sections": []} 
            
            # Determine the expected number of sections based on the current plan
            expected_section_count = len(state.get("sections", []))

            # If we received more sections than expected (likely due to revision state issue),
            # take only the last 'expected_section_count' sections.
            if expected_section_count > 0 and len(completed_sections) > expected_section_count:
                logger.warning(f"Synthesizer received {len(completed_sections)} sections, "
                               f"but expected {expected_section_count}. Using the last {expected_section_count}.")
                sections_to_use = completed_sections[-expected_section_count:]
            else:
                # Otherwise, use all received sections (normal first run or correct state)
                sections_to_use = completed_sections
            
            logger.info(f"Synthesizing report with sections: {completed_sections}")

            logger.info(f"Synthesizing report with {len(sections_to_use)} sections:")
            logger.info("SYNTHESIZER DEBUG:")
            logger.info(f"completed_sections count: {len(completed_sections)}")
            for i, section in enumerate(sections_to_use):
                # Log only the first few characters to avoid overly long logs
                logger.info(f"Section {i+1} (start): {section[:100]}...")
                logger.info(f"{'='*20}")


            # Join the selected sections to create the draft
            initial_draft = "\n\n---\n\n".join(sections_to_use)
            logger.info(f"Synthesized report draft generated (length: {len(initial_draft)}).")

            # Return the generated draft AND explicitly return an empty list
            # for completed_sections to update the state, clearing the old sections.
            return {
                "initial_draft": initial_draft,
                "completed_sections": []  # Explicitly clear the list in the returned state update
            }
    
    @log_entry_exit
    def feedback_collector(self, state: State) -> dict:
        logger.info(f"\n\n----------------:Entered feedback_collector with state:----------------------\n\n{state}")
        logger.info(f"Message count: {len(state.get('messages', []))}")
        logger.info(f"Last message type: {type(state['messages'][-1]) if state.get('messages') else 'None'}")
        
        if state.get("messages") and len(state["messages"]) > 0 and isinstance(state["messages"][-1], HumanMessage):
            try:
                feedback_data = json.loads(state["messages"][-1].content)
                is_approved = feedback_data.get("approved", False)
                comments = feedback_data.get("comments", "")
                logger.info(f"Parsed feedback: approved={is_approved}, comments={comments}")

                if is_approved:
                    logger.info("Content approved, preparing final report")
                    final_report = state.get("initial_draft", "")
                    collector_output = {
                        "feedback": comments,
                        "draft_approved": True,
                        "final_report": final_report
                    }
                else:
                    collector_output = {
                        "feedback": comments,
                        "draft_approved": False,
                        "final_report": ""
                    }
                logger.info(f"{'='*20}:feedback_collector output:{'='*20}\n{collector_output}") # Add this log
                return collector_output

            except json.JSONDecodeError:
                logger.warning("Invalid feedback format; returning default values")
                return {"feedback": "", "draft_approved": False, "final_report": ""}

        logger.info("No new feedback message found; returning default values")
        return {"feedback": "", "draft_approved": False, "final_report": ""}
    @log_entry_exit
    def file_generator(self, state: State) -> dict:
        """Generates the final report and ends the process."""
        final_report = state["final_report"]
        # In a real scenario, you would save this to a file
        logger.info(f"Final Report Generated:\n{final_report}")
        return {"final_report_path": "report.md"} # Simulate saving to a file

    @log_entry_exit # Conditional edge function to create llm_call workers
    def assign_workers(self, state: State):
        """Assign a worker to each section in the plan."""
        logger.info(f"\n{'='*10} State before assigning workers {'='*10}")
        logger.info(f"  Current sections plan: {len(state.get('sections', []))} sections")
        # Log the completed_sections list specifically
        logger.info(f"  Completed Sections before dispatch: {state.get('completed_sections', [])}")
        logger.info(f"{'='*40}\n")
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    @log_entry_exit# Conditional edge for feedback loop
    def route_feedback(self, state: State):
        """Route based on whether draft is approved."""
        draft_approved = state.get('draft_approved', False)
        logger.info(f"route_feedback: draft_approved = {draft_approved}")
        
        if draft_approved is True:  # Strict comparison
            logger.info("Draft approved; routing to file_generator")
            return "file_generator"
        else:
            logger.info("Draft not approved; routing back to orchestrator for revision")
            return "orchestrator"

