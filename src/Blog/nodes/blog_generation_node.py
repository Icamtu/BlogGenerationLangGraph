from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.Blog.state.state import BlogState as State, Sections, Section
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Added AIMessage
import streamlit as st
import json
from datetime import datetime
from typing import List
from src.Blog.logging.logging_utils import logger, log_entry_exit
class BlogGenerationNode:
    def __init__(self, model):
        """Initialize the BlogGenerationNode with an LLM."""
        self.llm = model
        self.planner = model.with_structured_output(Sections)

    @log_entry_exit
    def validate_and_standardize_structure(self, user_input: str) -> List[str]:
        """
        Parse the user input for section structure. If not found, use an LLM to extract/guess structure.
        Always returns a cleaned list of section names.
        """
        default_structure = ["Introduction", "Main Content", "Conclusion"]

        if not user_input or not user_input.strip():
            return default_structure

        # Try to extract structure from user input
        user_structure_str = next(
            (line.split(":", 1)[1].strip()
            for line in user_input.split("\n")
            if line.lower().startswith("structure:")),
            None,
        )

        if user_structure_str:
            # Clean and return user-given structure
            return [s.strip().capitalize() for s in user_structure_str.split(",") if s.strip()]

        # If not structured, ask LLM if available
        system_prompt = (
            "Given user input, extract a clear list of blog section names as JSON: "
            '{"sections": ["Section1", "Section2", "..."]} Capitalize each. Propose a logical structure if needed.'
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            # Try to parse as JSON, otherwise fallback
            result = json.loads(getattr(response, "content", str(response)))
            sections = result.get("sections", [])
            return [s.strip().capitalize() for s in sections if s.strip()] or default_structure
        except Exception as e:
            logger.warning(f"LLM error or bad JSON: {e}")
            return default_structure


    @log_entry_exit
    def user_input(self, state: State) -> dict:
        """Handle user input, distinguishing between initial requirements and feedback."""
        logger.info(f"Executing user_input with state: {state}")

        # Get the latest message, which is expected to be a HumanMessage from Streamlit UI
        user_message = ""
        if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
            user_message = state["messages"][-1].content
        
        if not user_message:
            logger.warning("No new user message found in state; potentially a re-run or initial empty state. Returning current requirements with content reset.")
            # If no user message, return existing requirements but ensure content is reset
            return {
                "topic": state.get("topic", "No topic provided"),
                "objective": state.get("objective", "Informative"),
                "target_audience": state.get("target_audience", "General Audience"),
                "tone_style": state.get("tone_style", "Casual"),
                "word_count": state.get("word_count", 1000),
                "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
                "feedback": state.get("feedback", "No feedback provided yet."),
                "initial_draft": "",
                "completed_sections": [],
                "sections": [], # Clear planned sections too on new input
                "draft_approved": False, # Reset approval status
                "final_report": "" # Clear final report content
            }

        # Initialize requirements with existing state values to preserve them if not overridden
        requirements = {
            "topic": state.get("topic", "No topic provided"),
            "objective": state.get("objective", "Informative"),
            "target_audience": state.get("target_audience", "General Audience"),
            "tone_style": state.get("tone_style", "Casual"),
            "word_count": state.get("word_count", 1000),
            "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
            "feedback": state.get("feedback", "No feedback provided yet."),
        }

        is_feedback = False

        try:
            # Attempt to parse as feedback JSON
            feedback_data = json.loads(user_message)
            if isinstance(feedback_data, dict) and "approved" in feedback_data:
                requirements["feedback"] = feedback_data.get("comments", "No feedback provided.")
                is_feedback = True
                logger.info(f"Processed feedback message: {requirements['feedback']}")
            else:
                # If not feedback, parse as initial requirements input
                temp_requirements = {}
                for line in user_message.split("\n"):
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        # Standardize keys: remove spaces, replace & with _
                        temp_requirements[key.lower().replace(" & ", "_").replace(" ", "_")] = value
                

                # Update requirements only for provided fields, keep existing for others
                requirements.update({
                    "topic": temp_requirements.get("topic", requirements["topic"]),
                    "objective": temp_requirements.get("objective", requirements["objective"]),
                    "target_audience": temp_requirements.get("target_audience", requirements["target_audience"]),
                    "tone_style": temp_requirements.get("tone_style", requirements["tone_style"]),
                    "word_count": int(temp_requirements.get("word_count", requirements["word_count"])),
                    "structure": temp_requirements.get("structure", requirements["structure"]),
                    "feedback": temp_requirements.get("feedback", requirements["feedback"]),
                })
                logger.info(f"Processed initial requirements input: {requirements}")

        except json.JSONDecodeError:
            # If it's not valid JSON feedback, it's treated as a new requirement string
            logger.info("User message is not JSON feedback; treating as initial requirements.")
            temp_requirements = {}
            for line in user_message.split("\n"):
                if ": " in line:
                    key, value = line.split(": ", 1)
                    temp_requirements[key.lower().replace(" & ", "_").replace(" ", "_")] = value

            requirements.update({
                "topic": temp_requirements.get("topic", requirements["topic"]),
                "objective": temp_requirements.get("objective", requirements["objective"]),
                "target_audience": temp_requirements.get("target_audience", requirements["target_audience"]),
                "tone_style": temp_requirements.get("tone_style", requirements["tone_style"]),
                "word_count": int(temp_requirements.get("word_count", requirements["word_count"])),
                "structure": temp_requirements.get("structure", requirements["structure"]),
                "feedback": temp_requirements.get("feedback", requirements["feedback"]), # Keep feedback if not explicitly overridden
            })
            logger.info(f"Processed initial requirements from plain text: {requirements}")

        except Exception as e:
            logger.error(f"Unexpected error processing user message in user_input: {e}")
            # If any other error, keep existing requirements, but ensure content is cleared
            return {
                **requirements, # Keep existing requirements
                "initial_draft": "",
                "completed_sections": [],
                "sections": [], # Clear planned sections too on error
                "draft_approved": False,
                "final_report": ""
            }

        # Validate and standardize structure based on the *latest* input or derived structure
        # For initial input, use the full user_message to extract structure.
        # For feedback, the structure should already be in state, so we use state['structure'] to ensure it persists.
        structure_source_for_validation = user_message if not is_feedback else requirements["structure"]
        standardized_structure = self.validate_and_standardize_structure(structure_source_for_validation)
        requirements["structure"] = ", ".join(standardized_structure)

        # Always reset content and approval status for new input or feedback,
        # forcing re-generation of draft.
        final_return_state = {
            **requirements,
            "initial_draft": "",
            "completed_sections": [],
            "sections": [], # Clear planned sections for regeneration by orchestrator
            "draft_approved": False,
            "final_report": ""
        }

        logger.info(f"Final parsed requirements from user_input: {final_return_state}")
        return final_return_state


    @log_entry_exit
    def orchestrator(self, state: State) -> dict:
        logger.info(f"Executing orchestrator with state: {state}")

        # Check if the last message was a feedback message indicating disapproval
        needs_revision = False
        if state.get("messages") and state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            last_message_content = state["messages"][-1].content
            try:
                feedback_data = json.loads(last_message_content)
                if isinstance(feedback_data, dict) and feedback_data.get("approved") is False:
                    needs_revision = True
                    logger.info("Orchestrator identified revision cycle triggered by disapproved feedback.")
            except json.JSONDecodeError:
                pass # Not a JSON feedback message
            except Exception as e:
                logger.warning(f"Error checking last message for revision trigger in orchestrator: {e}")

        # The 'sections' field in state might still hold old sections if it's not a new run.
        # It's crucial that user_input node clears `state["sections"]` when new requirements come in.
        # If it's a revision, we also want to clear completed sections to start fresh.
        if needs_revision:
            logger.info("Orchestrator clearing 'completed_sections' due to revision.")
            # This will be returned as part of the update.
            return_completed_sections = []
        else:
            # For a fresh run or if no revision, completed_sections should already be empty from user_input.
            # We still explicitly set it to empty to ensure consistency.
            return_completed_sections = []


        # Always re-parse structure string from state to get the latest plan
        structure_list = [s.strip() for s in state["structure"].split(",") if s.strip()]
        if not structure_list:
            logger.error("Structure list is empty, falling back to default.")
            structure_list = ["Introduction", "Main Content", "Conclusion"] # Fallback

        section_count = len(structure_list)
        feedback_str = state.get("feedback", "No specific feedback provided for this iteration.")

        # Construct the detailed prompt for the planner LLM to generate Section objects
        system_prompt_planner = (
            f"You are an expert blog post planner. Your task is to create a detailed plan for a blog post based on the following requirements. "
            f"The blog post must consist of exactly {section_count} sections. "
            f"**Topic:** '{state['topic']}'\n"
            f"**Objective:** To {state['objective']}\n"
            f"**Target Audience:** {state['target_audience']}\n"
            f"**Tone & Style:** {state['tone_style']}\n"
            f"**Approximate Total Word Count:** {state['word_count']} words (distribute content logically across sections).\n"
            f"**Required Section Names and Order:** {', '.join(structure_list)}\n"
            f"**Feedback for Revision (if any):** {feedback_str}\n\n"
            f"For EACH required section, provide a concise but highly descriptive `description` that outlines the specific content, "
            f"key points, and focus for that section, ensuring it is directly relevant to the blog's topic and respects the feedback. "
            f"The `description` should be detailed enough for another AI to write the section without further prompting about its scope or content. "
            f"Ensure the generated sections strictly adhere to the provided section names and count. DO NOT add extra sections. "
            f"Return the plan as a JSON object strictly conforming to the `Sections` Pydantic schema."
        )

        # Human message for the planner
        human_message_planner = f"Please generate the detailed section plan for a blog post on '{state['topic']}', incorporating the feedback: '{feedback_str}'."


        try:
            # Invoke the planner LLM to get structured Section objects
            report_sections_obj: Sections = self.planner.invoke([
                SystemMessage(content=system_prompt_planner),
                HumanMessage(content=human_message_planner)
            ])
            
            # Ensure the result is a Sections object and has a list of Section instances
            if not isinstance(report_sections_obj, Sections) or not report_sections_obj.sections:
                logger.error("Planner returned an invalid or empty Sections object.")
                planned_sections = []
            else:
                planned_sections = report_sections_obj.sections

            # Critical validation: Ensure planned sections match requested structure names
            # We want to use the names the user (or validate_and_standardize_structure) specified
            # but enrich them with descriptions from the LLM.
            
            # Get the exact names from the state (standardized by user_input)
            expected_names = [s.strip().capitalize() for s in state["structure"].split(",") if s.strip()]
            
            # Create a dictionary for quick lookup of planned descriptions by name
            planned_descriptions_map = {sec.name.capitalize(): sec.description for sec in planned_sections}
            
            # Reconstruct the final list of Section objects using the *expected names*
            # and filling in descriptions from the planner's output, or defaulting if not found.
            final_sections_for_state = []
            for name in expected_names:
                description = planned_descriptions_map.get(name, f"Write a detailed section on {name} for a blog post about {state['topic']}.")
                final_sections_for_state.append(Section(name=name, description=description))

            logger.info(f"Orchestrator successfully planned {len(final_sections_for_state)} sections.")
            for s in final_sections_for_state:
                logger.info(f" - Section: {s.name}, Description: {s.description[:100]}...") # Log first 100 chars of desc

            return {
                "sections": final_sections_for_state,
                "completed_sections": return_completed_sections # Will be empty if revision, or initially empty
            }

        except Exception as e:
            logger.error(f"Error generating plan with LLM in orchestrator: {e}")
            # If an error occurs, ensure sections are cleared to avoid infinite loops with invalid data
            return {
                "sections": [],
                "completed_sections": return_completed_sections
            }
    
    @log_entry_exit
    def assign_workers(self, state: State):
        """Assign a worker to each section in the plan, passing full context."""
        logger.info(f"\n{'='*10} State before assigning workers {'='*10}")
        logger.info(f" Current sections plan: {len(state.get('sections', []))} sections")
        logger.info(f" Completed Sections before dispatch: {state.get('completed_sections', [])}")
        logger.info(f"{'='*40}\n")

        sections = state.get("sections", [])
        if not sections:
            logger.warning("No sections found to assign workers. Skipping parallel calls.")
            return []   

        
        passthrough_keys = [
            "topic", "objective", "target_audience", "tone_style", "word_count",
            "structure", "feedback", "messages"
        ]

        send_list = []
        for s in sections:
            if not hasattr(s, "name"):
                logger.error(f"Found non-Section object in state['sections']: {s}. Skipping.")
                continue
            # Compose a new state dict for each worker, merging the section and full context needed
            worker_state = {"section": s}
            for k in passthrough_keys:
                if k in state:
                    worker_state[k] = state[k]
            # Always carry completed_sections so add_messages works
            worker_state["completed_sections"] = state.get("completed_sections", [])
            send_list.append(Send("llm_call", worker_state))
        
        logger.info(f"Assigning {len(send_list)} llm_call workers.")
        return send_list

    @log_entry_exit
    def llm_call(self, state: State) -> dict:
        """Worker writes a section of the report."""
        if "section" not in state or not isinstance(state["section"], Section):
            logger.error("llm_call received invalid or missing 'section' in state.")
            return {"completed_sections": state.get("completed_sections", [])}

        current_section: Section = state["section"]
        
        system_message_content = (
            "You are an expert content writer for a blog. Your task is to write a single section of a blog post "
            "based on the provided section name and detailed description. "
            "IMPORTANT: Do NOT include the section title/heading in your response. Do NOT start with the section name. "
            "Do NOT include any preambles (like 'Introduction to...', 'This section will cover...', 'Section X: Title'). "
            "Just write the actual content for this section directly. "
            "The section title will be added separately by the system. "
            "Use clear, concise language appropriate for the target audience. "
            "Format the content using Markdown for emphasis, lists, etc., but do NOT include any headers (#, ##, ###). "
            f"The overall blog topic is: {state.get('topic', 'UNKNOWN')}. The objective is: {state.get('objective', 'UNKNOWN')}. "
            f"The target audience is: {state.get('target_audience', 'UNKNOWN')}. The tone and style should be: {state.get('tone_style', 'UNKNOWN')}."
        )
        
        human_message_content = (
            f"Write the content for the section that will be titled '{current_section.name}'. "
            f"Here is the detailed description for this section's content: '{current_section.description}' "
            f"Remember: Do NOT include the section title '{current_section.name}' in your response. Start directly with the content."
        )

        try:
            section_content = self.llm.invoke([
                SystemMessage(content=system_message_content),
                HumanMessage(content=human_message_content)
            ])
            
            # Clean the content to remove any accidentally generated headers
            content = section_content.content.strip()
            
            # Remove any headers that might have been generated by the LLM
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip lines that are just the section name or header versions of it
                if line.strip() and not (
                    line.strip() == current_section.name or 
                    line.strip().startswith('#') and current_section.name.lower() in line.lower()
                ):
                    cleaned_lines.append(line)
                elif line.strip() and line.strip().startswith('#'):
                    # If it's a header but not the main section title, keep it (sub-headers are ok)
                    if current_section.name.lower() not in line.lower():
                        cleaned_lines.append(line)
            
            # Join the cleaned content
            cleaned_content = '\n'.join(cleaned_lines).strip()
            
            # Now add the section header properly
            formatted_section = f"## {current_section.name}\n\n{cleaned_content}"
            
            logger.info(f"\n{'='*20}:llm_call output for '{current_section.name}':{'='*20}\nGenerated content (first 200 chars): {formatted_section[:200]}...\n{'='*20}\n")
            
            return {"completed_sections": [formatted_section]}
            
        except Exception as e:
            logger.error(f"Error generating section '{current_section.name}' with LLM: {e}")
            return {"completed_sections": [f"## {current_section.name}\n\nError generating section '{current_section.name}': {e}"]}


        except Exception as e:
            logger.error(f"Error generating section '{current_section.name}' with LLM: {e}")
            # Return current completed_sections to avoid data loss, or an error message
            return {"completed_sections": [f"Error generating section '{current_section.name}': {e}"]}

    @log_entry_exit
    def synthesizer(self, state: State) -> dict:
        """Synthesize full report from sections and clear the sections list."""
        completed_sections = state.get("completed_sections", [])

        if not completed_sections:
            logger.warning("Synthesizer called but 'completed_sections' is empty or None.")
            return {"initial_draft": "", "completed_sections": []}

        # Determine the expected number of sections based on the current plan
        # This helps in case of multiple runs and sections accumulating unintentionally.
        expected_section_count = len(state.get("sections", []))

        sections_to_use = completed_sections
        if expected_section_count > 0 and len(completed_sections) > expected_section_count:
             # This scenario indicates residual sections from previous runs or over-generation.
             # If `user_input` correctly clears `completed_sections`, this shouldn't happen much.
             # However, it's safer to ensure we're using only the *relevant* last set of sections.
            logger.warning(f"Synthesizer received {len(completed_sections)} sections, "
                            f"but expected {expected_section_count}. Attempting to use the most recent sections.")
            # Use only the sections generated from the *current* orchestrator run.
            # This assumes llm_call appends one by one.
            sections_to_use = completed_sections[-expected_section_count:]
            logger.info(f"Reduced sections for synthesis to {len(sections_to_use)}.")


        logger.info(f"Synthesizing report with {len(sections_to_use)} sections:")
        for i, section_content in enumerate(sections_to_use):
            logger.info(f"Section {i+1} (start): {section_content[:100]}...")

        # Join the selected sections to create the draft
        initial_draft = "\n\n---\n\n".join(sections_to_use)
        logger.info(f"Synthesized report draft generated (length: {len(initial_draft)} characters).")

        # Return the generated draft AND explicitly clear completed_sections for the next cycle.
        return {
            "initial_draft": initial_draft,
            "completed_sections": [],  # Explicitly clear the list in the state update
            "sections": [] # Also clear the planned sections
        }


            
    @log_entry_exit
    def feedback_collector(self, state: State) -> dict:
        logger.info(f"\n\n----------------:Entered feedback_collector with state:----------------------\n\n{state}")
        
        # Look for the *latest* HumanMessage to act as feedback
        feedback_message_content = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                feedback_message_content = msg.content
                break

        if not feedback_message_content:
            logger.warning("No new feedback message found; returning default values")
            return {"feedback": "", "draft_approved": False, "final_report": ""}

        try:
            feedback_data = json.loads(feedback_message_content)
            is_approved = feedback_data.get("approved", False)
            comments = feedback_data.get("comments", "")
            logger.info(f"Parsed feedback: approved={is_approved}, comments={comments}")

            if is_approved:
                logger.info("Content approved, preparing final report")
                final_report = state.get("initial_draft", "No draft available for final report.")
                collector_output = {
                    "feedback": comments,
                    "draft_approved": True,
                    "final_report": final_report
                }
            else:
                logger.info("Content NOT approved, requesting revision.")
                collector_output = {
                    "feedback": comments,
                    "draft_approved": False,
                    "final_report": "" # Clear final report if not approved
                }
            logger.info(f"{'='*20}:feedback_collector output:{'='*20}\n{collector_output}")
            return collector_output

        except json.JSONDecodeError:
            logger.warning(f"Invalid feedback format. Expected JSON, got: {feedback_message_content[:100]}...")
            # If invalid format, assume not approved and set generic feedback for potential revision
            return {"feedback": "Invalid feedback format provided.", "draft_approved": False, "final_report": ""}
        except Exception as e:
            logger.error(f"An unexpected error occurred in feedback_collector: {e}")
            return {"feedback": f"Error during feedback collection: {e}", "draft_approved": False, "final_report": ""}
    
    @log_entry_exit
    def route_feedback(self, state: State):
        """Route based on whether draft is approved."""
        draft_approved = state.get('draft_approved', False)
        logger.info(f"route_feedback: draft_approved = {draft_approved}")

        if draft_approved is True:
            logger.info("Draft approved; routing to file_generator")
            return "file_generator"
        else:
            logger.info("Draft not approved; routing back to orchestrator for revision")
            return "orchestrator"


    @log_entry_exit
    def file_generator(self, state: State) -> dict:
        """Generates the final report and ends the process."""
        final_report = state.get("final_report", "No final report content available.")
        if not final_report:
            logger.warning("file_generator called but 'final_report' is empty.")
            return {"final_report_path": "error_report.md", "final_report": "Error: Final report content missing."}

        # Simulate saving to a file
        file_name = f"blog_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        try:
            # In a real Streamlit app, you might make this downloadable or display it
            # For this example, we'll just log it.
            # with open(file_name, "w", encoding="utf-8") as f:
            #     f.write(final_report)
            logger.info(f"Simulated: Final Report saved to {file_name}")
            logger.info(f"---BEGIN FINAL REPORT---\n{final_report}\n---END FINAL REPORT---")
            return {"final_report_path": file_name}
        except Exception as e:
            logger.error(f"Error saving final report file: {e}")
            return {"final_report_path": "error_saving_report.md", "final_report": final_report}