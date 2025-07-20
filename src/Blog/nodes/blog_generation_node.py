from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.Blog.state.state import BlogState as State, Sections, Section
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

        user_structure_str = next(
            (line.split(":", 1)[1].strip()
             for line in user_input.split("\n")
             if line.lower().startswith("structure:")),
            None,
        )

        if user_structure_str:
            return [s.strip().capitalize() for s in user_structure_str.split(",") if s.strip()]

        system_prompt = (
            "Given user input, extract a clear list of blog section names as JSON: "
            '{"sections": ["Section1", "Section2", "..."]} Capitalize each. Propose a logical structure if needed.'
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            result = json.loads(getattr(response, "content", str(response)))
            sections = result.get("sections", [])
            return [s.strip().capitalize() for s in sections if s.strip()] or default_structure
        except Exception as e:
            logger.warning(f"LLM error or bad JSON: {e}")
            return default_structure

    @log_entry_exit
    def user_input(self, state: State) -> dict:
        """Handle user input, distinguishing between initial requirements and feedback."""
        logger.info(f"Executing user_input with state keys: {list(state.keys())}")
        logger.info(f"Current state values - topic: {state.get('topic')}, audience: {state.get('target_audience')}, tone: {state.get('tone_style')}")

        # Get the latest message
        user_message = ""
        if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
            user_message = state["messages"][-1].content
        
        if not user_message:
            logger.warning("No new user message found in state")
            return {
                "topic": state.get("topic", "No topic provided"),
                "objective": state.get("objective", "Informative"),
                "target_audience": state.get("target_audience", "General Audience"),
                "tone_style": state.get("tone_style", "Casual"),
                "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
                "feedback": state.get("feedback", "No feedback provided yet."),
                "initial_draft": "", "completed_sections": [], "sections": [],
                "draft_approved": False, "final_report": "", "blog_title": ""
            }

        try:
            # ✅ CHECK IF THIS IS FEEDBACK JSON FIRST
            feedback_data = json.loads(user_message)
            if isinstance(feedback_data, dict) and "approved" in feedback_data:
                # ✅ THIS IS FEEDBACK - PRESERVE ALL EXISTING STATE
                logger.info(f"Processing feedback - preserving existing state")
                feedback_comments = feedback_data.get("comments", "No feedback provided.")
                
                # ✅ RETURN EXISTING STATE WITH ONLY FEEDBACK UPDATED
                preserved_state = {
                    "topic": state.get("topic", "No topic provided"),
                    "objective": state.get("objective", "Informative"), 
                    "target_audience": state.get("target_audience", "General Audience"),
                    "tone_style": state.get("tone_style", "Casual"),
                    "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
                    "feedback": feedback_comments,
                    "initial_draft": state.get("initial_draft", ""),
                    "completed_sections": [],
                    "sections": [],
                    "draft_approved": False,
                    "final_report": "",
                    "blog_title": state.get("blog_title", "")
                }
                
                logger.info(f"Preserved state for feedback - topic: {preserved_state['topic']}, audience: {preserved_state['target_audience']}")
                return preserved_state
                
        except json.JSONDecodeError:
            # This is NOT feedback JSON - treat as new requirements
            logger.info("User message is not JSON feedback; treating as initial requirements.")
            pass
        
        # ✅ PROCESS NEW REQUIREMENTS (NOT FEEDBACK)
        temp_requirements = {}
        for line in user_message.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                temp_requirements[key.lower().replace(" & ", "_").replace(" ", "_")] = value

        # Build new requirements
        new_requirements = {
            "topic": temp_requirements.get("topic", "No topic provided"),
            "objective": temp_requirements.get("objective", "Informative"),
            "target_audience": temp_requirements.get("target_audience", "General Audience"),
            "tone_style": temp_requirements.get("tone_style", "Casual"),
            "structure": temp_requirements.get("structure", ""),
            "feedback": temp_requirements.get("feedback", ""),
        }
        
        logger.info(f"Processed new requirements: {new_requirements}")

        # Validate and standardize structure for new requirements only
        if new_requirements["structure"]:
            standardized_structure = [s.strip().capitalize() for s in new_requirements["structure"].split(",") if s.strip()]
        else:
            standardized_structure = self.validate_and_standardize_structure(user_message)
        
        new_requirements["structure"] = ", ".join(standardized_structure)

        return {
            **new_requirements,
            "initial_draft": "", "completed_sections": [], "sections": [],
            "draft_approved": False, "final_report": "", "blog_title": ""
        }



    @log_entry_exit
    def orchestrator(self, state: State) -> dict:
        logger.info(f"Executing orchestrator with state: {state}")

        needs_revision = False
        if state.get("messages") and state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            last_message_content = state["messages"][-1].content
            try:
                feedback_data = json.loads(last_message_content)
                if isinstance(feedback_data, dict) and feedback_data.get("approved") is False:
                    needs_revision = True
                    logger.info("Orchestrator identified revision cycle triggered by disapproved feedback.")
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.warning(f"Error checking last message for revision trigger in orchestrator: {e}")

        if needs_revision:
            logger.info("Orchestrator clearing 'completed_sections' due to revision.")
            return_completed_sections = []
        else:
            return_completed_sections = []

        structure_list = [s.strip() for s in state["structure"].split(",") if s.strip()]
        if not structure_list:
            logger.error("Structure list is empty, falling back to default.")
            structure_list = ["Introduction", "Main Content", "Conclusion"]

        section_count = len(structure_list)
        feedback_str = state.get("feedback", "No specific feedback provided for this iteration.")

        title_system_prompt = (
            f"You are an expert blog title generator. Create a catchy, engaging title for a blog post based on the topic: '{state['topic']}'. "
            f"The objective is: {state['objective']}. Target audience: {state['target_audience']}. Tone: {state['tone_style']}. "
            f"Return only the title as a plain string, no additional text."
        )

        try:
            title_response = self.llm.invoke([
                SystemMessage(content=title_system_prompt),
                HumanMessage(content=f"Generate title for topic: {state['topic']}")
            ])
            blog_title = title_response.content.strip()
            logger.info(f"Generated blog title: {blog_title}")
        except Exception as e:
            logger.error(f"Error generating blog title: {e}")
            blog_title = state['topic']

        system_prompt_planner = (
            f"You are an expert blog post planner. Your task is to create a detailed plan for a blog post based on the following requirements. "
            f"The blog post must consist of exactly {section_count} sections. "
            f"**Topic:** '{state['topic']}'\n"
            f"**Objective:** To {state['objective']}\n"
            f"**Target Audience:** {state['target_audience']}\n"
            f"**Tone & Style:** {state['tone_style']}\n"
            f"**Required Section Names and Order:** {', '.join(structure_list)}\n"
            f"**Feedback for Revision (if any):** {feedback_str}\n\n"
            f"For EACH required section, provide a concise but highly descriptive `description` that outlines the specific content, "
            f"key points, and focus for that section, ensuring it is directly relevant to the blog's topic and respects the feedback. "
            f"The `description` should be detailed enough for another AI to write the section without further prompting about its scope or content. "
            f"Ensure the generated sections strictly adhere to the provided section names and count. DO NOT add extra sections. "
            f"Return the plan as a JSON object strictly conforming to the `Sections` Pydantic schema."
        )

        human_message_planner = f"Please generate the detailed section plan for a blog post on '{state['topic']}', incorporating the feedback: '{feedback_str}'."

        try:
            report_sections_obj: Sections = self.planner.invoke([
                SystemMessage(content=system_prompt_planner),
                HumanMessage(content=human_message_planner)
            ])

            if not isinstance(report_sections_obj, Sections) or not report_sections_obj.sections:
                logger.error("Planner returned an invalid or empty Sections object.")
                planned_sections = []
            else:
                planned_sections = report_sections_obj.sections

            expected_names = [s.strip().capitalize() for s in state["structure"].split(",") if s.strip()]
            planned_descriptions_map = {sec.name.capitalize(): sec.description for sec in planned_sections}

            final_sections_for_state = []
            for name in expected_names:
                description = planned_descriptions_map.get(name, f"Write a detailed section on {name} for a blog post about {state['topic']}.")
                final_sections_for_state.append(Section(name=name, description=description))

            logger.info(f"Orchestrator successfully planned {len(final_sections_for_state)} sections.")
            for s in final_sections_for_state:
                logger.info(f" - Section: {s.name}, Description: {s.description[:100]}...")

            return {
                "sections": final_sections_for_state,
                "completed_sections": return_completed_sections,
                "blog_title": blog_title
            }

        except Exception as e:
            logger.error(f"Error generating plan with LLM in orchestrator: {e}")
            return {
                "sections": [],
                "completed_sections": return_completed_sections,
                "blog_title": state['topic']
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
            "topic", "objective", "target_audience", "tone_style",
            "structure", "feedback", "messages", "blog_title"
        ]

        send_list = []
        for s in sections:
            if not hasattr(s, "name"):
                logger.error(f"Found non-Section object in state['sections']: {s}. Skipping.")
                continue

            worker_state = {"section": s}
            for k in passthrough_keys:
                if k in state:
                    worker_state[k] = state[k]

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

            content = section_content.content.strip()
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                if line.strip() and not (
                    line.strip() == current_section.name or
                    line.strip().startswith('#') and current_section.name.lower() in line.lower()
                ):
                    cleaned_lines.append(line)
                elif line.strip() and line.strip().startswith('#'):
                    if current_section.name.lower() not in line.lower():
                        cleaned_lines.append(line)

            cleaned_content = '\n'.join(cleaned_lines).strip()
            formatted_section = f"## {current_section.name}\n\n{cleaned_content}"

            logger.info(f"\n{'='*20}:llm_call output for '{current_section.name}':{'='*20}\nGenerated content (first 200 chars): {formatted_section[:200]}...\n{'='*20}\n")

            return {"completed_sections": [formatted_section]}

        except Exception as e:
            logger.error(f"Error generating section '{current_section.name}' with LLM: {e}")
            return {"completed_sections": [f"## {current_section.name}\n\nError generating section '{current_section.name}': {e}"]}

    @log_entry_exit
    def synthesizer(self, state: State) -> dict:
        """Synthesize full report from sections and clear the sections list."""
        completed_sections = state.get("completed_sections", [])
        if not completed_sections:
            logger.warning("Synthesizer called but 'completed_sections' is empty or None.")
            return {"initial_draft": "", "completed_sections": []}

        expected_section_count = len(state.get("sections", []))
        sections_to_use = completed_sections

        if expected_section_count > 0 and len(completed_sections) > expected_section_count:
            logger.warning(f"Synthesizer received {len(completed_sections)} sections, "
                         f"but expected {expected_section_count}. Attempting to use the most recent sections.")
            sections_to_use = completed_sections[-expected_section_count:]
            logger.info(f"Reduced sections for synthesis to {len(sections_to_use)}.")

        logger.info(f"Synthesizing report with {len(sections_to_use)} sections:")
        for i, section_content in enumerate(sections_to_use):
            logger.info(f"Section {i+1} (start): {section_content[:100]}...")

        blog_title = state.get("blog_title", state.get("topic", "Untitled Blog Post"))
        title_header = f"# {blog_title}\n\n"
        sections_content = "\n\n---\n\n".join(sections_to_use)
        initial_draft = title_header + sections_content

        logger.info(f"Synthesized report draft generated with title (length: {len(initial_draft)} characters).")


        original_draft = state.get("original_draft", "")
        draft_version = state.get("draft_version", 0)
        draft_history = state.get("draft_history", [])
        
        # If this is the first generation, save as original
        if not original_draft:
            original_draft = initial_draft
            draft_version = 1
            draft_history = [initial_draft]

        return {
            "initial_draft": initial_draft,
            "original_draft": original_draft,
            "draft_version": draft_version,
            "draft_history": draft_history,
            "completed_sections": [],
            "sections": []
        }

    @log_entry_exit
    def feedback_collector(self, state: State) -> dict:
        logger.info(f"\n\n----------------:Entered feedback_collector with state:----------------------\n\n{state}")

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
                    "final_report": ""
                }

            logger.info(f"{'='*20}:feedback_collector output:{'='*20}\n{collector_output}")
            return collector_output

        except json.JSONDecodeError:
            logger.warning(f"Invalid feedback format. Expected JSON, got: {feedback_message_content[:100]}...")
            return {"feedback": "Invalid feedback format provided.", "draft_approved": False, "final_report": ""}
        except Exception as e:
            logger.error(f"An unexpected error occurred in feedback_collector: {e}")
            return {"feedback": f"Error during feedback collection: {e}", "draft_approved": False, "final_report": ""}

    
    @log_entry_exit
    def file_generator(self, state: State) -> dict:
        """Generates the final report and ends the process."""
        final_report = state.get("final_report", "No final report content available.")
        
        if not final_report:
            logger.warning("file_generator called but 'final_report' is empty.")
            return {"final_report_path": "error_report.md", "final_report": "Error: Final report content missing."}

        file_name = f"blog_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        try:
            logger.info(f"Simulated: Final Report saved to {file_name}")
            logger.info(f"---BEGIN FINAL REPORT---\n{final_report}\n---END FINAL REPORT---")
            return {"final_report_path": file_name}
        except Exception as e:
            logger.error(f"Error saving final report file: {e}")
            return {"final_report_path": "error_saving_report.md", "final_report": final_report}

    @log_entry_exit
    def revise_blog(self, state: State) -> dict:
        """Revise existing draft while preserving original."""
        existing_draft = state.get("initial_draft", "")
        feedback = state.get("feedback", "")
        
        
        original_draft = state.get("original_draft", existing_draft)
        draft_version = state.get("draft_version", 1) + 1
        draft_history = state.get("draft_history", [])
        
        if not existing_draft:
            logger.error("No existing draft for revision - creating error response")
            
            return {
                "initial_draft": "Error: No draft available for revision. Please regenerate the blog.",
                "original_draft": original_draft,
                "draft_version": draft_version,
                "draft_history": draft_history,
                "completed_sections": [],
                "sections": []
            }
        
        # Create revision prompt
        revision_system_prompt = (
            "You are an expert blog editor. Revise the blog post based on specific feedback.\n"
            "CRITICAL: Make targeted revisions while preserving the overall quality and structure.\n"
            "Focus only on addressing the specific feedback concerns.\n\n"
            f"Original context:\n"
            f"- Topic: {state.get('topic', 'Unknown')}\n"
            f"- Objective: {state.get('objective', 'Unknown')}\n"
            f"- Target Audience: {state.get('target_audience', 'Unknown')}\n"
            f"- Tone: {state.get('tone_style', 'Unknown')}\n"
        )
        
        revision_prompt = (
            f"CURRENT BLOG POST TO REVISE:\n{existing_draft}\n\n"
            f"SPECIFIC FEEDBACK TO ADDRESS:\n{feedback}\n\n"
            f"Provide the complete revised blog post:"
        )
        
        try:
            revised_response = self.llm.invoke([
                SystemMessage(content=revision_system_prompt),
                HumanMessage(content=revision_prompt)
            ])
            
            revised_content = revised_response.content.strip()
            
            
            updated_history = draft_history + [revised_content]
            
            logger.info(f"Blog revision v{draft_version} completed")
            
            return {
                "initial_draft": revised_content,
                "original_draft": original_draft,     
                "draft_version": draft_version,       
                "draft_history": updated_history,     
                "completed_sections": [],
                "sections": []
            }
            
        except Exception as e:
            logger.error(f"Error during revision: {e}")
            return {
                "initial_draft": existing_draft,
                "original_draft": original_draft,
                "draft_version": draft_version - 1,
                "draft_history": draft_history,
                "completed_sections": [],
                "sections": []
            }

    @log_entry_exit
    def route_feedback(self, state: State):
        """Route based on whether draft is approved."""
        draft_approved = state.get('draft_approved', False)
        logger.info(f"🔍 ROUTING DEBUG: draft_approved = {draft_approved}")
        
        if draft_approved is True:
            logger.info("✅ ROUTING TO: file_generator")
            return "file_generator"
        else:
            logger.info("🔄 ROUTING TO: revise_blog")
            return "revise_blog"
