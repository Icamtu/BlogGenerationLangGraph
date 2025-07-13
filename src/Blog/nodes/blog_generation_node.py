from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.Blog.state.state import BlogState as State, Sections, Section
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
import json
from datetime import datetime
from typing import List
from src.Blog.logging.logging_utils import logger, log_entry_exit

class BlogGenerationNode:
    def __init__(self, model):
        self.llm = model
        self.planner = model.with_structured_output(Sections)

    def validate_and_standardize_structure(self, user_input: str) -> List[str]:
        default_structure = ["Introduction", "Main Content", "Conclusion"]
        if not user_input or not user_input.strip():
            return default_structure
        for line in user_input.splitlines():
            if line.lower().startswith("structure:"):
                structure_value = line.split(":", 1)[1].strip()
                user_sections = [s.strip().title() for s in structure_value.split(",") if s.strip()]
                return user_sections if user_sections else default_structure
        return default_structure

    @log_entry_exit
    def user_input(self, state: State) -> dict:
        requirements = {
            "topic": state.get("topic", "No topic provided"),
            "objective": state.get("objective", "Informative"),
            "target_audience": state.get("target_audience", "General Audience"),
            "tone_style": state.get("tone_style", "Casual"),
            "word_count": state.get("word_count", 1000),
            "structure": state.get("structure", "Introduction, Main Content, Conclusion"),
            "feedback": state.get("feedback", "No feedback provided yet."),
            "initial_draft": "",
            "completed_sections": []
        }

        user_message = state["messages"][-1].content if state.get("messages") else ""
        is_feedback = False

        try:
            feedback_data = json.loads(user_message)
            if isinstance(feedback_data, dict) and "approved" in feedback_data:
                requirements["feedback"] = feedback_data.get("comments", "No feedback provided.")
                is_feedback = True
        except Exception:
            for line in user_message.split("\n"):
                if ": " in line:
                    key, value = line.split(": ", 1)
                    key = key.lower().replace(" & ", "_").replace(" ", "_")
                    requirements[key] = value.strip()

        structure_input = requirements["structure"] if is_feedback else user_message
        standardized_structure = self.validate_and_standardize_structure(structure_input)
        requirements["structure"] = ", ".join(standardized_structure)
        return requirements

    @log_entry_exit
    def orchestrator(self, state: State) -> dict:
        return_state = {
            "sections": [],
            "completed_sections": [],
            "initial_draft": "",
            "title": ""
        }

        needs_revision = False
        if state.get("messages"):
            last_message_content = state["messages"][-1].content
            try:
                feedback_data = json.loads(last_message_content)
                if isinstance(feedback_data, dict) and not feedback_data.get("approved", True):
                    needs_revision = True
            except Exception:
                pass

        if needs_revision:
            return_state["completed_sections"] = []

        structure_input = state.get("structure", "")
        structure_list = [s.strip().title() for s in structure_input.split(",") if s.strip()]

        if structure_list == ["Introduction", "Main Content", "Conclusion"]:
            logger.info("Default structure detected. Invoking LLM to infer better structure.")
            sys_prompt = (
                f"You are a blog planning expert. Based on the topic '{state['topic']}', "
                f"objective '{state['objective']}', audience '{state['target_audience']}', "
                f"and tone '{state['tone_style']}', propose a clear, relevant structure as section names.\n"
                f"Word count: {state['word_count']} should be considered.\n"
                f"Return a JSON like: {{\"sections\": [\"Intro\", \"Benefits\", \"Future\"]}}."
            )
            try:
                infer_response = self.llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content="Please suggest blog structure sections.")
                ])
                infer_json = json.loads(infer_response.content if hasattr(infer_response, "content") else str(infer_response))
                inferred_sections = infer_json.get("sections", structure_list)
                structure_list = [s.strip().title() for s in inferred_sections if s.strip()]
                logger.info(f"Inferred structure from LLM: {structure_list}")
            except Exception as e:
                logger.warning(f"LLM inference failed; using fallback structure. Error: {e}")

        prompt = (
            f"Create a structured plan for a blog with {len(structure_list)} sections.\n"
            f"Topic: '{state['topic']}'\n"
            f"Objective: {state['objective']}\n"
            f"Target Audience: {state['target_audience']}\n"
            f"Tone: {state['tone_style']}\n"
            f"Word Count: ~{state['word_count']}\n"
            f"Structure: {', '.join(structure_list)}"
        )

        try:
            report_sections = self.planner.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=f"Topic: {state['topic']}")
            ])
            return_state["sections"] = report_sections.sections
        except Exception as e:
            logger.error(f"Error generating plan with LLM: {e}")

        try:
            title_prompt = (
                f"You are a professional blog writer. Based on the topic '{state['topic']}', audience '{state['target_audience']}', "
                f"and objective '{state['objective']}', suggest a clear, attention-grabbing blog title. Return only the title."
            )
            title_response = self.llm.invoke([
                SystemMessage(content=title_prompt),
                HumanMessage(content="Generate a blog title.")
            ])
            title_str = title_response.content if hasattr(title_response, "content") else str(title_response)
            return_state["title"] = title_str.strip().strip('\"\'')
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")

        return return_state

    @log_entry_exit
    def llm_call(self, state: State) -> dict:
        section = self.llm.invoke([
            SystemMessage(content="Write a report section using markdown. Do not include titles or intros."),
            HumanMessage(content=f"Section: {state['section'].name}\nDescription: {state['section'].description}")
        ])
        return {"completed_sections": state.get("completed_sections", []) + [section.content]}

    @log_entry_exit
    def synthesizer(self, state: State) -> dict:
        completed_sections = state.get("completed_sections", [])
        if not completed_sections:
            return {"initial_draft": "", "completed_sections": []}
        expected = len(state.get("sections", []))
        sections_to_use = completed_sections[-expected:] if expected and len(completed_sections) > expected else completed_sections
        initial_draft = "\n\n---\n\n".join(sections_to_use)
        return {"initial_draft": initial_draft, "completed_sections": []}

    @log_entry_exit
    def feedback_collector(self, state: State) -> dict:
        if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
            try:
                feedback_data = json.loads(state["messages"][-1].content)
                is_approved = feedback_data.get("approved", False)
                comments = feedback_data.get("comments", "")
                return {
                    "feedback": comments,
                    "draft_approved": is_approved,
                    "final_report": state.get("initial_draft", "") if is_approved else ""
                }
            except json.JSONDecodeError:
                pass
        return {"feedback": "", "draft_approved": False, "final_report": ""}

    @log_entry_exit
    def final_report(self, state: State) -> dict:
        if state.get("draft_approved", False):
            return {"final_report": state.get("initial_draft", "")}
        try:
            sys_prompt = (
                "Revise the blog based on the provided feedback. Ensure tone, style, and structure are preserved."
            )
            user_prompt = (
                f"Draft: {state.get('initial_draft', '')}\n"
                f"Feedback: {state.get('feedback', '')}"
            )
            revised_draft = self.llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])
            return {"final_report": revised_draft.content if hasattr(revised_draft, "content") else str(revised_draft)}
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            return {"final_report": ""}

    @log_entry_exit
    def file_generator(self, state: State) -> dict:
        return {"final_report_path": "report.md"}

    @log_entry_exit
    def assign_workers(self, state: State):
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    @log_entry_exit
    def route_feedback(self, state: State):
        return "file_generator" if state.get("draft_approved", False) else "orchestrator"
