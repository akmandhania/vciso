from typing import List, Dict, Optional, TypedDict
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os
import logging
import json

# Set up logger for debug output
logger = logging.getLogger("vciso.part1")
logging.basicConfig(level=logging.INFO)

MAX_TOKENS = 1024  # Limit for TavilySearch and any LLM calls
MAX_INTERACTIONS = 30  # Max number of user interactions per session

SYSTEM_PROMPT = """
You are an expert security consultant. Your job is to help the user build a comprehensive incident response plan (IRP) by asking them relevant questions, one at a time.
- After each user answer, ask the next most relevant question for the current section.
- When the user responds, ask: 'Is there anything else you'd like to add about [Section], or should we move on?'
- Only move to the next section if the user says to move on.
- The main sections to cover, in order, are: Preparation, Detection, Containment, Eradication, Recovery, Post-Mortem Analyses.
- For each response, output a JSON object with two fields: 'current_section' (one of the main sections) and 'message' (your reply to the user).
- Example output: {{"current_section": "Preparation", "message": "Can you describe your current preparation efforts? ..."}}
- Do not repeat the same question or explanation more than once.
- If the user does not know, suggest what information is typically needed, then proceed.
- Offer to explain any section if the user asks.
- Do not ask more than one question at a time.
"""

OVERVIEW_PROMPT = """
You are an expert security consultant. Please provide a clear, human-readable summary or overview based on the user's responses so far. Do not use JSON or any structured output—just write a helpful summary in plain English.
"""

class IRPBuilderState(TypedDict):
    """State for the IRP builder graph node."""
    chat_history: List[Dict[str, str]]
    answer: str
    last_plan: str
    current_section: str
    covered_sections: List[str]

class IRPChatHandler:
    def __init__(self, llm, search_tool, state):
        self.llm = llm
        self.search_tool = search_tool
        self.state = state

    def handle_web_search(self, message):
        query = message[7:].strip()
        results = self.search_tool.invoke({'query': query})
        result_texts = [r['content'][:MAX_TOKENS] for r in results]
        return f"Web search results for '{query}':\n" + '\n'.join(result_texts)

    def handle_progress(self):
        covered = self.state['covered_sections']
        remaining = [s for s in self.state['main_sections'] if s not in covered]
        return (
            f"Sections covered so far: {', '.join(covered) if covered else 'None'}\n"
            f"Sections remaining: {', '.join(remaining) if remaining else 'None'}\n"
            "Type 'overview' for a summary, 'download' to save your plan, or 'review' to see your plan so far."
        )

    def handle_overview(self, message, history):
        chat_history = []
        if history:
            for msg in history:
                chat_history.append({"role": msg["role"], "content": msg["content"]})
        chat_history.append({"role": "user", "content": message})
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        ])
        prompt = OVERVIEW_PROMPT + "\n\nCurrent section: " + self.state['current_section'] + "\nSections covered: " + ", ".join(self.state['covered_sections']) + "\n\nChat history so far:\n" + history_str + "\n\nPlease provide an overview of what is needed to create or update my incident response plan, based on my responses."
        overview = self.llm.invoke(prompt)
        if hasattr(overview, "content"):
            overview = overview.content
        return overview

    def handle_review(self, message, history):
        chat_history = []
        if history:
            for msg in history:
                chat_history.append({"role": msg["role"], "content": msg["content"]})
        chat_history.append({"role": "user", "content": message})
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        ])
        prompt = OVERVIEW_PROMPT + "\n\nCurrent section: " + self.state['current_section'] + "\nSections covered: " + ", ".join(self.state['covered_sections']) + "\n\nChat history so far:\n" + history_str + "\n\nPlease summarize the incident response plan we have built so far."
        summary = self.llm.invoke(prompt)
        if hasattr(summary, "content"):
            summary = summary.content
        self.state['last_plan'] = summary
        return summary

    def handle_download(self, message, history, create_download_func):
        chat_history = []
        if history:
            for msg in history:
                chat_history.append({"role": msg["role"], "content": msg["content"]})
        chat_history.append({"role": "user", "content": "review"})
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        ])
        prompt = OVERVIEW_PROMPT + "\n\nCurrent section: " + self.state['current_section'] + "\nSections covered: " + ", ".join(self.state['covered_sections']) + "\n\nChat history so far:\n" + history_str + "\n\nPlease summarize the incident response plan we have built so far."
        summary = self.llm.invoke(prompt)
        if hasattr(summary, "content"):
            summary = summary.content
        self.state['last_plan'] = summary
        path = create_download_func()
        return f"Download your plan here: {path} (copy this path to your browser or file explorer)"

    def handle_section_flow(self, message, history, graph, last_plan):
        chat_history = self._build_chat_history(history, message)
        if self._is_intro_repeat(message, chat_history):
            return "We've already discussed what an incident response plan is. Let's move on to building your plan."

        self._advance_section_if_needed(message)
        graph_state = self._build_graph_state(chat_history, last_plan)
        result = graph.invoke(graph_state)
        llm_message = self._parse_llm_output(result)

        options_msg = ("\n\nOptions: Type 'overview' for a summary, 'download' to save your plan, "
                       "'review' to see your plan so far, or 'progress' to check remaining steps.")
        if len(self.state['covered_sections']) == len(self.state['main_sections']):
            return llm_message + "\n\nAll main sections have been covered." + options_msg
        return llm_message + options_msg

    def _build_chat_history(self, history, message):
        chat_history = []
        if history:
            for msg in history:
                chat_history.append({"role": msg["role"], "content": msg["content"]})
        chat_history.append({"role": "user", "content": message})
        return chat_history

    def _is_intro_repeat(self, message, chat_history):
        intro_given = any(
            'incident response plan (IRP) is a documented strategy' in msg.get('content', '').lower()
            for msg in chat_history if msg.get('role') == 'assistant'
        )
        return (
            message.strip().lower() in [
                "what is an incident response plan?",
                "what is an irp?",
                "explain incident response plan"
            ] and intro_given
        )

    def _advance_section_if_needed(self, message):
        move_on_phrases = ["move on", "next section", "continue", "proceed"]
        user_move_on = any(phrase in message.lower() for phrase in move_on_phrases)
        current_section = self.state['current_section']
        covered_sections = self.state['covered_sections'][:]
        if user_move_on and current_section not in covered_sections:
            covered_sections.append(current_section)
            next_idx = self.state['main_sections'].index(current_section) + 1
            if next_idx < len(self.state['main_sections']):
                self.state['current_section'] = self.state['main_sections'][next_idx]
            else:
                self.state['current_section'] = self.state['main_sections'][-1]
        self.state['covered_sections'] = covered_sections

    def _build_graph_state(self, chat_history, last_plan):
        return {
            "chat_history": chat_history,
            "answer": "",
            "last_plan": last_plan,
            "current_section": self.state['current_section'],
            "covered_sections": self.state['covered_sections']
        }

    def _parse_llm_output(self, result):
        llm_message = None
        try:
            llm_json = json.loads(result["answer"])
            llm_message = llm_json.get("message", result["answer"])
            llm_section = llm_json.get("current_section", self.state['current_section'])
            self.state['current_section'] = llm_section
        except Exception:
            try:
                fixed = result["answer"].replace("'", '"')
                llm_json = json.loads(fixed)
                llm_message = llm_json.get("message", fixed)
                llm_section = llm_json.get("current_section", self.state['current_section'])
                self.state['current_section'] = llm_section
            except Exception:
                logger.warning(f"Failed to parse LLM output as JSON: {result['answer']}")
                llm_message = result["answer"]
        return llm_message

    def process(self, message, history, graph, last_plan, create_download_func):
        self.state['interaction_count'] += 1
        if self.state['interaction_count'] > MAX_INTERACTIONS:
            return ("Session limit reached. Please restart the app to continue. "
                    "This is to prevent excessive usage and potential infinite loops.")
        if message.strip().lower().startswith('search '):
            return self.handle_web_search(message)
        if message.strip().lower() == 'progress':
            return self.handle_progress()
        if message.strip().lower() == 'overview':
            return self.handle_overview(message, history)
        if message.strip().lower() == 'review':
            return self.handle_review(message, history)
        if message.strip().lower() == 'download':
            return self.handle_download(message, history, create_download_func)
        return self.handle_section_flow(message, history, graph, last_plan)

def get_main_sections_from_llm(llm):
    prompt = (
        "List the main sections (in order) that should be included in a comprehensive incident response plan. "
        "Return only a Python list of section names, nothing else."
    )
    response = llm.invoke(prompt)
    try:
        content = response.content if hasattr(response, "content") else response
        sections = eval(content)
        if isinstance(sections, list) and all(isinstance(s, str) for s in sections):
            return sections
    except Exception:
        pass
    # Fallback to default if parsing fails
    return [
        "Preparation",
        "Detection",
        "Containment",
        "Eradication",
        "Recovery",
        "Post-Mortem Analyses"
    ]

class VisoPart1:
    """
    Interactive Incident Response Plan (IRP) builder using LangGraph and LLM.
    - Guides the user through IRP creation by asking questions and tracking progress.
    - Supports commands for overview, review, download, and progress.
    - Uses TavilySearch for web search if requested.
    - Uses a state machine for section flow and LLM structured output.
    """
    def __init__(self):
        self.llm = None
        self.graph = None
        self.search_tool = None
        self.state = {
            'interaction_count': 0,
            'last_plan': '',
            'main_sections': [],
            'current_section': None,
            'covered_sections': [],
        }
        self.handler = None

    def initialize(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.search_tool = TavilySearch(max_results=3, max_tokens=MAX_TOKENS)
        # Dynamically get main sections from LLM
        self.state['main_sections'] = get_main_sections_from_llm(self.llm)
        self.state['current_section'] = self.state['main_sections'][0]
        self.prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPT + "\n\nCurrent section: {current_section}\nSections covered: {covered_sections}\n\nChat history so far:\n{history}\n\nUser: {user_message}\nAssistant (output JSON):"
        )
        graph = StateGraph(IRPBuilderState)
        graph.add_node("llm_interact", self.llm_node)
        graph.add_edge(START, "llm_interact")
        graph.add_edge("llm_interact", END)
        self.graph = graph.compile()
        self.handler = IRPChatHandler(self.llm, self.search_tool, self.state)

    def llm_node(self, state: IRPBuilderState):
        """
        LangGraph node: Given chat history and user message, ask the next relevant IRP question or summarize.
        Output is a JSON object with 'current_section' and 'message'.
        """
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in state['chat_history']
        ])
        user_message = ""
        for msg in reversed(state['chat_history']):
            if msg['role'] == 'user':
                user_message = msg['content']
                break
        chain = self.prompt | self.llm | StrOutputParser()
        llm_output = chain.invoke({
            "history": history_str,
            "user_message": user_message,
            "current_section": state['current_section'],
            "covered_sections": ", ".join(state['covered_sections'])
        })
        # Parse the LLM's JSON output
        try:
            parsed = json.loads(llm_output)
            current_section = parsed.get("current_section", state['current_section'])
            message = parsed.get("message", llm_output)
        except Exception as e:
            logger.warning(f"Failed to parse LLM output as JSON: {llm_output}")
            current_section = state['current_section']
            message = llm_output
        return {
            "answer": message,
            "chat_history": state['chat_history'],
            "last_plan": state.get('last_plan', ''),
            "current_section": current_section,
            "covered_sections": state['covered_sections']
            
        }

    def _create_download(self) -> str:
        plan_text = self.state['last_plan'] or "No plan summary available. Type 'review' to generate a summary."
        path = os.path.join(os.getcwd(), 'plan_outline.txt')
        with open(path, 'w') as f:
            f.write(plan_text)
        return path

    def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        return self.handler.process(message, history, self.graph, self.state.get('last_plan', ''), self._create_download)
