import os
import glob
import json
import logging
from typing import Dict, List, Optional, TypedDict
import chromadb
import PyPDF2
import docx
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from opik.integrations.langchain import OpikTracer
from pydantic import BaseModel
import difflib

# Command constants
CMD_NEXT = "next"
CMD_CLARIFY = "clarify"
CMD_COMPLETE = "complete"
CMD_SUMMARY = "summary"
CMD_ANALYSIS = "analysis"
CMD_LESSONS = "lessons"
CMD_DOWNLOAD = "download"
CMD_TEAM = "team"
CMD_OVERVIEW = "overview"
CMD_HELP = "help"
CMD_WELCOME = "welcome"
CMD_START = "start"

# User instruction string constants
INSTRUCTION_STEP = (
    f"Type '{CMD_NEXT}' to proceed, '{CMD_CLARIFY}' for detailed guidance, or '{CMD_COMPLETE}' to finish all steps and move to post-incident analysis."
)
INSTRUCTION_NO_SESSION = "No active incident response session. Please select an incident type first."
INSTRUCTION_SUMMARY = f"Type '{CMD_SUMMARY}' for session summary or '{CMD_ANALYSIS}' for detailed analysis."
INSTRUCTION_ALL_STEPS_DONE = f"All incident response steps completed! {INSTRUCTION_SUMMARY}"

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress ONNX provider warnings from chromadb embedding functions
logging.getLogger("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration flags
ENABLE_OPIK_TRACING = False  # Set to True to enable Opik tracing

# Suppress third-party library logs at INFO level
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("tiktoken").setLevel(logging.WARNING)

MAX_TOKENS = 1024
MAX_INTERACTIONS = 50
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "IRCurrentDocuments")

# Incident Response specific prompts
LLM_STEP_PROMPT = """
You are an expert security consultant. For each incident response step, output a JSON array of sub-actions. Each sub-action should have:
- "title": a short action name (e.g., "Forensic Investigation")
- "description": a clear, actionable description of what to do

Example:
[
  {"title": "Forensic Investigation", "description": "Conduct a thorough forensic analysis..."},
  {"title": "Evidence Collection", "description": "Securely collect and preserve all relevant logs..."}
]
"""

INCIDENT_CLARIFY_PROMPT = """
You are an expert security consultant providing detailed guidance for incident response steps.

Current step: {step_text}

Provide a comprehensive, step-by-step breakdown of this action including:
1. Specific technical procedures
2. Required tools and resources
3. Team member responsibilities
4. Potential challenges and solutions
5. Success criteria
6. Safety considerations

Format your response with clear headings and actionable details.
"""

ROLE_MAPPING_PROMPT = """
Based on this incident response step, identify which team member should be primarily responsible:

Step: {step_text}

Team members available:
{team_members}

Return only the name of the most appropriate team member for this step, or "Team Lead" if it requires leadership coordination.
"""

POST_INCIDENT_PROMPT = """
You are conducting a comprehensive post-incident review as a senior security consultant. Based on the incident response session:

Session history: {session_history}

Incident Type: {incident_type}
Team Members: {team_members}

Provide a detailed, professional post-incident analysis report with the following sections:

## EXECUTIVE SUMMARY
- Brief overview of the incident response session
- Overall effectiveness rating (1-10)
- Key achievements and critical issues

## DETAILED ANALYSIS

### Actions Taken
- Chronological summary of all steps completed
- Team member performance and responsibilities
- Adherence to incident response plan

### Effectiveness Assessment
- What worked well during the response
- Areas where the response was effective
- Identification of strengths in the current process

### Lessons Learned
- Key insights gained from this session
- What could have been done differently
- Unexpected challenges encountered

### Gap Analysis
- Comparison against industry best practices
- Missing elements in the current response plan
- Process improvements identified

### Recommendations for Improvement
- Specific, actionable recommendations
- Priority levels (High/Medium/Low)
- Timeline for implementation
- Resource requirements

### Follow-up Actions
- Immediate next steps required
- Long-term improvements needed
- Training and awareness recommendations

### Risk Assessment
- Potential risks if improvements aren't implemented
- Impact assessment of identified gaps
- Compliance considerations

Format the response with clear headings, bullet points, and actionable insights. Be specific and provide concrete examples where possible.
"""

SESSION_ANALYSIS_PROMPT = """
You are an expert incident response analyst. Analyze this incident response session for patterns and insights:

Session Data: {session_data}

Provide a structured analysis covering:

## PERFORMANCE METRICS
- Response time analysis
- Step completion efficiency
- Team coordination effectiveness

## BEHAVIORAL INSIGHTS
- Decision-making patterns
- Communication effectiveness
- Stress management during incident

## PROCESS OPTIMIZATION
- Bottlenecks identified
- Streamlining opportunities
- Automation potential

## TEAM DYNAMICS
- Role clarity and effectiveness
- Collaboration patterns
- Leadership effectiveness

## COMPLIANCE & GOVERNANCE
- Regulatory alignment
- Documentation quality
- Audit trail completeness

Provide specific, actionable insights with clear recommendations.
"""

LESSONS_LEARNED_PROMPT = """
As a senior incident response consultant, extract key lessons learned from this session:

Session Context: {session_context}

Focus on:

## CRITICAL LESSONS
- What surprised the team
- What worked unexpectedly well
- What failed or was challenging

## PROCESS IMPROVEMENTS
- Specific process changes needed
- Tool and technology gaps
- Training requirements

## TEAM DEVELOPMENT
- Individual and team skill gaps
- Communication improvements
- Leadership development needs

## STRATEGIC INSIGHTS
- Long-term planning implications
- Resource allocation recommendations
- Technology investment priorities

Provide actionable insights with clear next steps.
"""

class DocumentChunk(TypedDict):
    doc_name: str
    section: str
    content: str
    chunk_id: str

class IncidentStep(BaseModel):
    title: str
    description: str

class IncidentResponsePlan(BaseModel):
    steps: List[IncidentStep]

class IncidentResponseState(TypedDict):
    """State for the incident response workflow."""
    user_message: str
    incident_type: Optional[str]
    current_step_index: int
    incident_plan: Optional[IncidentResponsePlan]
    session_history: List[Dict[str, str]]
    team_members: List[str]
    current_step_text: str
    response_result: str
    interaction_count: int
    command_type: Optional[str]

class IncidentResponseGuide:
    """
    Guides the user through an incident response workflow, using LLM and RAG for step-by-step guidance, team info, and post-incident analysis.
    Supports commands for starting, progressing, clarifying, completing, and analyzing incident response sessions.
    """
    def __init__(self, rag_retriever=None):
        """
        Initialize the guide with optional RAG retriever for document-based answers.
        Sets up incident types, team members, and session state.
        """
        self.llm = None
        self.rag_retriever = rag_retriever
        self.incident_types = [
            "Data Breach",
            "Ransomware Attack",
            "Phishing Campaign",
            "DDoS Attack",
            "Insider Threat",
            "Malware Infection",
            "Social Engineering",
            "Physical Security Breach",
            "Zero-Day Exploit",
            "Supply Chain Attack",
            "Credential Compromise",
            "Web Application Attack",
            "Insider Misuse",
            "Physical Theft or Loss",
            "Data Leakage",
            "Business Email Compromise",
            "Rogue Device",
            "Advanced Persistent Threat",
            "Malicious Code Injection",
            "Social Media Account Compromise",
            "Third-Party Service Outage"
        ]
        self.team_members = [
            "Incident Response Lead",
            "Security Analyst",
            "Network Administrator",
            "System Administrator",
            "Legal Counsel",
            "Communications Officer",
            "Forensic Investigator",
            "IT Support"
        ]
        self.current_incident_type = None
        self.incident_plan = None
        self.current_step_index = 0
        self.session_history = []

    def set_llm(self, llm):
        """Set the language model to use for all LLM-based responses."""
        self.llm = llm

    def set_rag_retriever(self, rag_retriever):
        """Set the retriever for retrieving policy/team info from ingested documents."""
        self.rag_retriever = rag_retriever

    def _format_step(self, step_index: int, total_steps: int, step: 'IncidentStep') -> str:
        """
        Helper to format the current step with 'Step X of N', bold title, and description.
        """
        return (
            f"### ▶️ Step {step_index + 1} of {total_steps}: <b>{step.title}</b>\n"
            f"<i>{step.description}</i>\n"
        )

    def start_incident_response(self, incident_type: str) -> str:
        """
        Start a new incident response session for the given incident type.
        Generates a step-by-step plan using the LLM and resets session state.
        """
        incident_type = incident_type.strip().lower()
        matched = None
        for valid in self.incident_types:
            if incident_type in valid.lower():
                matched = valid
                break
        if not matched:
            return f"Invalid incident type. Please choose from: {', '.join(self.incident_types)}"
        self.current_incident_type = matched
        self.current_step_index = 0
        self.session_history = []
        plan_prompt = f"""
        Create a comprehensive incident response plan for a {matched} incident.
        Break down the response into specific, actionable steps.
        """
        structured_llm = self.llm.with_structured_output(IncidentResponsePlan)
        result = structured_llm.invoke(plan_prompt)
        self.incident_plan = result
        num_steps = len(self.incident_plan.steps) if self.incident_plan and self.incident_plan.steps else 0
        step = self.incident_plan.steps[0] if num_steps > 0 else None
        step_block = self._format_step(0, num_steps, step) if step else ''
        return (
            f"## 🚦 Incident Response Session Started\n"
            f"Incident type: **{matched}**  \n"
            f"Number of steps: **{num_steps}**\n"
            f"---\n"
            f"{step_block}"
            f"\n---\n"
            f"{INSTRUCTION_STEP}"
        )

    def get_current_step(self) -> tuple[str, bool]:
        """Return the current step's title and whether a session is active."""
        if not self.incident_plan or self.current_step_index >= len(self.incident_plan.steps):
            return "No active incident response session.", False
        
        step = self.incident_plan.steps[self.current_step_index]
        return step.title, True

    def confirm_step(self) -> str:
        """
        Mark the current step as completed and move to the next step.
        Updates session history and returns the next step or completion message.
        """
        if not self.incident_plan:
            return INSTRUCTION_NO_SESSION
        
        if self.current_step_index >= len(self.incident_plan.steps):
            # Use the same detailed message as complete_steps
            return (
                "✅ All incident response steps are now completed for this incident!\n"
                "You can now use the following commands to review and analyze the session:\n"
                "- 'summary' — Get a session summary\n"
                "- 'analysis' — Get a detailed session analysis\n"
                "- 'lessons' — Extract lessons learned\n"
                "- 'download' — Save the session analysis to a file\n"
                "To start a new incident, type 'start [incident_type]'."
            )
        
        current_step = self.incident_plan.steps[self.current_step_index]
        
        # Add to session history
        self.session_history.append({
            "role": "system",
            "content": f"Completed step: {current_step.title}"
        })
        
        self.current_step_index += 1
        
        if self.current_step_index >= len(self.incident_plan.steps):
            return INSTRUCTION_ALL_STEPS_DONE
        
        next_step = self.incident_plan.steps[self.current_step_index]
        num_steps = len(self.incident_plan.steps)
        step_block = self._format_step(self.current_step_index, num_steps, next_step)
        return (
            f"Step completed. Moving to next step:\n"
            f"{step_block}"
            f"\n{INSTRUCTION_STEP}"
        )

    def get_plan_overview(self) -> str:
        """Return a checklist-style overview of the current incident response plan and progress."""
        if not self.incident_plan:
            return "No active incident response plan."
        
        overview = f"## Incident Response Plan: {self.current_incident_type}\n"
        for i, step in enumerate(self.incident_plan.steps):
            status = "✓" if i < self.current_step_index else "○"
            overview += f"{status} Step {i+1}: {step.title}\n"
        
        return overview

    def clarify_current_step(self) -> str:
        """
        Provide detailed, step-by-step guidance for the current step using the LLM.
        """
        if not self.incident_plan or self.current_step_index >= len(self.incident_plan.steps):
            return "No active step to clarify."
        
        current_step = self.incident_plan.steps[self.current_step_index]
        step_text = f"{current_step.title}: {current_step.description}"
        
        prompt = ChatPromptTemplate.from_template(INCIDENT_CLARIFY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        clarification = chain.invoke({"step_text": step_text})
        return f"## Detailed Guidance for Current Step\n{clarification}"

    def get_team_info(self) -> str:
        """
        Retrieve and display the team roster (names, roles, responsibilities, and contact info if present) from ingested policy documents.
        Uses the LLM to extract and format as a visually aligned table for best readability in the UI.
        Ensures the full, untruncated document context is sent to the LLM. Falls back to a static list if no document is found or LLM fails.
        """
        if self.rag_retriever:
            roster_docs = self.rag_retriever.retrieve(
                "incident response team roster names roles responsibilities contact info", top_k=10
            )
            if roster_docs and self.llm:
                combined_content = "\n".join(doc['content'] for doc in roster_docs)
                prompt = (
                    "Extract every team member from the following text, regardless of formatting or length. "
                    "If any contact information (such as email or phone) is present for a team member, include a 'Contact Info' column in the table; otherwise, omit the column. "
                    "Output the result as a valid Markdown table (using | and --- for headers and separators). "
                    "Do not include any explanation, closing remarks, or extra lines—output only the Markdown table.\n"
                    f"{combined_content}"
                )
                response = self.llm.invoke(prompt)
                table = response.content if hasattr(response, "content") else str(response)
                table = table.strip()
                # Strict post-processing: ensure Markdown table format
                if table.startswith('|') and ('|---' in table):
                    return "## Incident Response Team Roster (from policy documents)\n" + table
                # Fallback to static Markdown table if not compliant
        # Fallback to static list if nothing found or LLM fails
        team_info = "## Incident Response Team\n"
        team_info += "| Name | Role |\n|---|---|\n"
        for member in self.team_members:
            team_info += f"| {member} | |\n"
        team_info += "\nEach team member has specific responsibilities during incident response."
        return team_info

    def get_summary(self) -> str:
        """
        Generate a post-incident summary using session history and policy documents.
        """
        if not self.session_history:
            return "No session history available."
        
        session_data = "\n".join([
            f"{entry['role']}: {entry['content']}" 
            for entry in self.session_history
        ])
        
        prompt = ChatPromptTemplate.from_template(POST_INCIDENT_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        summary = chain.invoke({
            "session_history": session_data,
            "incident_type": self.current_incident_type or "Unknown",
            "team_members": ", ".join(self.team_members)
        })
        
        return summary

    def get_session_analysis(self) -> str:
        """
        Provide a detailed analysis of the session (performance, team dynamics, etc.).
        """
        if not self.session_history:
            return "No session data available for analysis."
        
        session_data = json.dumps(self.session_history, indent=2)
        
        prompt = ChatPromptTemplate.from_template(SESSION_ANALYSIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        analysis = chain.invoke({"session_data": session_data})
        return analysis

    def get_lessons_learned(self) -> str:
        """
        Extract lessons learned from the session using LLM and session context.
        """
        if not self.session_history:
            return "No session data available for lessons learned analysis."
        
        session_context = f"Incident Type: {self.current_incident_type}\nSteps Completed: {self.current_step_index}\nTeam Members: {', '.join(self.team_members)}"
        
        prompt = ChatPromptTemplate.from_template(LESSONS_LEARNED_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        lessons = chain.invoke({"session_context": session_context})
        return lessons

    def get_welcome_screen(self) -> str:
        """
        Show a welcome/help message with all available commands and dynamically lists all supported incident types.
        """
        welcome = """
# Incident Response Guidance System

Welcome to the Incident Response Guidance System. I can help you:

## Start a New Incident Response Session
Choose an incident type to begin:
"""
        for incident_type in self.incident_types:
            welcome += f"- {incident_type}\n"
        
        welcome += """
## Available Commands
- `start [incident_type]` - Start a new incident response session
- `next` - Confirm current step and move to next
- `clarify` - Get detailed guidance for current step
- `overview` - Show incident response plan overview
- `team` - Show team member information
- `summary` - Get session summary
- `analysis` - Get detailed session analysis
- `lessons` - Extract lessons learned
- `help` - Show this help message

## Example Usage
```
start Data Breach
next
clarify
overview
summary
```

Type 'start [incident_type]' to begin an incident response session.
"""
        return welcome

    def download_analysis(self) -> str:
        """
        Download the session analysis to a local file named 'incident_response_analysis.txt'.
        Returns the file path to the user.
        """
        analysis = self.get_session_analysis()
        filename = "incident_response_analysis.txt"
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(analysis)
        return f"Analysis saved to: {filepath}"

    def complete_steps(self) -> str:
        """
        Mark all remaining steps as completed and move to the end of the plan.
        Only works if an incident session is active; otherwise prompts the user to select an incident type first.
        """
        if not self.incident_plan:
            return INSTRUCTION_NO_SESSION
        while self.current_step_index < len(self.incident_plan.steps):
            current_step = self.incident_plan.steps[self.current_step_index]
            self.session_history.append({
                "role": "system",
                "content": f"Completed step: {current_step.title}"
            })
            self.current_step_index += 1
        return (
            "✅ All incident response steps are now completed for this incident!\n"
            "You can now use the following commands to review and analyze the session:\n"
            "- 'summary' — Get a session summary\n"
            "- 'analysis' — Get a detailed session analysis\n"
            "- 'lessons' — Extract lessons learned\n"
            "- 'download' — Save the session analysis to a file\n"
            "To start a new incident, type 'start [incident_type]'."
        )

    # Command groups for routing
    INCIDENT_RELATED_COMMANDS = {
        CMD_NEXT, CMD_COMPLETE, CMD_CLARIFY, CMD_OVERVIEW, CMD_SUMMARY, CMD_ANALYSIS, CMD_LESSONS
    }
    INCIDENT_AGNOSTIC_COMMANDS = {
        CMD_TEAM, CMD_HELP, CMD_WELCOME, CMD_DOWNLOAD
    }

    def process_incident_message(self, message: str, chat_history: Optional[list] = None) -> str:
        """
        Main command router for user messages.
        - Organizes commands into incident-related and incident-agnostic groups for robust, future-proof handling.
        - Only incident-related commands can affect the workflow/session.
        - Incident-agnostic commands (team, help, download, etc.) never reset or break the session.
        - Fuzzy/partial incident type matching only occurs if no command matches.
        - If no command matches, provides a free-form LLM response.
        """
        message_lower = message.strip().lower()
        # Incident-agnostic commands (never reset session)
        if message_lower in self.INCIDENT_AGNOSTIC_COMMANDS:
            if message_lower in {CMD_HELP, CMD_WELCOME}:
                return self.get_welcome_screen()
            if message_lower == CMD_TEAM:
                return self.get_team_info()
            if message_lower == CMD_DOWNLOAD:
                return self.download_analysis()
        # Incident-related commands (affect workflow/session)
        if message_lower.startswith(CMD_START):
            incident_type = message[len(CMD_START):].strip()
            return self.start_incident_response(incident_type)
        if message_lower in self.INCIDENT_RELATED_COMMANDS:
            if message_lower == CMD_NEXT:
                return self.confirm_step()
            if message_lower == CMD_COMPLETE:
                return self.complete_steps()
            if message_lower == CMD_CLARIFY:
                return self.clarify_current_step()
            if message_lower == CMD_OVERVIEW:
                return self.get_plan_overview()
            if message_lower == CMD_SUMMARY:
                return self.get_summary()
            if message_lower == CMD_ANALYSIS:
                return self.get_session_analysis()
            if message_lower == CMD_LESSONS:
                return self.get_lessons_learned()
        # Fuzzy/partial match for incident types (only if not a command)
        best_match = None
        highest_ratio = 0.0
        for valid in self.incident_types:
            valid_lower = valid.lower()
            # Exact or substring match
            if message_lower in valid_lower:
                best_match = valid
                highest_ratio = 1.0
                break
            # Fuzzy match
            ratio = difflib.SequenceMatcher(None, message_lower, valid_lower).ratio()
            if ratio > highest_ratio:
                best_match = valid
                highest_ratio = ratio
        # Accept if substring or fuzzy match is reasonably close
        if highest_ratio >= 0.5:
            return self.start_incident_response(best_match)
        # If the user says "start a simulation" or similar, show the welcome/help screen
        if message_lower in ["start a simulation", "start simulation", "begin simulation", "launch simulation"]:
            return self.get_welcome_screen()
        # If no specific command, provide general guidance
        return self._llm_free_form_response(message, chat_history)

    def _llm_free_form_response(self, message: str, chat_history: Optional[list] = None) -> str:
        """
        Handle free-form questions about incident response using the LLM.
        """
        if not self.llm:
            return "LLM not initialized."
        
        prompt = f"""
        You are an expert incident response consultant. The user is asking about incident response.
        
        User question: {message}
        
        If this is about starting an incident response session, suggest using 'start [incident_type]'.
        If this is about the current session, provide helpful guidance.
        If this is a general question about incident response, provide expert advice.
        
        Keep your response concise and actionable.
        """
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

class RAGDocumentLoader:
    def __init__(self, folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.folder = folder
        self.documents = []
        self.chunks = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str, doc_name: str, section_prefix: str, chunk_prefix: str):
        # Split text into overlapping chunks
        start = 0
        chunk_id = 1
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                self.chunks.append({
                    "doc_name": doc_name,
                    "section": f"{section_prefix} {chunk_id}",
                    "content": chunk,
                    "chunk_id": f"{doc_name}_{chunk_prefix}_{chunk_id}"
                })
            if end == len(text):
                break
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

    def load_documents(self):
        self.documents = []
        self.chunks = []
        for path in glob.glob(os.path.join(self.folder, "*")):
            logger.info(f"Processing document: {path}")
            if os.path.isdir(path):
                continue  # Skip directories
            doc_name = os.path.basename(path)
            if path.lower().endswith(".pdf"):
                self._load_pdf(path, doc_name)
            elif path.lower().endswith(".docx"):
                self._load_docx(path, doc_name)
            elif path.lower().endswith((".html", ".htm")):
                self._load_html(path, doc_name)
            elif path.lower().endswith(".txt"):
                self._load_txt(path, doc_name)
            else:
                logger.warning(f"Unsupported file extension for document: {doc_name}")
        logger.info(f"Loaded {len(self.chunks)} document chunks.")

    def _load_pdf(self, path, doc_name):
        try:
            reader = PyPDF2.PdfReader(path)
            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            if full_text:
                self._chunk_text(full_text, doc_name, "PDF Chunk", "pdf")
        except Exception as e:
            logger.warning(f"Failed to load PDF {doc_name}: {e}")

    def _load_docx(self, path, doc_name):
        try:
            doc = docx.Document(path)
            full_text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            if full_text:
                self._chunk_text(full_text, doc_name, "DOCX Chunk", "docx")
        except Exception as e:
            logger.warning(f"Failed to load DOCX {doc_name}: {e}")

    def _load_txt(self, path, doc_name):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text:
                    self._chunk_text(text, doc_name, "TXT Chunk", "txt")
        except Exception as e:
            logger.warning(f"Failed to load TXT {doc_name}: {e}")

    def _load_html(self, path, doc_name):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator='\n')
                if text:
                    self._chunk_text(text, doc_name, "HTML Chunk", "html")
        except Exception as e:
            logger.warning(f"Failed to load HTML {doc_name}: {e}")

    def get_chunks(self) -> List[DocumentChunk]:
        return self.chunks

class RAGRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("irp_docs")
        self._index_chunks()

    def _index_chunks(self):
        for chunk in self.chunks:
            self.collection.add(
                documents=[chunk["content"]],
                metadatas=[{"doc_name": chunk["doc_name"], "section": chunk["section"]}],
                ids=[chunk["chunk_id"]]
            )

    def retrieve(self, query: str, top_k: int = 5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"doc_name": meta["doc_name"], "section": meta["section"], "content": doc})
        return docs

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

class VisoPart3:
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
        self.opik_tracer = None
        self.state = {
            'interaction_count': 0,
            'recommendations_by_section': {},
            'main_sections': [],
        }
        self.rag_loader = RAGDocumentLoader(DOCUMENTS_FOLDER)
        self.rag_loader.load_documents()
        self.rag_retriever = RAGRetriever(self.rag_loader.get_chunks())
        self.incident_guide = IncidentResponseGuide(self.rag_retriever)

    def initialize(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.search_tool = TavilySearch(max_results=3, max_tokens=MAX_TOKENS)
        self.state['main_sections'] = get_main_sections_from_llm(self.llm)
        self.incident_guide.set_llm(self.llm)
        self.incident_guide.set_rag_retriever(self.rag_retriever)
        
        # Create the LangGraph workflow
        graph = StateGraph(IncidentResponseState)
        
        # Add nodes
        graph.add_node("command_router", self._command_router_node)
        graph.add_node("response_generator", self._response_generator_node)
        
        # Add edges with conditional routing
        graph.add_edge(START, "command_router")
        graph.add_conditional_edges(
            "command_router",
            self._route_to_response,
            {
                "incident": "response_generator",
                "policy": "response_generator", 
                "welcome": "response_generator",
                "agnostic": "response_generator"
            }
        )
        graph.add_edge("response_generator", END)
        
        # Compile the graph
        self.graph = graph.compile()

        # Initialize Opik tracer only if enabled
        if ENABLE_OPIK_TRACING:
            self.opik_tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
            logger.info("Opik tracing enabled")
        else:
            self.opik_tracer = None
            logger.info("Opik tracing disabled")

    def _command_router_node(self, state: IncidentResponseState):
        """Route the user message to the appropriate handler."""
        message = state["user_message"].strip().lower()
        logger.info(f"Command router processing: '{message}'")

        # Check if this is an incident-agnostic command
        if message in self.incident_guide.INCIDENT_AGNOSTIC_COMMANDS:
            logger.info(f"Routing to agnostic handler for message: '{message}'")
            return {"command_type": "agnostic"}
        # Check if this is an incident response command
        if message in self.incident_guide.INCIDENT_RELATED_COMMANDS or message.startswith(CMD_START):
            logger.info(f"Routing to incident handler for message: '{message}'")
            return {"command_type": "incident"}
        # Check if this is a policy analysis command
        policy_commands = ["analyze", "review", "policy"]
        if any(cmd in message for cmd in policy_commands):
            logger.info(f"Routing to policy handler for message: '{message}'")
            return {"command_type": "policy"}
        # Default to welcome/help
        logger.info(f"Routing to welcome handler for message: '{message}'")
        return {"command_type": "welcome"}

    def _route_to_response(self, state: IncidentResponseState):
        """Route to the appropriate response type."""
        return state.get("command_type", "welcome")

    def _response_generator_node(self, state: IncidentResponseState):
        """Generate the appropriate response based on command type."""
        command_type = state.get("command_type", "welcome")
        message = state["user_message"]
        logger.info(f"Response generator - command_type: '{command_type}', message: '{message}'")

        if command_type == "incident":
            logger.info("Generating incident response")
            response = self.incident_guide.process_incident_message(message)
        elif command_type == "policy":
            logger.info("Generating policy response")
            response = "Policy analysis functionality coming soon. Use incident response commands for now."
        elif command_type == "agnostic":
            logger.info("Generating agnostic response")
            response = self.incident_guide.process_incident_message(message)
        else:  # welcome
            logger.info("Generating welcome response")
            response = self.incident_guide.get_welcome_screen()

        logger.info(f"Generated response length: {len(response)}")
        return {"response_result": response}

    def normalize_command(self, message: str) -> str:
        """Normalize user input to standard commands."""
        message_lower = message.strip().lower()
        
        # Incident response commands
        if message_lower in ["help", "welcome", "intro"]:
            return CMD_HELP
        if message_lower.startswith(CMD_START):
            return message  # Keep as-is
        if message_lower in ["next", "continue", "proceed"]:
            return CMD_NEXT
        if message_lower in ["clarify", "explain", "details"]:
            return CMD_CLARIFY
        if message_lower in ["overview", "plan", "steps"]:
            return CMD_OVERVIEW
        if message_lower in ["team", "members", "roles"]:
            return CMD_TEAM
        if message_lower in ["summary", "report"]:
            return CMD_SUMMARY
        if message_lower in ["analysis", "analyze"]:
            return CMD_ANALYSIS
        if message_lower in ["lessons", "learned"]:
            return CMD_LESSONS
        
        # Policy analysis commands
        if message_lower in ["analyze", "policy", "review"]:
            return "analyze" # This normalization is not in the new_code, so keep it as is
        if message_lower in ["download", "save", "export"]:
            return CMD_DOWNLOAD
        
        # If no specific command, return as-is for free-form processing
        return message

    def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the LangGraph workflow."""
        logger.info(f"Processing message: {message}")
        
        # Intercept simulation start phrases and map to 'welcome'
        simulation_phrases = [
            "start a simulation.", "start a simulation", "start simulation", "begin simulation", "launch simulation"
        ]
        if message.strip().lower() in simulation_phrases:
            message = CMD_WELCOME # Changed from "welcome" to CMD_WELCOME
        
        # Increment interaction count
        self.state['interaction_count'] += 1
        if self.state['interaction_count'] > MAX_INTERACTIONS:
            return "Session limit reached. Please restart the app to continue."
        
        # Create initial state
        initial_state = {
            "user_message": message,  # Use possibly remapped message
            "incident_type": self.incident_guide.current_incident_type,
            "current_step_index": self.incident_guide.current_step_index,
            "incident_plan": self.incident_guide.incident_plan,
            "session_history": self.incident_guide.session_history,
            "team_members": self.incident_guide.team_members,
            "current_step_text": "",
            "response_result": "",
            "interaction_count": self.state['interaction_count'],
            "command_type": None
        }
        
        # Prepare config with conditional Opik tracer
        config = {}
        if ENABLE_OPIK_TRACING and self.opik_tracer:
            config["callbacks"] = [self.opik_tracer]
            logger.debug("Using Opik tracer for this query")
        else:
            logger.debug("Opik tracer not used for this query")
        
        # Execute the workflow
        result = self.graph.invoke(initial_state, config=config)
        
        logger.info(f"Incident response completed. Response length: {len(result.get('response_result', ''))}")
        return result.get("response_result", "No response generated")

    def _find_section(self, user_input: str) -> Optional[str]:
        """Find which section the user is referring to."""
        user_input_lower = user_input.lower()
        for section in self.state['main_sections']:
            if section.lower() in user_input_lower:
                return section
        return None

    def _analyze_section(self, section: str) -> str:
        """Analyze a specific section using RAG."""
        logger.info(f"Analyzing section: {section}")
        
        # Retrieve relevant documents
        query = f"incident response plan {section.lower()} policy procedures guidelines"
        rag_docs = self.rag_retriever.retrieve(query, top_k=5)
        
        if not rag_docs:
            return f"No relevant documents found for {section} analysis."
        
        # For now, return a simple analysis
        return f"Analysis for {section} would be performed here using the retrieved documents."

    def _process_policy_analysis(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process policy analysis requests."""
        # This would integrate the full policy analysis functionality
        return "Policy analysis functionality is being integrated. Please use incident response commands for now."

    def review_recommendations(self) -> str:
        """Review all recommendations made so far."""
        if not self.state['recommendations_by_section']:
            return "No recommendations have been generated yet."
        
        review = "## Policy Analysis Summary\n"
        for section, analysis in self.state['recommendations_by_section'].items():
            review += f"### {section}\n{analysis}\n"
        
        return review

    def download_recommendations(self) -> str:
        """Download recommendations to a file."""
        if not self.state['recommendations_by_section']:
            return "No recommendations to download."
        
        content = self.review_recommendations()
        filename = "incident_response_recommendations.txt"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return f"Recommendations saved to: {filepath}"

