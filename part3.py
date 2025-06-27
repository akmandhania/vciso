import os
import glob
import json
import re
import logging
from typing import Dict, List, Optional, Tuple, TypedDict
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress ONNX provider warnings from chromadb embedding functions
logging.getLogger("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
- Resource allocation insights
- Risk management improvements

Provide actionable, specific lessons with clear implementation guidance.
"""

ANALYSIS_PROMPT = """
You are an expert security consultant. The user has selected the '{section}' section of their Incident Response Plan for review.

Here are the relevant excerpts from their existing policy documents:
{rag_docs}

Based on these documents, please perform a gap analysis against industry best practices for the '{section}' section.

Your response should include:
1.  A summary of the current state of their policy for this section based on the documents provided.
2.  A list of identified gaps when compared to industry best practices.
3.  A comprehensive, actionable list of recommendations for changes to their policy documents. These recommendations should be clear and detailed enough for someone to implement them.

If no relevant documents are found, please state that and provide recommendations for what should be included in this section from scratch, based on best practices.
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

class IncidentResponseGuide:
    """Interactive incident response guidance system (RAG-only, no direct file parsing)"""
    
    def __init__(self, rag_retriever=None):
        self.incident_types = {
            "data_breach": "Data Breach",
            "phishing": "Phishing Attack", 
            "ransomware": "Ransomware Attack"
        }
        self.current_incident = None
        self.current_step = 0
        self.step_responses = []
        self.llm = None  # Will be set by VisoPart3
        self.rag_retriever = rag_retriever
        self.session_plan = None  # LLM-generated plan for the current session
        self.team_members = {}  # Optionally, can be filled by RAG as well

    def set_llm(self, llm):
        self.llm = llm

    def set_rag_retriever(self, rag_retriever):
        self.rag_retriever = rag_retriever

    def start_incident_response(self, incident_type: str) -> str:
        logger.info(f"Entered start_incident_response for: {incident_type}")
        incident_type = incident_type.lower().strip()
        if incident_type not in self.incident_types:
            logger.warning(f"Invalid incident type: {incident_type}")
            return "Invalid incident type. Please choose from: data_breach, phishing, ransomware"
        logger.debug("Set incident type, step, responses")
        self.current_incident = incident_type
        self.current_step = 0
        self.step_responses = []
        logger.debug("About to retrieve RAG docs")
        rag_docs = self.rag_retriever.retrieve(incident_type, top_k=10) if self.rag_retriever else []
        logger.info(f"RAG docs retrieved: {len(rag_docs)}")
        rag_text = "\n\n".join([f"From {doc['doc_name']} ({doc['section']}): {doc['content']}" for doc in rag_docs])
        logger.debug("RAG text prepared")
        plan_prompt = f"""
You are an expert security consultant. Based ONLY on the following company policy excerpts, generate a step-by-step incident response plan for a {self.incident_types[incident_type]}.

IMPORTANT: Use ONLY the information provided in the policy excerpts below. Do NOT use any external knowledge, best practices, or general security expertise. If the policy excerpts don't contain enough information for a complete plan, indicate what information is missing.

Create a structured response with a list of steps. Each step should have a clear title and detailed description based solely on the company's documented procedures.

The response should be structured as a JSON object with a "steps" array containing objects with "title" and "description" fields.

Example format:
{{
  "steps": [
    {{
      "title": "Immediate Containment",
      "description": "Based on company policy: [specific procedure from documents]..."
    }},
    {{
      "title": "Evidence Preservation", 
      "description": "According to company procedures: [specific steps from documents]..."
    }}
  ]
}}

Company Policy Excerpts:
{rag_text}

Generate a comprehensive incident response plan with 5-8 steps for this {self.incident_types[incident_type]} using ONLY the above policy information. If the policy excerpts don't provide enough detail for certain steps, clearly indicate what additional information is needed from the company's procedures.
"""
        logger.debug("Plan prompt prepared")
        try:
            logger.debug("Before LLM call")
            # Use structured output with Pydantic model
            structured_llm = self.llm.with_structured_output(IncidentResponsePlan)
            result = structured_llm.invoke(plan_prompt)
            logger.debug("After LLM call")
            logger.debug(f"LLM structured output: {result}")
            
            if not result.steps or len(result.steps) == 0:
                logger.warning("No steps returned from LLM")
                return f"[DEBUG] LLM returned no steps. Raw output: {result}"
            
            logger.info(f"Parsed {len(result.steps)} steps from LLM response")
            self.session_plan = [{"title": step.title, "description": step.description} for step in result.steps]
            incident_name = self.incident_types[incident_type]
            total_steps = len(self.session_plan)
            response = f"🚨 **INCIDENT RESPONSE ACTIVATED** 🚨\n\n"
            response += f"**Incident Type:** {incident_name}\n"
            response += f"**Total Steps:** {total_steps}\n"
            response += f"**Incident Commander:** (see team roster)\n\n"
            response += "**IMMEDIATE ACTION REQUIRED:**\n"
            response += "1. Do NOT panic - follow the plan step by step\n"
            response += "2. Confirm each step before proceeding to the next\n"
            response += "3. Document all actions taken\n\n"
            response += "Type 'next' to begin the response process.\n"
            response += "Type 'team' to see the incident response team.\n"
            response += "Type 'overview' to see the complete plan.\n"
            logger.debug("Returning response to user")
            return response
        except Exception as e:
            logger.error(f"Exception in start_incident_response: {e}")
            return f"Exception in start_incident_response: {e}"

    def get_current_step(self) -> tuple[str, bool]:
        if not self.session_plan or self.current_step >= len(self.session_plan):
            return "All steps completed. Type 'summary' to see the response summary.", True
        step = self.session_plan[self.current_step]
        response = f"**Step {self.current_step + 1}: {step.get('title', 'Untitled')}**\n\n"
        response += f"{step.get('description', '')}\n\n"
        response += "**Action Required:** Please confirm that you have completed this step.\n"
        response += "Type 'next' when ready to proceed to the next step.\n"
        response += "Type 'clarify' if you need more details about this step.\n"
        return response, False

    def confirm_step(self) -> str:
        if not self.session_plan or self.current_step >= len(self.session_plan):
            return "All steps have been completed. Type 'summary' to see the response summary."
        step = self.session_plan[self.current_step]
        self.step_responses.append({
            'step': self.current_step + 1,
            'title': step.get('title', ''),
            'status': 'completed'
        })
        self.current_step += 1
        if self.current_step >= len(self.session_plan):
            return "✅ **All steps completed!**\n\nType 'summary' to see the complete incident response summary."
        else:
            next_step, _ = self.get_current_step()
            return f"✅ **Step {self.current_step} completed successfully!**\n\n{next_step}"

    def get_plan_overview(self) -> str:
        if not self.session_plan:
            return "No plan available for this incident."
        response = f"**INCIDENT RESPONSE PLAN OVERVIEW**\n\n"
        response += f"**Incident Type:** {self.incident_types.get(self.current_incident, 'Unknown')}\n\n"
        for i, step in enumerate(self.session_plan):
            response += f"**Step {i+1}:** {step.get('title', 'Untitled')}\n  {step.get('description', '')[:80]}...\n"
        return response

    def clarify_current_step(self) -> str:
        if not self.session_plan or self.current_step >= len(self.session_plan):
            return "No current step to clarify."
        step = self.session_plan[self.current_step]
        prompt = INCIDENT_CLARIFY_PROMPT.format(step_text=f"{step.get('title', '')}: {step.get('description', '')}")
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            return f"Error generating clarification: {e}"

    def get_team_info(self) -> str:
        """Display the incident response team information using focused RAG and LLM prompt"""
        if self.rag_retriever and self.llm:
            # Use a more focused query
            rag_docs = self.rag_retriever.retrieve("incident response team roster names roles responsibilities contact information", top_k=10)
            rag_text = "\n\n".join([f"From {doc['doc_name']} ({doc['section']}): {doc['content']}" for doc in rag_docs])
            prompt = f"""
You are an expert security consultant. Using ONLY the information in the following policy excerpts, extract the incident response team roster: names, roles, main responsibilities, and contact information (if available).
- Do NOT use any best practices, external knowledge, or invent any information.
- If a field is missing, write 'No information available' for that field.
- Return the output as a Markdown table with columns: Name, Role, Responsibility, Contact.
- Only output the table. Do not include any other text.

{rag_text}
"""
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        return "No team information available."

    def get_summary(self) -> str:
        """Generate comprehensive post-incident summary using LLM and RAG"""
        if not self.step_responses:
            return "No incident response session to summarize."
        # Retrieve team info via RAG
        team_info = self.get_team_info()
        # Prepare session data
        history_text = "\n".join([
            f"Step {item['step']}: {item.get('title', '')} - {item['status']}" for item in self.step_responses
        ])
        incident_type = self.incident_types.get(self.current_incident, "Unknown")
        prompt = POST_INCIDENT_PROMPT.format(
            session_history=history_text,
            incident_type=incident_type,
            team_members=team_info
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            return f"Error generating comprehensive post-incident review: {e}"

    def get_session_analysis(self) -> str:
        """Generate detailed session analysis using LLM and RAG"""
        if not self.step_responses:
            return "No incident response session to analyze."
        # Retrieve team info via RAG
        team_info = self.get_team_info()
        session_data = {
            "incident_type": self.incident_types.get(self.current_incident, "Unknown"),
            "total_steps": len(self.step_responses),
            "completed_steps": len([s for s in self.step_responses if s['status'] == 'completed']),
            "step_details": self.step_responses,
            "team_info": team_info
        }
        prompt = SESSION_ANALYSIS_PROMPT.format(session_data=str(session_data))
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            return f"Error generating session analysis: {e}"

    def get_lessons_learned(self) -> str:
        """Extract specific lessons learned using LLM and RAG"""
        if not self.step_responses:
            return "No incident response session to analyze for lessons learned."
        # Retrieve team info via RAG
        team_info = self.get_team_info()
        session_context = {
            "incident_type": self.incident_types.get(self.current_incident, "Unknown"),
            "steps_completed": self.step_responses,
            "team_info": team_info,
            "response_duration": len(self.step_responses),
        }
        prompt = LESSONS_LEARNED_PROMPT.format(session_context=str(session_context))
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            return f"Error generating lessons learned: {e}"

    def get_welcome_screen(self) -> str:
        """Display the welcome screen with available simulation options"""
        response = "🚨 **INCIDENT RESPONSE SIMULATION CENTER** 🚨\n\n"
        response += "Welcome to the VCISO Incident Response Simulation System!\n\n"
        response += "**Available Simulations:**\n"
        response += "1. **Data Breach** - Simulate a data breach incident response\n"
        response += "2. **Phishing Attack** - Simulate a phishing attack response\n"
        response += "3. **Ransomware Attack** - Simulate a ransomware incident response\n\n"
        response += "**To start a simulation, simply type:**\n"
        response += "• `data_breach` (or `data breach`, `breach`)\n"
        response += "• `phishing` (or `phish`)\n"
        response += "• `ransomware` (or `ransom`)\n\n"
        response += "**Other Commands:**\n"
        response += "• `welcome` - Show this screen again\n"
        response += "• `next` - Continue to next step (during simulation)\n"
        response += "• `team` - View incident response team\n"
        response += "• `overview` - See complete response plan\n"
        response += "• `summary` - Generate post-incident summary\n"
        response += "• `analysis` - Get detailed session analysis\n"
        response += "• `lessons` - Extract lessons learned\n"
        response += "• `clarify` - Get detailed explanation of current step\n\n"
        response += "**Natural Language Support:**\n"
        response += "You can also type natural phrases like:\n"
        response += "• \"Start a phishing simulation\"\n"
        response += "• \"Begin data breach response\"\n"
        response += "• \"Launch ransomware incident\"\n\n"
        response += "Ready to begin? Choose your simulation type!"
        return response

    def process_incident_message(self, message: str, chat_history: Optional[list] = None) -> str:
        """Process user messages for incident response"""
        logger.debug(f"Processing incident message: {message}")
        message = message.lower().strip()
        if message == 'welcome':
            return self.get_welcome_screen()
        elif message == 'next':
            return self.confirm_step()
        elif message == 'team':
            return self.get_team_info()
        elif message == 'overview':
            return self.get_plan_overview()
        elif message == 'summary':
            return self.get_summary()
        elif message == 'analysis':
            return self.get_session_analysis()
        elif message == 'lessons':
            return self.get_lessons_learned()
        elif message == 'clarify':
            return self.clarify_current_step()
        elif message in self.incident_types:
            return self.start_incident_response(message)
        else:
            # Use LLM for free-form questions, with RAG context
            logger.debug(f"Received message for free-form response: {message}")
            return self._llm_free_form_response(message, chat_history)

    def _llm_free_form_response(self, message: str, chat_history: Optional[list] = None) -> str:
        """Handle free-form user questions using LLM and RAG context"""
        context = ""
        if self.current_incident:
            context = f"Current incident: {self.incident_types[self.current_incident]}\n"
            if self.session_plan and self.current_step < len(self.session_plan):
                step = self.session_plan[self.current_step]
                context += f"Current step: {step.get('title', '')}\n"
        # Retrieve relevant docs for the message
        rag_docs = self.rag_retriever.retrieve(message, top_k=5) if self.rag_retriever else []
        rag_text = "\n\n".join([f"From {doc['doc_name']} ({doc['section']}): {doc['content']}" for doc in rag_docs])
        prompt = f"""You are an expert incident response consultant.\n\n{context}\nRelevant policy excerpts:\n{rag_text}\n\nUser question: {message}\n\nProvide a helpful, actionable response based on incident response best practices and the current context."""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else response
        except Exception as e:
            return f"I'm sorry, I encountered an error processing your question: {e}"

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
            logger.debug(f"Processing document: {path}")
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
        if self.chunks:
            logger.debug(f"First chunk: {self.chunks[0]}")

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
        # Use settings to disable telemetry
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
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
        self.state = {
            'interaction_count': 0,
            'recommendations_by_section': {},
            'main_sections': [],
        }
        self.rag_loader = RAGDocumentLoader(DOCUMENTS_FOLDER)
        self.rag_loader.load_documents()
        self.rag_retriever = RAGRetriever(self.rag_loader.get_chunks())
        # Initialize incident response guide
        self.incident_guide = IncidentResponseGuide(self.rag_retriever)

    def initialize(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.search_tool = TavilySearch(max_results=3, max_tokens=MAX_TOKENS)
        self.incident_guide.set_llm(self.llm)

    def normalize_command(self, message: str) -> str:
        """Normalize user input by stripping common phrases and mapping variations"""
        message = message.lower().strip()
        
        # Handle special cases first
        if any(phrase in message for phrase in ['start a simulation', 'start simulation', 'begin simulation', 'launch simulation']):
            return 'welcome'
        
        # Strip common phrases
        phrases_to_remove = [
            'start', 'initiate', 'begin', 'launch', 'run', 'execute',
            'simulation', 'simulate', 'a simulation', 'the simulation',
            'incident', 'response', 'incident response'
        ]
        
        for phrase in phrases_to_remove:
            message = message.replace(phrase, '').strip()
        
        # Remove extra whitespace and punctuation
        message = re.sub(r'\s+', ' ', message).strip()
        message = message.strip('.,!?')
        
        # If we're left with just "a" or empty, return welcome
        if message in ['a', 'the', '']:
            return 'welcome'
        
        # Map variations to standard commands
        command_mapping = {
            # Incident types
            'data breach': 'data_breach',
            'databreach': 'data_breach',
            'breach': 'data_breach',
            'phishing': 'phishing',
            'phish': 'phishing',
            'ransomware': 'ransomware',
            'ransom': 'ransomware',
            
            # Command variations
            'welcome': 'welcome',
            'help': 'welcome',
            'menu': 'welcome',
            'next': 'next',
            'continue': 'next',
            'proceed': 'next',
            'team': 'team',
            'roster': 'team',
            'members': 'team',
            'overview': 'overview',
            'plan': 'overview',
            'summary': 'summary',
            'report': 'summary',
            'analysis': 'analysis',
            'analyze': 'analysis',
            'lessons': 'lessons',
            'learned': 'lessons',
            'clarify': 'clarify',
            'explain': 'clarify',
            'details': 'clarify'
        }
        
        return command_mapping.get(message, message)

    def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process user messages for both RAG analysis and incident response"""
        if not self.llm:
            self.initialize()
        
        # Increment interaction count and check limit (consistent with part1.py and part2.py)
        self.state['interaction_count'] += 1
        if self.state['interaction_count'] > MAX_INTERACTIONS:
            return ("Session limit reached. Please restart the app to continue. "
                    "This is to prevent excessive usage and potential infinite loops.")
        
        # Normalize the command
        normalized_message = self.normalize_command(message)
        logger.debug(f"Original message: '{message}' -> Normalized: '{normalized_message}'")
        
        # Check if this is an incident response command
        if (normalized_message in ['welcome', 'next', 'team', 'overview', 'summary', 'analysis', 'lessons', 'clarify'] or 
            normalized_message in self.incident_guide.incident_types):
            return self.incident_guide.process_incident_message(normalized_message, history)
        logger.debug(f"Processing message: {normalized_message}")
        
        # Check if this is a section analysis request
        section = self._find_section(normalized_message)
        if section:
            return self._analyze_section(section)
        
        # Default to RAG-based policy analysis
        return self._process_policy_analysis(normalized_message, history)

    def _find_section(self, user_input: str) -> Optional[str]:
        """Find if user input matches any main section"""
        for section in self.state['main_sections']:
            if section.lower() in user_input.lower():
                return section
        return None

    def _analyze_section(self, section: str) -> str:
        """Analyze a specific section using RAG"""
        # Retrieve relevant documents
        rag_docs = self.rag_retriever.retrieve(section)
        rag_text = "\n\n".join([f"From {doc['doc_name']} ({doc['section']}): {doc['content']}" for doc in rag_docs])
        
        # Generate analysis
        prompt = ANALYSIS_PROMPT.format(section=section, rag_docs=rag_text)
        response = self.llm.invoke(prompt)
        
        # Store recommendations
        self.state['recommendations_by_section'][section] = response.content if hasattr(response, "content") else response
        
        return response.content if hasattr(response, "content") else response

    def _process_policy_analysis(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process general policy analysis questions"""
        # Retrieve relevant documents
        rag_docs = self.rag_retriever.retrieve(message)
        rag_text = "\n\n".join([f"From {doc['doc_name']} ({doc['section']}): {doc['content']}" for doc in rag_docs])
        
        # Generate response
        prompt = f"""You are an expert security consultant. The user asked: {message}

Here are relevant excerpts from their policy documents:
{rag_text}

Provide a comprehensive, helpful response based on the documents and security best practices."""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def review_recommendations(self) -> str:
        """Review all recommendations made during the session"""
        if not self.state['recommendations_by_section']:
            return "No recommendations have been generated yet. Please analyze some sections first."
        
        return self.download_recommendations()

    def download_recommendations(self) -> str:
        """Format recommendations for download"""
        if not self.state['recommendations_by_section']:
            return "No recommendations available."
        
        report = "VCISO Security Recommendations Report\n"
        report += "=" * 50 + "\n\n"
        
        for section, recommendation in self.state['recommendations_by_section'].items():
            report += f"Section: {section}\n"
            report += "-" * 30 + "\n"
            report += recommendation + "\n\n"
        
        return report

