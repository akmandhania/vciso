from typing import List, Dict, Optional, TypedDict
import gradio as gr
import PyPDF2
import docx
import os
import glob
import json
import logging
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from opik.integrations.langchain import OpikTracer
import chromadb
from bs4 import BeautifulSoup

logger = logging.getLogger("vciso.part2")
logging.basicConfig(level=logging.INFO)

# Configuration flags
ENABLE_OPIK_TRACING = False  # Set to True to enable Opik tracing

MAX_TOKENS = 1024
MAX_INTERACTIONS = 30
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "IRCurrentDocuments")

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

class PolicyAnalysisState(TypedDict):
    """State for the policy analysis workflow."""
    user_message: str
    current_section: Optional[str]
    rag_docs: List[Dict[str, str]]
    analysis_result: str
    recommendations_by_section: Dict[str, str]
    interaction_count: int

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
        if self.chunks:
            logger.info(f"First chunk: {self.chunks[0]}")

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

class VisoPart2:
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

    def initialize(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.search_tool = TavilySearch(max_results=3, max_tokens=MAX_TOKENS)
        self.state['main_sections'] = get_main_sections_from_llm(self.llm)
        
        # Create the LangGraph workflow
        graph = StateGraph(PolicyAnalysisState)
        
        # Add nodes
        graph.add_node("command_router", self._command_router_node)
        graph.add_node("section_analyzer", self._section_analyzer_node)
        graph.add_node("welcome_handler", self._welcome_handler_node)
        graph.add_node("selection_handler", self._selection_handler_node)
        graph.add_node("review_handler", self._review_handler_node)
        graph.add_node("download_handler", self._download_handler_node)
        
        # Add edges
        graph.add_edge(START, "command_router")
        graph.add_edge("command_router", "welcome_handler")
        graph.add_edge("command_router", "selection_handler")
        graph.add_edge("command_router", "section_analyzer")
        graph.add_edge("command_router", "review_handler")
        graph.add_edge("command_router", "download_handler")
        graph.add_edge("welcome_handler", END)
        graph.add_edge("selection_handler", END)
        graph.add_edge("section_analyzer", END)
        graph.add_edge("review_handler", END)
        graph.add_edge("download_handler", END)
        
        # Compile the graph
        self.graph = graph.compile()

        # Initialize Opik tracer only if enabled
        if ENABLE_OPIK_TRACING:
            self.opik_tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
            logger.info("Opik tracing enabled")
        else:
            self.opik_tracer = None
            logger.info("Opik tracing disabled")

    def _command_router_node(self, state: PolicyAnalysisState):
        """Route the user message to the appropriate handler."""
        message = state["user_message"].strip().lower()
        
        if message in ["start", "begin", "help", "welcome"]:
            return {"next_step": "welcome"}
        elif any(section.lower() in message for section in self.state['main_sections']):
            return {"next_step": "analyze"}
        elif message in ["review", "summary", "recommendations"]:
            return {"next_step": "review"}
        elif message in ["download", "save", "export"]:
            return {"next_step": "download"}
        else:
            return {"next_step": "selection"}

    def _welcome_handler_node(self, state: PolicyAnalysisState):
        """Handle welcome and help messages."""
        return {"analysis_result": self._get_welcome_message()}

    def _selection_handler_node(self, state: PolicyAnalysisState):
        """Handle section selection."""
        return {"analysis_result": self._get_selection_prompt()}

    def _section_analyzer_node(self, state: PolicyAnalysisState):
        """Analyze a specific section."""
        section = self._find_section(state["user_message"])
        if section:
            analysis = self._analyze_section(section)
            # Store the analysis result
            self.state['recommendations_by_section'][section] = analysis
            return {"analysis_result": analysis, "current_section": section}
        else:
            return {"analysis_result": "I couldn't identify a specific section. Please specify which section you'd like to analyze."}

    def _review_handler_node(self, state: PolicyAnalysisState):
        """Review all recommendations."""
        return {"analysis_result": self.review_recommendations()}

    def _download_handler_node(self, state: PolicyAnalysisState):
        """Download recommendations."""
        return {"analysis_result": self.download_recommendations()}

    def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the LangGraph workflow."""
        logger.info(f"Processing message: {message}")
        
        # Increment interaction count
        self.state['interaction_count'] += 1
        if self.state['interaction_count'] > MAX_INTERACTIONS:
            return "Session limit reached. Please restart the app to continue."
        
        # Create initial state
        initial_state = {
            "user_message": message,
            "current_section": None,
            "rag_docs": [],
            "analysis_result": "",
            "recommendations_by_section": self.state['recommendations_by_section'],
            "interaction_count": self.state['interaction_count']
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
        
        # Update state with results
        if "current_section" in result:
            self.state['current_section'] = result["current_section"]
        if "recommendations_by_section" in result:
            self.state['recommendations_by_section'].update(result["recommendations_by_section"])
        
        logger.info(f"Policy analysis completed. Response length: {len(result.get('analysis_result', ''))}")
        return result.get("analysis_result", "No response generated")

    def _find_section(self, user_input: str) -> Optional[str]:
        """Find which section the user is referring to."""
        user_input_lower = user_input.lower()
        for section in self.state['main_sections']:
            if section.lower() in user_input_lower:
                return section
        return None

    def _get_welcome_message(self) -> str:
        """Get the welcome message."""
        return (
            "Welcome to the Incident Response Plan Policy Analyzer!\n\n"
            "I can help you analyze your existing policy documents and provide recommendations "
            "for improving your incident response plan.\n\n"
            "Available sections to analyze:\n" +
            "\n".join(f"- {section}" for section in self.state['main_sections']) +
            "\n\nTo get started, tell me which section you'd like to analyze, or type 'review' to see all recommendations."
        )

    def _get_selection_prompt(self) -> str:
        """Get the section selection prompt."""
        return (
            "Please specify which section of your Incident Response Plan you'd like me to analyze:\n\n" +
            "\n".join(f"- {section}" for section in self.state['main_sections']) +
            "\n\nYou can also type:\n"
            "- 'review' to see all recommendations\n"
            "- 'download' to save recommendations to a file"
        )

    def _analyze_section(self, section: str) -> str:
        """Analyze a specific section using RAG."""
        logger.info(f"Analyzing section: {section}")
        
        # Retrieve relevant documents
        query = f"incident response plan {section.lower()} policy procedures guidelines"
        rag_docs = self.rag_retriever.retrieve(query, top_k=5)
        
        if not rag_docs:
            rag_docs_text = "No relevant documents found in the current document set."
        else:
            rag_docs_text = "\n\n".join([
                f"Document: {doc['doc_name']}\nSection: {doc['section']}\nContent: {doc['content'][:500]}..."
                for doc in rag_docs
            ])
        
        # Create the analysis prompt
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        # Perform the analysis
        analysis = chain.invoke({
            "section": section,
            "rag_docs": rag_docs_text
        })
        
        return f"## Analysis for {section}\n\n{analysis}"

    def review_recommendations(self) -> str:
        """Review all recommendations made so far."""
        if not self.state['recommendations_by_section']:
            return "No recommendations have been generated yet. Please analyze at least one section first."
        
        review = "## Policy Analysis Summary\n\n"
        for section, analysis in self.state['recommendations_by_section'].items():
            review += f"### {section}\n{analysis}\n\n"
        
        return review

    def download_recommendations(self) -> str:
        """Download recommendations to a file."""
        if not self.state['recommendations_by_section']:
            return "No recommendations to download. Please analyze at least one section first."
        
        content = self.review_recommendations()
        filename = "policy_recommendations.txt"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return f"Recommendations saved to: {filepath}"
