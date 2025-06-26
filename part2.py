from typing import List, Dict, Optional, TypedDict
import gradio as gr
import PyPDF2
import docx
import os
import glob
import json
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

logger = logging.getLogger("vciso.part2")
logging.basicConfig(level=logging.INFO)

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

    def process_message(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        self.state['interaction_count'] += 1
        if self.state['interaction_count'] > MAX_INTERACTIONS:
            return ("Session limit reached. Please restart the app to continue. "
                    "This is to prevent excessive usage and potential infinite loops.")

        user_input = message.strip().lower()

        if user_input == 'review':
            return self.review_recommendations()
        if user_input == 'download':
            return self.download_recommendations()

        selected_section = self._find_section(user_input)
        if selected_section:
            return self._analyze_section(selected_section)

        return self._get_welcome_message()

    def _find_section(self, user_input: str) -> Optional[str]:
        for section in self.state['main_sections']:
            if user_input.lower() == section.lower():
                return section
        return None

    def _get_welcome_message(self) -> str:
        welcome = "Welcome! I can help you evaluate your incident response policies against industry best practices.\n\n"
        return welcome + self._get_selection_prompt()

    def _get_selection_prompt(self) -> str:
        all_sections = self.state['main_sections']
        evaluated_sections = list(self.state['recommendations_by_section'].keys())
        unevaluated_sections = [s for s in all_sections if s not in evaluated_sections]

        message = ""
        if unevaluated_sections:
            message += "Here are the sections yet to be evaluated:\n"
            message += "- " + "\n- ".join(unevaluated_sections) + "\n\nPlease select a section to analyze.\n\n"
        else:
            message += "All sections have been evaluated.\n\n"
        
        if evaluated_sections:
            message += "Evaluated sections:\n"
            message += "- " + "\n- ".join(evaluated_sections) + "\n\n"
        
        message += "You can also type 'review' to see all recommendations so far, or 'download' to save them."
        return message

    def _analyze_section(self, section: str) -> str:
        if section in self.state['recommendations_by_section']:
            return (
                f"The '{section}' section has already been evaluated.\n\n" +
                self._get_selection_prompt()
            )
        logger.info(f"Analyzing section: {section}")
        rag_docs = self.rag_retriever.retrieve(section, top_k=10)
        logger.info(f"Retrieved {len(rag_docs)} docs for section: {section}")

        rag_docs_str = "\n\n".join([
            f"Document: {doc['doc_name']} ({doc['section']})\nContent: {doc['content']}"
            for doc in rag_docs
        ]) if rag_docs else "No relevant documents found."

        prompt = ANALYSIS_PROMPT.format(section=section, rag_docs=rag_docs_str)
        
        response = self.llm.invoke(prompt)
        analysis = response.content if hasattr(response, "content") else response

        self.state['recommendations_by_section'][section] = analysis
        
        return (
            analysis + 
            "\n\n------------------------------------------------------------\n" +
            self._get_selection_prompt()
        )

    def review_recommendations(self) -> str:
        if not self.state['recommendations_by_section']:
            return "No recommendations have been generated yet. Please select a section to analyze first."

        full_report = "--- Incident Response Plan Recommendations ---\n\n"
        for section, recommendations in self.state['recommendations_by_section'].items():
            full_report += f"--- Section: {section} ---\n"
            full_report += recommendations + "\n\n"
        
        return full_report

    def download_recommendations(self) -> str:
        report = self.review_recommendations()
        if report.startswith("No recommendations"):
            return report
        
        path = os.path.join(os.getcwd(), 'irp_recommendations.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        return f"Recommendations saved to: {path}"
