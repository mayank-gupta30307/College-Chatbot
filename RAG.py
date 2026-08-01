import json
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
import re
import requests
from pypdf import PdfReader
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage



CURR_DIR = os.path.dirname(__file__)

with open(os.path.join(CURR_DIR , "Data" , "ContactUs.json"),"r",encoding="utf-8") as f:
    CONTACT_DATA = json.load(f)


@tool
def get_contact(category: str, value: str) -> str:
    """
    Retrieve official IGNOU contact information.

    Use this tool whenever the user asks for:
    - Regional centre phone numbers or contact details.
    - IGNOU head office department phone numbers.
    - IGNOU address.
    - IGNOU office timings.

    Arguments:
        category:
            One of:
            - "regional_centre"
            - "head_office"
            - "office_info"

        value:
            If category is "regional_centre":
                Pass only the regional centre name.
                Examples:
                - Chennai
                - Agartala
                - Delhi 1
                - Mumbai

            If category is "head_office":
                Pass only the department name.
                Examples:
                - Student Service Centre
                - Assignment Marks
                - Certificate Programmes

            If category is "office_info":
                Pass either:
                - address
                - office timings

    Return the exact information from the IGNOU contact database.
    """


    category = category.strip().lower()
    value = value.strip().lower()

    if category == "regional_centre":
        centres = CONTACT_DATA["Regional Centres Officials Contact Number"]
        for centre , number in centres.items():
            if centre.lower() == value:
                return (
                    f"Regional Centre: {centre}\n"
                    f"Phone Number(s): {number}"
                )

        # Partial match
        for centre, number in centres.items():
            if value in centre.lower():
                return (
                    f"Regional Centre: {centre}\n"
                    f"Phone Number(s): {number}"
                )
        return """
NO_RELEVANT_CONTEXT_FOUND

No matching regional centre found.
"""


    elif category == "head_office":
        departments = CONTACT_DATA["IGNOU Head Office Contact Number"]
        for department, number in departments.items():
            if department.lower() == value:
                return (
                    f"Department: {department}\n"
                    f"Contact: {number}"
                )

        # Partial match
        for department, number in departments.items():
            if value in department.lower():
                return (
                    f"Department: {department}\n"
                    f"Contact: {number}"
                )

        return """
NO_RELEVANT_CONTEXT_FOUND

No matching head office department found.
"""

    elif category == "office_info":
        office = CONTACT_DATA["Address"]
        if "address" in value:
            return office["Address"]
        if ("timing" in value or "office timing" in value or "office timings" in value or "hours" in value ):
            return office["Office Timings"]
        return "Available office information are address and office timings."
    return (
        "Invalid category. "
        "Use one of: regional_centre, head_office, office_info."
    )




@tool
def get_question_paper(query: str) -> str:
    """
    Retrieve an IGNOU Previous Year Question Paper.

    Use this tool whenever the user asks about:
    - Previous year papers (PYQs)
    - Question papers
    - Questions from a paper
    - First/last question
    - Summaries of a paper
    - Contents of a paper

    Input:
        query: The user's complete request.

    Examples:
        "Show ACC-01 June 2025 paper"

        "First question of June 2025 ACC-01"

        "Summarize December 2024 BCS-042"

        "Question 5 of MCS-021 December 2025"

    Returns:
        The complete text extracted from the requested paper.
    """

    query_lower = query.lower()
    month = None
    if "june 2025" in query_lower or "june2025" in query_lower:
        month = "June2025"
    elif "december 2025" in query_lower or "dec2025" in query_lower:
        month = "December2025"
    elif "december 2024" in query_lower or "dec2024" in query_lower:
        month = "December2024"

    if month is None:
        return (
            "Could not determine the session.\n"
            "Supported sessions are:\n"
            "- June 2025\n"
            "- December 2025\n"
            "- December 2024"
        )
    json_path = os.path.join(os.curdir , "Data" , month + ".json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except Exception as e:
        return f"Could not load metadata.\n{e}"
    paper_name = None
    for filename in papers.keys():
        base = filename.lower().replace(".pdf", "")
        if base in query_lower:
            paper_name = filename
            break


    if paper_name is None:
        match = re.search(r"[A-Za-z]{2,5}-\d{2,3}",query)
        if match:
            code = match.group(0).upper()
            for filename in papers.keys():
                if filename.upper().startswith(code):
                    paper_name = filename
                    break
    if paper_name is None:
        return """
NO_RELEVANT_CONTEXT_FOUND

Could not determine which paper was requested.
"""
    link = papers[paper_name]
    pdf_dir = os.path.join(os.curdir , "Data" , "PDFs")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir , paper_name)

    if not os.path.exists(pdf_path):
        try:
            response = requests.get(link , stream=True , timeout=30)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            return (
                f"Failed to download paper.\n"
                f"{e}"
            )
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip() == "":
            return """
NO_RELEVANT_CONTEXT_FOUND

The PDF contains no extractable text.
"""
        return (
            f"Paper: {paper_name}\n"
            f"Session: {month}\n\n"
            f"{text}"
        )
    except Exception as e:
        return (
            f"Failed to read PDF.\n"
            f"{e}"
        )


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_db = Chroma(
    persist_directory=os.path.join(os.curdir , "Vectorstores" , "QuestionPapers"),
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 10
    }
)


@tool
def search_question_papers(query: str) -> str:
    """
    Search the contents of the vector database containing IGNOU Previous Year Question Papers.

    IMPORTANT:
    Use this tool ONLY if the required answer cannot be obtained using the
    `get_question_paper` tool.

    Prefer `get_question_paper` when:
    - The user requests a specific paper.
    - The user specifies the paper code and/or session.
    - The answer can be obtained directly from a single question paper.

    Use this tool only when:
    - `get_question_paper` could not find the requested paper.
    - The user does not know the paper code or session.
    - The user asks broad questions requiring searching across multiple papers.
    - The user asks about recurring topics or patterns across papers.
    - The user asks which paper contains a particular question or topic.
    - Information may be spread across multiple question papers.

    Input:
        query: The user's complete request.

    Returns:
        The most relevant excerpts retrieved from the vector database.
    """


    docs = retriever.invoke(query)
    if not docs:
        return "No relevant question paper found."
    context = []
    for doc in docs:
        context.append(
            f"""
Paper: {doc.metadata.get("source","Unknown")}

{doc.page_content}
"""
        )

    return f"""
========== RETRIEVED CONTEXT ==========

{chr(10).join(context)}

=======================================

Everything above is retrieved from the vector database.
If it does not contain the answer, answer:

"I don't know the answer based on the available IGNOU data."
"""










load_dotenv()

SYSTEM_PROMPT = """
You are an official IGNOU assistant.

You have access to three tools:
1. get_contact
2. get_question_paper
3. search_question_papers

Rules:

- Never answer from your own knowledge.
- Never guess.
- Never infer missing information.
- Never fabricate phone numbers, addresses, dates, policies, courses or question papers.
- If none of the tools provide enough information, reply exactly:
- Answer in bullet points
- Always cite the sources

"I don't know the answer based on the available IGNOU data."

- If search_question_papers returns NO_RELEVANT_CONTEXT_FOUND, reply exactly:

"I don't know the answer based on the available IGNOU data."

Treat tool outputs as the only source of truth.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
tools = [get_contact, 
         get_question_paper , 
         search_question_papers
         ]
llm = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages":[response]}


def should_continue(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools",ToolNode(tools))
builder.set_entry_point("chatbot")
builder.add_conditional_edges("chatbot" , should_continue, {"tools" : "tools" , END : END})
builder.add_edge("tools" , "chatbot")
graph = builder.compile()



def get_graph():
    return graph