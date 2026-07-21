import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv() 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def build_retriver(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

academic_retriever = build_retriver("academics_handbook.pdf")
fee_retriever = build_retriver("fee_structure.pdf")

llm = ChatMistralAI(model_name="open-mistral-7b", api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.4)

class State(TypedDict):
    programme : str
    messages : Annotated[list, add_messages]
    query_type : str
    retrieved_context : str

def classifier_node(state : State) -> dict:
    """Stage 1 : Classifies the query type based on the user's input."""

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {state['messages'][-1]['content']}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}

def academic_retriever_node(state : State) -> dict:
    """Retrieves relevant academic information based on the user's query."""

    query = state['messages'][-1]['content']
    retrieved_docs = academic_retriever.get_relevant_documents(query)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_texts)

    return {"retrieved_context": retrieved_context}

def fee_retriever_node(state : State) -> dict:
    """Retrieves relevant fee information based on the user's query."""

    query = state['messages'][-1]['content']
    retrieved_docs = fee_retriever.get_relevant_documents(query)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_texts)

    return {"retrieved_context": retrieved_context} 

def general_response_node(state : State) -> dict: 
    """Answer directly using LLM's own knowledge, no retrieval needed.""" 
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state : State) -> dict:
    """Stage 4 : Generates a response based on the retrieved context and user's query."""

    prompt = (
        "You are a helpful assistant. Use the following context to answer the user's query. "
        "If the context does not contain enough information, respond with 'I don't know'.\n\n"
        f"Context: {state['retrieved_context']}\n\n"
        f"User Query: {state['messages'][-1]['content']}\n\n"
        "Provide a clear and concise answer."
    )
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

def route_query(state:State):
    if state['query_type'] == "academic":
        return "academic_retriever_node"        
    elif state['query_type'] == "fee":
        return "fee_retriever_node"
    else:
        return "general_response_node"
    
graph = StateGraph(State)

graph.add_node("classifier_node", classifier_node)
graph.add_node("academic_retriever_node", academic_retriever_node)  
graph.add_node("fee_retriever_node", fee_retriever_node)
graph.add_node("general_response_node", general_response_node)
graph.add_node("response_node", response_node)

graph.add_edge(START, "classifier_node")
graph.add_conditional_edges("classifier_node", route_query)
graph.add_edge("academic_retriever_node", "response_node")
graph.add_edge("fee_retriever_node", "response_node")
graph.add_edge("general_response_node", "response_node")

graph.add_edge("response_node", END)
app = graph.compile()

print("Conditional RAG pipeline is ready. You can now send queries to the app.")

print("which programme are you in? ")
print("1. BCA")
print("2. BBA")
print("3. B.com(H)")

choice = input("Enter the number corresponding to your programme: ")
programme_map = {"1": "BCA", "2": "BBA", "3": "B.com(H)"}
programme = programme_map.get(choice, "BCA")
print(f"You selected: {programme}")

while True:
    user_query = input("You:  ")

    if user_query.lower() in ["exit","quit"]:
        break
    
    result = app.invoke({
        "programme": programme,
        "messages": [("human",user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")
