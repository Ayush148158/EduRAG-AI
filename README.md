# EduRAG-AI
EduRAG AI is a LangGraph-powered RAG chatbot that intelligently classifies student queries into academic, fee, or general categories. It retrieves relevant information from PDF knowledge bases using FAISS and Hugging Face embeddings, then generates accurate, context-aware responses with Mistral AI.
# 🎓 CampusRAG AI – Intelligent Student Assistant

> An AI-powered student assistant built using **LangGraph**, **Mistral AI**, **FAISS**, and **Retrieval-Augmented Generation (RAG)** to answer academic and fee-related queries from institutional documents.

---

## 📖 Overview

CampusRAG AI is an intelligent chatbot designed to assist students by providing accurate, context-aware answers from official college documents. Instead of relying on the LLM's memory, it retrieves relevant information from PDF handbooks and fee documents before generating responses.

The system uses a **Conditional RAG Pipeline** where every user query is first classified and then routed to the appropriate knowledge base.

---

## ✨ Features

* 🤖 AI-powered Student Assistant
* 📚 Retrieval-Augmented Generation (RAG)
* 🧠 Intelligent Query Classification
* 📄 PDF Knowledge Base Support
* ⚡ Fast Semantic Search using FAISS
* 🔍 Hugging Face Sentence Embeddings
* 🌐 LangGraph Workflow
* 💬 Context-Aware Responses
* 🎯 Multi-domain Retrieval (Academic & Fee)
* 🔄 Expandable Architecture

---

## 🏗️ System Architecture

```text
                User Query
                     │
                     ▼
          Query Classification
       (Academic / Fee / General)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
 Academic RAG     Fee RAG     General LLM
        │            │            │
        └────────────┼────────────┘
                     ▼
            Mistral AI Response
                     ▼
                 Final Answer
```

---

# 🛠 Tech Stack

| Technology                     | Purpose                |
| ------------------------------ | ---------------------- |
| Python                         | Core Development       |
| LangGraph                      | Workflow Orchestration |
| LangChain                      | LLM Framework          |
| Mistral AI                     | Large Language Model   |
| Hugging Face Embeddings        | Semantic Embeddings    |
| FAISS                          | Vector Database        |
| PyPDFLoader                    | PDF Processing         |
| RecursiveCharacterTextSplitter | Document Chunking      |
| dotenv                         | Environment Variables  |

---

# 📂 Project Structure

```text
CampusRAG-AI/
│
├── academics_handbook.pdf
├── fee_structure.pdf
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# ⚙️ Workflow

### 1️⃣ User enters a question

Example:

> What is the minimum attendance required?

↓

### 2️⃣ Query Classifier

The AI classifies the question into one of three categories:

* Academic
* Fee
* General

↓

### 3️⃣ Conditional Routing

Depending on the category, LangGraph routes the query to the appropriate retriever.

↓

### 4️⃣ Document Retrieval

Relevant document chunks are retrieved using:

* FAISS Vector Store
* HuggingFace Embeddings

↓

### 5️⃣ Response Generation

The retrieved context is passed to **Mistral AI**, which generates an accurate response.

---

# 📚 Knowledge Sources

Current knowledge bases include:

* 📘 Academic Handbook
* 💰 Fee Structure

Additional PDFs can easily be added for domains such as:

* Hostel
* Library
* Placement
* Examination
* Scholarships
* Rules & Regulations

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/CampusRAG-AI.git
```

```bash
cd CampusRAG-AI
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Create Environment File

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

### Run

```bash
python app.py
```

---

# 💬 Example

### Input

```
How many credits are required to graduate?
```

↓

### Output

```
According to the academic handbook, students must complete the required credit requirements specified by their programme to be eligible for graduation.
```

---

# 🔮 Future Improvements

* 🌐 Web Interface (Streamlit / React)
* 🔊 Voice Assistant
* 📱 WhatsApp Integration
* 📧 Email Support
* 🗂 Multiple Knowledge Bases
* 💾 Persistent Vector Database
* 📊 Admin Dashboard
* 🔐 Authentication
* 🎤 Speech-to-Text
* 🌍 Multi-language Support

#

#
