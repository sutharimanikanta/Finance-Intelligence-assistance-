NeoStats — AI-Powered Investment Research Assistant
An intelligent financial research assistant that combines portfolio analytics, document retrieval, and real-time market intelligence into a unified chatbot interface.

Built with Streamlit, LLMs, SQL analytics, RAG pipelines, and live web search.

The system allows users to query portfolio data, analyze financial documents, and retrieve market insights through a single conversational interface.

Project Architecture
text
project/
│
├── config/
│   └── config.py
│
├── models/
│   ├── llm.py
│   └── embeddings.py
│
├── utils/
│   ├── intent_sql_engine.py
│   ├── db_manager.py
│   ├── rag_engine.py
│   ├── web_search.py
│   └── synthesizer.py
│
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml
Folder Overview
Folder	Description
config/	Stores configuration variables and API keys
models/	LLM wrapper and embedding model
utils/	Core system logic (SQL engine, RAG pipeline, web search)
app.py	Main Streamlit application
.streamlit/	Streamlit UI configuration
Key Features
1. Reliable SQL Generation (Zero-Hallucination SQL)
The LLM does not generate SQL directly. Instead:

The model classifies the user query into a structured QueryIntent

SQL queries are generated deterministically using a rule-based generator

Every column reference is validated against the live SQLite schema

This prevents SQL hallucinations and ensures accurate analytics.

2. Intelligent Query Routing
A triage router determines which data sources should be used before answering a query.

Query Example	Sources Used
Show Garfield holdings	SQL
Explain what Nvidia does	RAG / Web
How is Nvidia affecting my portfolio?	SQL + Web
Explain Berry Brand and show my exposure	SQL + RAG
3. Multi-Index RAG System
Uploaded documents are indexed using vector embeddings.

Features:

Each document creates its own vector index

Queries can search across multiple indexes

Results are deduplicated before retrieval

Supported formats:

PDF

DOCX

4. Cache-Aside Retrieval
To reduce repeated LLM calls, the system uses in-memory caching.

Property	Value
Cache type	In-memory
TTL	5 minutes
Max entries	200
Cache key format	MD5(index_name + query)
5. Clarification Handling
If the system detects an ambiguous query, it requests clarification.

Example:

text
User: Show profit
System: Which portfolio do you mean?
A retry limit prevents infinite clarification loops.

Installation
Clone the repository:

bash
git clone <repo-url>
cd project
Install dependencies:

bash
pip install -r requirements.txt
Set API keys:

bash
export GROQ_API_KEY="your_key"
export TAVILY_API_KEY="your_key"
Run the application:

bash
streamlit run app.py
Streamlit Cloud Deployment
Push the repository to GitHub

Go to Streamlit Cloud

Create a new app

Add secrets in deployment settings:

text
GROQ_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
Set the main file to: app.py

Deploy the application

Usage
Step 1 — Load datasets
Upload the following files:

holdings.csv

trades.csv

Click Load Datasets in the sidebar.

Step 2 — Upload documents (optional)
Upload financial research documents.

Supported formats:

PDF

DOCX

Step 3 — Ask questions
Query	Data Sources
Total market value for Garfield portfolio	SQL
Which portfolios hold bonds?	SQL
Yearly trade volume by fund	SQL
Explain bond duration	RAG / Web
How is the Fed rate decision affecting HoldCo 1?	SQL + Web
Show Nvidia exposure and latest news	SQL + Web
Expected Dataset Format
holdings.csv
text
AsOfDate,PortfolioName,SecurityTypeName,SecName,Qty,Price,MV_Base
trades.csv
text
TradeDate,SettleDate,TradeTypeName,SecurityId,SecurityName,Ticker,Quantity,Price,PortfolioName,TotalCash,CustodianName
Tech Stack
Python

Streamlit

SQLite

Sentence Transformers

Vector Search (FAISS-style)

Groq LLM API

Tavily Search API
