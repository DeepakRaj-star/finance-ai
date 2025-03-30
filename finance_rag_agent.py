"""
Finance RAG (Retrieval Augmented Generation) Agent Module.

This module implements a sophisticated RAG system that uses LangChain and uAgents 
to provide enhanced financial insights and document-based Q&A capabilities.
"""

import os
import asyncio
import json
import datetime
import threading
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile

# uAgents for autonomous agent capabilities
from uagents import Agent, Context, Protocol, Model
from pydantic import Field, BaseModel

# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.base import Chain

# Financial libraries
import pandas as pd
import numpy as np

# Import app utils
import ai_agents
import db_service as db

# Ensure OpenAI API key is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# Define document types and categories
FINANCIAL_DOCUMENT_TYPES = ["statement", "invoice", "receipt", "contract", "report"]
FINANCIAL_CATEGORIES = [
    "banking", "investment", "retirement", "taxation", "insurance", 
    "mortgage", "loan", "credit", "budgeting", "market_analysis"
]

# Define message models for RAG agent communication
class RagQueryRequest(Model):
    """Request for RAG-based financial query."""
    user_id: str
    query: str
    document_type: Optional[str] = None
    category: Optional[str] = None

class RagQueryResponse(Model):
    """Response with RAG-based financial insights."""
    user_id: str
    answer: str
    sources: List[str] = []
    confidence: float = 0.0

class DocumentIndexRequest(Model):
    """Request to index financial documents."""
    user_id: str
    document_path: str
    document_type: str
    category: str

class DocumentIndexResponse(Model):
    """Response after document indexing."""
    user_id: str
    success: bool
    document_id: Optional[str] = None
    error: Optional[str] = None

class FinancialReportRequest(Model):
    """Request for comprehensive financial report."""
    user_id: str
    report_type: str = "overview"  # overview, detailed, forecast
    time_period: str = "month"    # week, month, quarter, year

class FinancialReportResponse(Model):
    """Response with financial report."""
    user_id: str
    report: str = ""
    insights: List[str] = []
    action_items: List[str] = []

# Define RAG Agent
class FinanceRagAgent:
    """Financial RAG Agent that retrieves relevant information and generates insights."""
    
    def __init__(self):
        """Initialize the Finance RAG Agent."""
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.1, model="gpt-4o")
        self.vector_stores = {}
        self.user_indexes = {}
        self.documents_path = Path("user_data")
        self.knowledge_base_path = Path("knowledge_base")
        self._initialize_folders()
        
    def _initialize_folders(self):
        """Initialize necessary folders."""
        # User data for user-specific documents
        self.documents_path.mkdir(exist_ok=True)
        
        # Knowledge base for financial information
        self.knowledge_base_path.mkdir(exist_ok=True)
        for category in FINANCIAL_CATEGORIES:
            (self.knowledge_base_path / category).mkdir(exist_ok=True)
    
    def _get_user_index_path(self, user_id: str) -> Path:
        """Get the path to a user's vector index."""
        user_path = self.documents_path / user_id / "vector_index"
        user_path.mkdir(exist_ok=True, parents=True)
        return user_path
    
    def _load_or_create_index(self, user_id: str) -> FAISS:
        """Load existing index or create a new one for the user."""
        if user_id in self.vector_stores:
            return self.vector_stores[user_id]
        
        index_path = self._get_user_index_path(user_id)
        try:
            # Try to load existing index
            if (index_path / "index.faiss").exists():
                vector_store = FAISS.load_local(str(index_path), self.embeddings)
                self.vector_stores[user_id] = vector_store
                return vector_store
        except Exception as e:
            print(f"Error loading index for {user_id}: {str(e)}")
            # Fall through to create new index
        
        # Create new empty index
        vector_store = FAISS.from_texts(["Financial assistant initialization"], self.embeddings)
        self.vector_stores[user_id] = vector_store
        return vector_store
    
    def _save_index(self, user_id: str):
        """Save the user's vector index."""
        if user_id in self.vector_stores:
            index_path = self._get_user_index_path(user_id)
            self.vector_stores[user_id].save_local(str(index_path))
    
    def index_financial_document(self, user_id: str, document_path: str, 
                               document_type: str, category: str) -> Dict[str, Any]:
        """
        Index a financial document for RAG retrieval.
        
        Args:
            user_id: User ID
            document_path: Path to the document
            document_type: Type of financial document
            category: Financial category
            
        Returns:
            Dict with indexing status
        """
        try:
            # Create user folder if it doesn't exist
            user_docs_path = self.documents_path / user_id / "documents"
            user_docs_path.mkdir(exist_ok=True, parents=True)
            
            # Determine loader based on file extension
            doc_path = Path(document_path)
            if not doc_path.exists():
                return {
                    "success": False,
                    "error": f"Document not found: {document_path}"
                }
            
            # Load document based on file type
            if doc_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(document_path)
            elif doc_path.suffix.lower() == ".csv":
                loader = CSVLoader(document_path)
            else:
                # Default to text loader for txt and other formats
                loader = TextLoader(document_path)
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(doc_path),
                    "user_id": user_id,
                    "document_type": document_type,
                    "category": category,
                    "date_indexed": datetime.datetime.now().isoformat()
                })
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Load or create vector store
            vector_store = self._load_or_create_index(user_id)
            
            # Add documents to vector store
            vector_store.add_documents(chunks)
            
            # Save index
            self._save_index(user_id)
            
            # Generate a document ID
            document_id = f"{user_id}_{document_type}_{int(time.time())}"
            
            # Record document in user's index
            if user_id not in self.user_indexes:
                self.user_indexes[user_id] = []
            
            self.user_indexes[user_id].append({
                "document_id": document_id,
                "path": str(doc_path),
                "type": document_type,
                "category": category,
                "chunks": len(chunks),
                "indexed_at": datetime.datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_indexed": len(chunks)
            }
        
        except Exception as e:
            print(f"Error indexing document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def index_transaction_history(self, user_id: str) -> Dict[str, Any]:
        """
        Index a user's transaction history for RAG retrieval.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with indexing status
        """
        try:
            # Get user transactions
            transactions = db.get_transactions(user_id)
            
            # Convert to dictionary format if needed
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
            
            if not txn_dicts:
                return {
                    "success": False,
                    "error": "No transactions found"
                }
            
            # Create a temporary JSON file to store transactions
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(txn_dicts, tmp, default=str)
                temp_path = tmp.name
            
            # Now index this temporary file
            result = self.index_financial_document(
                user_id=user_id,
                document_path=temp_path,
                document_type="transaction_history",
                category="banking"
            )
            
            # Remove temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            print(f"Error indexing transaction history: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def answer_financial_query(self, user_id: str, query: str, 
                             document_type: Optional[str] = None,
                             category: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a financial query using RAG.
        
        Args:
            user_id: User ID
            query: The query text
            document_type: Optional document type filter
            category: Optional category filter
            
        Returns:
            Dict with answer and sources
        """
        try:
            # Load or create vector store
            vector_store = self._load_or_create_index(user_id)
            
            # Create retriever with metadata filtering if specified
            search_kwargs = {}
            
            if document_type or category:
                filter_dict = {}
                if document_type:
                    filter_dict["document_type"] = document_type
                if category:
                    filter_dict["category"] = category
                
                search_kwargs["filter"] = filter_dict
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, **search_kwargs}
            )
            
            # Create multi-query retriever for better results
            llm = ChatOpenAI(temperature=0.0)
            multi_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=llm
            )
            
            # Define prompt template
            template = """
            You are an expert financial advisor assistant specialized in providing accurate, 
            helpful information about financial topics. Use the context provided to answer 
            the question. If you can't find the answer in the context, say "I don't have enough 
            information to answer this question" rather than making something up. Use a friendly,
            professional tone. If the user's question is about specific transactions or financial
            data, be precise with numbers and calculations. Always explain financial concepts
            clearly without assuming expert knowledge.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=multi_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Run chain
            result = qa_chain({"query": query})
            
            # Extract sources
            sources = []
            seen_sources = set()
            
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "unknown")
                if source not in seen_sources:
                    sources.append({
                        "source": source,
                        "document_type": doc.metadata.get("document_type", "unknown"),
                        "category": doc.metadata.get("category", "unknown")
                    })
                    seen_sources.add(source)
            
            # Generate confidence score
            confidence_query = f"""
            Based on the answer "{result['result']}" to the question "{query}",
            rate your confidence in the answer on a scale from 0.0 to 1.0, where:
            - 0.0 means completely uncertain or the information was not found
            - 0.5 means moderately confident but with some uncertainty
            - 1.0 means extremely confident with direct evidence
            Return only the numeric value.
            """
            
            confidence_response = self.llm.invoke([HumanMessage(content=confidence_query)])
            confidence = 0.7  # Default moderate confidence
            
            try:
                # Extract just the number from the response
                import re
                confidence_text = confidence_response.content
                confidence_match = re.search(r'(\d+\.\d+|\d+)', confidence_text)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))  # Ensure in range [0,1]
            except:
                # If parsing fails, keep default confidence
                pass
            
            return {
                "user_id": user_id,
                "answer": result["result"],
                "sources": sources,
                "confidence": confidence
            }
        
        except Exception as e:
            print(f"Error answering query: {str(e)}")
            return {
                "user_id": user_id,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def generate_financial_report(self, user_id: str, report_type: str = "overview",
                                time_period: str = "month") -> Dict[str, Any]:
        """
        Generate a comprehensive financial report using transactions and indexed documents.
        
        Args:
            user_id: User ID
            report_type: Type of report (overview, detailed, forecast)
            time_period: Time period to analyze (week, month, quarter, year)
            
        Returns:
            Dict with report and insights
        """
        try:
            # Get user transactions
            transactions = db.get_transactions(user_id)
            
            # Convert to dictionary format if needed
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
            
            if not txn_dicts:
                return {
                    "user_id": user_id,
                    "report": "No transaction data available for analysis.",
                    "insights": [],
                    "action_items": []
                }
            
            # Filter by time period
            filtered_txns = txn_dicts
            if time_period != "all":
                today = datetime.datetime.now()
                
                if time_period == "week":
                    start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                elif time_period == "month":
                    start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
                elif time_period == "quarter":
                    start_date = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
                elif time_period == "year":
                    start_date = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                else:
                    start_date = None
                
                if start_date:
                    filtered_txns = [t for t in txn_dicts if t.get("date", "") >= start_date]
            
            # Calculate key financial metrics
            income = sum(t.get("amount", 0) for t in filtered_txns if t.get("type") == "income")
            expenses = sum(t.get("amount", 0) for t in filtered_txns if t.get("type") == "expense")
            net_flow = income - expenses
            
            # Calculate expense categories
            categories = {}
            for t in filtered_txns:
                if t.get("type") == "expense":
                    cat = t.get("category", "Other")
                    if cat in categories:
                        categories[cat] += t.get("amount", 0)
                    else:
                        categories[cat] = t.get("amount", 0)
            
            # Sort categories by amount
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            
            # Get user's financial goals if available
            goals = []
            try:
                goals = db.get_goals(user_id)
            except:
                # If goals not available, continue without them
                pass
            
            # Get user's budgets if available
            budgets = []
            try:
                budgets = db.get_budgets(user_id)
            except:
                # If budgets not available, continue without them
                pass
            
            # Prepare report based on report type
            financial_summary = ""
            
            if report_type == "overview":
                financial_summary = f"""
                # Financial Overview ({time_period.title()})
                
                ## Summary
                - Total Income: ${income:.2f}
                - Total Expenses: ${expenses:.2f}
                - Net Cash Flow: ${net_flow:.2f}
                
                ## Top Expense Categories
                {', '.join([f"{cat}: ${amt:.2f}" for cat, amt in sorted_categories[:3]])}
                
                ## Savings Rate
                {(income - expenses) / income * 100:.1f}% of income saved
                """
            
            elif report_type == "detailed":
                # Additional detailed metrics
                avg_daily_expense = expenses / (30 if time_period == "month" else 
                                              7 if time_period == "week" else 
                                              90 if time_period == "quarter" else 365)
                
                # Build category breakdown
                category_breakdown = "\n".join([f"- {cat}: ${amt:.2f} ({amt/expenses*100:.1f}%)" 
                                             for cat, amt in sorted_categories])
                
                financial_summary = f"""
                # Detailed Financial Report ({time_period.title()})
                
                ## Income & Expense Summary
                - Total Income: ${income:.2f}
                - Total Expenses: ${expenses:.2f}
                - Net Cash Flow: ${net_flow:.2f}
                - Average Daily Expense: ${avg_daily_expense:.2f}
                
                ## Expense Breakdown by Category
                {category_breakdown}
                
                ## Budget Status
                """
                
                # Add budget information if available
                if budgets:
                    for budget in budgets:
                        budget_category = budget.get("category", "Unknown")
                        budget_limit = budget.get("amount", 0)
                        actual_spend = categories.get(budget_category, 0)
                        percent_used = (actual_spend / budget_limit * 100) if budget_limit > 0 else 0
                        
                        status = "✅ Under budget"
                        if percent_used > 90 and percent_used <= 100:
                            status = "⚠️ Approaching limit"
                        elif percent_used > 100:
                            status = "❌ Over budget"
                        
                        financial_summary += f"\n- {budget_category}: ${actual_spend:.2f} / ${budget_limit:.2f} ({percent_used:.1f}%) {status}"
                else:
                    financial_summary += "\nNo budget information available."
                
                # Add goal tracking if available
                financial_summary += "\n\n## Goal Progress\n"
                if goals:
                    for goal in goals:
                        goal_name = goal.get("name", "Unknown")
                        goal_target = goal.get("target_amount", 0)
                        goal_current = goal.get("current_amount", 0)
                        goal_deadline = goal.get("target_date", "No deadline")
                        
                        percent_complete = (goal_current / goal_target * 100) if goal_target > 0 else 0
                        
                        financial_summary += f"\n- {goal_name}: ${goal_current:.2f} / ${goal_target:.2f} ({percent_complete:.1f}% complete, Deadline: {goal_deadline})"
                else:
                    financial_summary += "\nNo financial goals set."
            
            elif report_type == "forecast":
                # Simple forecast based on current spending patterns
                months_forecast = 6
                
                # Monthly averages
                monthly_income = income
                monthly_expense = expenses
                
                if time_period == "week":
                    monthly_income = income * 4.33
                    monthly_expense = expenses * 4.33
                elif time_period == "quarter":
                    monthly_income = income / 3
                    monthly_expense = expenses / 3
                elif time_period == "year":
                    monthly_income = income / 12
                    monthly_expense = expenses / 12
                
                # Generate 6-month forecast
                forecast_data = []
                current_month = datetime.datetime.now().month
                current_year = datetime.datetime.now().year
                
                cumulative_savings = 0
                for i in range(months_forecast):
                    forecast_month = (current_month + i) % 12
                    if forecast_month == 0:
                        forecast_month = 12
                    forecast_year = current_year + (current_month + i - 1) // 12
                    
                    month_income = monthly_income
                    month_expense = monthly_expense
                    month_savings = month_income - month_expense
                    cumulative_savings += month_savings
                    
                    forecast_data.append({
                        "month": f"{forecast_month}/{forecast_year}",
                        "income": month_income,
                        "expenses": month_expense,
                        "savings": month_savings,
                        "cumulative_savings": cumulative_savings
                    })
                
                # Build forecast table
                forecast_table = "\n".join([
                    f"- {d['month']}: Income: ${d['income']:.2f}, Expenses: ${d['expenses']:.2f}, "
                    f"Savings: ${d['savings']:.2f}, Cumulative: ${d['cumulative_savings']:.2f}"
                    for d in forecast_data
                ])
                
                financial_summary = f"""
                # Financial Forecast (Next {months_forecast} Months)
                
                ## Current Monthly Averages
                - Average Monthly Income: ${monthly_income:.2f}
                - Average Monthly Expenses: ${monthly_expense:.2f}
                - Average Monthly Savings: ${monthly_income - monthly_expense:.2f}
                
                ## 6-Month Projection
                {forecast_table}
                
                ## Projected 6-Month Savings
                ${cumulative_savings:.2f}
                """
            
            # Generate insights using LLM
            insights_prompt = f"""
            You are a financial analyst providing actionable insights based on financial data.
            Please analyze this financial summary and provide 3-5 key insights and 2-3 specific 
            action items the user could take to improve their financial health.
            
            Financial Summary:
            {financial_summary}
            
            Respond with a JSON object with two arrays:
            1. "insights": Array of insight strings
            2. "action_items": Array of specific action item strings
            
            Each insight should be concise and data-driven. Each action item should be specific and actionable.
            """
            
            insights_response = self.llm.invoke([HumanMessage(content=insights_prompt)])
            
            insights = []
            action_items = []
            
            try:
                # Parse the JSON response
                import json
                import re
                
                response_text = insights_response.content
                # Extract JSON if it's wrapped in markdown code blocks or other text
                json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1).strip()
                
                # Handle cases where there might be text before/after the JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
                
                response_data = json.loads(response_text)
                insights = response_data.get("insights", [])
                action_items = response_data.get("action_items", [])
            except Exception as e:
                print(f"Error parsing insights: {str(e)}")
                # Fallback to extracting insights from text
                insights = [line.strip() for line in insights_response.content.split('\n') 
                          if line.strip().startswith('-') or line.strip().startswith('*')][:5]
                action_items = []
            
            # Store the report in the database
            db.create_financial_analysis(
                user_id=user_id,
                analysis_type=f"report_{report_type}",
                content=financial_summary,
                meta_data={
                    "time_period": time_period,
                    "insights": insights,
                    "action_items": action_items
                }
            )
            
            return {
                "user_id": user_id,
                "report": financial_summary,
                "insights": insights,
                "action_items": action_items
            }
        
        except Exception as e:
            print(f"Error generating financial report: {str(e)}")
            return {
                "user_id": user_id,
                "report": f"Error generating report: {str(e)}",
                "insights": [],
                "action_items": []
            }

# Create global FinanceRagAgent instance
rag_agent = FinanceRagAgent()

# Define RAG agent protocol
rag_protocol = Protocol("finance-rag")

# Register message handlers for RAG protocol
@rag_protocol.on_message(model=RagQueryRequest)
async def handle_rag_query(ctx: Context, sender: str, msg: RagQueryRequest):
    """Handle RAG-based financial queries."""
    try:
        # Process query with RAG agent
        result = rag_agent.answer_financial_query(
            user_id=msg.user_id,
            query=msg.query,
            document_type=msg.document_type,
            category=msg.category
        )
        
        # Create response
        response = RagQueryResponse(
            user_id=msg.user_id,
            answer=result["answer"],
            sources=[s["source"] for s in result.get("sources", [])],
            confidence=result.get("confidence", 0.0)
        )
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error processing RAG query: {str(e)}")

@rag_protocol.on_message(model=DocumentIndexRequest)
async def handle_document_index(ctx: Context, sender: str, msg: DocumentIndexRequest):
    """Handle financial document indexing requests."""
    try:
        # Process document indexing with RAG agent
        result = rag_agent.index_financial_document(
            user_id=msg.user_id,
            document_path=msg.document_path,
            document_type=msg.document_type,
            category=msg.category
        )
        
        # Create response
        response = DocumentIndexResponse(
            user_id=msg.user_id,
            success=result["success"],
            document_id=result.get("document_id"),
            error=result.get("error")
        )
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error indexing document: {str(e)}")

@rag_protocol.on_message(model=FinancialReportRequest)
async def handle_financial_report(ctx: Context, sender: str, msg: FinancialReportRequest):
    """Handle financial report generation requests."""
    try:
        # Generate financial report with RAG agent
        result = rag_agent.generate_financial_report(
            user_id=msg.user_id,
            report_type=msg.report_type,
            time_period=msg.time_period
        )
        
        # Create response
        response = FinancialReportResponse(
            user_id=msg.user_id,
            report=result["report"],
            insights=result.get("insights", []),
            action_items=result.get("action_items", [])
        )
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error generating financial report: {str(e)}")

# Helper functions for external use
async def query_financial_rag(user_id: str, query: str, 
                             document_type: Optional[str] = None,
                             category: Optional[str] = None) -> Dict[str, Any]:
    """
    Query the finance RAG system.
    
    Args:
        user_id: User ID
        query: The query text
        document_type: Optional document type filter
        category: Optional category filter
        
    Returns:
        Dict with answer and sources
    """
    try:
        return rag_agent.answer_financial_query(
            user_id=user_id,
            query=query,
            document_type=document_type,
            category=category
        )
    except Exception as e:
        print(f"Error in RAG query: {str(e)}")
        return {
            "user_id": user_id,
            "answer": f"Error processing your query: {str(e)}",
            "sources": [],
            "confidence": 0.0
        }

async def index_document(user_id: str, document_path: str, 
                       document_type: str, category: str) -> Dict[str, Any]:
    """
    Index a financial document.
    
    Args:
        user_id: User ID
        document_path: Path to the document
        document_type: Type of financial document
        category: Financial category
        
    Returns:
        Dict with indexing status
    """
    try:
        return rag_agent.index_financial_document(
            user_id=user_id,
            document_path=document_path,
            document_type=document_type,
            category=category
        )
    except Exception as e:
        print(f"Error indexing document: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def generate_report(user_id: str, report_type: str = "overview",
                        time_period: str = "month") -> Dict[str, Any]:
    """
    Generate a financial report.
    
    Args:
        user_id: User ID
        report_type: Type of report (overview, detailed, forecast)
        time_period: Time period to analyze (week, month, quarter, year)
        
    Returns:
        Dict with report and insights
    """
    try:
        return rag_agent.generate_financial_report(
            user_id=user_id,
            report_type=report_type,
            time_period=time_period
        )
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return {
            "user_id": user_id,
            "report": f"Error generating report: {str(e)}",
            "insights": [],
            "action_items": []
        }

async def index_transaction_history(user_id: str) -> Dict[str, Any]:
    """
    Index a user's transaction history.
    
    Args:
        user_id: User ID
        
    Returns:
        Dict with indexing status
    """
    try:
        return rag_agent.index_transaction_history(user_id=user_id)
    except Exception as e:
        print(f"Error indexing transaction history: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Simple synchronous wrappers for easier integration
def query_rag(user_id: str, query: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for query_financial_rag."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(query_financial_rag(user_id, query, **kwargs))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in synchronous RAG query: {str(e)}")
        return {
            "user_id": user_id,
            "answer": f"Error processing your query: {str(e)}",
            "sources": [],
            "confidence": 0.0
        }

def index_user_document(user_id: str, document_path: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for index_document."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(index_document(user_id, document_path, **kwargs))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in synchronous document indexing: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_financial_report(user_id: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for generate_report."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_report(user_id, **kwargs))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in synchronous report generation: {str(e)}")
        return {
            "user_id": user_id,
            "report": f"Error generating report: {str(e)}",
            "insights": [],
            "action_items": []
        }

def index_user_transactions(user_id: str) -> Dict[str, Any]:
    """Synchronous wrapper for index_transaction_history."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(index_transaction_history(user_id))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in synchronous transaction indexing: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Initialize the RAG system when the module is imported
print("Initializing Finance RAG system...")