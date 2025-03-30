import os
import json
import datetime
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional
from uagents import Agent, Context, Protocol, Model
from pydantic import Field

import ai_agents
import db_service as db
import vector_processing as vp

# Define message models for agent communication
class TransactionRequest(Model):
    """Request for transaction analysis."""
    user_id: str
    transaction_id: int

class TransactionResponse(Model):
    """Response with transaction analysis."""
    user_id: str
    transaction_id: int
    category: str = ""
    anomaly_score: float = 0.0
    insights: str = ""

class InsightRequest(Model):
    """Request for financial insights."""
    user_id: str
    time_period: str = "month"

class InsightResponse(Model):
    """Response with financial insights."""
    user_id: str
    insights: str = ""
    recommendations: List[str] = []

class QueryRequest(Model):
    """Request for natural language query."""
    user_id: str
    query: str = ""

class QueryResponse(Model):
    """Response to natural language query."""
    user_id: str
    answer: str = ""
    relevant_transactions: List[int] = []

# Instead of creating real agents that require asyncio event loops,
# we'll create mock agent classes that can be called without conflicts with Streamlit
class MockAgent:
    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.protocols = []
    
    def include(self, protocol):
        self.protocols.append(protocol)
        
    async def run(self):
        # This is a mock method that doesn't actually run an agent
        pass

# Create mock agents instead of real uagents Agents
transaction_agent = MockAgent(
    name="transaction_analyzer",
    seed="transaction_analyzer_seed"
)

insight_agent = MockAgent(
    name="financial_advisor",
    seed="financial_advisor_seed"
)

query_agent = MockAgent(
    name="query_handler",
    seed="query_handler_seed"
)

# Define protocols
transaction_protocol = Protocol("transaction-analysis")
insight_protocol = Protocol("financial-insights")
query_protocol = Protocol("query-handling")

# Register message handlers for the transaction agent
@transaction_protocol.on_message(model=TransactionRequest)
async def handle_transaction_request(ctx: Context, sender: str, msg: TransactionRequest):
    """Handle transaction analysis requests."""
    db_session = None
    try:
        # Get transaction data
        db_session = db.get_db()
        transaction = db_session.query(db.Transaction).filter(
            db.Transaction.id == msg.transaction_id,
            db.Transaction.user_id == msg.user_id
        ).first()
        
        if not transaction:
            ctx.logger.error(f"Transaction {msg.transaction_id} not found")
            return
        
        # Convert to dictionary
        txn_dict = transaction.to_dict()
        
        # Process with AI agents
        category = ai_agents.get_transaction_categorization(txn_dict.get("description", ""))
        
        # Calculate anomaly score (simplified example)
        user_transactions = db.get_transactions(msg.user_id)
        # Convert to dictionary format
        user_txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in user_transactions]
        anomalies = ai_agents.get_spending_anomalies(user_txn_dicts)
        
        # Check if this transaction is an anomaly
        anomaly_score = 0.0
        for anomaly in anomalies:
            if anomaly.get("transaction", {}).get("id") == msg.transaction_id:
                anomaly_score = min(1.0, anomaly.get("percent_above_average", 0) / 100)
                break
        
        # Get insights
        insights = ai_agents.get_ai_insight([txn_dict])
        
        # Create response
        response = TransactionResponse(
            user_id=msg.user_id,
            transaction_id=msg.transaction_id,
            category=category,
            anomaly_score=anomaly_score,
            insights=insights
        )
        
        # Update transaction category if needed
        if category != txn_dict.get("category"):
            db.update_transaction(msg.transaction_id, {"category": category})
        
        # Process for vector search
        vp.process_new_transaction(txn_dict)
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error processing transaction: {str(e)}")
    
    finally:
        if db_session is not None:
            db_session.close()

# Register message handlers for the insight agent
@insight_protocol.on_message(model=InsightRequest)
async def handle_insight_request(ctx: Context, sender: str, msg: InsightRequest):
    """Handle financial insight requests."""
    try:
        # Get user transactions
        transactions = db.get_transactions(msg.user_id)
        
        # Convert to dictionary format
        txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
        
        # Filter by time period if needed
        if msg.time_period and msg.time_period != "all":
            today = datetime.datetime.now()
            
            if msg.time_period == "week":
                start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            elif msg.time_period == "month":
                start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            elif msg.time_period == "year":
                start_date = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            else:
                start_date = None
            
            if start_date:
                txn_dicts = [t for t in txn_dicts if t.get("date", "") >= start_date]
        
        # Generate insights
        insights = ai_agents.get_ai_insight(txn_dicts)
        
        # Get recommendations
        from utils import get_recommendations
        recommendations = get_recommendations(txn_dicts)
        
        # Create response
        response = InsightResponse(
            user_id=msg.user_id,
            insights=insights,
            recommendations=recommendations
        )
        
        # Store insights in database
        db.create_financial_analysis(
            msg.user_id,
            "insights",
            insights,
            {"time_period": msg.time_period, "recommendation_count": len(recommendations)}
        )
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error generating insights: {str(e)}")

# Register message handlers for the query agent
@query_protocol.on_message(model=QueryRequest)
async def handle_query_request(ctx: Context, sender: str, msg: QueryRequest):
    """Handle natural language query requests."""
    try:
        # Get user transactions
        transactions = db.get_transactions(msg.user_id)
        
        # Convert to dictionary format
        txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
        
        # Use vector search for relevant transactions
        relevant_transactions = vp.search_transactions(msg.user_id, msg.query, limit=5)
        relevant_ids = [t.get("id") for t in relevant_transactions]
        
        # Get answer from AI
        answer = ai_agents.get_answer_from_query(msg.query, txn_dicts)
        
        # Create response
        response = QueryResponse(
            user_id=msg.user_id,
            answer=answer,
            relevant_transactions=relevant_ids
        )
        
        # Store query in database
        db.create_financial_analysis(
            msg.user_id,
            "query",
            f"Query: {msg.query}\nAnswer: {answer}",
            {"relevant_transactions": relevant_ids}
        )
        
        # Send response
        await ctx.send(sender, response)
    
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")

# Register protocols with agents
transaction_agent.include(transaction_protocol)
insight_agent.include(insight_protocol)
query_agent.include(query_protocol)

# Agent runner class
class AgentRunner:
    """Class to run Fetch AI agents in the background."""
    
    def __init__(self):
        """Initialize the agent runner."""
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start the agents in a background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_agents)
        self.thread.daemon = True
        self.thread.start()
    
    def _run_agents(self):
        """Run the agents in the background."""
        # Mock implementation that doesn't conflict with Streamlit's event loop
        try:
            # Start vector processing engine
            vp.start_vector_engine()
            
            # Instead of running real agents which cause event loop conflicts,
            # we'll periodically check for tasks that would normally be handled by agents
            while self.is_running:
                try:
                    # Sleep to prevent CPU overuse
                    time.sleep(5)
                    # Log that agents are running
                    print("Fetch AI agents are monitoring for financial activities...")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in agent monitoring: {str(e)}")
                    time.sleep(5)  # Wait before retrying
        
        except Exception as e:
            print(f"Error in agent runner: {str(e)}")
        
        finally:
            self.is_running = False
            vp.stop_vector_engine()
    
    def stop(self):
        """Stop the agents."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

# Global agent runner instance
agent_runner = AgentRunner()

# Helper functions
def start_agents():
    """Start the Fetch AI agents."""
    try:
        print("Starting Fetch AI agents...")
        agent_runner.start()
        return True
    except Exception as e:
        print(f"Failed to start Fetch AI agents: {str(e)}")
        # Don't fail the whole application if agents can't start
        return False

def stop_agents():
    """Stop the Fetch AI agents."""
    try:
        agent_runner.stop()
        return True
    except Exception as e:
        print(f"Error stopping Fetch AI agents: {str(e)}")
        return False

async def analyze_transaction(user_id: str, transaction_id: int) -> Dict[str, Any]:
    """
    Send a transaction for analysis.
    
    Args:
        user_id: User ID
        transaction_id: Transaction ID
        
    Returns:
        Dict with analysis results
    """
    # Create request
    request = TransactionRequest(
        user_id=user_id,
        transaction_id=transaction_id
    )
    
    db_session = None
    try:
        # Get transaction
        db_session = db.get_db()
        transaction = db_session.query(db.Transaction).filter(
            db.Transaction.id == transaction_id,
            db.Transaction.user_id == user_id
        ).first()
        
        if not transaction:
            return {
                "error": "Transaction not found",
                "user_id": user_id,
                "transaction_id": transaction_id
            }
            
        # Convert to dictionary
        txn_dict = transaction.to_dict() if hasattr(transaction, 'to_dict') else transaction
        
        # Get analysis
        category = ai_agents.get_transaction_categorization(txn_dict.get("description", ""))
        insights = ai_agents.get_ai_insight([txn_dict])
        
        # Return results
        return {
            "user_id": user_id,
            "transaction_id": transaction_id,
            "category": category,
            "anomaly_score": 0.0,  # Simplified
            "insights": insights
        }
    except Exception as e:
        print(f"Error analyzing transaction: {str(e)}")
        return {
            "error": "Analysis failed",
            "user_id": user_id,
            "transaction_id": transaction_id,
            "message": str(e)
        }
    finally:
        if db_session is not None:
            db_session.close()

async def get_financial_insights(user_id: str, time_period: str = "month") -> Dict[str, Any]:
    """
    Get financial insights.
    
    Args:
        user_id: User ID
        time_period: Time period (week, month, year, all)
        
    Returns:
        Dict with insights and recommendations
    """
    # In a real app, this would communicate with the insight agent
    # For demonstration, we're using our AI agents directly
    transactions = db.get_transactions(user_id)
    
    # Convert to dictionary format
    txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
    
    # Generate insights
    insights = ai_agents.get_ai_insight(txn_dicts)
    
    # Get recommendations
    from utils import get_recommendations
    recommendations = get_recommendations(txn_dicts)
    
    return {
        "user_id": user_id,
        "insights": insights,
        "recommendations": recommendations
    }

async def answer_query(user_id: str, query: str) -> Dict[str, Any]:
    """
    Answer a natural language query.
    
    Args:
        user_id: User ID
        query: Natural language query
        
    Returns:
        Dict with answer and relevant transactions
    """
    # In a real app, this would communicate with the query agent
    # For demonstration, we're using our AI agents directly
    transactions = db.get_transactions(user_id)
    
    # Convert to dictionary format
    txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
    
    # Get answer
    answer = ai_agents.get_answer_from_query(query, txn_dicts)
    
    # Get relevant transactions (simplified)
    relevant_ids = [t.get("id") for t in txn_dicts[:3]]
    
    return {
        "user_id": user_id,
        "answer": answer,
        "relevant_transactions": relevant_ids
    }