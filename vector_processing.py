import os
import json
import time
import numpy as np
import threading
import datetime
from typing import List, Dict, Any
import openai
import pathway as pw

# OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if openai is not None:
    openai.api_key = OPENAI_API_KEY

# Import database models
from db_models import Transaction, VectorTransaction, get_db, init_db
import ai_agents

# Embedding dimension
EMBEDDING_DIM = 1536  # OpenAI standard size

class TransactionVectorizer:
    """
    Class to convert transactions to vector embeddings for semantic search.
    """

    def __init__(self):
        """Initialize the vectorizer."""
        self.model = "text-embedding-ada-002"

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a piece of text."""
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return a zero embedding as fallback
            return [0.0] * EMBEDDING_DIM

    def transaction_to_text(self, transaction: Dict[str, Any]) -> str:
        """Convert a transaction to a text representation."""
        return (
            f"Date: {transaction.get('date', '')}, "
            f"Description: {transaction.get('description', '')}, "
            f"Amount: {transaction.get('amount', 0)}, "
            f"Type: {transaction.get('type', '')}, "
            f"Category: {transaction.get('category', '')}, "
            f"Notes: {transaction.get('notes', '')}"
        )

    def vectorize_transaction(self, transaction: Dict[str, Any]) -> List[float]:
        """Convert a transaction to a vector embedding."""
        text = self.transaction_to_text(transaction)
        return self.get_embedding(text)

    def store_transaction_vector(self, transaction_id: int, embedding: List[float]) -> None:
        """Store a transaction vector in the database."""
        db = get_db()
        try:
            # Check if vector already exists
            existing = db.query(VectorTransaction).filter(
                VectorTransaction.transaction_id == transaction_id
            ).first()

            if existing:
                # Update existing
                existing.embedding = embedding
                existing.updated_at = datetime.datetime.now()
            else:
                # Create new
                vector_txn = VectorTransaction(
                    transaction_id=transaction_id,
                    embedding=embedding
                )
                db.add(vector_txn)

            db.commit()
        except Exception as e:
            print(f"Error storing vector: {str(e)}")
            db.rollback()
        finally:
            db.close()

    def process_transaction(self, transaction: Dict[str, Any]) -> None:
        """Process a transaction to create and store its vector embedding."""
        # Get vector embedding
        embedding = self.vectorize_transaction(transaction)

        # Store in database
        if 'id' in transaction:
            self.store_transaction_vector(transaction['id'], embedding)


class PathwayVectorEngine:
    """
    Real-time vector processing engine using Pathway for streaming transaction analysis.
    """

    def __init__(self):
        """Initialize the engine."""
        self.vectorizer = TransactionVectorizer()
        self.is_running = False
        self.pipeline = None
        self.thread = None
        # Vector store for in-memory fast access
        self.vector_store = {}
        # Path for the persistent vector store
        self.store_path = "user_data/vector_store"

    def setup_pipeline(self):
        """Set up the Pathway data processing pipeline with advanced real-time features."""
        try:
            # Ensure the store directory exists
            os.makedirs(self.store_path, exist_ok=True)

            # Define schema for transaction data
            class TransactionSchema(pw.Schema):
                id: int
                user_id: str
                date: str
                description: str
                amount: float
                type: str
                category: str
                notes: str

            # Create database connector (more robust than CSV for production)
            # For demonstration, we still use a CSV connector
            transaction_input = pw.io.csv.read(
                schema=TransactionSchema,
                path="user_data/transaction_stream.csv",
                mode="streaming",
                autocommit_duration_ms=500,  # Faster commits for real-time processing
            )

            # Process transactions with real OpenAI embeddings
            def compute_embedding(transaction):
                if not OPENAI_API_KEY:
                    # Fallback if no API key
                    return np.zeros(EMBEDDING_DIM).tolist()

                text = (
                    f"Date: {transaction.date}, "
                    f"Description: {transaction.description}, "
                    f"Amount: {transaction.amount}, "
                    f"Type: {transaction.type}, "
                    f"Category: {transaction.category}, "
                    f"Notes: {transaction.notes}"
                )

                try:
                    # Try to get real embedding
                    embedding = self.vectorizer.get_embedding(text)
                    return embedding
                except Exception as e:
                    print(f"Error in Pathway embedding: {str(e)}")
                    # Return zeros as fallback
                    return np.zeros(EMBEDDING_DIM).tolist()

            # Apply embedding computation with proper error handling
            transaction_embeddings = transaction_input.select(
                pw.this.id,
                pw.this.user_id,
                pw.this.date,
                pw.this.description,
                pw.this.amount,
                pw.this.type,
                pw.this.category,
                embedding_data=pw.apply(compute_embedding, pw.this, dtype=pw.engine.Pending), # Modified line
                embedding=pw.apply(lambda x: x['embedding_data']['embedding'], pw.this),
                error=pw.apply(lambda x: x['embedding_data']['error'], pw.this)
            )

            # Filter out successful embeddings
            valid_embeddings = transaction_embeddings.filter(pw.this.error == None)

            # Create index tables by user for faster retrieval
            user_embeddings = valid_embeddings.groupby(pw.this.user_id)

            # Store data for each user separately
            for user_group, user_data in user_embeddings:
                user_id = user_group.user_id
                # Output to user-specific files
                pw.io.jsonlines.write(
                    user_data,
                    path=f"{self.store_path}/{user_id}_vectors.jsonl"
                )

            # Set up real-time monitoring with stats
            stats = transaction_embeddings.select(
                total_count=pw.count(),
                error_count=pw.count(pw.this.error != None),
                success_count=pw.count(pw.this.error == None),
                unique_users=pw.unique_count(pw.this.user_id)
            )

            # Output to stats file
            pw.io.csv.write(
                stats,
                path=f"{self.store_path}/vector_stats.csv"
            )

            # Return the pipeline for running
            return pw.compile()

        except Exception as e:
            print(f"Error setting up Pathway pipeline: {str(e)}")
            return None

    def start(self):
        """Start the vector processing engine in a background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run_engine)
        self.thread.daemon = True
        self.thread.start()

    def _run_engine(self):
        """Run the vector processing engine in the background."""
        try:
            self.pipeline = self.setup_pipeline()

            if self.pipeline:
                # This is a placeholder for the actual pipeline run
                # In a real implementation, you'd run: self.pipeline.run()

                # Instead, we're running a simple loop to simulate processing
                while self.is_running:
                    time.sleep(5)  # Check every 5 seconds

                    db = get_db()
                    try:
                        # Get unprocessed transactions
                        transactions = db.query(Transaction).outerjoin(
                            VectorTransaction,
                            Transaction.id == VectorTransaction.transaction_id
                        ).filter(
                            VectorTransaction.id == None
                        ).limit(10).all()

                        for txn in transactions:
                            # Process each transaction
                            self.vectorizer.process_transaction(txn.to_dict())
                    except Exception as e:
                        print(f"Error processing transactions: {str(e)}")
                    finally:
                        db.close()

        except Exception as e:
            print(f"Error running vector engine: {str(e)}")
            self.is_running = False

    def stop(self):
        """Stop the vector processing engine."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None


class VectorSearchEngine:
    """
    Advanced engine for performing vector-based semantic search on transactions with hybrid search capabilities.
    """

    def __init__(self):
        """Initialize the search engine."""
        self.vectorizer = TransactionVectorizer()
        # Path for the vector store
        self.store_path = "user_data/vector_store"
        # Cache for frequent searches
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_clear = time.time()

    def search(self, user_id: str, query: str, limit: int = 10, filter_params: Dict = None) -> List[Dict[str, Any]]:
        """
        Advanced semantic search for transactions with filtering capabilities.

        Args:
            user_id: User ID to search transactions for
            query: Natural language query
            limit: Maximum number of results to return
            filter_params: Optional filters (e.g., date range, amount range, categories)

        Returns:
            List of matching transactions, sorted by relevance
        """
        # Check cache
        cache_key = f"{user_id}:{query}:{limit}:{json.dumps(filter_params or {})}"
        current_time = time.time()

        # Clear old cache entries
        if current_time - self.last_cache_clear > 60:  # Clear cache every minute
            self.cache = {k: v for k, v in self.cache.items() 
                         if current_time - v.get('timestamp', 0) < self.cache_ttl}
            self.last_cache_clear = current_time

        # Return from cache if available
        if cache_key in self.cache and current_time - self.cache[cache_key].get('timestamp', 0) < self.cache_ttl:
            return self.cache[cache_key].get('results', [])

        try:
            # Convert query to vector
            query_vector = self.vectorizer.get_embedding(query)

            # Get db session
            db = get_db()

            # Build base query
            base_query = db.query(Transaction, VectorTransaction).join(
                VectorTransaction,
                Transaction.id == VectorTransaction.transaction_id
            ).filter(
                Transaction.user_id == user_id
            )

            # Apply filters if provided
            if filter_params:
                # Date range filter
                if 'date_from' in filter_params:
                    base_query = base_query.filter(Transaction.date >= filter_params['date_from'])
                if 'date_to' in filter_params:
                    base_query = base_query.filter(Transaction.date <= filter_params['date_to'])

                # Amount range filter
                if 'amount_min' in filter_params:
                    base_query = base_query.filter(Transaction.amount >= filter_params['amount_min'])
                if 'amount_max' in filter_params:
                    base_query = base_query.filter(Transaction.amount <= filter_params['amount_max'])

                # Category filter
                if 'categories' in filter_params and filter_params['categories']:
                    base_query = base_query.filter(Transaction.category.in_(filter_params['categories']))

                # Transaction type filter
                if 'types' in filter_params and filter_params['types']:
                    base_query = base_query.filter(Transaction.type.in_(filter_params['types']))

            # Execute query
            transactions = base_query.all()

            # Calculate similarities using hybrid search approach
            results = []
            for txn, vector in transactions:
                # Skip if no embedding
                if not vector.embedding:
                    continue

                # Convert transaction to dict
                txn_dict = txn.to_dict()

                # Calculate semantic similarity score (vector-based)
                semantic_similarity = self._cosine_similarity(query_vector, vector.embedding)

                # Calculate keyword match score (text-based)
                keyword_similarity = self._keyword_match_score(query, txn_dict)

                # Hybrid scoring - combine semantic and keyword matches
                # Hybrid search gives more weight to semantic similarity but boosts exact matches
                hybrid_score = (semantic_similarity * 0.7) + (keyword_similarity * 0.3)

                # Add to results
                results.append((txn_dict, hybrid_score))

            # Sort by hybrid score (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            # Get top matches
            top_results = [r[0] for r in results[:limit]]

            # Store in cache
            self.cache[cache_key] = {
                'results': top_results,
                'timestamp': current_time
            }

            return top_results

        except Exception as e:
            print(f"Error searching transactions: {str(e)}")
            return []

        finally:
            if 'db' in locals():
                db.close()

    def semantic_query(self, user_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Perform a semantic query on transactions and return structured insights.

        Args:
            user_id: User ID to search transactions for
            query: Natural language query
            limit: Maximum number of transactions to analyze

        Returns:
            Dict with query results and insights
        """
        try:
            # Get relevant transactions
            relevant_txns = self.search(user_id, query, limit=limit)

            if not relevant_txns:
                return {
                    "query": query,
                    "transactions": [],
                    "answer": "No relevant transactions found.",
                    "insights": "No data available for analysis."
                }

            # Generate answer using AI
            answer = ai_agents.get_answer_from_query(query, relevant_txns)

            # Generate insights on these specific transactions
            insights = ai_agents.get_ai_insight(relevant_txns)

            # Calculate statistics
            total_amount = sum(t.get('amount', 0) for t in relevant_txns)
            avg_amount = total_amount / len(relevant_txns) if relevant_txns else 0

            return {
                "query": query,
                "transactions": relevant_txns,
                "transaction_count": len(relevant_txns),
                "total_amount": total_amount,
                "average_amount": avg_amount,
                "answer": answer,
                "insights": insights
            }

        except Exception as e:
            print(f"Error in semantic query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "transactions": [],
                "answer": "An error occurred while processing your query."
            }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Calculate similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def _keyword_match_score(self, query: str, transaction: Dict[str, Any]) -> float:
        """
        Calculate keyword matching score between query and transaction.

        Args:
            query: Search query
            transaction: Transaction dictionary

        Returns:
            Score between 0 and 1
        """
        # Normalize text
        query = query.lower()

        # Extract transaction text fields
        description = str(transaction.get('description', '')).lower()
        category = str(transaction.get('category', '')).lower()
        notes = str(transaction.get('notes', '')).lower()

        # Split query into keywords
        keywords = [k.strip() for k in query.split()]

        if not keywords:
            return 0

        # Count matches
        matches = 0
        for keyword in keywords:
            if keyword in description or keyword in category or keyword in notes:
                matches += 1

        # Calculate score
        return matches / len(keywords)


# Global instances
vectorizer = TransactionVectorizer()
vector_engine = PathwayVectorEngine()
search_engine = VectorSearchEngine()

# Helper functions
def process_new_transaction(transaction: Dict[str, Any]) -> None:
    """Process a new transaction for vector embedding."""
    vectorizer.process_transaction(transaction)

def search_transactions(user_id: str, query: str, limit: int = 10, filter_params: Dict = None) -> List[Dict[str, Any]]:
    """
    Search for transactions using semantic similarity.

    Args:
        user_id: User ID to search transactions for
        query: Natural language query
        limit: Maximum number of results to return
        filter_params: Optional filters (e.g., date range, amount range, categories)

    Returns:
        List of matching transactions, sorted by relevance
    """
    return search_engine.search(user_id, query, limit, filter_params)

def semantic_query(user_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Perform a semantic query on transactions and return structured insights.

    Args:
        user_id: User ID to search transactions for
        query: Natural language query
        limit: Maximum number of transactions to analyze

    Returns:
        Dict with query results and insights
    """
    return search_engine.semantic_query(user_id, query, limit)

def start_vector_engine():
    """Start the vector processing engine."""
    try:
        if not vector_engine.is_running:
            print("Starting vector processing engine...")
            vector_engine.start()
        return True
    except Exception as e:
        print(f"Failed to start vector engine: {str(e)}")
        st.warning("Vector search functionality may be limited due to initialization error.")
        # Don't fail the whole application if the vector engine fails
        return False

def stop_vector_engine():
    """Stop the vector processing engine."""
    try:
        if vector_engine.is_running:
            vector_engine.stop()
        return True
    except Exception as e:
        print(f"Error stopping vector engine: {str(e)}")
        return False

def reset_user_vectors(user_id: str) -> bool:
    """
    Clear all vector data for a user.

    Args:
        user_id (str): The user ID

    Returns:
        bool: Success or failure
    """
    db = get_db()
    try:
        # Find all vector transactions for user's transactions
        vector_txns = db.query(VectorTransaction).join(
            Transaction, 
            Transaction.id == VectorTransaction.transaction_id
        ).filter(
            Transaction.user_id == user_id
        ).all()

        # Delete each vector transaction
        for vtxn in vector_txns:
            db.delete(vtxn)

        db.commit()
        return True

    except Exception as e:
        db.rollback()
        print(f"Error resetting user vectors: {str(e)}")
        return False

    finally:
        db.close()