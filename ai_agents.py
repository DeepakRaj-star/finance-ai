import os
import json
import time
import datetime
from typing import List, Dict, Any, Optional

# OpenAI setup with error handling
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

try:
    import openai
    from openai import OpenAI

    # Initialize OpenAI client if API key is available
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("WARNING: OPENAI_API_KEY environment variable not found. AI features will be unavailable.")
        client = None
except ImportError:
    print("WARNING: OpenAI package not installed. AI features will be unavailable.")
    client = None

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL = "gpt-4o"

def generate_response(prompt: str, system_prompt: str = None) -> str:
    """
    Generate a response using OpenAI's GPT model with improved error handling.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt

    Returns:
        Generated text response
    """
    # Check if OpenAI client is available
    if client is None or not OPENAI_API_KEY:
        return "AI features require an OpenAI API key. Please configure it in your environment variables."

    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Add exponential backoff retry
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                if "quota" in str(e).lower() or "insufficient_quota" in str(e):
                    return "AI features are temporarily unavailable due to API quota limits. Please try again later."
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating response: {error_msg}")

        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return "AI features are currently unavailable due to OpenAI API quota limitations. Please check your OpenAI API plan, billing details, or try again later."
        return "AI features are currently unavailable. Please try again later."

def generate_structured_response(prompt: str, system_prompt: str = None) -> Dict[str, Any]:
    """
    Generate a structured JSON response using OpenAI's GPT model.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt

    Returns:
        Generated response as a dictionary
    """
    # Check if OpenAI client is available
    if client is None:
        return {
            "error": "OPENAI_NOT_CONFIGURED",
            "message": "AI features are unavailable. Please ensure the OpenAI API key is properly configured."
        }

    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating structured response: {error_msg}")

        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return {
                "error": "API_QUOTA_EXCEEDED",
                "message": "AI features are currently unavailable due to OpenAI API quota limitations. Please check your OpenAI API plan, billing details, or try again later."
            }

        return {
            "error": "AI_UNAVAILABLE", 
            "message": "AI features are currently unavailable. Please try again later."
        }

def get_transaction_categorization(description: str) -> str:
    """
    Categorize a transaction based on its description.

    Args:
        description: Transaction description

    Returns:
        Category string
    """
    # Define standard categories
    categories = [
        "Food", "Transportation", "Housing", "Utilities", "Entertainment", 
        "Shopping", "Healthcare", "Education", "Travel", "Income", "Investments", "Other"
    ]

    # Simple keyword-based fallback categorization
    description_lower = description.lower()
    category_keywords = {
        "Food": ["restaurant", "cafe", "food", "grocery", "meal", "takeout", "dining"],
        "Transportation": ["gas", "fuel", "car", "uber", "lyft", "taxi", "train", "transit", "bus", "subway"],
        "Housing": ["rent", "mortgage", "apartment", "house", "property"],
        "Utilities": ["electric", "water", "gas", "internet", "phone", "utility", "bill"],
        "Entertainment": ["movie", "theater", "concert", "subscription", "netflix", "spotify", "game"],
        "Shopping": ["amazon", "walmart", "target", "store", "shop", "mall", "purchase"],
        "Healthcare": ["doctor", "hospital", "medical", "health", "dental", "pharmacy", "prescription"],
        "Education": ["tuition", "school", "university", "college", "course", "book", "education"],
        "Travel": ["hotel", "flight", "airline", "vacation", "trip", "travel"],
        "Income": ["salary", "paycheck", "deposit", "income", "payment received", "wage"],
        "Investments": ["investment", "stock", "bond", "dividend", "crypto", "bitcoin"]
    }

    # Try AI categorization first
    try:
        system_prompt = (
            "You are a financial transaction categorization agent. "
            "Your task is to categorize financial transactions into the most appropriate category. "
            "Available categories are: Food, Transportation, Housing, Utilities, Entertainment, Shopping, "
            "Healthcare, Education, Travel, Income, Investments, and Other."
        )

        prompt = f"Please categorize this transaction: {description}"

        response = generate_response(prompt, system_prompt)

        # Extract just the category name
        for category in categories:
            if category.lower() in response.lower():
                return category

        # If AI failed to categorize (didn't match any standard category), fall back to keyword matching
        raise Exception("AI categorization failed to match standard categories")

    except Exception as e:
        print(f"Using fallback categorization due to AI error: {str(e)}")

        # Fallback to keyword-based categorization
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category

    # Default category if all else fails
    return "Other"

def get_spending_anomalies(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify spending anomalies in transactions.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        List of anomalies with transaction and anomaly details
    """
    # Calculate average spending by category
    category_totals = {}
    category_counts = {}

    for txn in transactions:
        category = txn.get("category", "Other")
        amount = txn.get("amount", 0)
        txn_type = txn.get("type", "expense")

        if txn_type.lower() == "expense":
            if category not in category_totals:
                category_totals[category] = 0
                category_counts[category] = 0

            category_totals[category] += amount
            category_counts[category] += 1

    # Calculate averages
    category_averages = {}
    for category in category_totals:
        if category_counts[category] > 0:
            category_averages[category] = category_totals[category] / category_counts[category]

    # Identify anomalies (transactions > 50% above average for their category)
    anomalies = []
    for txn in transactions:
        category = txn.get("category", "Other")
        amount = txn.get("amount", 0)
        txn_type = txn.get("type", "expense")

        if txn_type.lower() == "expense" and category in category_averages:
            avg = category_averages[category]

            if amount > avg * 1.5:
                percent_above = ((amount - avg) / avg) * 100
                anomalies.append({
                    "transaction": txn,
                    "average_for_category": avg,
                    "percent_above_average": percent_above
                })

    # Sort by percent_above_average descending
    anomalies.sort(key=lambda x: x["percent_above_average"], reverse=True)

    return anomalies

def get_budget_status(budgets: List[Dict[str, Any]], spending: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Check budget status based on current spending.

    Args:
        budgets: List of budget dictionaries
        spending: Dictionary of spending by category

    Returns:
        List of budget status with warnings
    """
    status = []

    for budget in budgets:
        category = budget.get("category", "")
        limit = budget.get("amount", 0)
        period = budget.get("period", "monthly")

        spent = spending.get(category, 0)
        percent_used = (spent / limit) * 100 if limit > 0 else 0

        warning_level = None
        if percent_used >= 100:
            warning_level = "critical"
        elif percent_used >= 80:
            warning_level = "warning"
        elif percent_used >= 50:
            warning_level = "notice"

        status.append({
            "budget": budget,
            "spent": spent,
            "percent_used": percent_used,
            "remaining": max(0, limit - spent),
            "warning_level": warning_level
        })

    # Sort by percent_used descending
    status.sort(key=lambda x: x["percent_used"], reverse=True)

    return status

def get_ai_insight(transactions: List[Dict[str, Any]]) -> str:
    """Get AI-powered insight from transactions."""
    try:
        if not transactions:
            return "No transactions to analyze."

        # Add rate limiting
        time.sleep(1)  # Basic rate limiting

        # Process with LLM
        txn_text = ""
        for i, txn in enumerate(transactions[:20]):  # Limit to 20 transactions
            txn_text += (
                f"Transaction {i+1}:\n"
                f"  Date: {txn.get('date', 'N/A')}\n"
                f"  Description: {txn.get('description', 'N/A')}\n"
                f"  Amount: {txn.get('amount', 0)}\n"
                f"  Type: {txn.get('type', 'N/A')}\n"
                f"  Category: {txn.get('category', 'N/A')}\n\n"
            )

        system_prompt = (
            "You are a financial advisor AI. "
            "Based on the transaction data provided, generate insights about "
            "spending patterns, potential savings opportunities, and financial habits. "
            "Be specific, actionable, and concise."
        )

        prompt = (
            f"Please analyze these transactions and provide financial insights:\n\n"
            f"{txn_text}"
        )

        return generate_response(prompt, system_prompt)

    except openai.error.RateLimitError:
        return "AI analysis temporarily unavailable. Please try again in a few moments."
    except Exception as e:
        return f"Error analyzing transactions: {str(e)}"

def get_answer_from_query(query: str, transactions: List[Dict[str, Any]]) -> str:
    """
    Answer a natural language query about transactions.

    Args:
        query: The natural language query
        transactions: List of transaction dictionaries

    Returns:
        Answer string
    """
    if not transactions:
        return "No transaction data available to answer your query."

    # Prepare transaction data as text
    txn_text = ""
    for i, txn in enumerate(transactions[:50]):  # Limit to 50 transactions
        txn_text += (
            f"Transaction {i+1}:\n"
            f"  Date: {txn.get('date', 'N/A')}\n"
            f"  Description: {txn.get('description', 'N/A')}\n"
            f"  Amount: {txn.get('amount', 0)}\n"
            f"  Type: {txn.get('type', 'N/A')}\n"
            f"  Category: {txn.get('category', 'N/A')}\n\n"
        )

    system_prompt = (
        "You are a financial analysis AI assistant. "
        "Based on the transaction data provided, answer the user's query "
        "accurately and concisely. If you can't answer based on the data, "
        "say so clearly."
    )

    prompt = (
        f"User query: {query}\n\n"
        f"Transaction data:\n{txn_text}\n\n"
        f"Please answer the user's query based on this transaction data."
    )

    return generate_response(prompt, system_prompt)

def get_financial_summary(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive financial summary from transactions.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        Summary dictionary with various financial metrics
    """
    if not transactions:
        return {
            "total_income": 0,
            "total_expenses": 0,
            "net_cash_flow": 0,
            "top_expense_categories": [],
            "summary_text": "No transactions to analyze."
        }

    # Calculate totals
    total_income = 0
    total_expenses = 0
    category_expenses = {}

    for txn in transactions:
        amount = txn.get("amount", 0)
        txn_type = txn.get("type", "").lower()
        category = txn.get("category", "Other")

        if txn_type == "income":
            total_income += amount
        elif txn_type == "expense":
            total_expenses += amount

            if category not in category_expenses:
                category_expenses[category] = 0

            category_expenses[category] += amount

    # Calculate net cash flow
    net_cash_flow = total_income - total_expenses

    # Get top expense categories
    top_categories = sorted(
        category_expenses.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]  # Top 5

    # Prepare transaction text for AI
    txn_text = ""
    for i, txn in enumerate(transactions[:20]):  # Limit to 20 transactions
        txn_text += (
            f"Transaction {i+1}:\n"
            f"  Date: {txn.get('date', 'N/A')}\n"
            f"  Description: {txn.get('description', 'N/A')}\n"
            f"  Amount: {txn.get('amount', 0)}\n"
            f"  Type: {txn.get('type', 'N/A')}\n"
            f"  Category: {txn.get('category', 'N/A')}\n\n"
        )

    # Generate AI summary
    system_prompt = (
        "You are a financial advisor AI. "
        "Based on the financial data provided, generate a concise summary of "
        "the person's overall financial situation. Focus on the key metrics, "
        "patterns, and potential areas for improvement."
    )

    prompt = (
        f"Please analyze this financial data and provide a summary:\n\n"
        f"Total Income: {total_income}\n"
        f"Total Expenses: {total_expenses}\n"
        f"Net Cash Flow: {net_cash_flow}\n"
        f"Top Expense Categories: {dict(top_categories)}\n\n"
        f"Recent transactions:\n{txn_text}"
    )

    summary_text = generate_response(prompt, system_prompt)

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_cash_flow": net_cash_flow,
        "top_expense_categories": [{"category": cat, "amount": amt} for cat, amt in top_categories],
        "summary_text": summary_text
    }

def get_transaction_predictions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict future transactions based on historical data.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        List of predicted transactions
    """
    if not transactions:
        return []

    # Organize transactions by category and recurring patterns
    recurring_txns = {}
    for txn in transactions:
        desc = txn.get("description", "").lower()
        amount = txn.get("amount", 0)
        category = txn.get("category", "Other")
        txn_type = txn.get("type", "expense")

        key = f"{desc}-{category}-{txn_type}"

        if key not in recurring_txns:
            recurring_txns[key] = {
                "transactions": [],
                "description": txn.get("description", ""),
                "category": category,
                "type": txn_type,
                "count": 0
            }

        recurring_txns[key]["transactions"].append(txn)
        recurring_txns[key]["count"] += 1

    # Filter for recurring transactions (appear at least 2 times)
    recurring = {k: v for k, v in recurring_txns.items() if v["count"] >= 2}

    # Prepare for prediction
    predictions = []
    today = datetime.datetime.now()

    for _, data in recurring.items():
        txns = data["transactions"]

        # Sort by date
        txns.sort(key=lambda x: x.get("date", ""))

        # Get average amount
        avg_amount = sum(t.get("amount", 0) for t in txns) / len(txns)

        # Get most recent date
        most_recent_date = txns[-1].get("date", "")
        if not most_recent_date:
            continue

        # Parse most recent date
        try:
            recent_dt = datetime.datetime.strptime(most_recent_date, "%Y-%m-%d")

            # If monthly recurring (roughly 30 days between transactions)
            if len(txns) >= 2:
                date_diffs = []
                for i in range(len(txns) - 1):
                    date1 = datetime.datetime.strptime(txns[i].get("date", ""), "%Y-%m-%d")
                    date2 = datetime.datetime.strptime(txns[i+1].get("date", ""), "%Y-%m-%d")
                    date_diffs.append((date2 - date1).days)

                avg_diff = sum(date_diffs) / len(date_diffs)

                # If appears to be monthly (25-35 days)
                if 25 <= avg_diff <= 35:
                    next_date = recent_dt + datetime.timedelta(days=30)

                    # Only if next date is in the future
                    if next_date > today:
                        predictions.append({
                            "description": data["description"],
                            "category": data["category"],
                            "type": data["type"],
                            "amount": avg_amount,
                            "predicted_date": next_date.strftime("%Y-%m-%d"),
                            "confidence": min(0.9, 0.5 + (0.1 * len(txns)))  # Higher confidence with more occurrences
                        })

        except Exception as e:
            # Skip if date parsing fails
            print(f"Error parsing date: {str(e)}")
            continue

    # Sort by predicted date
    predictions.sort(key=lambda x: x.get("predicted_date", ""))

    return predictions