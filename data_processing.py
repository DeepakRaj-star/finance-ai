import pandas as pd
from typing import List, Dict, Any
import datetime
import json
import os

def load_transactions() -> List[Dict[str, Any]]:
    """
    Load transactions from storage or return sample data if not available.
    
    Returns:
        List[Dict]: List of transaction dictionaries
    """
    # In a real application, this would load from a database
    # For this MVP, we'll use session state as storage
    return []

def add_transaction(transactions: List[Dict[str, Any]], 
                   new_transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Add a new transaction to the list of transactions.
    
    Args:
        transactions (List[Dict]): Existing transactions
        new_transaction (Dict): New transaction to add
        
    Returns:
        List[Dict]: Updated transaction list
    """
    # Create a copy to avoid modifying the original list
    updated_transactions = transactions.copy()
    updated_transactions.append(new_transaction)
    return updated_transactions

def get_transaction_summary(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from transaction data.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        Dict: Summary statistics
    """
    if not transactions:
        return {
            "total_income": 0,
            "total_expenses": 0,
            "net_cashflow": 0,
            "monthly_income": 0,
            "monthly_expenses": 0,
            "savings_rate": 0
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(transactions)
    
    # Handle empty dataframe
    if df.empty:
        return {
            "total_income": 0,
            "total_expenses": 0,
            "net_cashflow": 0,
            "monthly_income": 0,
            "monthly_expenses": 0,
            "savings_rate": 0
        }
    
    # Convert date strings to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate total income and expenses
    total_income = df[df['type'] == 'income']['amount'].sum()
    total_expenses = df[df['type'] == 'expense']['amount'].sum()
    net_cashflow = total_income - total_expenses
    
    # Calculate monthly figures
    today = pd.Timestamp(datetime.datetime.now())
    start_of_month = pd.Timestamp(datetime.datetime(today.year, today.month, 1))
    
    if 'date' in df.columns:
        monthly_df = df[df['date'] >= start_of_month]
        monthly_income = monthly_df[monthly_df['type'] == 'income']['amount'].sum()
        monthly_expenses = monthly_df[monthly_df['type'] == 'expense']['amount'].sum()
    else:
        # If no date column, use all data as a fallback
        monthly_income = total_income
        monthly_expenses = total_expenses
    
    # Calculate savings rate
    savings_rate = 0
    if monthly_income > 0:
        savings_rate = max(0, (monthly_income - monthly_expenses) / monthly_income)
    
    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_cashflow": net_cashflow,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "savings_rate": savings_rate
    }

def calculate_balance(transactions: List[Dict[str, Any]]) -> float:
    """
    Calculate the current balance from all transactions.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        float: Current balance
    """
    if not transactions:
        return 0.0
    
    balance = 0.0
    for t in transactions:
        if t.get('type') == 'income':
            balance += t.get('amount', 0)
        elif t.get('type') == 'expense':
            balance -= t.get('amount', 0)
    
    return balance

def get_category_spending(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Get spending by category from transaction data.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        Dict: Dictionary mapping categories to total spending
    """
    if not transactions:
        return {}
    
    category_spending = {}
    for t in transactions:
        if t.get('type') == 'expense':
            category = t.get('category', 'Other')
            amount = t.get('amount', 0)
            
            if category in category_spending:
                category_spending[category] += amount
            else:
                category_spending[category] = amount
    
    return category_spending

def get_daily_spending(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Get daily spending totals from transaction data.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        Dict: Dictionary mapping dates to total spending
    """
    if not transactions:
        return {}
    
    df = pd.DataFrame(transactions)
    
    # Handle empty dataframe
    if df.empty or 'date' not in df.columns:
        return {}
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for expenses only
    expenses_df = df[df['type'] == 'expense']
    
    # Group by date and sum amounts
    daily_spending = expenses_df.groupby(expenses_df['date'].dt.date)['amount'].sum()
    
    # Convert to dictionary
    return daily_spending.to_dict()

def get_monthly_spending(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Get monthly spending totals from transaction data.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        Dict: Dictionary mapping month strings to total spending
    """
    if not transactions:
        return {}
    
    df = pd.DataFrame(transactions)
    
    # Handle empty dataframe
    if df.empty or 'date' not in df.columns:
        return {}
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for expenses only
    expenses_df = df[df['type'] == 'expense']
    
    # Create month string
    expenses_df['month'] = expenses_df['date'].dt.strftime('%Y-%m')
    
    # Group by month and sum amounts
    monthly_spending = expenses_df.groupby('month')['amount'].sum()
    
    # Convert to dictionary
    return monthly_spending.to_dict()
