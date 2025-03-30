import json
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from collections import defaultdict

def setup_data_system():
    """Ensure the user data directory exists."""
    user_data_path = Path("user_data")
    user_data_path.mkdir(exist_ok=True)

def get_transaction_file_path(user_id: str) -> Path:
    """Get the path to a user's transaction file."""
    return Path("user_data") / user_id / "transactions.json"

def get_goals_file_path(user_id: str) -> Path:
    """Get the path to a user's goals file."""
    return Path("user_data") / user_id / "goals.json"

def get_budgets_file_path(user_id: str) -> Path:
    """Get the path to a user's budgets file."""
    return Path("user_data") / user_id / "budgets.json"

def load_transactions(user_id: str) -> List[Dict[str, Any]]:
    """
    Load transactions for a specific user.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        List[Dict]: List of transaction dictionaries
    """
    file_path = get_transaction_file_path(user_id)
    
    if not file_path.exists():
        # Create user directory if it doesn't exist
        file_path.parent.mkdir(exist_ok=True)
        
        # Create empty transactions file
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or doesn't exist
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []

def save_transactions(user_id: str, transactions: List[Dict[str, Any]]) -> bool:
    """
    Save transactions for a specific user.
    
    Args:
        user_id (str): The user ID
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        bool: Success or failure
    """
    file_path = get_transaction_file_path(user_id)
    
    # Create user directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(transactions, f, indent=2)
        return True
    except Exception:
        return False

def add_transaction(user_id: str, new_transaction: Dict[str, Any]) -> bool:
    """
    Add a new transaction for a specific user.
    
    Args:
        user_id (str): The user ID
        new_transaction (Dict): The new transaction to add
        
    Returns:
        bool: Success or failure
    """
    transactions = load_transactions(user_id)
    
    # Add timestamp if not present
    if "created_at" not in new_transaction:
        new_transaction["created_at"] = datetime.datetime.now().isoformat()
    
    transactions.append(new_transaction)
    
    return save_transactions(user_id, transactions)

def update_transaction(user_id: str, transaction_id: int, updated_transaction: Dict[str, Any]) -> bool:
    """
    Update an existing transaction.
    
    Args:
        user_id (str): The user ID
        transaction_id (int): The index of the transaction to update
        updated_transaction (Dict): The updated transaction data
        
    Returns:
        bool: Success or failure
    """
    transactions = load_transactions(user_id)
    
    if 0 <= transaction_id < len(transactions):
        # Add updated timestamp
        updated_transaction["updated_at"] = datetime.datetime.now().isoformat()
        
        transactions[transaction_id] = updated_transaction
        return save_transactions(user_id, transactions)
    
    return False

def delete_transaction(user_id: str, transaction_id: int) -> bool:
    """
    Delete a transaction.
    
    Args:
        user_id (str): The user ID
        transaction_id (int): The index of the transaction to delete
        
    Returns:
        bool: Success or failure
    """
    transactions = load_transactions(user_id)
    
    if 0 <= transaction_id < len(transactions):
        transactions.pop(transaction_id)
        return save_transactions(user_id, transactions)
    
    return False

def load_goals(user_id: str) -> List[Dict[str, Any]]:
    """
    Load financial goals for a specific user.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        List[Dict]: List of goal dictionaries
    """
    file_path = get_goals_file_path(user_id)
    
    if not file_path.exists():
        # Create user directory if it doesn't exist
        file_path.parent.mkdir(exist_ok=True)
        
        # Create empty goals file
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or doesn't exist
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []

def save_goals(user_id: str, goals: List[Dict[str, Any]]) -> bool:
    """
    Save financial goals for a specific user.
    
    Args:
        user_id (str): The user ID
        goals (List[Dict]): List of goal dictionaries
        
    Returns:
        bool: Success or failure
    """
    file_path = get_goals_file_path(user_id)
    
    # Create user directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(goals, f, indent=2)
        return True
    except Exception:
        return False

def add_goal(user_id: str, new_goal: Dict[str, Any]) -> bool:
    """
    Add a new financial goal for a specific user.
    
    Args:
        user_id (str): The user ID
        new_goal (Dict): The new goal to add
        
    Returns:
        bool: Success or failure
    """
    goals = load_goals(user_id)
    
    # Add timestamp if not present
    if "created_at" not in new_goal:
        new_goal["created_at"] = datetime.datetime.now().isoformat()
    
    goals.append(new_goal)
    
    return save_goals(user_id, goals)

def update_goal(user_id: str, goal_id: int, updated_goal: Dict[str, Any]) -> bool:
    """
    Update an existing financial goal.
    
    Args:
        user_id (str): The user ID
        goal_id (int): The index of the goal to update
        updated_goal (Dict): The updated goal data
        
    Returns:
        bool: Success or failure
    """
    goals = load_goals(user_id)
    
    if 0 <= goal_id < len(goals):
        # Add updated timestamp
        updated_goal["updated_at"] = datetime.datetime.now().isoformat()
        
        goals[goal_id] = updated_goal
        return save_goals(user_id, goals)
    
    return False

def delete_goal(user_id: str, goal_id: int) -> bool:
    """
    Delete a financial goal.
    
    Args:
        user_id (str): The user ID
        goal_id (int): The index of the goal to delete
        
    Returns:
        bool: Success or failure
    """
    goals = load_goals(user_id)
    
    if 0 <= goal_id < len(goals):
        goals.pop(goal_id)
        return save_goals(user_id, goals)
    
    return False

def load_budgets(user_id: str) -> List[Dict[str, Any]]:
    """
    Load budgets for a specific user.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        List[Dict]: List of budget dictionaries
    """
    file_path = get_budgets_file_path(user_id)
    
    if not file_path.exists():
        # Create user directory if it doesn't exist
        file_path.parent.mkdir(exist_ok=True)
        
        # Create empty budgets file
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or doesn't exist
        with open(file_path, 'w') as f:
            json.dump([], f)
        
        return []

def save_budgets(user_id: str, budgets: List[Dict[str, Any]]) -> bool:
    """
    Save budgets for a specific user.
    
    Args:
        user_id (str): The user ID
        budgets (List[Dict]): List of budget dictionaries
        
    Returns:
        bool: Success or failure
    """
    file_path = get_budgets_file_path(user_id)
    
    # Create user directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(budgets, f, indent=2)
        return True
    except Exception:
        return False

def add_budget(user_id: str, new_budget: Dict[str, Any]) -> bool:
    """
    Add a new budget for a specific user.
    
    Args:
        user_id (str): The user ID
        new_budget (Dict): The new budget to add
        
    Returns:
        bool: Success or failure
    """
    budgets = load_budgets(user_id)
    
    # Add timestamp if not present
    if "created_at" not in new_budget:
        new_budget["created_at"] = datetime.datetime.now().isoformat()
    
    budgets.append(new_budget)
    
    return save_budgets(user_id, budgets)

def update_budget(user_id: str, budget_id: int, updated_budget: Dict[str, Any]) -> bool:
    """
    Update an existing budget.
    
    Args:
        user_id (str): The user ID
        budget_id (int): The index of the budget to update
        updated_budget (Dict): The updated budget data
        
    Returns:
        bool: Success or failure
    """
    budgets = load_budgets(user_id)
    
    if 0 <= budget_id < len(budgets):
        # Add updated timestamp
        updated_budget["updated_at"] = datetime.datetime.now().isoformat()
        
        budgets[budget_id] = updated_budget
        return save_budgets(user_id, budgets)
    
    return False

def delete_budget(user_id: str, budget_id: int) -> bool:
    """
    Delete a budget.
    
    Args:
        user_id (str): The user ID
        budget_id (int): The index of the budget to delete
        
    Returns:
        bool: Success or failure
    """
    budgets = load_budgets(user_id)
    
    if 0 <= budget_id < len(budgets):
        budgets.pop(budget_id)
        return save_budgets(user_id, budgets)
    
    return False

def get_transaction_summary(user_id: str) -> Dict[str, Any]:
    """
    Calculate summary statistics from transaction data.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        Dict: Summary statistics
    """
    transactions = load_transactions(user_id)
    
    if not transactions:
        return {
            "total_transactions": 0,
            "total_income": 0,
            "total_expenses": 0,
            "balance": 0,
            "monthly_income": 0,
            "monthly_expenses": 0,
            "savings_rate": 0
        }
    
    # Get total income and expenses
    total_income = sum(t.get("amount", 0) for t in transactions 
                      if t.get("type", "").lower() == "income")
    total_expenses = sum(t.get("amount", 0) for t in transactions 
                        if t.get("type", "").lower() == "expense")
    
    # Get current balance
    balance = total_income - total_expenses
    
    # Calculate monthly income and expenses
    current_month = datetime.datetime.now().replace(day=1).strftime("%Y-%m")
    
    monthly_income = sum(t.get("amount", 0) for t in transactions 
                        if t.get("type", "").lower() == "income"
                        and t.get("date", "").startswith(current_month))
    
    monthly_expenses = sum(t.get("amount", 0) for t in transactions 
                          if t.get("type", "").lower() == "expense"
                          and t.get("date", "").startswith(current_month))
    
    # Calculate savings rate
    savings_rate = 0
    if monthly_income > 0:
        savings_rate = max(0, (monthly_income - monthly_expenses) / monthly_income)
    
    return {
        "total_transactions": len(transactions),
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": balance,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "savings_rate": savings_rate
    }

def calculate_balance(user_id: str) -> float:
    """
    Calculate the current balance from all transactions.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        float: Current balance
    """
    transactions = load_transactions(user_id)
    
    total_income = sum(t.get("amount", 0) for t in transactions 
                      if t.get("type", "").lower() == "income")
    total_expenses = sum(t.get("amount", 0) for t in transactions 
                        if t.get("type", "").lower() == "expense")
    
    return total_income - total_expenses

def get_category_spending(user_id: str) -> Dict[str, float]:
    """
    Get spending by category from transaction data.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        Dict: Dictionary mapping categories to total spending
    """
    transactions = load_transactions(user_id)
    
    # Filter to only include expenses in the current month
    current_month = datetime.datetime.now().replace(day=1).strftime("%Y-%m")
    month_expenses = [t for t in transactions 
                     if t.get("type", "").lower() == "expense"
                     and t.get("date", "").startswith(current_month)]
    
    # Group by category
    category_spending = defaultdict(float)
    
    for expense in month_expenses:
        category = expense.get("category", "Other")
        amount = expense.get("amount", 0)
        category_spending[category] += amount
    
    return dict(category_spending)

def get_daily_spending(user_id: str) -> Dict[str, float]:
    """
    Get daily spending totals from transaction data.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        Dict: Dictionary mapping dates to total spending
    """
    transactions = load_transactions(user_id)
    
    # Filter to only include expenses
    expenses = [t for t in transactions if t.get("type", "").lower() == "expense"]
    
    # Group by date
    daily_spending = defaultdict(float)
    
    for expense in expenses:
        date = expense.get("date", "")
        amount = expense.get("amount", 0)
        daily_spending[date] += amount
    
    return dict(daily_spending)

def get_monthly_spending(user_id: str) -> Dict[str, float]:
    """
    Get monthly spending totals from transaction data.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        Dict: Dictionary mapping month strings to total spending
    """
    transactions = load_transactions(user_id)
    
    # Filter to only include expenses
    expenses = [t for t in transactions if t.get("type", "").lower() == "expense"]
    
    # Group by month
    monthly_spending = defaultdict(float)
    
    for expense in expenses:
        date = expense.get("date", "")
        if date:
            # Extract YYYY-MM format
            month = date[:7]
            amount = expense.get("amount", 0)
            monthly_spending[month] += amount
    
    return dict(monthly_spending)

def check_budget_status(user_id: str) -> List[Dict[str, Any]]:
    """
    Check the status of all budgets for warnings.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        List[Dict]: List of budget warnings
    """
    budgets = load_budgets(user_id)
    category_spending = get_category_spending(user_id)
    
    # Calculate total spending for "All Categories" budgets
    total_spending = sum(category_spending.values())
    
    # Check each budget for warnings
    warnings = []
    
    for budget in budgets:
        category = budget.get("category", "Uncategorized")
        amount = budget.get("amount", 0)
        
        # Get current spending for this category
        if category == "All Categories":
            spent = total_spending
        else:
            spent = category_spending.get(category, 0)
        
        # Calculate remaining amount and percent used
        remaining = max(0, amount - spent)
        percent_used = (spent / amount) * 100 if amount > 0 else 0
        
        # Determine severity based on percent used
        severity = "low"
        if percent_used >= 100:
            severity = "high"
        elif percent_used >= 90:
            severity = "medium"
        elif percent_used >= 75:
            severity = "low"
        else:
            # No warning needed if under 75%
            continue
        
        warnings.append({
            "budget": budget,
            "spent": spent,
            "remaining": remaining,
            "percent_used": percent_used,
            "severity": severity
        })
    
    return warnings