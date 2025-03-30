import os
import hashlib
import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
import json

from db_models import (
    User, 
    Transaction, 
    Goal, 
    Budget, 
    VectorTransaction, 
    FinancialAnalysis,
    UserSetting,
    get_db,
    init_db
)

# Initialize the database on import
def init_db():
    """Initialize the database on import."""
    try:
        init_db()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        # Continue without failing to allow basic functionality
        pass

# User operations
def get_user_by_email(email: str) -> Optional[User]:
    """Get a user by email."""
    db = get_db()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()

def get_user_by_id(user_id: str) -> Optional[User]:
    """Get a user by ID."""
    db = get_db()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()

def create_user(email: str, password: str, name: str) -> User:
    """Create a new user."""
    db = get_db()
    try:
        # Generate user ID from email
        user_id = hashlib.md5(email.encode()).hexdigest()

        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Create user
        user = User(
            id=user_id,
            email=email,
            password_hash=password_hash,
            name=name
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        # Create default settings
        settings = UserSetting(user_id=user_id)
        db.add(settings)
        db.commit()

        return user
    finally:
        db.close()

def verify_password(stored_hash: str, password: str) -> bool:
    """Verify a password against a hash."""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return stored_hash == password_hash

# Transaction operations
def get_transactions(user_id: str) -> List[Transaction]:
    """Get all transactions for a user."""
    db = get_db()
    try:
        return db.query(Transaction).filter(Transaction.user_id == user_id).all()
    finally:
        db.close()

def create_transaction(user_id: str, transaction_data: Dict[str, Any]) -> Transaction:
    """Create a new transaction."""
    db = get_db()
    try:
        # Create transaction
        transaction = Transaction(
            user_id=user_id,
            date=transaction_data.get("date"),
            description=transaction_data.get("description"),
            amount=transaction_data.get("amount"),
            type=transaction_data.get("type"),
            category=transaction_data.get("category"),
            notes=transaction_data.get("notes")
        )

        db.add(transaction)
        db.commit()
        db.refresh(transaction)

        return transaction
    finally:
        db.close()

def update_transaction(transaction_id: int, transaction_data: Dict[str, Any]) -> Optional[Transaction]:
    """Update a transaction."""
    db = get_db()
    try:
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()

        if not transaction:
            return None

        # Update fields
        for key, value in transaction_data.items():
            if hasattr(transaction, key):
                setattr(transaction, key, value)

        db.commit()
        db.refresh(transaction)

        return transaction
    finally:
        db.close()

def delete_transaction(transaction_id: int) -> bool:
    """Delete a transaction."""
    db = get_db()
    try:
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()

        if not transaction:
            return False

        db.delete(transaction)
        db.commit()

        return True
    finally:
        db.close()

# Goal operations
def get_goals(user_id: str) -> List[Goal]:
    """Get all goals for a user."""
    db = get_db()
    try:
        return db.query(Goal).filter(Goal.user_id == user_id).all()
    finally:
        db.close()

def create_goal(user_id: str, goal_data: Dict[str, Any]) -> Goal:
    """Create a new goal."""
    db = get_db()
    try:
        # Create goal
        goal = Goal(
            user_id=user_id,
            name=goal_data.get("name"),
            target_amount=goal_data.get("target_amount"),
            current_amount=goal_data.get("current_amount", 0.0),
            deadline=goal_data.get("deadline"),
            category=goal_data.get("category"),
            notes=goal_data.get("notes")
        )

        db.add(goal)
        db.commit()
        db.refresh(goal)

        return goal
    finally:
        db.close()

def update_goal(goal_id: int, goal_data: Dict[str, Any]) -> Optional[Goal]:
    """Update a goal."""
    db = get_db()
    try:
        goal = db.query(Goal).filter(Goal.id == goal_id).first()

        if not goal:
            return None

        # Update fields
        for key, value in goal_data.items():
            if hasattr(goal, key):
                setattr(goal, key, value)

        db.commit()
        db.refresh(goal)

        return goal
    finally:
        db.close()

def delete_goal(goal_id: int) -> bool:
    """Delete a goal."""
    db = get_db()
    try:
        goal = db.query(Goal).filter(Goal.id == goal_id).first()

        if not goal:
            return False

        db.delete(goal)
        db.commit()

        return True
    finally:
        db.close()

# Budget operations
def get_budgets(user_id: str) -> List[Budget]:
    """Get all budgets for a user."""
    db = get_db()
    try:
        return db.query(Budget).filter(Budget.user_id == user_id).all()
    finally:
        db.close()

def create_budget(user_id: str, budget_data: Dict[str, Any]) -> Budget:
    """Create a new budget."""
    db = get_db()
    try:
        # Create budget
        budget = Budget(
            user_id=user_id,
            category=budget_data.get("category"),
            amount=budget_data.get("amount"),
            period=budget_data.get("period"),
            notes=budget_data.get("notes")
        )

        db.add(budget)
        db.commit()
        db.refresh(budget)

        return budget
    finally:
        db.close()

def update_budget(budget_id: int, budget_data: Dict[str, Any]) -> Optional[Budget]:
    """Update a budget."""
    db = get_db()
    try:
        budget = db.query(Budget).filter(Budget.id == budget_id).first()

        if not budget:
            return None

        # Update fields
        for key, value in budget_data.items():
            if hasattr(budget, key):
                setattr(budget, key, value)

        db.commit()
        db.refresh(budget)

        return budget
    finally:
        db.close()

def delete_budget(budget_id: int) -> bool:
    """Delete a budget."""
    db = get_db()
    try:
        budget = db.query(Budget).filter(Budget.id == budget_id).first()

        if not budget:
            return False

        db.delete(budget)
        db.commit()

        return True
    finally:
        db.close()

# Analysis operations
def create_financial_analysis(user_id: str, analysis_type: str, content: str, metadata: Dict[str, Any] = None) -> FinancialAnalysis:
    """Create a new financial analysis."""
    db = get_db()
    try:
        # Create analysis
        analysis = FinancialAnalysis(
            user_id=user_id,
            analysis_type=analysis_type,
            content=content,
            meta_data=metadata
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return analysis
    finally:
        db.close()

def get_financial_analyses(user_id: str, analysis_type: str = None) -> List[FinancialAnalysis]:
    """Get financial analyses for a user."""
    db = get_db()
    try:
        query = db.query(FinancialAnalysis).filter(FinancialAnalysis.user_id == user_id)

        if analysis_type:
            query = query.filter(FinancialAnalysis.analysis_type == analysis_type)

        return query.order_by(FinancialAnalysis.created_at.desc()).all()
    finally:
        db.close()

# User settings operations
def get_user_settings(user_id: str) -> Optional[UserSetting]:
    """Get user settings."""
    db = get_db()
    try:
        return db.query(UserSetting).filter(UserSetting.user_id == user_id).first()
    finally:
        db.close()

def update_user_settings(user_id: str, settings_data: Dict[str, Any]) -> Optional[UserSetting]:
    """Update user settings."""
    db = get_db()
    try:
        settings = db.query(UserSetting).filter(UserSetting.user_id == user_id).first()

        if not settings:
            # Create settings if they don't exist
            settings = UserSetting(user_id=user_id)
            db.add(settings)

        # Update fields
        for key, value in settings_data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        db.commit()
        db.refresh(settings)

        return settings
    finally:
        db.close()

# Helper functions for analytics
def get_category_spending(user_id: str, start_date: str = None, end_date: str = None) -> Dict[str, float]:
    """Get spending by category."""
    db = get_db()
    try:
        query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense"
        )

        if start_date:
            query = query.filter(Transaction.date >= start_date)

        if end_date:
            query = query.filter(Transaction.date <= end_date)

        transactions = query.all()

        # Group by category
        category_spending = {}
        for transaction in transactions:
            category = transaction.category
            amount = transaction.amount

            if category not in category_spending:
                category_spending[category] = 0

            category_spending[category] += amount

        return category_spending
    finally:
        db.close()

def get_daily_spending(user_id: str, start_date: str = None, end_date: str = None) -> Dict[str, float]:
    """Get daily spending."""
    db = get_db()
    try:
        query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense"
        )

        if start_date:
            query = query.filter(Transaction.date >= start_date)

        if end_date:
            query = query.filter(Transaction.date <= end_date)

        transactions = query.all()

        # Group by date
        daily_spending = {}
        for transaction in transactions:
            date = transaction.date
            amount = transaction.amount

            if date not in daily_spending:
                daily_spending[date] = 0

            daily_spending[date] += amount

        return daily_spending
    finally:
        db.close()

def get_monthly_spending(user_id: str, year: int = None) -> Dict[str, float]:
    """Get monthly spending."""
    db = get_db()
    try:
        query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense"
        )

        if year:
            query = query.filter(Transaction.date.like(f"{year}-%"))

        transactions = query.all()

        # Group by month
        monthly_spending = {}
        for transaction in transactions:
            # Extract year and month (YYYY-MM)
            month = transaction.date[:7]
            amount = transaction.amount

            if month not in monthly_spending:
                monthly_spending[month] = 0

            monthly_spending[month] += amount

        return monthly_spending
    finally:
        db.close()

def calculate_balance(user_id: str) -> float:
    """Calculate the current balance."""
    db = get_db()
    try:
        # Get income
        income_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "income"
        )

        total_income = sum(transaction.amount for transaction in income_query.all())

        # Get expenses
        expense_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense"
        )

        total_expenses = sum(transaction.amount for transaction in expense_query.all())

        # Calculate balance
        return total_income - total_expenses
    finally:
        db.close()

def check_budget_status(user_id: str) -> List[Dict[str, Any]]:
    """Check budget status and return warnings."""
    db = get_db()
    try:
        # Get budgets
        budgets = db.query(Budget).filter(Budget.user_id == user_id).all()

        # Get current month
        current_month = datetime.datetime.now().strftime("%Y-%m")

        # Get monthly spending by category
        query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense",
            Transaction.date.like(f"{current_month}%")
        )

        transactions = query.all()

        # Calculate spending by category
        category_spending = {}
        total_spending = 0

        for transaction in transactions:
            category = transaction.category
            amount = transaction.amount

            if category not in category_spending:
                category_spending[category] = 0

            category_spending[category] += amount
            total_spending += amount

        # Check budgets
        warnings = []

        for budget in budgets:
            category = budget.category
            amount = budget.amount

            # Get spending for this category
            if category == "All Categories":
                spent = total_spending
            else:
                spent = category_spending.get(category, 0)

            # Calculate remaining and percent used
            remaining = max(0, amount - spent)
            percent_used = (spent / amount) * 100 if amount > 0 else 0

            # Determine severity
            severity = "low"
            if percent_used >= 100:
                severity = "high"
            elif percent_used >= 90:
                severity = "medium"
            elif percent_used >= 75:
                severity = "low"
            else:
                # Skip if under 75%
                continue

            warnings.append({
                "budget": budget.to_dict(),
                "spent": spent,
                "remaining": remaining,
                "percent_used": percent_used,
                "severity": severity
            })

        return warnings
    finally:
        db.close()

def reset_user_data(user_id: str) -> bool:
    """
    Reset all user data - delete all transactions, goals, budgets, and analyses.

    Args:
        user_id (str): The user ID

    Returns:
        bool: Success or failure
    """
    db = get_db()
    try:
        # Start a transaction
        db.begin()

        # Delete all vector transactions for user's transactions
        vector_txns = db.query(VectorTransaction).join(
            Transaction, 
            Transaction.id == VectorTransaction.transaction_id
        ).filter(
            Transaction.user_id == user_id
        ).all()

        for vtxn in vector_txns:
            db.delete(vtxn)

        # Delete all financial analyses
        analyses = db.query(FinancialAnalysis).filter(
            FinancialAnalysis.user_id == user_id
        ).all()

        for analysis in analyses:
            db.delete(analysis)

        # Delete all transactions
        transactions = db.query(Transaction).filter(
            Transaction.user_id == user_id
        ).all()

        for txn in transactions:
            db.delete(txn)

        # Delete all goals
        goals = db.query(Goal).filter(
            Goal.user_id == user_id
        ).all()

        for goal in goals:
            db.delete(goal)

        # Delete all budgets
        budgets = db.query(Budget).filter(
            Budget.user_id == user_id
        ).all()

        for budget in budgets:
            db.delete(budget)

        # Commit changes
        db.commit()
        return True

    except Exception as e:
        db.rollback()
        print(f"Error resetting user data: {str(e)}")
        return False

    finally:
        db.close()

def get_transaction_summary(user_id: str) -> Dict[str, Any]:
    """Get transaction summary for a user."""
    db = get_db()
    try:
        # Get total transactions
        total_transactions = db.query(Transaction).filter(Transaction.user_id == user_id).count()

        # Get total income and expenses
        income_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "income"
        )

        expense_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense"
        )

        total_income = sum(transaction.amount for transaction in income_query.all())
        total_expenses = sum(transaction.amount for transaction in expense_query.all())

        # Calculate balance
        balance = total_income - total_expenses

        # Get monthly income and expenses
        current_month = datetime.datetime.now().strftime("%Y-%m")

        monthly_income_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "income",
            Transaction.date.like(f"{current_month}%")
        )

        monthly_expense_query = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == "expense",
            Transaction.date.like(f"{current_month}%")
        )

        monthly_income = sum(transaction.amount for transaction in monthly_income_query.all())
        monthly_expenses = sum(transaction.amount for transaction in monthly_expense_query.all())

        # Calculate savings rate
        savings_rate = 0
        if monthly_income > 0:
            savings_rate = max(0, (monthly_income - monthly_expenses) / monthly_income)

        return {
            "total_transactions": total_transactions,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "balance": balance,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "savings_rate": savings_rate
        }
    finally:
        db.close()