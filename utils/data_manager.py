import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    # Financial summary metrics
    if 'total_balance' not in st.session_state:
        st.session_state.total_balance = 12568.42
    if 'balance_change' not in st.session_state:
        st.session_state.balance_change = 2.3
    if 'monthly_income' not in st.session_state:
        st.session_state.monthly_income = 5200.00
    if 'income_change' not in st.session_state:
        st.session_state.income_change = 0.0
    if 'monthly_expenses' not in st.session_state:
        st.session_state.monthly_expenses = 3450.75
    if 'expense_change' not in st.session_state:
        st.session_state.expense_change = 1.2
    
    # Daily tip
    if 'daily_tip' not in st.session_state:
        st.session_state.daily_tip = """
        **Build an emergency fund first** - Before focusing on other financial goals, 
        aim to save 3-6 months of essential expenses in an easily accessible account. 
        This provides a crucial safety net for unexpected situations.
        """
    
    # Transactions data - only initialize if not already present
    if 'transactions' not in st.session_state:
        # Generate dates for the last 30 days
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        
        # Define categories with specific characteristics
        categories = {
            'Groceries': {'type': 'expense', 'min': 15, 'max': 120},
            'Dining': {'type': 'expense', 'min': 10, 'max': 85},
            'Utilities': {'type': 'expense', 'min': 40, 'max': 200},
            'Transport': {'type': 'expense', 'min': 5, 'max': 60},
            'Entertainment': {'type': 'expense', 'min': 15, 'max': 100},
            'Shopping': {'type': 'expense', 'min': 20, 'max': 200},
            'Healthcare': {'type': 'expense', 'min': 15, 'max': 300},
            'Salary': {'type': 'income', 'min': 2000, 'max': 3000},
            'Freelance': {'type': 'income', 'min': 200, 'max': 1000},
            'Investments': {'type': 'income', 'min': 50, 'max': 500}
        }
        
        # Create transactions with realistic patterns
        transactions = []
        
        # Add regular income (salary twice a month)
        salary_days = [3, 18]  # Typical salary days in a month
        for day in salary_days:
            if day <= today.day:
                transactions.append({
                    'date': f"{today.year}-{today.month:02d}-{day:02d}",
                    'description': 'Monthly Salary',
                    'amount': round(np.random.uniform(2500, 2600), 2),
                    'category': 'Salary',
                    'type': 'income'
                })
        
        # Add regular expenses (bills, rent)
        rent_day = 1
        if rent_day <= today.day:
            transactions.append({
                'date': f"{today.year}-{today.month:02d}-{rent_day:02d}",
                'description': 'Monthly Rent',
                'amount': round(1200.00, 2),
                'category': 'Housing',
                'type': 'expense'
            })
        
        # Add utilities around the 15th
        utilities_day = 15
        if utilities_day <= today.day:
            transactions.append({
                'date': f"{today.year}-{today.month:02d}-{utilities_day:02d}",
                'description': 'Utility Bills',
                'amount': round(np.random.uniform(150, 180), 2),
                'category': 'Utilities',
                'type': 'expense'
            })
        
        # Add random transactions
        for _ in range(40):  # Generate about 40 random transactions
            date = random.choice(dates)
            category = random.choice(list(categories.keys()))
            cat_props = categories[category]
            
            # Generate a realistic amount based on category
            amount = round(np.random.uniform(cat_props['min'], cat_props['max']), 2)
            
            # Generate a realistic description based on category
            descriptions = {
                'Groceries': ['Trader Joe\'s', 'Whole Foods', 'Safeway', 'Kroger', 'Costco'],
                'Dining': ['Restaurant', 'Coffee Shop', 'Fast Food', 'Pizza Delivery', 'Sushi Bar'],
                'Utilities': ['Electricity Bill', 'Water Bill', 'Internet Service', 'Phone Bill', 'Gas Bill'],
                'Transport': ['Uber', 'Lyft', 'Gas Station', 'Public Transit', 'Parking Fee'],
                'Entertainment': ['Movie Theater', 'Concert Tickets', 'Streaming Service', 'Game Purchase', 'Book Store'],
                'Shopping': ['Amazon', 'Target', 'Walmart', 'Best Buy', 'Clothing Store'],
                'Healthcare': ['Pharmacy', 'Doctor Visit', 'Dental Checkup', 'Eye Exam', 'Health Insurance'],
                'Salary': ['Salary Payment', 'Paycheck', 'Direct Deposit'],
                'Freelance': ['Client Payment', 'Contract Work', 'Consulting Fee', 'Side Project'],
                'Investments': ['Dividend', 'Stock Sale', 'Interest Income', 'Investment Return']
            }
            
            description = random.choice(descriptions.get(category, ['Payment']))
            
            transactions.append({
                'date': date,
                'description': description,
                'amount': amount,
                'category': category,
                'type': cat_props['type']
            })
        
        # Sort transactions by date (most recent first)
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Convert to DataFrame and store in session state
        st.session_state.transactions = pd.DataFrame(transactions)
    
    # Financial goals
    if 'goals' not in st.session_state:
        st.session_state.goals = pd.DataFrame({
            'name': ['Emergency Fund', 'Vacation', 'New Laptop', 'Home Down Payment'],
            'target': [10000, 2500, 1200, 60000],
            'current': [6500, 1800, 450, 12000],
            'deadline': ['2024-10-01', '2024-07-15', '2024-05-30', '2025-12-31']
        })

def add_transaction(date, description, amount, category, transaction_type):
    """Add a new transaction to the session state."""
    new_transaction = {
        'date': date,
        'description': description,
        'amount': float(amount),
        'category': category,
        'type': transaction_type
    }
    
    # Append the new transaction
    st.session_state.transactions = pd.concat([
        pd.DataFrame([new_transaction]),
        st.session_state.transactions
    ], ignore_index=True)
    
    # Update the summary metrics
    if transaction_type == 'income':
        st.session_state.total_balance += float(amount)
        if date.startswith(f"{datetime.now().year}-{datetime.now().month:02d}"):
            st.session_state.monthly_income += float(amount)
    else:  # expense
        st.session_state.total_balance -= float(amount)
        if date.startswith(f"{datetime.now().year}-{datetime.now().month:02d}"):
            st.session_state.monthly_expenses += float(amount)

def update_goal(index, current_value):
    """Update the current value of a financial goal."""
    st.session_state.goals.at[index, 'current'] = current_value

def add_goal(name, target, current, deadline):
    """Add a new financial goal."""
    new_goal = {
        'name': name,
        'target': float(target),
        'current': float(current),
        'deadline': deadline
    }
    
    # Append the new goal
    st.session_state.goals = pd.concat([
        st.session_state.goals,
        pd.DataFrame([new_goal])
    ], ignore_index=True)

def get_monthly_summary():
    """Calculate monthly income and expenses summary."""
    current_month = datetime.now().strftime('%Y-%m')
    monthly_data = st.session_state.transactions[
        st.session_state.transactions['date'].str.startswith(current_month)
    ]
    
    income = monthly_data[monthly_data['type'] == 'income']['amount'].sum()
    expenses = monthly_data[monthly_data['type'] == 'expense']['amount'].sum()
    
    return income, expenses, income - expenses  # income, expenses, savings

def get_category_breakdown(transaction_type='expense'):
    """Get breakdown of transactions by category for the current month."""
    current_month = datetime.now().strftime('%Y-%m')
    monthly_data = st.session_state.transactions[
        (st.session_state.transactions['date'].str.startswith(current_month)) &
        (st.session_state.transactions['type'] == transaction_type)
    ]
    
    category_totals = monthly_data.groupby('category')['amount'].sum().reset_index()
    return category_totals
