"""Core utility functions for the finance application."""
import datetime
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_date_range(period: str) -> Tuple[str, str]:
    """
    Get date range for a specific period.
    
    Args:
        period: Period name ("week", "month", "quarter", "year", "all")
        
    Returns:
        Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
    """
    today = datetime.datetime.now()
    end_date = today.strftime("%Y-%m-%d")
    
    if period == "week":
        start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    elif period == "month":
        start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    elif period == "quarter":
        start_date = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    elif period == "year":
        start_date = f"{today.year}-01-01"
    else:  # "all"
        start_date = "2000-01-01"  # Far enough in the past
    
    return start_date, end_date

def get_recommendations(transactions: List[Dict[str, Any]]) -> List[str]:
    """
    Get personalized financial recommendations based on transaction history.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Default recommendations if no transactions
    if not transactions:
        return [
            "Start by adding your income and expenses to get personalized recommendations.",
            "Set up budget categories to track your spending effectively.",
            "Add savings goals to monitor your progress toward financial targets."
        ]
    
    # Simple recommendations based on transaction patterns
    expenses = [t for t in transactions if t.get("type") == "expense"]
    income = [t for t in transactions if t.get("type") == "income"]
    
    # Analyze expense categories
    if expenses:
        categories = {}
        for expense in expenses:
            cat = expense.get("category", "Other")
            if cat in categories:
                categories[cat] += expense.get("amount", 0)
            else:
                categories[cat] = expense.get("amount", 0)
        
        # Find highest expense category
        if categories:
            highest_cat = max(categories.items(), key=lambda x: x[1])
            recommendations.append(f"Your highest spending is in '{highest_cat[0]}' category. Consider setting a budget limit for this area.")
    
    # Check income vs expenses
    if income and expenses:
        total_income = sum(t.get("amount", 0) for t in income)
        total_expenses = sum(t.get("amount", 0) for t in expenses)
        
        if total_expenses > total_income * 0.9:
            recommendations.append("Your expenses are approaching your income. Consider reducing non-essential spending to improve your savings rate.")
        
        # Calculate savings rate
        if total_income > 0:
            savings_rate = (total_income - total_expenses) / total_income
            if savings_rate < 0.1:
                recommendations.append("Your savings rate is less than 10%. Consider the 50/30/20 rule: 50% on needs, 30% on wants, and 20% on savings.")
            elif savings_rate > 0.2:
                recommendations.append("Great job saving! Consider investing some of your savings for long-term growth.")
    
    # Common budget categories
    budget_categories = [
        "Housing & Utilities",
        "Groceries & Food",
        "Transportation",
        "Healthcare",
        "Insurance",
        "Entertainment",
        "Shopping",
        "Personal Care",
        "Education",
        "Savings",
        "Investments",
        "Emergency Fund",
        "Debt Payments",
        "Subscriptions",
        "Travel",
        "Gifts & Donations",
        "Home Maintenance",
        "Pet Expenses",
        "Hobbies",
        "Dining Out",
        "Fitness & Health",
        "Electronics",
        "Clothing",
        "Children Expenses",
        "Professional Development"
    ]

    # Add some general recommendations
    general_recs = [
        {"type": "Cost Optimization", "tip": "Review subscriptions monthly to eliminate unused services."},
        {"type": "Savings", "tip": "Consider automated transfers to a savings account on payday."},
        {"type": "Financial Tracking", "tip": "Track your net worth monthly to measure your financial progress."},
        {"type": "Emergency Planning", "tip": "Maintain an emergency fund of 3-6 months of expenses."},
        {"type": "Investment", "tip": "Review your investment portfolio at least quarterly to ensure alignment with your goals."}
    ]
    
    # Add 2-3 general recommendations
    import random
    sample_size = min(3, len(general_recs))
    recommendations.extend(random.sample(general_recs, sample_size))
    
    return recommendations

def format_currency(amount: float, currency: str = "$") -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Amount to format
        currency: Currency symbol (default: $)
    
    Returns:
        Formatted currency string
    """
    if amount >= 0:
        return f"{currency}{amount:,.2f}"
    else:
        return f"-{currency}{abs(amount):,.2f}"

def create_spending_by_category_chart(category_data: Dict[str, float]):
    """
    Create a chart for spending by category.
    
    Args:
        category_data: Dictionary mapping categories to amounts
        
    Returns:
        Plotly figure
    """
    if not category_data:
        # Return empty chart if no data
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'category': list(category_data.keys()),
        'amount': list(category_data.values())
    })
    
    # Sort by amount descending
    df = df.sort_values('amount', ascending=False)
    
    # Create pie chart
    fig = px.pie(
        df,
        values='amount',
        names='category',
        title="Spending by Category",
        color_discrete_sequence=px.colors.sequential.Purp,
        hole=0.4
    )
    
    # Custom styling
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#161B22', width=1))
    )
    
    fig.update_layout(
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(
            bgcolor='rgba(22, 27, 34, 0.6)',
            bordercolor='rgba(147, 112, 219, 0.3)',
            borderwidth=1
        )
    )
    
    return fig

def create_spending_trend_chart(daily_spending: Dict[str, float]):
    """
    Create a chart for spending trends over time.
    
    Args:
        daily_spending: Dictionary mapping dates to spending amounts
        
    Returns:
        Plotly figure
    """
    if not daily_spending:
        # Return empty chart if no data
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(list(daily_spending.keys())),
        'amount': list(daily_spending.values())
    })
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create line chart
    fig = px.line(
        df,
        x='date',
        y='amount',
        title="Daily Spending Trend",
        markers=True
    )
    
    # Add area below line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['amount'],
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(147, 112, 219, 0.2)',
            showlegend=False
        )
    )
    
    # Custom styling
    fig.update_traces(
        line=dict(color='#9370DB', width=2),
        marker=dict(color='#C8A2FF', size=8)
    )
    
    fig.update_layout(
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(147, 112, 219, 0.1)',
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(147, 112, 219, 0.1)',
            title=None
        )
    )
    
    return fig

def create_balance_chart(transactions: List[Dict[str, Any]]):
    """
    Create a chart showing balance over time.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        Plotly figure
    """
    if not transactions:
        # Return empty chart if no data
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Ensure date is datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate running balance
    df['amount_signed'] = df.apply(
        lambda x: x['amount'] if x['type'] == 'income' else -x['amount'],
        axis=1
    )
    df['balance'] = df['amount_signed'].cumsum()
    
    # Create line chart
    fig = px.line(
        df,
        x='date',
        y='balance',
        title="Account Balance Over Time",
        markers=True
    )
    
    # Custom styling
    fig.update_traces(
        line=dict(color='#9370DB', width=2),
        marker=dict(color='#C8A2FF', size=8)
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=df['date'].min(),
        y0=0,
        x1=df['date'].max(),
        y1=0,
        line=dict(
            color="rgba(255, 255, 255, 0.3)",
            width=1,
            dash="dash",
        )
    )
    
    # Shade area below zero in red, above zero in green
    for i, row in df.iterrows():
        if i == 0:
            continue
        
        prev_balance = df.iloc[i-1]['balance']
        curr_balance = row['balance']
        prev_date = df.iloc[i-1]['date']
        curr_date = row['date']
        
        if prev_balance <= 0 and curr_balance <= 0:
            # Both points below zero - red
            fig.add_trace(
                go.Scatter(
                    x=[prev_date, curr_date],
                    y=[prev_balance, curr_balance],
                    mode='none',
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    showlegend=False
                )
            )
        elif prev_balance >= 0 and curr_balance >= 0:
            # Both points above zero - green
            fig.add_trace(
                go.Scatter(
                    x=[prev_date, curr_date],
                    y=[prev_balance, curr_balance],
                    mode='none',
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    showlegend=False
                )
            )
    
    fig.update_layout(
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(147, 112, 219, 0.1)',
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(147, 112, 219, 0.1)',
            title=None
        )
    )
    
    return fig