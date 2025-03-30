"""Utility functions for the finance application."""
import datetime
from typing import Tuple, List, Dict, Any, Optional

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
    
    # Analyze transactions
    total_income = sum(t.get("amount", 0) for t in transactions if t.get("type") == "income")
    total_expenses = sum(t.get("amount", 0) for t in transactions if t.get("type") == "expense")
    
    # Income vs. Expenses
    if total_expenses > total_income * 0.9:
        recommendations.append(
            "Your expenses are approaching your income level. Consider reducing non-essential spending."
        )
    
    # Check if there are any regular savings
    savings_transactions = [t for t in transactions if "saving" in t.get("category", "").lower()]
    if not savings_transactions:
        recommendations.append(
            "Consider setting up automatic transfers to a savings account for financial security."
        )
    
    # Categorize expenses
    expense_categories = {}
    for t in transactions:
        if t.get("type") == "expense":
            category = t.get("category", "Other")
            amount = t.get("amount", 0)
            if category not in expense_categories:
                expense_categories[category] = 0
            expense_categories[category] += amount
    
    # Check for high spending categories
    if expense_categories:
        top_category = max(expense_categories, key=expense_categories.get)
        top_category_pct = (expense_categories[top_category] / total_expenses) * 100 if total_expenses > 0 else 0
        
        if top_category_pct > 40:
            recommendations.append(
                f"Your spending on {top_category} represents {top_category_pct:.1f}% of your expenses. "
                f"Consider if there are opportunities to reduce this category."
            )
    
    # Add more generic recommendations
    recommendations.extend([
        "Review your subscriptions regularly to identify services you no longer use.",
        "Consider using cash-back credit cards for everyday purchases to earn rewards.",
        "Track your progress toward financial goals to stay motivated.",
        "Set up alerts for unusual transactions to prevent fraud.",
        "Review your investment strategy regularly to ensure it aligns with your goals."
    ])
    
    # Limit to 5 recommendations
    return recommendations[:5]

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
    import plotly.express as px
    import pandas as pd
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame([
        {"Category": cat, "Amount": amount} 
        for cat, amount in category_data.items() if amount > 0
    ])
    
    if df.empty:
        # Create an empty figure if no data
        return px.pie(title="No spending data available")
    
    # Sort by amount (descending)
    df = df.sort_values("Amount", ascending=False)
    
    # Create pie chart
    fig = px.pie(
        df,
        values="Amount",
        names="Category",
        title="Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    
    # Update layout for better visualization
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Format the labels
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}: $%{value:.2f} (%{percent})",
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
    import plotly.express as px
    import pandas as pd
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame([
        {"Date": date, "Amount": amount} 
        for date, amount in daily_spending.items()
    ])
    
    if df.empty:
        # Create an empty figure if no data
        return px.line(title="No spending data available")
    
    # Convert dates to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Create line chart
    fig = px.line(
        df,
        x="Date",
        y="Amount",
        title="Daily Spending Trend",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#9370DB"],
    )
    
    # Add 7-day moving average
    if len(df) >= 7:
        df["7-Day Avg"] = df["Amount"].rolling(window=7).mean()
        fig.add_scatter(
            x=df["Date"],
            y=df["7-Day Avg"],
            name="7-Day Avg",
            line=dict(color="#FFA07A", dash="dash"),
        )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Amount",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix="$")
    
    return fig

def create_balance_chart(transactions: List[Dict[str, Any]]):
    """
    Create a chart showing balance over time.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        Plotly figure
    """
    import plotly.express as px
    import pandas as pd
    
    if not transactions:
        # Create an empty figure if no data
        return px.line(title="No data available to calculate balance")
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Convert dates to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date
    df = df.sort_values("date")
    
    # Create a new DataFrame for calculating running balance
    balance_data = []
    running_balance = 0
    
    for _, row in df.iterrows():
        amount = row["amount"]
        txn_type = row["type"]
        
        # Update running balance
        if txn_type == "income":
            running_balance += amount
        else:  # expense
            running_balance -= amount
        
        balance_data.append({
            "Date": row["date"],
            "Balance": running_balance
        })
    
    # Create balance DataFrame
    balance_df = pd.DataFrame(balance_data)
    
    # Create line chart
    fig = px.line(
        balance_df,
        x="Date",
        y="Balance",
        title="Balance Over Time",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#9370DB"],
    )
    
    # Add horizontal line at zero
    fig.add_shape(
        type="line",
        x0=balance_df["Date"].min(),
        y0=0,
        x1=balance_df["Date"].max(),
        y1=0,
        line=dict(color="#FF6B6B", dash="dash", width=1),
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Balance",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix="$")
    
    return fig