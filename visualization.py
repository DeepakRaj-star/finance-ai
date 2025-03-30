import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import datetime

def create_balance_chart(transactions: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a line chart showing balance over time.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not transactions:
        # Return empty figure if no transactions
        fig = go.Figure()
        fig.update_layout(
            title="Balance Over Time",
            xaxis_title="Date",
            yaxis_title="Balance ($)",
            height=400
        )
        return fig
    
    # Sort transactions by date
    sorted_transactions = sorted(transactions, key=lambda x: x.get('date', ''))
    
    # Calculate running balance
    dates = []
    balances = []
    balance = 0
    
    for txn in sorted_transactions:
        date = txn.get('date', '')
        amount = txn.get('amount', 0)
        txn_type = txn.get('type', '').lower()
        
        if txn_type == 'income':
            balance += amount
        else:  # expense
            balance -= amount
            
        dates.append(date)
        balances.append(balance)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'balance': balances
    })
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Fill in missing dates with last balance
    date_range = pd.date_range(start=df['date'].min(), end=datetime.datetime.now())
    full_df = pd.DataFrame({'date': date_range})
    full_df = full_df.merge(df, on='date', how='left')
    full_df['balance'] = full_df['balance'].ffill()
    
    # Create figure
    fig = px.line(
        full_df, 
        x='date', 
        y='balance',
        title="Balance Over Time",
        labels={'date': 'Date', 'balance': 'Balance ($)'},
        color_discrete_sequence=['#4F8BF9']
    )
    
    # Add reference line at zero
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="red", 
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            tickformat="%b %d",
            tickangle=-45,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    return fig

def create_spending_by_category_chart(category_spending: Dict[str, float]) -> go.Figure:
    """
    Create a pie chart showing spending by category.
    
    Args:
        category_spending (Dict[str, float]): Dictionary mapping categories to amounts
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not category_spending:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Spending by Category",
            height=400
        )
        return fig
    
    # Convert to lists
    categories = list(category_spending.keys())
    values = list(category_spending.values())
    
    # Create figure
    fig = px.pie(
        names=categories,
        values=values,
        title="Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4
    )
    
    # Update layout
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )
    
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    return fig

def create_spending_trend_chart(daily_spending: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart showing spending trends over time.
    
    Args:
        daily_spending (Dict[str, float]): Dictionary mapping dates to spending amounts
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not daily_spending:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Daily Spending",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            height=400
        )
        return fig
    
    # Convert to dataframe
    df = pd.DataFrame({
        'date': list(daily_spending.keys()),
        'amount': list(daily_spending.values())
    })
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Fill in missing dates with zero
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
    full_df = pd.DataFrame({'date': date_range})
    full_df = full_df.merge(df, on='date', how='left')
    full_df['amount'] = full_df['amount'].fillna(0)
    
    # Create 7-day moving average
    full_df['moving_avg'] = full_df['amount'].rolling(window=7, min_periods=1).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for daily spending
    fig.add_trace(go.Bar(
        x=full_df['date'],
        y=full_df['amount'],
        name='Daily Spending',
        marker_color='rgba(63, 81, 181, 0.6)'
    ))
    
    # Add line chart for moving average
    fig.add_trace(go.Scatter(
        x=full_df['date'],
        y=full_df['moving_avg'],
        name='7-Day Average',
        line=dict(color='#FF5722', width=3)
    ))
    
    # Update layout
    fig.update_layout(
        title="Daily Spending Trends",
        xaxis=dict(
            title="Date",
            tickformat="%b %d",
            tickangle=-45,
            showgrid=True
        ),
        yaxis=dict(
            title="Amount ($)",
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    return fig

def create_goal_progress_chart(goals: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing progress towards financial goals.
    
    Args:
        goals (List[Dict]): List of goal dictionaries
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not goals:
        # Return empty figure if no goals
        fig = go.Figure()
        fig.update_layout(
            title="Goal Progress",
            xaxis_title="Goal",
            yaxis_title="Amount ($)",
            height=400
        )
        return fig
    
    # Extract data
    names = [goal.get('name', f'Goal {i+1}') for i, goal in enumerate(goals)]
    target_amounts = [goal.get('target_amount', 0) for goal in goals]
    current_amounts = [goal.get('current_amount', 0) for goal in goals]
    
    # Calculate remaining amounts
    remaining_amounts = [max(0, target - current) for target, current in zip(target_amounts, current_amounts)]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for current amount
    fig.add_trace(go.Bar(
        x=names,
        y=current_amounts,
        name='Current Amount',
        marker_color='rgba(63, 81, 181, 0.8)'
    ))
    
    # Add bar for remaining amount
    fig.add_trace(go.Bar(
        x=names,
        y=remaining_amounts,
        name='Remaining',
        marker_color='rgba(200, 200, 200, 0.5)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Progress Towards Financial Goals",
        barmode='stack',
        xaxis=dict(
            title="",
            tickangle=-45
        ),
        yaxis=dict(
            title="Amount ($)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    # Add target markers
    for i, (name, target) in enumerate(zip(names, target_amounts)):
        fig.add_shape(
            type="line",
            x0=i - 0.4,
            y0=target,
            x1=i + 0.4,
            y1=target,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
    
    return fig

def create_income_expense_comparison_chart(monthly_income: Dict[str, float], monthly_expenses: Dict[str, float]) -> go.Figure:
    """
    Create a chart comparing monthly income and expenses.
    
    Args:
        monthly_income (Dict[str, float]): Dictionary mapping months to income amounts
        monthly_expenses (Dict[str, float]): Dictionary mapping months to expense amounts
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not monthly_income and not monthly_expenses:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Income vs. Expenses",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            height=400
        )
        return fig
    
    # Combine all months
    all_months = set(list(monthly_income.keys()) + list(monthly_expenses.keys()))
    
    # Sort months chronologically
    sorted_months = sorted(all_months)
    
    # Create lists for plotting
    months = []
    income_values = []
    expense_values = []
    
    for month in sorted_months:
        months.append(month)
        income_values.append(monthly_income.get(month, 0))
        expense_values.append(monthly_expenses.get(month, 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for income
    fig.add_trace(go.Bar(
        x=months,
        y=income_values,
        name='Income',
        marker_color='rgba(75, 192, 192, 0.8)'
    ))
    
    # Add bar for expenses
    fig.add_trace(go.Bar(
        x=months,
        y=expense_values,
        name='Expenses',
        marker_color='rgba(255, 99, 132, 0.8)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Monthly Income vs. Expenses",
        barmode='group',
        xaxis=dict(
            title="Month",
            tickangle=-45
        ),
        yaxis=dict(
            title="Amount ($)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    return fig