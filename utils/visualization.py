import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def plot_monthly_balance(transactions_df):
    """Create a line chart showing daily balance over the current month."""
    # Filter for current month
    current_month = datetime.now().strftime('%Y-%m')
    monthly_data = transactions_df[transactions_df['date'].str.startswith(current_month)].copy()
    
    if monthly_data.empty:
        return go.Figure().update_layout(
            title="No data available for the current month",
            xaxis_title="Date",
            yaxis_title="Balance ($)"
        )
    
    # Convert to datetime for proper sorting
    monthly_data['date'] = pd.to_datetime(monthly_data['date'])
    monthly_data = monthly_data.sort_values('date')
    
    # Create a date range for the entire month
    start_date = datetime(datetime.now().year, datetime.now().month, 1)
    if datetime.now().month == 12:
        next_month = datetime(datetime.now().year + 1, 1, 1)
    else:
        next_month = datetime(datetime.now().year, datetime.now().month + 1, 1)
    
    end_date = next_month - timedelta(days=1)
    
    # Create a DataFrame with all dates in the month
    date_range = pd.date_range(start=start_date, end=min(end_date, datetime.now()))
    daily_balance = pd.DataFrame(date_range, columns=['date'])
    
    # Calculate running balance
    monthly_data['value'] = monthly_data.apply(
        lambda x: x['amount'] if x['type'] == 'income' else -x['amount'], axis=1
    )
    
    # Group by date and calculate daily net
    daily_net = monthly_data.groupby(monthly_data['date'].dt.date)['value'].sum().reset_index()
    
    # Merge with date range
    daily_balance = daily_balance.merge(
        daily_net, left_on=daily_balance['date'].dt.date, 
        right_on='date', how='left'
    ).fillna(0)
    
    # Calculate cumulative balance
    daily_balance['cumulative_balance'] = daily_balance['value'].cumsum() + st.session_state.total_balance
    
    # Create the line chart
    fig = px.line(
        daily_balance, 
        x='date', 
        y='cumulative_balance',
        title="Daily Balance This Month",
        labels={'date': 'Date', 'cumulative_balance': 'Balance ($)'},
        markers=True
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        height=400
    )
    
    return fig

def plot_category_pie_chart(transaction_type='expense'):
    """Create a pie chart showing breakdown by category."""
    category_data = st.session_state.transactions[
        st.session_state.transactions['type'] == transaction_type
    ].groupby('category')['amount'].sum().reset_index()
    
    # Check if data exists
    if category_data.empty:
        return go.Figure().update_layout(
            title=f"No {transaction_type} data available",
            template="plotly_dark"
        )
    
    # Create a pie chart
    title = 'Expense Breakdown by Category' if transaction_type == 'expense' else 'Income Breakdown by Category'
    fig = px.pie(
        category_data, 
        values='amount', 
        names='category',
        title=title,
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    # Customize the layout
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400
    )
    
    # Customize text
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        insidetextfont=dict(size=12)
    )
    
    return fig

def plot_income_vs_expense_bar():
    """Create a bar chart comparing income vs expenses over time."""
    # Get the last 6 months of data
    months = []
    for i in range(5, -1, -1):
        date = datetime.now() - timedelta(days=30*i)
        months.append(date.strftime('%Y-%m'))
    
    # Filter and process data
    monthly_summary = []
    for month in months:
        monthly_data = st.session_state.transactions[
            st.session_state.transactions['date'].str.startswith(month)
        ]
        
        income = monthly_data[monthly_data['type'] == 'income']['amount'].sum()
        expense = monthly_data[monthly_data['type'] == 'expense']['amount'].sum()
        savings = income - expense
        
        monthly_summary.append({
            'month': month,
            'Income': income,
            'Expenses': expense,
            'Savings': savings
        })
    
    summary_df = pd.DataFrame(monthly_summary)
    
    # Handle empty data
    if summary_df.empty or summary_df['Income'].sum() == 0 and summary_df['Expenses'].sum() == 0:
        return go.Figure().update_layout(
            title="No income or expense data available",
            template="plotly_dark"
        )
    
    # Format month labels
    summary_df['month_display'] = summary_df['month'].apply(
        lambda x: datetime.strptime(x, '%Y-%m').strftime('%b %Y')
    )
    
    # Convert to long format for Plotly
    plot_data = pd.melt(
        summary_df, 
        id_vars=['month_display'], 
        value_vars=['Income', 'Expenses', 'Savings'],
        var_name='Category', 
        value_name='Amount'
    )
    
    # Create a grouped bar chart
    fig = px.bar(
        plot_data,
        x='month_display',
        y='Amount',
        color='Category',
        title="Monthly Income vs Expenses",
        barmode='group',
        color_discrete_map={
            'Income': '#3366CC',
            'Expenses': '#DC3912',
            'Savings': '#109618'
        }
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title=None,
        height=400
    )
    
    return fig

def plot_spending_trends():
    """Create a line chart showing spending trends by category over time."""
    # Get expense data from transactions
    expense_data = st.session_state.transactions[
        st.session_state.transactions['type'] == 'expense'
    ].copy()
    
    if expense_data.empty:
        return go.Figure().update_layout(
            title="No expense data available",
            template="plotly_dark"
        )
    
    # Convert to datetime and extract month
    expense_data['date'] = pd.to_datetime(expense_data['date'])
    expense_data['month'] = expense_data['date'].dt.strftime('%Y-%m')
    
    # Get the last 6 months
    today = datetime.now()
    months = [(today - timedelta(days=30*i)).strftime('%Y-%m') for i in range(5, -1, -1)]
    
    # Filter for top categories to avoid cluttered chart
    top_categories = expense_data.groupby('category')['amount'].sum().nlargest(5).index.tolist()
    filtered_data = expense_data[expense_data['category'].isin(top_categories)]
    
    # Group by month and category
    monthly_category_totals = filtered_data.groupby(['month', 'category'])['amount'].sum().reset_index()
    
    # Create line chart
    fig = px.line(
        monthly_category_totals,
        x='month',
        y='amount',
        color='category',
        title="Monthly Spending Trends by Category",
        markers=True
    )
    
    # Customize x-axis to show all months
    fig.update_xaxes(
        categoryorder='array',
        categoryarray=months
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title="Category",
        height=400
    )
    
    return fig

def plot_goal_progress(goals_df):
    """Create a horizontal bar chart showing progress towards financial goals."""
    if goals_df.empty:
        return go.Figure().update_layout(
            title="No financial goals available",
            template="plotly_dark"
        )
    
    # Calculate progress percentage
    goals_df = goals_df.copy()
    goals_df['progress_pct'] = (goals_df['current'] / goals_df['target'] * 100).clip(0, 100)
    goals_df['remaining'] = 100 - goals_df['progress_pct']
    
    # Create figure
    fig = go.Figure()
    
    # Add progress bars
    fig.add_trace(go.Bar(
        y=goals_df['name'],
        x=goals_df['progress_pct'],
        name='Progress',
        orientation='h',
        marker=dict(color='rgba(58, 171, 115, 0.8)'),
        text=goals_df.apply(lambda x: f"${x['current']:,.0f} / ${x['target']:,.0f} ({x['progress_pct']:.1f}%)", axis=1),
        textposition='auto',
        hovertemplate='%{y}: %{text}<extra></extra>'
    ))
    
    # Add "remaining" bars (transparent/lighter color)
    fig.add_trace(go.Bar(
        y=goals_df['name'],
        x=goals_df['remaining'],
        name='Remaining',
        orientation='h',
        marker=dict(color='rgba(58, 171, 115, 0.2)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Customize the layout
    fig.update_layout(
        title="Progress Towards Financial Goals",
        barmode='stack',
        xaxis=dict(
            title="Progress (%)",
            range=[0, 100]
        ),
        yaxis=dict(
            title=""
        ),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        showlegend=False
    )
    
    return fig

def plot_daily_spending(transactions_df):
    """Create a grouped bar chart showing daily spending for the current week."""
    # Get dates for the current week (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    date_range = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Filter expenses for current week
    weekly_expenses = transactions_df[
        (transactions_df['type'] == 'expense') & 
        (transactions_df['date'].isin(date_range))
    ].copy()
    
    if weekly_expenses.empty:
        return go.Figure().update_layout(
            title="No expense data available for the current week",
            template="plotly_dark"
        )
    
    # Group by date and category
    daily_category_totals = weekly_expenses.groupby(['date', 'category'])['amount'].sum().reset_index()
    
    # Format the dates for display (e.g., "Mon 04")
    daily_category_totals['date'] = pd.to_datetime(daily_category_totals['date'])
    daily_category_totals['day_display'] = daily_category_totals['date'].dt.strftime('%a %d')
    
    # Create bar chart
    fig = px.bar(
        daily_category_totals,
        x='day_display',
        y='amount',
        color='category',
        title="Daily Spending This Week",
        barmode='stack'
    )
    
    # Ensure all days of the week are shown
    day_displays = [(start_date + timedelta(days=i)).strftime('%a %d') for i in range(7)]
    fig.update_xaxes(
        categoryorder='array',
        categoryarray=day_displays
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Amount ($)",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title="Category",
        height=400
    )
    
    return fig
