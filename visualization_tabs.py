import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Callable
import datetime

from utils import format_currency
from db_manager import (
    get_category_spending,
    get_daily_spending,
    get_monthly_spending
)
import db_service as db

def create_tabbed_charts(user_id: str, time_period: str = "month"):
    """
    Create a tabbed interface for visualizing financial data with different chart types.
    
    Args:
        user_id (str): The user ID
        time_period (str): Time period for filtering data ("week", "month", "year", "all")
    """
    # Define the tabs
    tab1, tab2, tab3 = st.tabs(["Income vs Expense", "Spending by Category", "Daily Spending"])
    
    # Get today's date
    today = datetime.datetime.now()
    
    # Define date ranges based on time period
    if time_period == "week":
        start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        period_name = "Last 7 Days"
    elif time_period == "month":
        start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        period_name = "Last 30 Days"
    elif time_period == "year":
        start_date = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        period_name = "Last Year"
    else:  # all
        start_date = None
        period_name = "All Time"
        
    end_date = today.strftime("%Y-%m-%d")
    
    # Income vs Expense tab
    with tab1:
        st.subheader(f"Income vs Expense ({period_name})")
        
        # Get transactions
        transactions = db.get_transactions(user_id)
        txn_dicts = [t.to_dict() for t in transactions]
        
        # Filter by date if needed
        if start_date:
            txn_dicts = [t for t in txn_dicts if t.get('date', '') >= start_date]
        
        # Group by month and type
        monthly_data = {}
        for txn in txn_dicts:
            date = txn.get('date', '')
            if not date:
                continue
                
            # Get month and year
            month_year = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%b %Y")
            
            # Initialize if not exists
            if month_year not in monthly_data:
                monthly_data[month_year] = {"income": 0, "expense": 0}
            
            # Add to appropriate category
            txn_type = txn.get('type', 'expense').lower()
            amount = txn.get('amount', 0)
            
            if txn_type == 'income':
                monthly_data[month_year]['income'] += amount
            else:
                monthly_data[month_year]['expense'] += amount
        
        # Convert to dataframe for plotting
        plot_data = []
        for month, values in monthly_data.items():
            plot_data.append({
                "Month": month,
                "Income": values["income"],
                "Expense": values["expense"],
                "Net": values["income"] - values["expense"]
            })
        
        # Sort by date
        if plot_data:
            try:
                # Get a list of month names sorted by date
                months = sorted(list(monthly_data.keys()), 
                              key=lambda x: datetime.datetime.strptime(x, "%b %Y"))
                
                # Sort plot data by this order
                plot_data = sorted(plot_data, key=lambda x: months.index(x["Month"]))
            except:
                # Fallback if there's an issue with sorting
                pass
        
        # Create plot
        if plot_data:
            df = pd.DataFrame(plot_data)
            
            # Create a grouped bar chart
            fig = go.Figure()
            
            # Add income bars
            fig.add_trace(go.Bar(
                x=df["Month"],
                y=df["Income"],
                name="Income",
                marker_color='green'
            ))
            
            # Add expense bars
            fig.add_trace(go.Bar(
                x=df["Month"],
                y=df["Expense"],
                name="Expense",
                marker_color='red'
            ))
            
            # Add net line
            fig.add_trace(go.Scatter(
                x=df["Month"],
                y=df["Net"],
                name="Net",
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Income vs. Expense by Month ({period_name})",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                barmode='group',
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add hover templates with formatted currency
            fig.update_traces(
                hovertemplate="%{y:$,.2f}",
                selector=dict(type="bar")
            )
            
            fig.update_traces(
                hovertemplate="%{y:$,.2f}",
                selector=dict(type="scatter")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            total_income = sum(d["Income"] for d in plot_data)
            total_expense = sum(d["Expense"] for d in plot_data)
            total_net = total_income - total_expense
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Income", format_currency(total_income))
            with col2:
                st.metric("Total Expenses", format_currency(total_expense))
            with col3:
                st.metric("Net Savings", format_currency(total_net), 
                         delta=f"{(total_net/total_income*100) if total_income > 0 else 0:.1f}%")
        else:
            st.info("No transaction data available for this time period.")
    
    # Spending by Category tab
    with tab2:
        st.subheader(f"Spending by Category ({period_name})")
        
        # Get category spending
        category_spending = get_category_spending(user_id)
        
        if category_spending:
            # Convert dictionary to dataframe
            df = pd.DataFrame({
                "Category": list(category_spending.keys()),
                "Amount": list(category_spending.values())
            })
            
            # Sort by amount (descending)
            df = df.sort_values("Amount", ascending=False)
            
            # Create pie chart
            fig = px.pie(
                df,
                values="Amount",
                names="Category",
                title=f"Spending by Category ({period_name})",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:$,.2f} (%{percent})"
            )
            
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display top categories in a table
            st.subheader("Top Spending Categories")
            
            # Format for display
            df["Amount"] = df["Amount"].apply(lambda x: format_currency(x))
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No spending data available for this time period.")
    
    # Daily Spending tab
    with tab3:
        st.subheader(f"Daily Spending ({period_name})")
        
        # Get daily spending
        daily_spending = get_daily_spending(user_id)
        
        if daily_spending:
            # Convert dictionary to dataframe
            df = pd.DataFrame({
                "Date": list(daily_spending.keys()),
                "Amount": list(daily_spending.values())
            })
            
            # Parse dates
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Sort by date
            df = df.sort_values("Date")
            
            # Format date for display
            df["Date"] = df["Date"].dt.strftime("%b %d, %Y")
            
            # Create line chart
            fig = px.line(
                df,
                x="Date",
                y="Amount",
                title=f"Daily Spending ({period_name})",
                markers=True
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode="x unified"
            )
            
            # Add hover template with formatted currency
            fig.update_traces(
                hovertemplate="%{y:$,.2f}"
            )
            
            # Adding a moving average trendline (7-day)
            if len(df) > 7:
                df_with_ma = df.copy()
                df_with_ma["Moving_Avg"] = df["Amount"].rolling(window=7).mean()
                
                fig.add_scatter(
                    x=df_with_ma["Date"],
                    y=df_with_ma["Moving_Avg"],
                    mode="lines",
                    name="7-Day Average",
                    line=dict(color="red", width=1, dash="dash")
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            total_spending = sum(daily_spending.values())
            avg_daily = total_spending / len(daily_spending) if daily_spending else 0
            max_day = max(daily_spending.values()) if daily_spending else 0
            max_day_date = max(daily_spending.items(), key=lambda x: x[1])[0] if daily_spending else "N/A"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Spending", format_currency(total_spending))
            with col2:
                st.metric("Average Daily", format_currency(avg_daily))
            with col3:
                st.metric(f"Highest Day ({max_day_date})", format_currency(max_day))
        else:
            st.info("No daily spending data available for this time period.")