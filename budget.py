import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from db_manager import (
    load_budgets, 
    save_budgets, 
    add_budget, 
    update_budget, 
    delete_budget,
    get_category_spending,
    check_budget_status
)
from utils import format_currency
from ai_agents import get_budget_status

def budget_management_page(user_id: str):
    """
    Display the budget management page.
    
    Args:
        user_id (str): The ID of the current user
    """
    st.title("Budget Management")
    
    # Load user budgets
    budgets = load_budgets(user_id)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Budget Overview", "Create/Edit Budgets", "Smart Budget Analysis"])
    
    with tab1:
        display_budget_overview(user_id, budgets)
    
    with tab2:
        manage_budgets(user_id, budgets)
    
    with tab3:
        smart_budget_analysis(user_id, budgets)

def display_budget_overview(user_id: str, budgets: List[Dict[str, Any]]):
    """
    Display the budget overview tab.
    
    Args:
        user_id (str): The user ID
        budgets (List[Dict]): List of budget dictionaries
    """
    st.subheader("Budget Overview")
    
    if not budgets:
        st.info("You haven't set up any budgets yet. Go to the 'Create/Edit Budgets' tab to get started.")
        return
    
    # Get current month spending by category
    category_spending = get_category_spending(user_id)
    
    # Check budget status
    budget_status = check_budget_status(user_id)
    
    # Display budget warnings
    if budget_status:
        st.warning("Budget Alerts")
        for warning in budget_status:
            budget = warning["budget"]
            category = budget.get("category", "All Categories")
            limit = budget.get("amount", 0)
            spent = warning["spent"]
            remaining = warning["remaining"]
            percent_used = warning["percent_used"]
            
            # Create colored message based on severity
            if warning["severity"] == "high":
                message = f"⚠️ **{category}**: Budget exceeded! Spent {format_currency(spent)} of {format_currency(limit)} ({percent_used:.1f}%)"
                st.error(message)
            elif warning["severity"] == "medium":
                message = f"⚠️ **{category}**: Budget nearly exceeded. Spent {format_currency(spent)} of {format_currency(limit)} ({percent_used:.1f}%)"
                st.warning(message)
            else:
                message = f"⚠️ **{category}**: Budget approaching limit. Spent {format_currency(spent)} of {format_currency(limit)} ({percent_used:.1f}%)"
                st.info(message)
    
    # Display budget vs. actual spending chart
    st.subheader("Budget vs. Actual Spending")
    
    # Prepare data for chart
    budget_data = []
    
    for budget in budgets:
        category = budget.get("category", "Uncategorized")
        limit = budget.get("amount", 0)
        
        # Get actual spending for this category
        if category == "All Categories":
            spent = sum(category_spending.values())
        else:
            spent = category_spending.get(category, 0)
        
        # Calculate percentage and remaining amount
        percent_used = (spent / limit * 100) if limit > 0 else 0
        remaining = max(0, limit - spent)
        
        budget_data.append({
            "category": category,
            "budget": limit,
            "spent": spent,
            "remaining": remaining,
            "percent_used": percent_used
        })
    
    # Sort by percent used (descending)
    budget_data.sort(key=lambda x: x["percent_used"], reverse=True)
    
    # Create dataframe for chart
    df = pd.DataFrame(budget_data)
    
    if not df.empty:
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add budget bars (background)
        fig.add_trace(go.Bar(
            y=df["category"],
            x=df["budget"],
            name="Budget",
            orientation="h",
            marker=dict(color="rgba(200, 200, 200, 0.5)"),
            hovertemplate="Budget: %{x:$,.2f}<extra></extra>"
        ))
        
        # Add spent bars (foreground)
        fig.add_trace(go.Bar(
            y=df["category"],
            x=df["spent"],
            name="Spent",
            orientation="h",
            marker=dict(
                color=df["percent_used"].apply(
                    lambda x: "red" if x >= 100 else "orange" if x >= 90 else "rgba(63, 81, 181, 0.8)"
                )
            ),
            hovertemplate="Spent: %{x:$,.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Budget vs. Actual Spending",
            barmode="overlay",
            xaxis=dict(title="Amount ($)"),
            yaxis=dict(title="Category", autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, l=20, r=20, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display budget details in a table
        st.subheader("Budget Details")
        
        # Format dataframe for display
        display_df = df.copy()
        display_df["budget"] = display_df["budget"].apply(format_currency)
        display_df["spent"] = display_df["spent"].apply(format_currency)
        display_df["remaining"] = display_df["remaining"].apply(format_currency)
        display_df["percent_used"] = display_df["percent_used"].apply(lambda x: f"{x:.1f}%")
        
        # Rename columns for display
        display_df.columns = ["Category", "Budget", "Spent", "Remaining", "% Used"]
        
        # Show table
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No budget data available to display.")

def manage_budgets(user_id: str, budgets: List[Dict[str, Any]]):
    """
    Display the budget management tab.
    
    Args:
        user_id (str): The user ID
        budgets (List[Dict]): List of budget dictionaries
    """
    st.subheader("Create New Budget")
    
    # Get available categories
    from utils.utils_core import budget_categories
    available_categories = ["All Categories"] + budget_categories + list(get_category_spending(user_id).keys())
    available_categories = list(dict.fromkeys(available_categories))  # Remove duplicates
    
    # Create new budget form
    with st.form("budget_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Category", available_categories)
            amount = st.number_input("Budget Amount", min_value=0.01, step=10.0, value=100.0)
        
        with col2:
            time_period = st.selectbox("Time Period", ["Monthly", "Weekly", "Yearly"])
            notes = st.text_area("Notes (Optional)")
        
        submit = st.form_submit_button("Add Budget")
        
        if submit:
            # Check if budget for this category already exists
            category_exists = any(b.get("category") == category for b in budgets)
            
            if category_exists:
                st.error(f"A budget for {category} already exists. Please edit the existing budget instead.")
            else:
                # Create new budget
                new_budget = {
                    "category": category,
                    "amount": float(amount),
                    "period": time_period.lower(),
                    "notes": notes,
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                success = add_budget(user_id, new_budget)
                
                if success:
                    st.success(f"Budget for {category} added successfully!")
                    # Refresh budgets
                    st.rerun()
                else:
                    st.error("Failed to add budget. Please try again.")
    
    # Display existing budgets for editing/deletion
    st.subheader("Existing Budgets")
    
    if not budgets:
        st.info("You haven't set up any budgets yet.")
    else:
        for i, budget in enumerate(budgets):
            with st.expander(f"{budget.get('category', 'Uncategorized')} - {format_currency(budget.get('amount', 0))} ({budget.get('period', 'monthly').title()})"):
                with st.form(f"edit_budget_{i}"):
                    amount = st.number_input(
                        "Budget Amount", 
                        min_value=0.01, 
                        step=10.0, 
                        value=float(budget.get("amount", 0))
                    )
                    
                    time_period = st.selectbox(
                        "Time Period", 
                        ["Monthly", "Weekly", "Yearly"],
                        index=0 if budget.get("period") == "monthly" else 
                              1 if budget.get("period") == "weekly" else 2
                    )
                    
                    notes = st.text_area(
                        "Notes (Optional)",
                        value=budget.get("notes", "")
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        update = st.form_submit_button("Update Budget")
                    
                    with col2:
                        delete = st.form_submit_button("Delete Budget")
                    
                    if update:
                        # Update budget
                        updated_budget = budget.copy()
                        updated_budget.update({
                            "amount": float(amount),
                            "period": time_period.lower(),
                            "notes": notes,
                            "updated_at": datetime.datetime.now().isoformat()
                        })
                        
                        success = update_budget(user_id, i, updated_budget)
                        
                        if success:
                            st.success("Budget updated successfully!")
                            # Refresh budgets
                            st.rerun()
                        else:
                            st.error("Failed to update budget. Please try again.")
                    
                    if delete:
                        # Delete budget
                        success = delete_budget(user_id, i)
                        
                        if success:
                            st.success("Budget deleted successfully!")
                            # Refresh budgets
                            st.rerun()
                        else:
                            st.error("Failed to delete budget. Please try again.")

def smart_budget_analysis(user_id: str, budgets: List[Dict[str, Any]]):
    """
    Display the smart budget analysis tab.
    
    Args:
        user_id (str): The user ID
        budgets (List[Dict]): List of budget dictionaries
    """
    st.subheader("Smart Budget Analysis")
    
    # Get spending data
    category_spending = get_category_spending(user_id)
    
    if not category_spending:
        st.info("Add some transactions to get AI-powered budget analysis.")
        return
    
    # Display spending breakdown chart
    fig = px.pie(
        names=list(category_spending.keys()),
        values=list(category_spending.values()),
        title="Current Spending Breakdown",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=50, l=20, r=20, b=20))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Budget recommendations
    st.subheader("AI Budget Recommendations")
    
    with st.spinner("Generating smart budget recommendations..."):
        try:
            # Create a transactions list to pass to the AI
            from db_manager import load_transactions
            transactions = load_transactions(user_id)
            
            if transactions:
                # Get AI-powered budget recommendations
                from ai_agents import generate_response
                
                budget_prompt = f"Based on these transactions, provide budget insights and recommendations: {str(transactions)[:1000]}"
                budget_insights = generate_response(budget_prompt, "You are a financial advisor providing budget insights and recommendations.")
                st.info(budget_insights)
                
                # Show budget optimization options
                st.subheader("Budget Optimization")
                
                # Create suggested budgets based on spending patterns
                suggested_budgets = {}
                total_spending = sum(category_spending.values())
                
                for category, amount in category_spending.items():
                    # Suggest a budget slightly higher than current spending
                    percent_of_total = amount / total_spending if total_spending > 0 else 0
                    suggested_budgets[category] = amount * 1.1  # 10% buffer
                
                # Display suggested budgets
                st.write("Suggested Budget Allocation:")
                
                col1, col2 = st.columns(2)
                
                for i, (category, amount) in enumerate(suggested_budgets.items()):
                    with col1 if i % 2 == 0 else col2:
                        st.metric(
                            category, 
                            format_currency(amount),
                            f"{amount/total_spending*100:.1f}% of total" if total_spending > 0 else "0% of total"
                        )
                
                # One-click budget creation
                if st.button("Create Suggested Budgets"):
                    # Check which budgets already exist
                    existing_categories = [b.get("category") for b in budgets]
                    budgets_created = 0
                    
                    for category, amount in suggested_budgets.items():
                        if category not in existing_categories:
                            # Create new budget
                            new_budget = {
                                "category": category,
                                "amount": float(amount),
                                "period": "monthly",
                                "notes": "Auto-created from AI suggestion",
                                "created_at": datetime.datetime.now().isoformat()
                            }
                            
                            success = add_budget(user_id, new_budget)
                            if success:
                                budgets_created += 1
                    
                    if budgets_created > 0:
                        st.success(f"Created {budgets_created} new budgets successfully!")
                        # Refresh budgets
                        st.rerun()
                    else:
                        st.info("No new budgets created. All suggested categories already have budgets.")
            else:
                st.warning("Add transactions to get AI-powered budget recommendations.")
        except Exception as e:
            st.error(f"Error generating budget recommendations: {str(e)}")
            st.info("Please try again later or add more transaction data for better recommendations.")