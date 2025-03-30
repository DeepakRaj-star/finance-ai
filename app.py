import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import os
import asyncio
from pathlib import Path

# Import custom modules
import vector_processing
from auth_db import auth_page, is_authenticated, get_current_user, require_auth
import ai_agents
import db_service as db
import vector_processing as vp
import finance_agent
from finance_agent import FinanceAgent
from fetch_agents import start_agents, stop_agents, analyze_transaction, get_financial_insights, answer_query
from budget import budget_management_page
from visualization import (
    create_balance_chart, 
    create_spending_by_category_chart, 
    create_spending_trend_chart,
    create_goal_progress_chart
)
# Import from utils
from utils import (
    get_recommendations, 
    format_currency, 
    get_date_range,
    create_spending_by_category_chart,
    create_spending_trend_chart,
    create_balance_chart
)

# Import custom styling
from utils.custom_style import (
    init_retro_ui,
    inject_custom_css,
    inject_custom_js,
    render_card,
    badge,
    neon_text,
    retro_metric,
    animated_progress,
    retro_table
)
import quantitative_finance
from finance_quiz import finance_quiz_page
from visualization_tabs import create_tabbed_charts
from credit_management import credit_management_page
from financial_coaching import financial_coaching_page
from crypto_finance import crypto_finance_page
from finance_rag_ui import finance_rag_page

# Check for required API keys with retry mechanism
def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found. Please add it in the Secrets tab.")
        st.stop()
    return api_key

OPENAI_API_KEY = get_api_key()

# Initialize services
try:
    # Initialize database
    db.init_db()
    
    # Start vector processing if API keys are available
    if OPENAI_API_KEY:
        vp.start_vector_engine()
        # Start Fetch AI agents
        start_agents()
    else:
        print("Skipping AI service initialization due to missing API keys")
except Exception as e:
    print(f"Failed to initialize services: {str(e)}")

# Create finance agent singleton
finance_agent = FinanceAgent()

# App configuration
st.set_page_config(
    page_title="AI Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply retro UI styling
inject_custom_css()
inject_custom_js()

# Initialize session state for data paths
if "user_data_path" not in st.session_state:
    st.session_state.user_data_path = Path("user_data")
    st.session_state.user_data_path.mkdir(exist_ok=True)

# Create a function to display a page
def show_page(page_name, user=None):
    if page_name == "Dashboard":
        show_dashboard_page(user)
    elif page_name == "Transactions":
        show_transactions_page(user)
    elif page_name == "Goals":
        show_goals_page(user)
    elif page_name == "AI Assistant":
        show_ai_assistant_page(user)
    elif page_name == "Budgets":
        budget_management_page(user["id"])
    elif page_name == "Quantitative Finance":
        quantitative_finance.run()
    elif page_name == "Finance Quiz":
        finance_quiz_page(user["id"])
    elif page_name == "Credit Management":
        credit_management_page(user["id"])
    elif page_name == "Financial Coaching":
        financial_coaching_page(user["id"])
    elif page_name == "Crypto Finance":
        crypto_finance_page(user["id"])
    elif page_name == "Financial RAG":
        finance_rag_page()
    elif page_name == "Settings":
        show_settings_page(user)
    else:
        st.error("Page not found")

# Dashboard page
def show_dashboard_page(user):
    st.title("Financial Dashboard")
    
    # Set user in the finance agent
    user_id = user["id"]
    finance_agent.set_user(user_id)
    
    # Get user's transactions
    transactions = db.get_transactions(user_id)
    txn_dicts = [t.to_dict() for t in transactions]
    
    # Current balance and summary metrics
    balance = db.calculate_balance(user_id)
    summary = db.get_transaction_summary(user_id)
    financial_health = finance_agent.get_financial_health_score()
    
    # Check for budget status
    budget_status = finance_agent.get_budget_status()
    budget_warnings = [b for b in budget_status if b.get("warning_level") in ["warning", "critical"]]
    
    # Display budget warnings at the top if any exist
    if budget_warnings:
        with st.expander("⚠️ Budget Alerts", expanded=True):
            for warning in budget_warnings:
                budget = warning.get("budget", {})
                category = budget.get("category", "All Categories")
                limit = budget.get("amount", 0)
                spent = warning.get("spent", 0)
                remaining = warning.get("remaining", 0)
                percent_used = warning.get("percent_used", 0)
                warning_level = warning.get("warning_level")
                
                if warning_level == "critical":
                    st.error(f"**{category}**: Budget exceeded! Spent {format_currency(spent)} of {format_currency(limit)} ({percent_used:.1f}%)")
                elif warning_level == "warning":
                    st.warning(f"**{category}**: Budget nearly exceeded. Spent {format_currency(spent)} of {format_currency(limit)} ({percent_used:.1f}%)")
    
    # Top metrics row
    st.subheader("Financial Overview")
    
    # Get advanced financial data
    anomalies = finance_agent.get_spending_anomalies()
    predictions = finance_agent.predict_transactions()
    
    # Define metrics data
    primary_metrics = [
        {"label": "Current Balance", "value": format_currency(balance), "delta": None},
        {"label": "Monthly Income", "value": format_currency(summary.get("monthly_income", 0)), "delta": None},
        {"label": "Monthly Expenses", "value": format_currency(summary.get("monthly_expenses", 0)), "delta": None},
        {"label": "Savings Rate", "value": f"{summary.get('savings_rate', 0) * 100:.1f}%", "delta": None}
    ]
    
    # Render primary metrics in a grid with retro styling
    col1, col2, col3, col4 = st.columns(4)
    
    # Display metrics in retro style
    with col1:
        retro_metric(
            primary_metrics[0]["label"],
            primary_metrics[0]["value"],
            primary_metrics[0]["delta"]
        )
    with col2:
        retro_metric(
            primary_metrics[1]["label"],
            primary_metrics[1]["value"],
            primary_metrics[1]["delta"]
        )
    with col3:
        retro_metric(
            primary_metrics[2]["label"],
            primary_metrics[2]["value"],
            primary_metrics[2]["delta"]
        )
    with col4:
        retro_metric(
            primary_metrics[3]["label"],
            primary_metrics[3]["value"],
            primary_metrics[3]["delta"]
        )
    
    # Add some space
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Advanced metrics with more context
    advanced_metrics = [
        {
            "label": "Financial Health Score", 
            "value": f"{financial_health.get('score', 0)}/100", 
            "delta": financial_health.get('change', 0),
            "delta_color": "normal" if financial_health.get('change', 0) >= 0 else "inverse"
        },
        {
            "label": "Top Spending Category", 
            "value": summary.get("top_spending_category", {"category": "None"}).get('category', 'None'),
            "delta": None
        },
        {
            "label": "Upcoming Expenses", 
            "value": f"{len(predictions)}",
            "delta": None
        },
        {
            "label": "Spending Anomalies", 
            "value": f"{len(anomalies)}",
            "delta": None
        }
    ]
    
    # Render advanced metrics in a grid with retro styling
    col1, col2, col3, col4 = st.columns(4)
    
    # Display metrics in retro style
    with col1:
        retro_metric(
            advanced_metrics[0]["label"],
            advanced_metrics[0]["value"],
            advanced_metrics[0]["delta"],
            delta_color=advanced_metrics[0].get("delta_color", "normal")
        )
    with col2:
        retro_metric(
            advanced_metrics[1]["label"],
            advanced_metrics[1]["value"],
            advanced_metrics[1]["delta"]
        )
    with col3:
        retro_metric(
            advanced_metrics[2]["label"],
            advanced_metrics[2]["value"],
            advanced_metrics[2]["delta"]
        )
    with col4:
        retro_metric(
            advanced_metrics[3]["label"],
            advanced_metrics[3]["value"],
            advanced_metrics[3]["delta"]
        )
    
    # AI-powered greeting card with enhanced styling
    ai_greeting = finance_agent.generate_assistant_message()
    
    # Use our custom card renderer
    render_card(
        title="🤖 AI Assistant Insights", 
        content=f"""
        <div style="padding: 10px; line-height: 1.5;">
            {ai_greeting}
        </div>
        """
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending by category
        get_date_range("month")  # Not used now
        category_data = db.get_category_spending(user_id)
        fig = create_spending_by_category_chart(category_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomalies if any exist with custom styling
        if anomalies:
            st.subheader("⚠️ Unusual Spending Detected")
            # Limit to top 3 anomalies
            for anomaly in anomalies[:3]:
                txn = anomaly.get("transaction", {})
                pct_above = anomaly.get("percent_above_average", 0)
                
                # Use custom card with badge
                content = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: bold; font-size: 1.1rem;">{txn.get('description', 'Unknown')}</span>
                    <span style="font-weight: bold; color: #e74c3c;">{format_currency(txn.get('amount', 0))}</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span style="margin-right: 8px;">{txn.get('category', 'expense')}</span>
                    {badge(f"{pct_above:.1f}% above normal", "danger")}
                </div>
                <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 8px;">
                    This amount is significantly higher than your usual spending in this category.
                </div>
                """
                
                render_card(
                    title=None,
                    content=content,
                    icon="warning"
                )
    
    with col2:
        # Balance trend
        balance_fig = create_balance_chart(txn_dicts)
        st.plotly_chart(balance_fig, use_container_width=True)
        
        # Predicted expenses
        try:
            if predictions:
                st.subheader("Predicted Upcoming Transactions")
                
                # Convert predictions to DataFrame
                pred_data = []
                for pred in predictions[:5]:  # Limit to top 5
                    pred_data.append({
                        "Date": pred.get("predicted_date", "Unknown"),
                        "Description": pred.get("description", "Unknown"),
                        "Amount": pred.get("amount", 0),
                        "Confidence": f"{int(pred.get('confidence', 0) * 100)}%"
                    })
                
                if pred_data:
                    pred_df = pd.DataFrame(pred_data)
                    st.dataframe(pred_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to display predictions: {str(e)}")
    
    # Tabbed Financial Charts
    st.subheader("Financial Charts")
    # Add time period selector
    time_period = st.selectbox(
        "Time Period",
        ["Last 30 Days", "Last 7 Days", "Last 90 Days", "This Year", "All Time"],
        index=0
    )
    
    # Convert selection to parameter
    if time_period == "Last 7 Days":
        period_param = "week"
    elif time_period == "Last 30 Days":
        period_param = "month"
    elif time_period == "Last 90 Days":
        period_param = "quarter"
    elif time_period == "This Year":
        period_param = "year"
    else:  # All Time
        period_param = "all"
        
    # Show tabbed charts
    create_tabbed_charts(user_id, period_param)
    
    # Financial summary from AI
    st.subheader("AI Financial Analysis")
    if txn_dicts:
        with st.spinner("Generating financial analysis..."):
            try:
                financial_summary = finance_agent.get_financial_summary()
                
                # Display summary text with enhanced card
                render_card(
                    title="💡 Financial Insights", 
                    content=f"""
                    <div style="padding: 10px; line-height: 1.6;">
                        {financial_summary.get('summary_text', 'No summary available.')}
                    </div>
                    """
                )
                
                # Display chart for top expense categories
                top_categories = financial_summary.get('top_expense_categories', [])
                if top_categories:
                    st.subheader("Top Expense Categories")
                    top_cat_df = pd.DataFrame(top_categories)
                    
                    fig = px.pie(
                        top_cat_df,
                        values='amount',
                        names='category',
                        title="Top Expense Categories",
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to generate AI financial analysis: {str(e)}")
    else:
        st.info("Add transactions to get AI-powered financial insights.")
    
    # Personalized recommendations with enhanced styling
    st.subheader("Personalized Recommendations")
    recommendations = get_recommendations(txn_dicts)
    
    if recommendations:
        # Create a card with all recommendations
        rec_content = ""
        for i, rec in enumerate(recommendations):
            rec_type = "success" if i % 3 == 0 else ("warning" if i % 3 == 1 else "info")
            rec_content += f'<div style="margin-bottom: 12px;">{badge("Tip", rec_type)} {rec}</div>'
        
        # Display in a card
        render_card(
            title="🔍 Smart Finance Tips",
            content=rec_content
        )
    else:
        st.info("Add more transaction data to receive personalized recommendations.")

# Transactions page
def show_transactions_page(user):
    st.title("Transaction Management")
    
    # Set user in finance agent
    user_id = user["id"]
    finance_agent.set_user(user_id)
    
    # Get user's transactions from database
    transactions = db.get_transactions(user_id)
    txn_dicts = [t.to_dict() for t in transactions]
    
    # Add transaction section
    st.subheader("Add New Transaction")
    with st.form(key="transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            transaction_date = st.date_input("Date", datetime.datetime.now())
            description = st.text_input("Description")
            amount = st.number_input("Amount", min_value=0.0, step=0.01)
        with col2:
            transaction_type = st.selectbox("Type", ["Expense", "Income"])
            category_options = [
                "Food", "Shopping", "Housing", "Transportation", 
                "Entertainment", "Healthcare", "Education", 
                "Travel", "Utilities", "Income", "Investments", "Other"
            ]
            # Option to use AI categorization
            use_ai_category = st.checkbox("Use AI Categorization", value=True)
            
            if use_ai_category:
                st.info("Category will be determined automatically by AI")
                category = None  # Will be determined by AI
            else:
                category = st.selectbox("Category", category_options)
                
            notes = st.text_area("Notes (Optional)")
        
        submit = st.form_submit_button(label="Add Transaction")
        
        if submit:
            if description and amount > 0:
                try:
                    # Determine category with AI if needed
                    if use_ai_category and description:
                        category = finance_agent.categorize_transaction(description)
                    
                    # Create transaction data
                    transaction_data = {
                        "date": transaction_date.strftime("%Y-%m-%d"),
                        "description": description,
                        "amount": float(amount),
                        "type": transaction_type.lower(),
                        "category": category,
                        "notes": notes
                    }
                    
                    # Add to database
                    new_transaction = db.create_transaction(user_id, transaction_data)
                    
                    if new_transaction:
                        # Process with vector engine
                        vp.process_new_transaction(new_transaction.to_dict())
                        
                        # Get insights about the transaction
                        asyncio.run(finance_agent.analyze_transaction(new_transaction.id))
                        
                        st.success("Transaction added successfully!")
                        # Refresh transactions
                        st.rerun()
                    else:
                        st.error("Failed to add transaction. Please try again.")
                except Exception as e:
                    st.error(f"Error adding transaction: {str(e)}")
            else:
                st.error("Please fill in all required fields.")
    
    # Transaction list
    st.subheader("Transaction History")
    
    # View options
    view_tabs = st.tabs(["Table View", "Analytics View", "Search"])
    
    with view_tabs[0]:  # Table View
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Type", 
                                      ["All", "Income", "Expense"])
        with col2:
            if txn_dicts:
                categories = ["All"] + list(set([t.get("category", "Other") 
                              for t in txn_dicts]))
                filter_category = st.selectbox("Filter by Category", categories)
            else:
                filter_category = "All"
                categories = []
        with col3:
            search_term = st.text_input("Search", key="table_search")
        
        # Apply filters
        filtered_transactions = txn_dicts.copy()
        
        if filter_type != "All":
            filtered_transactions = [t for t in filtered_transactions 
                                    if t.get("type", "").lower() == filter_type.lower()]
        
        if filter_category != "All":
            filtered_transactions = [t for t in filtered_transactions 
                                    if t.get("category", "Other") == filter_category]
        
        if search_term:
            filtered_transactions = [t for t in filtered_transactions 
                                    if search_term.lower() in t.get("description", "").lower()]
        
        # Sort by date (newest first)
        filtered_transactions.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Display transactions
        if filtered_transactions:
            # Convert to dataframe
            df = pd.DataFrame(filtered_transactions)
            
            # Format dataframe
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%b %d, %Y')
            df['amount_display'] = df.apply(
                lambda x: format_currency(x['amount']), axis=1
            )
            
            # Conditional styling
            def color_type(val):
                if val == 'income':
                    return 'color: green'
                else:
                    return 'color: red'
            
            # Select columns for display
            styled_df = df[['date', 'description', 'category', 'type', 'amount_display']]
            styled_df = styled_df.rename(columns={
                'date': 'Date',
                'description': 'Description',
                'category': 'Category',
                'type': 'Type',
                'amount_display': 'Amount'
            })
            
            # Display the table
            st.dataframe(
                styled_df.style.applymap(color_type, subset=['Type']),
                use_container_width=True,
                height=400
            )
            
            # Export transactions
            if st.button("Export Transactions"):
                # Convert to CSV
                csv = df.to_csv(index=False)
                
                # Offer download
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="transactions_export.csv",
                    mime="text/csv"
                )
        else:
            st.info("No transactions found with the selected filters.")
    
    with view_tabs[1]:  # Analytics View
        if txn_dicts:
            # Time period selector
            time_periods = ["Last 7 days", "Last 30 days", "Last 90 days", "This year", "All time"]
            selected_period = st.selectbox("Time period", time_periods)
            
            # Convert selection to date range
            today = datetime.datetime.now()
            if selected_period == "Last 7 days":
                start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            elif selected_period == "Last 30 days":
                start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            elif selected_period == "Last 90 days":
                start_date = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
            elif selected_period == "This year":
                start_date = f"{today.year}-01-01"
            else:  # All time
                start_date = None
                
            end_date = today.strftime("%Y-%m-%d")
            
            # Filter transactions by date
            if start_date:
                filtered_txns = [t for t in txn_dicts if t["date"] >= start_date and t["date"] <= end_date]
            else:
                filtered_txns = txn_dicts
                
            if filtered_txns:
                # Create analytics tabs
                analytics_tabs = st.tabs(["Spending by Category", "Income vs Expense", "Daily Spending"])
                
                with analytics_tabs[0]:  # Spending by Category
                    # Get spending by category
                    category_data = db.get_category_spending(user_id)
                    fig = create_spending_by_category_chart(category_data)
                    st.plotly_chart(fig, use_container_width=True, key="spending_category_chart")
                
                with analytics_tabs[1]:  # Income vs Expense
                    # Calculate income and expense totals
                    income_total = sum(t["amount"] for t in filtered_txns if t["type"] == "income")
                    expense_total = sum(t["amount"] for t in filtered_txns if t["type"] == "expense")
                    
                    # Create bar chart
                    comparison_df = pd.DataFrame({
                        "Type": ["Income", "Expenses"],
                        "Amount": [income_total, expense_total]
                    })
                    
                    fig = px.bar(
                        comparison_df,
                        x="Type",
                        y="Amount",
                        color="Type",
                        text="Amount",
                        color_discrete_map={"Income": "#76D7C4", "Expenses": "#F1948A"}
                    )
                    
                    # Format the text labels to show currency
                    fig.update_traces(texttemplate='%{text:$,.2f}', textposition='outside')
                    
                    fig.update_layout(
                        title="Income vs Expenses",
                        xaxis_title="",
                        yaxis_title="Amount",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show net result
                    net = income_total - expense_total
                    if net >= 0:
                        st.success(f"Net positive cash flow: {format_currency(net)}")
                    else:
                        st.warning(f"Net negative cash flow: {format_currency(net)}")
                
                with analytics_tabs[2]:  # Daily Spending
                    # Get daily spending
                    daily_spending = db.get_daily_spending(user_id)
                    spending_fig = create_spending_trend_chart(daily_spending)
                    st.plotly_chart(spending_fig, use_container_width=True, key="daily_spending_chart")
            else:
                st.info(f"No transactions found in the selected time period.")
        else:
            st.info("Add some transactions to see analytics.")
    
    with view_tabs[2]:  # Search
        st.subheader("Vector-Based Transaction Search")
        st.write("Search your transactions using natural language. The system will find semantically similar transactions.")
        
        search_query = st.text_input("Describe what you're looking for:", key="vector_search_box")
        search_button = st.button("Search", key="vector_search_button")
        
        if search_button and search_query:
            with st.spinner("Searching..."):
                try:
                    # Use vector search through finance agent
                    search_results = finance_agent.search_transactions(search_query)
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} matching transactions")
                        
                        # Convert to DataFrame for display
                        result_df = pd.DataFrame(search_results)
                        
                        # Format for display
                        result_df['date'] = pd.to_datetime(result_df['date']) 
                        result_df['date'] = result_df['date'].dt.strftime('%b %d, %Y')
                        result_df['amount_display'] = result_df.apply(
                            lambda x: format_currency(x['amount']), axis=1
                        )
                        
                        display_cols = ['date', 'description', 'category', 'type', 'amount_display']
                        display_df = result_df[display_cols].rename(columns={
                            'date': 'Date',
                            'description': 'Description',
                            'category': 'Category',
                            'type': 'Type',
                            'amount_display': 'Amount'
                        })
                        
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No matching transactions found. Try a different search query.")
                except Exception as e:
                    st.warning(f"Search error: {str(e)}")
        elif search_button:
            st.warning("Please enter a search query.")

# Goals page
def show_goals_page(user):
    st.title("Financial Goals")
    
    # Set user in finance agent
    user_id = user["id"]
    finance_agent.set_user(user_id)
    
    # Get user's data from database
    goals = db.get_goals(user_id)
    transactions = db.get_transactions(user_id)
    
    # Convert to dictionaries for processing
    goal_dicts = [g.to_dict() for g in goals]
    txn_dicts = [t.to_dict() for t in transactions]
    
    # Goal overview
    if goal_dicts:
        st.subheader("Goal Overview")
        
        # Calculate overall goal progress
        total_target = sum(g.get("target_amount", 0) for g in goal_dicts)
        total_current = sum(g.get("current_amount", 0) for g in goal_dicts)
        overall_progress = total_current / total_target if total_target > 0 else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Goals", len(goal_dicts))
        with col2:
            st.metric("Target Amount", format_currency(total_target))
        with col3:
            st.metric("Overall Progress", f"{overall_progress * 100:.1f}%")
    
    # Generate an AI savings plan
    st.subheader("AI Savings Plan Generator")
    with st.form(key="savings_plan_form"):
        col1, col2 = st.columns(2)
        with col1:
            goal_amount = st.number_input("Goal Amount", min_value=1.0, step=100.0, value=1000.0)
        with col2:
            timeframe = st.number_input("Timeframe (months)", min_value=1, max_value=60, value=6)
        
        submit = st.form_submit_button(label="Generate Savings Plan")
        
        if submit:
            if txn_dicts:
                with st.spinner("Generating your personalized savings plan..."):
                    try:
                        # Use AI to analyze transactions and generate a plan
                        financial_summary = finance_agent.get_financial_summary()
                        
                        # Calculate monthly income and expenses
                        monthly_income = financial_summary.get("total_income", 0) / 3  # Assume 3 months of data
                        monthly_expenses = financial_summary.get("total_expenses", 0) / 3
                        
                        # Calculate potential monthly savings
                        potential_savings = max(0, monthly_income - monthly_expenses)
                        
                        # Calculate months needed
                        months_needed = goal_amount / potential_savings if potential_savings > 0 else float('inf')
                        
                        # Generate plan text
                        if months_needed <= timeframe:
                            plan = (
                                f"### Your Savings Plan\n\n"
                                f"Based on your financial history, you could save approximately "
                                f"{format_currency(potential_savings)} per month. At this rate, you can reach your "
                                f"goal of {format_currency(goal_amount)} in about {months_needed:.1f} months, "
                                f"which is within your desired timeframe of {timeframe} months.\n\n"
                                f"**Recommended actions:**\n"
                                f"- Set up an automatic transfer of {format_currency(potential_savings)} to a savings account each month\n"
                                f"- Track your progress regularly\n"
                                f"- Consider cutting expenses in the following categories: "
                                f"{', '.join([cat['category'] for cat in financial_summary.get('top_expense_categories', [])[:2]])}"
                            )
                        else:
                            # Calculate required savings
                            required_savings = goal_amount / timeframe
                            expense_reduction = required_savings - potential_savings
                            
                            plan = (
                                f"### Your Savings Plan\n\n"
                                f"Based on your financial history, you currently save about "
                                f"{format_currency(potential_savings)} per month. To reach your "
                                f"goal of {format_currency(goal_amount)} within {timeframe} months, "
                                f"you'll need to save {format_currency(required_savings)} per month. "
                                f"This means you need to find an additional {format_currency(expense_reduction)} "
                                f"in savings each month.\n\n"
                                f"**Recommended actions:**\n"
                                f"- Reduce spending in: "
                                f"{', '.join([cat['category'] for cat in financial_summary.get('top_expense_categories', [])[:3]])}\n"
                                f"- Set up an automatic transfer of {format_currency(required_savings)} to a savings account\n"
                                f"- Look for additional income opportunities"
                            )
                        
                        st.success("Plan generated!")
                        st.markdown(f"""
                        <div class="insight-card">
                            {plan}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Unable to generate savings plan: {str(e)}")
            else:
                st.warning("Add some transactions first to get a personalized savings plan.")
    
    # Add goal section
    st.subheader("Create a New Goal")
    with st.form(key="goal_form"):
        col1, col2 = st.columns(2)
        with col1:
            goal_name = st.text_input("Goal Name")
            goal_amount = st.number_input("Target Amount", min_value=1.0, step=10.0)
        with col2:
            goal_deadline = st.date_input("Deadline", 
                                          min_value=datetime.datetime.now())
            goal_category = st.selectbox(
                "Category", 
                ["Savings", "Investment", "Emergency Fund", "Travel", 
                 "Education", "Home", "Vehicle", "Other"]
            )
        
        submit = st.form_submit_button(label="Add Goal")
        
        if submit:
            if goal_name and goal_amount > 0:
                try:
                    # Create goal data
                    goal_data = {
                        "name": goal_name,
                        "target_amount": float(goal_amount),
                        "current_amount": 0.0,
                        "deadline": goal_deadline.strftime("%Y-%m-%d"),
                        "category": goal_category,
                        "notes": ""
                    }
                    
                    # Add to database
                    new_goal = db.create_goal(user_id, goal_data)
                    
                    if new_goal:
                        st.success("Goal added successfully!")
                        # Refresh goals
                        st.rerun()
                    else:
                        st.error("Failed to add goal. Please try again.")
                except Exception as e:
                    st.error(f"Error adding goal: {str(e)}")
            else:
                st.error("Please fill in all required fields.")
    
    # Goal list and progress
    st.subheader("Your Financial Goals")
    
    if goal_dicts:
        goals_tab, chart_tab = st.tabs(["Goals List", "Visualization"])
        
        with goals_tab:
            for i, goal in enumerate(goal_dicts):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {goal['name']}")
                    st.markdown(f"Target: **{format_currency(goal['target_amount'])}** by "
                               f"{goal['deadline']} ({goal['category']})")
                    
                    # Progress bar
                    progress = min(1.0, goal['current_amount'] / goal['target_amount'])
                    st.progress(progress)
                    
                    # Progress text
                    st.markdown(f"**{format_currency(goal['current_amount'])}** of "
                               f"**{format_currency(goal['target_amount'])}** "
                               f"({progress*100:.1f}%)")
                
                with col2:
                    # Update progress
                    form_key = f"update_goal_{goal['id']}"
                    with st.form(key=form_key):
                        current_amount = st.number_input(
                            "Current Amount", 
                            min_value=0.0, 
                            max_value=float(goal['target_amount']),
                            value=float(goal['current_amount']),
                            key=f"goal_amt_{goal['id']}"
                        )
                        if st.form_submit_button("Update"):
                            try:
                                # Create updated goal data
                                updated_data = {
                                    "current_amount": current_amount
                                }
                                
                                # Update in database
                                updated_goal = db.update_goal(goal['id'], updated_data)
                                
                                if updated_goal:
                                    st.success("Goal progress updated!")
                                    # Refresh goals
                                    st.rerun()
                                else:
                                    st.error("Failed to update goal. Please try again.")
                            except Exception as e:
                                st.error(f"Error updating goal: {str(e)}")
                
                st.divider()
        
        with chart_tab:
            # Create visualization tabs
            viz_tabs = st.tabs(["Progress Chart", "Timeline", "Completion Forecast"])
            
            with viz_tabs[0]:  # Progress Chart
                # Convert goals for chart
                goal_data = []
                for g in goal_dicts:
                    goal_data.append({
                        "name": g["name"],
                        "target": g["target_amount"],
                        "current": g["current_amount"],
                        "percent": (g["current_amount"] / g["target_amount"]) * 100 if g["target_amount"] > 0 else 0
                    })
                
                # Create a DataFrame
                goal_df = pd.DataFrame(goal_data)
                
                # Create a horizontal bar chart
                fig = px.bar(
                    goal_df,
                    y="name",
                    x="percent",
                    orientation='h',
                    labels={"name": "Goal", "percent": "Progress (%)"},
                    text="percent",
                    color="percent",
                    color_continuous_scale="Blues",
                    range_x=[0, 100]
                )
                
                # Add text labels
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                
                # Add a vertical line at 100%
                fig.add_vline(x=100, line_dash="dash", line_color="green", annotation_text="Target")
                
                # Layout
                fig.update_layout(
                    title="Goal Progress",
                    yaxis=dict(autorange="reversed"),  # Reverse the order of goals
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[1]:  # Timeline
                # Create a DataFrame with goal deadlines
                timeline_data = []
                for g in goal_dicts:
                    deadline = datetime.datetime.strptime(g["deadline"], "%Y-%m-%d")
                    days_left = (deadline - datetime.datetime.now()).days
                    status = "Completed" if g["current_amount"] >= g["target_amount"] else "In Progress"
                    
                    timeline_data.append({
                        "name": g["name"],
                        "deadline": deadline,
                        "days_left": max(0, days_left),
                        "status": status
                    })
                
                # Create a DataFrame
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df["deadline_str"] = timeline_df["deadline"].dt.strftime("%b %d, %Y")
                timeline_df = timeline_df.sort_values("deadline")
                
                # Create a horizontal bar chart
                fig = px.bar(
                    timeline_df,
                    y="name",
                    x="days_left",
                    orientation='h',
                    labels={"name": "Goal", "days_left": "Days Left"},
                    text="deadline_str",
                    color="status",
                    color_discrete_map={"Completed": "#4CAF50", "In Progress": "#2196F3"}
                )
                
                # Add text labels
                fig.update_traces(textposition='outside')
                
                # Layout
                fig.update_layout(
                    title="Goal Timeline",
                    yaxis=dict(autorange="reversed"),  # Reverse the order of goals
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[2]:  # Completion Forecast
                # Calculate estimated completion dates
                forecast_data = []
                try:
                    financial_summary = finance_agent.get_financial_summary()
                    total_monthly_savings = financial_summary.get("total_income", 0) - financial_summary.get("total_expenses", 0)
                    monthly_savings = max(0, total_monthly_savings / 3)  # Assume 3 months of data
                    
                    for g in goal_dicts:
                        remaining = g["target_amount"] - g["current_amount"]
                        if remaining <= 0:
                            # Already completed
                            forecast_data.append({
                                "name": g["name"],
                                "status": "Completed",
                                "target_date": g["deadline"],
                                "forecast_date": "Completed",
                                "on_track": True
                            })
                        else:
                            # Calculate months needed
                            months_needed = remaining / monthly_savings if monthly_savings > 0 else float('inf')
                            forecast_date = datetime.datetime.now() + datetime.timedelta(days=int(months_needed * 30))
                            deadline = datetime.datetime.strptime(g["deadline"], "%Y-%m-%d")
                            
                            forecast_data.append({
                                "name": g["name"],
                                "status": "In Progress",
                                "target_date": g["deadline"],
                                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                                "on_track": forecast_date <= deadline
                            })
                    
                    # Display the forecast
                    for i, row in enumerate(forecast_data):
                        if row["status"] == "Completed":
                            st.success(f"**{row['name']}**: Goal completed ✓")
                        elif row["on_track"]:
                            st.info(f"**{row['name']}**: On track to complete by {row['forecast_date']} (before deadline of {row['target_date']})")
                        else:
                            st.warning(f"**{row['name']}**: Projected completion on {row['forecast_date']} - **after** deadline of {row['target_date']}")
                except Exception as e:
                    st.warning(f"Could not generate forecast: {str(e)}")
                    st.info("Add more transaction data for better forecasting.")
    else:
        st.info("No goals set yet. Create a goal to get started!")

# AI Assistant page
def show_ai_assistant_page(user):
    st.title("AI Financial Assistant")
    
    # Set user in the finance agent
    user_id = user["id"]
    finance_agent.set_user(user_id)
    
    # Get transactions
    transactions = db.get_transactions(user_id)
    txn_dicts = [t.to_dict() for t in transactions]
    
    # Display AI assistant greeting
    ai_greeting = finance_agent.generate_assistant_message()
    st.markdown(f"""
    <div class="insight-card">
        <h3>Welcome to Your AI Financial Assistant</h3>
        {ai_greeting}
    </div>
    """, unsafe_allow_html=True)
    
    # Example queries and features
    st.markdown("""
    ## Ask any question about your finances
    
    Our AI assistant can answer complex questions and provide deep financial insights:
    
    **Example queries:**
    - Can I afford a $50 purchase today?
    - What were my top 3 expenses last week?
    - Show my spending trends in the entertainment category
    - Am I on track to meet my savings goals?
    - Is my spending on groceries higher than average?
    
    **Advanced features:**
    - Natural language understanding
    - Vector-based transaction search
    - Multi-agent AI processing
    - Anomaly detection
    - Financial forecasting
    """)
    
    # Search and query section
    st.subheader("Search & Query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Natural language query
        query_tab, search_tab = st.tabs(["Ask a Question", "Search Transactions"])
        
        with query_tab:
            query = st.text_input("Ask a financial question:", key="nl_query")
            
            if st.button("Get Answer", key="query_button"):
                if txn_dicts:
                    with st.spinner("Processing with AI..."):
                        try:
                            # Use async function to get answer
                            answer_dict = asyncio.run(finance_agent.answer_question(query))
                            response = answer_dict.get("answer", "Unable to generate answer.")
                            relevant_ids = answer_dict.get("relevant_transactions", [])
                            
                            # Add to query history
                            if "queries" not in st.session_state:
                                st.session_state.queries = []
                            
                            st.session_state.queries.append({
                                "query": query,
                                "response": response,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "relevant_ids": relevant_ids
                            })
                            
                            st.success("Analysis complete!")
                            st.markdown(f"""
                            <div class="insight-card">
                                <h3>Answer</h3>
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display relevant transactions if any
                            if relevant_ids:
                                st.subheader("Relevant Transactions")
                                relevant_txns = [t for t in txn_dicts if t.get("id") in relevant_ids]
                                if relevant_txns:
                                    df = pd.DataFrame(relevant_txns)
                                    st.dataframe(df[["date", "description", "amount", "category"]])
                        except Exception as e:
                            st.warning(f"Unable to process your query: {str(e)}")
                else:
                    st.warning("You need to add some transactions first before asking questions.")
        
        with search_tab:
            search_query = st.text_input("Search for transactions:", key="vector_search")
            
            if st.button("Search", key="search_button"):
                if txn_dicts:
                    with st.spinner("Searching with vector similarity..."):
                        try:
                            # Use vector search through finance agent
                            search_results = finance_agent.search_transactions(search_query)
                            
                            if search_results:
                                st.success(f"Found {len(search_results)} matching transactions")
                                
                                # Convert to DataFrame for display
                                result_df = pd.DataFrame(search_results)
                                
                                # Format for display
                                display_df = result_df[["date", "description", "amount", "category"]]
                                display_df = display_df.sort_values("date", ascending=False)
                                
                                st.dataframe(display_df, use_container_width=True)
                            else:
                                st.info("No matching transactions found.")
                        except Exception as e:
                            st.warning(f"Search error: {str(e)}")
                else:
                    st.warning("You need to add some transactions first before searching.")
    
    with col2:
        # Financial insights based on vector analysis
        st.subheader("AI Insights & Analysis")
        
        insight_types = ["Spending Patterns", "Budget Analysis", "Financial Health"]
        selected_insight = st.selectbox("Select analysis type:", insight_types)
        
        if st.button("Generate Insights", key="insight_button"):
            if txn_dicts:
                with st.spinner(f"Generating {selected_insight.lower()} analysis..."):
                    try:
                        if selected_insight == "Spending Patterns":
                            # Get spending anomalies
                            anomalies = finance_agent.get_spending_anomalies()
                            
                            st.markdown("### Spending Anomalies")
                            if anomalies:
                                for anomaly in anomalies[:3]:  # Show top 3
                                    txn = anomaly.get("transaction", {})
                                    pct = anomaly.get("percent_above_average", 0)
                                    st.markdown(f"""
                                    <div class="warning-card">
                                        <strong>{txn.get('description', 'Unknown')}</strong> ({format_currency(txn.get('amount', 0))})
                                        <br/>
                                        This {txn.get('category', 'expense')} is {pct:.1f}% higher than your usual spending in this category.
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No spending anomalies detected.")
                                
                        elif selected_insight == "Budget Analysis":
                            # Get budget status
                            budget_status = finance_agent.get_budget_status()
                            
                            st.markdown("### Budget Status")
                            if budget_status:
                                for status in budget_status:
                                    budget = status.get("budget", {})
                                    percent = status.get("percent_used", 0)
                                    warning = status.get("warning_level")
                                    
                                    if warning == "critical":
                                        st.markdown(f"""
                                        <div class="warning-card">
                                            <strong>{budget.get('category', 'Unknown')}</strong> - {percent:.1f}% used
                                            <br/>
                                            Budget of {format_currency(budget.get('amount', 0))} exceeded.
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif warning == "warning":
                                        st.markdown(f"""
                                        <div class="warning-card" style="border-left: 4px solid #FF9800;">
                                            <strong>{budget.get('category', 'Unknown')}</strong> - {percent:.1f}% used
                                            <br/>
                                            Approaching budget limit of {format_currency(budget.get('amount', 0))}.
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="insight-card">
                                            <strong>{budget.get('category', 'Unknown')}</strong> - {percent:.1f}% used
                                            <br/>
                                            Within budget of {format_currency(budget.get('amount', 0))}.
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No budgets set. Add budgets in the Budget section.")
                            
                        elif selected_insight == "Financial Health":
                            # Get financial health score
                            health = finance_agent.get_financial_health_score()
                            
                            st.markdown("### Financial Health Assessment")
                            score = health.get("score", 0)
                            details = health.get("details", {})
                            message = health.get("message", "No assessment available")
                            
                            # Display score with a gauge chart
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Financial Health Score"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#4F8BF9"},
                                    'steps' : [
                                        {'range': [0, 40], 'color': "#FF5722"},
                                        {'range': [40, 60], 'color': "#FF9800"},
                                        {'range': [60, 80], 'color': "#8BC34A"},
                                        {'range': [80, 100], 'color': "#4CAF50"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "white", 'width': 4},
                                        'thickness': 0.75,
                                        'value': score
                                    }
                                }
                            ))
                            
                            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display assessment message
                            st.markdown(f"""
                            <div class="insight-card">
                                {message}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display score breakdown
                            if details:
                                st.subheader("Score Breakdown")
                                score_data = []
                                for k, v in details.items():
                                    # Convert keys from snake_case to human readable
                                    key_name = k.replace('_', ' ').title()
                                    score_data.append({"Category": key_name, "Score": v})
                                
                                score_df = pd.DataFrame(score_data)
                                fig = px.bar(
                                    score_df,
                                    x="Score",
                                    y="Category",
                                    orientation='h',
                                    color="Score",
                                    color_continuous_scale="Blues"
                                )
                                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except Exception as e:
                        st.warning(f"Unable to generate insights: {str(e)}")
            else:
                st.warning("You need to add some transactions first to get insights.")
    
    # Query history
    if "queries" in st.session_state and st.session_state.queries:
        st.subheader("Recent Questions")
        for i, q in enumerate(reversed(st.session_state.queries[-5:])):
            with st.expander(f"Q: {q['query']} ({q['timestamp']})"):
                st.markdown(q['response'])

# Settings page
def show_settings_page(user):
    st.title("Settings")
    
    # User info section
    st.subheader("User Information")
    st.write(f"Name: {user['name']}")
    st.write(f"Email: {user['email']}")
    st.write(f"Account Created: {user['created_at']}")
    
    # Data management section
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Data"):
            # Get user's data
            user_id = user["id"]
            transactions = db.get_transactions(user_id)
            goals = db.get_goals(user_id)
            
            # Convert to dictionaries
            txn_dicts = [t.to_dict() for t in transactions]
            goal_dicts = [g.to_dict() for g in goals]
            
            # Prepare data for export
            export_data = {
                "user_info": {
                    "name": user["name"],
                    "email": user["email"],
                    "created_at": user["created_at"]
                },
                "transactions": txn_dicts,
                "goals": goal_dicts,
                "export_date": datetime.datetime.now().isoformat()
            }
            
            # Convert to JSON
            export_json = json.dumps(export_data, indent=2)
            
            # Offer download
            st.download_button(
                label="Download JSON",
                data=export_json,
                file_name="finance_data_export.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Reset All Data"):
            # Confirm deletion
            st.warning("⚠️ This will delete all your financial data and cannot be undone.")
            confirm = st.checkbox("I understand this will delete all my data")
            if confirm and st.button("Confirm Reset"):
                # Reset user's data using database operations
                user_id = user["id"]
                
                # Delete all user data
                try:
                    # Delete all user data with db transactions
                    db.reset_user_data(user_id)
                    
                    # Process vector store cleanup
                    vp.reset_user_vectors(user_id)
                    
                    st.success("All data has been reset!")
                    # Refresh the page
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting data: {str(e)}")

# Main app
def main():
    # Create modern styled sidebar
    with st.sidebar:
        st.image("generated-icon.png", width=80)
        st.title("AI Finance Tracker")
        st.markdown("---")
        
        # Display system status
        with st.expander("System Status", expanded=False):
            # Check database connection
            try:
                db.get_db()
                st.success("Database: Connected")
            except Exception as e:
                st.error(f"Database Error: {str(e)}")
            
            # Check vector engine
            if hasattr(vector_processing.vector_engine, 'is_running') and \
               vector_processing.vector_engine.is_running:
                st.success("Vector Engine: Running")
            else:
                st.warning("Vector Engine: Not Running")
            
            # Check AI services
            if os.environ.get("OPENAI_API_KEY"):
                st.success("AI Services: Available")
            else:
                st.error("AI Services: API Key Missing")
    
    # Check if user is authenticated
    is_logged_in = is_authenticated()
    
    if is_logged_in:
        user = get_current_user()
        
        # User profile section
        with st.sidebar:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background-color: #7B61FF; 
                           display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                    <span style="color: white; font-weight: bold;">{user['name'][0].upper()}</span>
                </div>
                <div>
                    <div style="font-weight: bold;">{user['name']}</div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">{user['email']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Logout button with improved styling
            if st.button("🚪 Logout", key="logout_btn"):
                st.session_state.user_id = None
                st.rerun()
            
            st.markdown("---")
        
        # Sidebar navigation with icons
        with st.sidebar:
            st.markdown("### Navigation")
            
            # Define navigation options with icons
            nav_options = {
                "Dashboard": "📊",
                "Transactions": "💸",
                "Goals": "🎯",
                "Budgets": "💰",
                "Credit Management": "💳",
                "Financial Coaching": "🧠",
                "Crypto Finance": "🪙",
                "AI Assistant": "🤖",
                "Financial RAG": "🔍",
                "Quantitative Finance": "📈",
                "Finance Quiz": "🎮",
                "Settings": "⚙️"
            }
            
            # Create styled navigation buttons
            selected_page = None
            for page_name, icon in nav_options.items():
                if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                    selected_page = page_name
            
            # If no button is clicked, default to session state or Dashboard
            if selected_page is None:
                if "current_page" in st.session_state:
                    selected_page = st.session_state.current_page
                else:
                    selected_page = "Dashboard"
            
            # Store current page in session state
            st.session_state.current_page = selected_page
            
            st.markdown("---")
        
        # Quick Add Transaction with enhanced styling
        with st.sidebar:
            st.markdown("""
            <div style="background-color: rgba(123, 97, 255, 0.1); padding: 10px; border-radius: 10px;">
                <h3 style="margin-top: 0; color: #7B61FF; font-size: 1rem;">⚡ Quick Add</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form(key="add_transaction_form"):
                transaction_date = st.date_input("Date", datetime.datetime.now(), key="sidebar_date")
                description = st.text_input("Description", key="sidebar_desc")
                amount = st.number_input("Amount", min_value=0.0, step=0.01, key="sidebar_amount")
                transaction_type = st.selectbox("Type", ["Expense", "Income"], key="sidebar_type")
                submit_button = st.form_submit_button(label="Add Transaction")
            
            if submit_button:
                if description and amount > 0:
                    try:
                        # Set up finance agent for this user
                        user_id = user["id"]
                        finance_agent.set_user(user_id)
                        
                        # Get AI categorization
                        category = finance_agent.categorize_transaction(description)
                        
                        # Create transaction data
                        transaction_data = {
                            "date": transaction_date.strftime("%Y-%m-%d"),
                            "description": description,
                            "amount": float(amount),
                            "type": transaction_type.lower(),
                            "category": category,
                            "notes": ""
                        }
                        
                        # Add to database
                        new_transaction = db.create_transaction(user_id, transaction_data)
                        
                        if new_transaction:
                            # Process with vector engine
                            vp.process_new_transaction(new_transaction.to_dict())
                            
                            # Get insights about the transaction
                            asyncio.run(finance_agent.analyze_transaction(new_transaction.id))
                            
                            st.sidebar.success("Transaction added successfully!")
                        else:
                            st.sidebar.error("Failed to add transaction. Please try again.")
                    except Exception as e:
                        st.sidebar.error(f"Error: {str(e)}")
                else:
                    st.sidebar.error("Please fill in all fields.")
        
        # Show selected page
        show_page(selected_page, user)
    else:
        # User is not logged in, show authentication page
        auth_page()

if __name__ == "__main__":
    main()