"""
Quantitative Finance page for Streamlit application.
This page displays advanced financial analyses using quantitative methods.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
from typing import Dict, Any, List, Tuple

import finance_agent
import db_service as db
from auth_db import get_current_user

def run():
    """Run the Quantitative Finance page."""
    st.title("Quantitative Finance")

    user = get_current_user()
    if not user:
        st.warning("Please log in to access Quantitative Finance features.")
        return

    # Initialize finance agent with current user
    agent = finance_agent.FinanceAgent(user["id"])

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Monte Carlo Simulation", "Risk Assessment", "Budget Optimization", "Portfolio Optimization"])

    with tab1:
        run_monte_carlo_tab(agent)

    with tab2:
        run_risk_assessment_tab(agent)

    with tab3:
        run_budget_optimization_tab(agent)

    with tab4:
        run_portfolio_optimization_tab(agent)

def run_monte_carlo_tab(agent: finance_agent.FinanceAgent):
    """Run the Monte Carlo Simulation tab."""
    st.header("Monte Carlo Simulation")
    st.write("""
    This simulation uses your historical spending and income patterns to predict 
    possible future financial scenarios. The results show a range of possible outcomes 
    based on your current financial behavior.
    """)

    col1, col2 = st.columns(2)

    with col1:
        months_ahead = st.slider("Months to simulate", min_value=1, max_value=24, value=6, step=1)

    with col2:
        simulations = st.slider("Number of simulations", min_value=100, max_value=5000, value=1000, step=100)

    if st.button("Run Simulation", key="run_monte_carlo"):
        with st.spinner("Running Monte Carlo simulation..."):
            results = agent.run_monte_carlo_simulation(months_ahead, simulations)

            if "error" in results:
                st.error(f"Error: {results.get('message', 'Unknown error')}")
            else:
                display_monte_carlo_results(results)

def display_monte_carlo_results(results: Dict[str, Any]):
    """Display Monte Carlo simulation results."""
    # Extract data
    current_balance = results.get("current_balance", 0)
    dates = results.get("prediction_dates", [])
    percentiles = results.get("percentiles", {})
    negative_prob = results.get("negative_balance_probability", 0) * 100

    # Display summary stats
    st.subheader("Simulation Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Balance", f"${current_balance:.2f}")

    with col2:
        final_median = percentiles.get(str(len(dates)-1), {}).get("median", 0)
        delta = final_median - current_balance
        st.metric("Projected Median Balance", f"${final_median:.2f}", delta=f"${delta:.2f}")

    with col3:
        risk_color = "green" if negative_prob < 10 else "yellow" if negative_prob < 30 else "red"
        st.markdown(f"<div style='text-align: center'><h4>Risk of Negative Balance</h4><p style='color:{risk_color}; font-size:24px; font-weight:bold'>{negative_prob:.1f}%</p></div>", unsafe_allow_html=True)

    # Plot the projection
    fig = go.Figure()

    # Convert dates to datetime for better x-axis
    x_dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]

    # Plot current balance point
    fig.add_trace(go.Scatter(
        x=[x_dates[0]],
        y=[current_balance],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Current Balance'
    ))

    # Plot median projection line
    median_values = [percentiles.get(str(i), {}).get("median", 0) for i in range(len(dates))]
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=median_values,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Median Projection'
    ))

    # Plot the 10th percentile (pessimistic)
    p10_values = [percentiles.get(str(i), {}).get("p10", 0) for i in range(len(dates))]
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=p10_values,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Pessimistic (10th percentile)'
    ))

    # Plot the 90th percentile (optimistic)
    p90_values = [percentiles.get(str(i), {}).get("p90", 0) for i in range(len(dates))]
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=p90_values,
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Optimistic (90th percentile)'
    ))

    # Fill the area between 10th and 90th percentile
    fig.add_trace(go.Scatter(
        x=x_dates + x_dates[::-1],
        y=p90_values + p10_values[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # Add a horizontal line at zero
    fig.add_shape(
        type="line",
        x0=x_dates[0],
        y0=0,
        x1=x_dates[-1],
        y1=0,
        line=dict(
            color="red",
            width=1,
            dash="dot",
        )
    )

    # Update layout
    fig.update_layout(
        title="Balance Projection (Monte Carlo Simulation)",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display stats
    st.subheader("Financial Statistics")
    stats = results.get("statistics", {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Monthly Income", f"${stats.get('avg_monthly_income', 0):.2f}")
        st.metric("Income Volatility", f"{stats.get('income_volatility', 0)*100:.1f}%")
    with col2:
        st.metric("Average Monthly Expenses", f"${stats.get('avg_monthly_expenses', 0):.2f}")
        st.metric("Expense Volatility", f"{stats.get('expense_volatility', 0)*100:.1f}%")

    # Display recommendations
    st.subheader("Recommendations")

    if negative_prob > 50:
        st.error("⚠️ High risk of negative balance! Consider reducing expenses or increasing income immediately.")
    elif negative_prob > 20:
        st.warning("⚠️ Moderate risk of negative balance. Review your spending patterns.")
    else:
        st.success("✅ Low risk of negative balance. Your financial trajectory looks stable.")

    if stats.get('expense_volatility', 0) > 0.3:
        st.info("💡 Your expenses show high volatility. Consider more consistent spending patterns.")

    if stats.get('income_volatility', 0) > 0.3:
        st.info("💡 Your income shows high volatility. Consider more stable income sources if possible.")

    savings_rate = (stats.get('avg_monthly_income', 0) - stats.get('avg_monthly_expenses', 0)) / stats.get('avg_monthly_income', 1)
    if savings_rate < 0.1 and stats.get('avg_monthly_income', 0) > 0:
        st.info("💡 Your savings rate is below 10%. Try to increase savings for better financial security.")

def run_risk_assessment_tab(agent: finance_agent.FinanceAgent):
    """Run the Financial Risk Assessment tab."""
    st.header("Financial Risk Assessment")
    st.write("""
    This assessment analyzes your financial data to identify strengths, weaknesses, and potential risks.
    The model evaluates multiple factors including income stability, expense volatility, savings rate, and more.
    """)

    if st.button("Run Assessment", key="run_risk_assessment"):
        with st.spinner("Analyzing financial health..."):
            results = agent.get_financial_risk_assessment()

            if "error" in results:
                st.error(f"Error: {results.get('message', 'Unknown error')}")
            else:
                display_risk_assessment_results(results)

def display_risk_assessment_results(results: Dict[str, Any]):
    """Display risk assessment results."""
    # Extract data
    health_score = results.get("financial_health_score", 0)
    risk_level = results.get("risk_level", "Unknown")
    metrics = results.get("metrics", {})
    recommendations = results.get("recommendations", [])

    # Set risk color
    risk_color = "green" if risk_level.lower() == "low" else "orange" if risk_level.lower() == "medium" else "red"

    # Display header stats
    st.subheader("Financial Health Overview")
    col1, col2 = st.columns(2)

    with col1:
        # Create a gauge for financial health score
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Financial Health Score"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': health_score
                }
            }
        ))

        gauge.update_layout(height=250)
        st.plotly_chart(gauge, use_container_width=True)

    with col2:
        st.markdown(f"<div style='text-align: center; margin-top: 50px;'><h3>Risk Level</h3><p style='color:{risk_color}; font-size:36px; font-weight:bold'>{risk_level}</p></div>", unsafe_allow_html=True)

    # Display detailed metrics
    st.subheader("Financial Health Metrics")

    # Prepare metrics data for radar chart
    metric_names = {
        "income_stability": "Income Stability",
        "expense_volatility": "Expense Volatility (Lower is Better)",
        "savings_rate": "Savings Rate",
        "emergency_fund_ratio": "Emergency Fund",
        "debt_to_income_ratio": "Debt-to-Income (Lower is Better)",
        "budget_adherence": "Budget Adherence"
    }

    # Convert metrics for radar chart
    radar_metrics = {}
    for key, label in metric_names.items():
        if key in metrics:
            # For metrics where lower is better, invert the value so higher is always better visually
            if "Lower is Better" in label:
                radar_metrics[label] = 1 - metrics[key]
            else:
                radar_metrics[label] = metrics[key]

    # Create radar chart
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())

    # Close the loop for the radar chart
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Metrics',
        line_color='blue'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display raw metrics in an expander
    with st.expander("View Detailed Metrics"):
        metric_df = pd.DataFrame({
            "Metric": [metric_names.get(key, key) for key in metrics.keys()],
            "Value": [f"{value:.2f}" for value in metrics.values()]
        })
        st.dataframe(metric_df, use_container_width=True)

    # Display recommendations
    st.subheader("Recommendations")
    for rec in recommendations:
        st.info(f"💡 {rec}")

def run_budget_optimization_tab(agent: finance_agent.FinanceAgent):
    """Run the Budget Optimization tab."""
    st.header("Budget Optimization")
    st.write("""
    This module applies portfolio theory to personal finance, optimizing your budget allocations
    based on your spending patterns, necessity of expenses, and financial goals.
    """)

    if st.button("Optimize Budget", key="run_budget_optimization"):
        with st.spinner("Analyzing spending patterns and optimizing budget allocation..."):
            results = agent.optimize_budget()

            if "error" in results:
                st.error(f"Error: {results.get('message', 'Unknown error')}")
            else:
                display_budget_optimization_results(results)

def display_budget_optimization_results(results: Dict[str, Any]):
    """Display budget optimization results."""
    # Check if it's an initial recommendation
    if "recommended_budget" in results:
        display_initial_budget_recommendation(results)
        return

    # Extract data for existing budget optimization
    current = results.get("current_allocation", [])
    optimized = results.get("optimized_allocation", [])
    insights = results.get("category_insights", [])
    improvement = results.get("effectiveness_improvement", 0)

    # Display header stats
    st.subheader("Budget Optimization Results")
    st.metric("Effectiveness Improvement", f"{improvement:.1f}%")

    # Compare current vs optimized allocations
    st.subheader("Current vs Optimized Budget")

    # Prepare data for comparison
    categories = []
    current_amounts = []
    optimized_amounts = []

    for c_item in current:
        cat = c_item.get("category", "")
        categories.append(cat)
        current_amounts.append(c_item.get("amount", 0))

        # Find matching optimized budget
        opt_amount = next((o.get("amount", 0) for o in optimized if o.get("category", "") == cat), 0)
        optimized_amounts.append(opt_amount)

    # Add any categories that are only in optimized budget
    for o_item in optimized:
        cat = o_item.get("category", "")
        if cat not in categories:
            categories.append(cat)
            current_amounts.append(0)
            optimized_amounts.append(o_item.get("amount", 0))

    # Create comparison bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=current_amounts,
        name='Current Budget',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=optimized_amounts,
        name='Optimized Budget',
        marker_color='green'
    ))

    fig.update_layout(
        title="Budget Comparison by Category",
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        barmode='group',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display a table with detailed changes
    st.subheader("Detailed Budget Changes")

    changes = []
    for i, cat in enumerate(categories):
        current_val = current_amounts[i]
        optimized_val = optimized_amounts[i]
        dollar_change = optimized_val - current_val
        percent_change = (dollar_change / current_val * 100) if current_val > 0 else 100

        changes.append({
            "Category": cat,
            "Current Budget": f"${current_val:.2f}",
            "Optimized Budget": f"${optimized_val:.2f}",
            "Dollar Change": f"${dollar_change:.2f}",
            "Percent Change": f"{percent_change:.1f}%"
        })

    changes_df = pd.DataFrame(changes)
    st.dataframe(changes_df, use_container_width=True)

    # Display category insights
    st.subheader("Category Insights")

    # Create a scatter plot of volatility vs avg monthly spending
    fig = px.scatter(
        insights, 
        x="avg_monthly", 
        y="volatility", 
        size="avg_monthly",
        color="necessity_score",
        hover_name="category",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={
            "avg_monthly": "Average Monthly Spending ($)",
            "volatility": "Spending Volatility",
            "necessity_score": "Necessity Score"
        }
    )

    fig.update_layout(
        title="Category Analysis",
        xaxis_title="Average Monthly Spending ($)",
        yaxis_title="Spending Volatility",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display recommendations
    st.subheader("Recommendations")

    # Generate recommendations based on the data
    if improvement >= 10:
        st.success("✅ Significant improvements possible! Consider adopting the optimized budget.")
    else:
        st.info("ℹ️ Your current budget is fairly well optimized, but some improvements are still possible.")

    # Find categories with the biggest changes
    big_increases = []
    big_decreases = []

    for i, cat in enumerate(categories):
        current_val = current_amounts[i]
        optimized_val = optimized_amounts[i]
        if current_val > 0:
            percent_change = (optimized_val - current_val) / current_val * 100
            if percent_change >= 20:
                big_increases.append((cat, percent_change))
            elif percent_change <= -20:
                big_decreases.append((cat, percent_change))

    if big_increases:
        for cat, pct in big_increases:
            st.info(f"💡 Consider increasing your '{cat}' budget by approximately {pct:.1f}%")

    if big_decreases:
        for cat, pct in big_decreases:
            st.info(f"💡 Consider decreasing your '{cat}' budget by approximately {-pct:.1f}%")

    # Find high volatility categories
    high_volatility = [item for item in insights if item.get("volatility", 0) > 0.3]
    if high_volatility:
        volatile_cats = ", ".join([item.get("category", "") for item in high_volatility])
        st.info(f"💡 Your spending in {volatile_cats} shows high volatility. Consider more consistent spending patterns.")

def display_initial_budget_recommendation(results: Dict[str, Any]):
    """Display initial budget recommendation when no current budget exists."""
    recommended = results.get("recommended_budget", [])
    total = results.get("total_monthly", 0)

    st.subheader("Recommended Initial Budget")
    st.write(results.get("message", ""))

    # Create pie chart
    labels = [item.get("category", "") for item in recommended]
    values = [item.get("amount", 0) for item in recommended]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title=f"Recommended Budget Allocation (Total: ${total:.2f})"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display detailed table
    st.subheader("Detailed Budget Recommendation")

    budget_df = pd.DataFrame([{
        "Category": item.get("category", ""),
        "Recommended Amount": f"${item.get('amount', 0):.2f}",
        "Percentage": f"{item.get('percent', 0):.1f}%"
    } for item in recommended])

    st.dataframe(budget_df, use_container_width=True)

    # Display recommendations
    st.subheader("Getting Started with Budgeting")
    st.info("💡 This is a starting point based on your current spending patterns. Adjust as needed based on your priorities.")
    st.info("💡 Set up budget categories in the Budgets section using these recommendations as a guideline.")
    st.info("💡 Monitor your spending against these budgets to improve your financial health over time.")

def run_portfolio_optimization_tab(agent: finance_agent.FinanceAgent):
    """Run the Portfolio Optimization tab."""
    st.header("Investment Portfolio Optimization")
    st.write("""
    This tool helps you optimize your investment portfolio based on your risk tolerance and time horizon.
    It applies Modern Portfolio Theory to suggest an asset allocation that balances risk and return.
    """)

    col1, col2 = st.columns(2)

    with col1:
        risk_tolerance = st.slider(
            "Risk Tolerance", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Lower values indicate less willingness to take risk, higher values indicate more risk tolerance."
        )

    with col2:
        time_horizon = st.slider(
            "Investment Time Horizon (years)", 
            min_value=1, 
            max_value=30, 
            value=5, 
            step=1,
            help="The length of time you plan to hold your investments before needing the money."
        )

    if st.button("Optimize Portfolio", key="run_portfolio_optimization"):
        with st.spinner("Calculating optimal asset allocation..."):
            results = agent.optimize_portfolio(risk_tolerance, time_horizon)

            if "error" in results:
                st.error(f"Error: {results.get('message', 'Unknown error')}")
            else:
                display_portfolio_optimization_results(results, risk_tolerance, time_horizon)

def display_portfolio_optimization_results(results: Dict[str, Any], risk_tolerance: float, time_horizon: int):
    """Display portfolio optimization results."""
    # Extract data
    risk_profile = results.get("risk_profile", "Moderate")
    allocation = results.get("allocation", [])
    explanation = results.get("explanation", "")
    expected_returns = results.get("expected_returns", {})

    # Display header stats
    st.subheader("Portfolio Optimization Results")
    st.write(f"**Risk Profile:** {risk_profile}")
    st.write(explanation)

    # Create pie chart for asset allocation
    labels = [item.get("category", "") for item in allocation]
    values = [item.get("percent", 0) for item in allocation]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title=f"Recommended Asset Allocation for {risk_profile} Risk Profile"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display expected returns
    st.subheader("Expected Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Expected Annual Return", f"{expected_returns.get('expected_annual_return', 0):.1f}%")

    with col2:
        st.metric("Expected Volatility", f"{expected_returns.get('expected_volatility', 0):.1f}%")

    with col3:
        st.metric("Time Horizon", f"{time_horizon} years")

    # Display range of outcomes
    st.subheader("Potential Annual Returns")

    worst_case = expected_returns.get('worst_case_annual_return', 0)
    expected = expected_returns.get('expected_annual_return', 0)
    best_case = expected_returns.get('best_case_annual_return', 0)

    # Create a horizontal bar chart for the range
    fig = go.Figure()

    # Add the range bar
    fig.add_trace(go.Bar(
        x=[best_case - worst_case],
        y=["Return Range"],
        orientation='h',
        base=worst_case,
        marker=dict(color='lightblue'),
        name='Potential Range',
        text=f"{worst_case:.1f}% to {best_case:.1f}%",
        textposition='auto'
    ))

    # Add a marker for the expected return
    fig.add_trace(go.Scatter(
        x=[expected],
        y=["Return Range"],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='diamond'),
        name='Expected Return'
    ))

    fig.update_layout(
        title="Range of Potential Annual Returns",
        xaxis_title="Annual Return (%)",
        yaxis=dict(showticklabels=False),
        showlegend=True,
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display compound growth projection
    st.subheader("Projected Growth Over Time")

    # Calculate compound growth
    initial_investment = 10000  # Assume $10,000 initial investment
    years = list(range(time_horizon + 1))

    # Calculate low, expected, and high growth trajectories
    worst_growth = [initial_investment * ((1 + (worst_case / 100)) ** year) for year in years]
    expected_growth = [initial_investment * ((1 + (expected / 100)) ** year) for year in years]
    best_growth = [initial_investment * ((1 + (best_case / 100)) ** year) for year in years]

    # Create the chart
    fig = go.Figure()

    # Add the expected growth line
    fig.add_trace(go.Scatter(
        x=years,
        y=expected_growth,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Expected Growth'
    ))

    # Add the worst-case line
    fig.add_trace(go.Scatter(
        x=years,
        y=worst_growth,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Worst Case'
    ))

    # Add the best-case line
    fig.add_trace(go.Scatter(
        x=years,
        y=best_growth,
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Best Case'
    ))

    # Fill the area between worst and best case
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=best_growth + worst_growth[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f"Projected Growth of $10,000 Over {time_horizon} Years",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display recommendations
    st.subheader("Recommendations")

    recommended_frequency = "monthly" if risk_tolerance < 0.3 else "quarterly" if risk_tolerance < 0.7 else "semi-annually"

    st.info(f"💡 Consider setting up automatic investments on a {recommended_frequency} basis to implement this allocation.")

    if time_horizon >= 10:
        st.info("💡 With your long time horizon, consider increasing your allocation to equities for potentially higher returns.")

    if risk_tolerance < 0.3 and time_horizon > 15:
        st.info("💡 Your risk tolerance is low compared to your time horizon. Consider whether you can accept more risk for better potential returns.")

    if risk_tolerance > 0.7 and time_horizon < 5:
        st.info("💡 Your risk tolerance is high compared to your short time horizon. Consider reducing risk to protect your capital.")

    st.info("💡 Remember to rebalance your portfolio periodically to maintain your target allocation.")