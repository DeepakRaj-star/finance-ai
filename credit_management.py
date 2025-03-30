import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime
import random  # For demo purposes
from typing import Dict, List, Any, Optional

from utils import format_currency
import ai_agents

class CreditManager:
    def __init__(self, user_id: str):
        """
        Initialize the Credit Manager for a specific user.
        
        Args:
            user_id (str): The user ID
        """
        self.user_id = user_id
        
    def get_credit_score(self) -> Dict[str, Any]:
        """
        Get the user's credit score and related metrics.
        
        Returns:
            Dict: Credit score information
        """
        # In a real implementation, this would pull from a credit API
        # For demo, we'll generate a simulated credit score
        
        # Ask AI for assessment based on transactions and financial behavior
        credit_assessment = ai_agents.generate_structured_response(
            f"Based on user {self.user_id}'s financial behaviors, generate a detailed credit assessment",
            system_prompt="You are a credit analysis expert. Generate a realistic credit assessment with score, factors, and recommendations."
        )
        
        # If AI fails, use demo data
        if not credit_assessment:
            # Demo data
            score = random.randint(650, 850)
            
            # Calculate score category
            if score >= 800:
                category = "Excellent"
                color = "green"
            elif score >= 740:
                category = "Very Good"
                color = "lightgreen" 
            elif score >= 670:
                category = "Good"
                color = "yellow"
            elif score >= 580:
                category = "Fair"
                color = "orange"
            else:
                category = "Poor"
                color = "red"
                
            return {
                "score": score,
                "category": category,
                "color": color,
                "factors": [
                    {"name": "Payment History", "impact": "High", "status": "Good", "score": random.randint(70, 100)},
                    {"name": "Credit Utilization", "impact": "High", "status": "Fair", "score": random.randint(60, 95)},
                    {"name": "Credit Age", "impact": "Medium", "status": "Good", "score": random.randint(65, 100)},
                    {"name": "Credit Mix", "impact": "Low", "status": "Very Good", "score": random.randint(75, 100)},
                    {"name": "Recent Inquiries", "impact": "Low", "status": "Excellent", "score": random.randint(80, 100)}
                ],
                "history": [
                    {"date": (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d"), "score": score - random.randint(5, 15)},
                    {"date": (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d"), "score": score - random.randint(3, 10)},
                    {"date": (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d"), "score": score - random.randint(0, 8)},
                    {"date": (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d"), "score": score - random.randint(0, 5)},
                    {"date": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"), "score": score - random.randint(-3, 3)},
                    {"date": datetime.datetime.now().strftime("%Y-%m-%d"), "score": score}
                ],
                "recommendations": [
                    "Consider paying down high-interest debt first",
                    "Maintain low credit card balances (below 30% of limit)",
                    "Avoid opening multiple new accounts in a short period",
                    "Continue making on-time payments for all accounts",
                    "Monitor your credit report regularly for errors"
                ]
            }
        
        return credit_assessment
        
    def analyze_credit_utilization(self) -> Dict[str, Any]:
        """
        Analyze the user's credit utilization ratio.
        
        Returns:
            Dict: Credit utilization analysis
        """
        # In a real implementation, this would pull from actual credit accounts
        # For demo, we'll generate simulated credit accounts
        
        credit_analysis = ai_agents.generate_structured_response(
            f"Based on user {self.user_id}'s financial data, analyze their credit utilization",
            system_prompt="You are a credit utilization expert. Generate a detailed analysis with account breakdowns and recommendations."
        )
        
        # If AI fails, use demo data
        if not credit_analysis:
            accounts = [
                {"name": "Credit Card 1", "limit": 5000, "balance": 1250, "utilization": 25.0},
                {"name": "Credit Card 2", "limit": 10000, "balance": 2200, "utilization": 22.0},
                {"name": "Store Card", "limit": 2500, "balance": 950, "utilization": 38.0},
                {"name": "Credit Line", "limit": 15000, "balance": 3000, "utilization": 20.0}
            ]
            
            total_limit = sum(a["limit"] for a in accounts)
            total_balance = sum(a["balance"] for a in accounts)
            overall_utilization = (total_balance / total_limit) * 100 if total_limit > 0 else 0
            
            return {
                "accounts": accounts,
                "total_limit": total_limit,
                "total_balance": total_balance,
                "overall_utilization": overall_utilization,
                "status": "Good" if overall_utilization < 30 else "Fair" if overall_utilization < 50 else "Poor",
                "recommendations": [
                    "Pay down Store Card balance to reduce high utilization",
                    "Consider requesting a credit limit increase on Credit Card 1",
                    "Maintain current payment strategy on Credit Line",
                    "Target overall utilization below 30% for optimal credit impact"
                ]
            }
            
        return credit_analysis
            
    def get_payment_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get AI-powered recommendations for optimal payment strategies.
        
        Returns:
            List[Dict]: Payment recommendations
        """
        # Generate payment recommendations using AI
        recommendations = ai_agents.generate_structured_response(
            f"Based on user {self.user_id}'s current accounts and credit status, provide payment optimization recommendations",
            system_prompt="You are a payment strategy expert. Generate actionable recommendations for improving credit through optimal payment timing and amounts."
        )
        
        # If AI fails, use demo data
        if not recommendations or not isinstance(recommendations, list):
            return [
                {
                    "account": "Credit Card 2",
                    "current_balance": 2200,
                    "min_payment": 44,
                    "optimal_payment": 500,
                    "due_date": (datetime.datetime.now() + datetime.timedelta(days=8)).strftime("%Y-%m-%d"),
                    "recommendation": "Pay $500 on or before due date to reduce utilization below 20%",
                    "impact": "Medium"
                },
                {
                    "account": "Store Card",
                    "current_balance": 950,
                    "min_payment": 35,
                    "optimal_payment": 400,
                    "due_date": (datetime.datetime.now() + datetime.timedelta(days=12)).strftime("%Y-%m-%d"),
                    "recommendation": "Pay $400 to bring utilization below 30% threshold",
                    "impact": "High"
                },
                {
                    "account": "Credit Card 1",
                    "current_balance": 1250,
                    "min_payment": 25,
                    "optimal_payment": 250,
                    "due_date": (datetime.datetime.now() + datetime.timedelta(days=15)).strftime("%Y-%m-%d"),
                    "recommendation": "Schedule payment now to ensure it posts before statement date",
                    "impact": "Low"
                }
            ]
            
        return recommendations
        
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """
        Generate a comprehensive credit improvement plan.
        
        Returns:
            Dict: Credit improvement plan
        """
        # Generate improvement plan using AI
        plan = ai_agents.generate_structured_response(
            f"Generate a 6-month credit improvement plan for user {self.user_id}",
            system_prompt="You are a credit improvement expert. Generate a detailed 6-month plan with monthly goals and specific actions."
        )
        
        # If AI fails, use demo data
        if not plan:
            return {
                "current_score": self.get_credit_score()["score"],
                "target_score": min(self.get_credit_score()["score"] + 50, 850),
                "timeframe": "6 months",
                "monthly_goals": [
                    {
                        "month": "Month 1",
                        "focus": "Reduce Credit Utilization",
                        "target": "Utilization below 40%",
                        "actions": [
                            "Pay down highest utilization card by $400",
                            "Request credit limit increase on oldest card",
                            "Do not make new purchases on credit cards"
                        ]
                    },
                    {
                        "month": "Month 2",
                        "focus": "Payment History Optimization",
                        "target": "All payments on time",
                        "actions": [
                            "Set up automatic payments for all accounts",
                            "Pay down second highest utilization card by $300",
                            "Review credit report for errors and dispute if needed"
                        ]
                    },
                    {
                        "month": "Month 3",
                        "focus": "Continue Utilization Improvement",
                        "target": "Utilization below 30%",
                        "actions": [
                            "Continue debt paydown strategy",
                            "Keep all utilization below 30% on individual cards",
                            "Avoid applying for new credit"
                        ]
                    },
                    {
                        "month": "Month 4",
                        "focus": "Credit Mix Optimization",
                        "target": "Balanced credit portfolio",
                        "actions": [
                            "Consider adding a secured loan if no installment accounts exist",
                            "Continue consistent payment history",
                            "Maintain low utilization"
                        ]
                    },
                    {
                        "month": "Month 5",
                        "focus": "Long-term Account Management",
                        "target": "Utilization below 20%",
                        "actions": [
                            "Keep oldest accounts active with small purchases",
                            "Pay in full each month to avoid interest",
                            "Continue debt reduction plan"
                        ]
                    },
                    {
                        "month": "Month 6",
                        "focus": "Final Assessment",
                        "target": "Credit score improvement",
                        "actions": [
                            "Review full credit report from all bureaus",
                            "Assess score improvement and adjust strategy if needed",
                            "Maintain good credit habits going forward"
                        ]
                    }
                ]
            }
            
        return plan

def credit_management_page(user_id: str):
    """
    Display the credit management page.
    
    Args:
        user_id (str): The ID of the current user
    """
    st.title("Dynamic Credit Management")
    
    # Initialize credit manager
    credit_manager = CreditManager(user_id)
    
    # Create tabs for different credit features
    tabs = st.tabs(["Credit Score", "Credit Utilization", "Payment Optimization", "Improvement Plan"])
    
    # Credit Score tab
    with tabs[0]:
        st.subheader("Credit Score Analysis")
        
        # Get credit score data
        credit_data = credit_manager.get_credit_score()
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            # Create a gauge chart for credit score
            # Get score with default value
            score = credit_data.get("score", 650)
            category = credit_data.get("category", "Good")
            color = credit_data.get("color", "yellow")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Credit Score: {category}"},
                gauge={
                    'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [300, 580], 'color': 'firebrick'},
                        {'range': [580, 670], 'color': 'darkorange'},
                        {'range': [670, 740], 'color': 'gold'},
                        {'range': [740, 800], 'color': 'yellowgreen'},
                        {'range': [800, 850], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "darkblue", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display factors affecting score
            st.subheader("Credit Score Factors")
            
            for factor in credit_data["factors"]:
                factor_color = "green" if factor["status"] == "Excellent" else \
                              "lightgreen" if factor["status"] == "Very Good" else \
                              "gold" if factor["status"] == "Good" else \
                              "orange" if factor["status"] == "Fair" else "red"
                              
                st.markdown(f"""
                <div style="
                    background-color: rgba(0,0,0,0.05); 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-bottom: 10px;
                    border-left: 5px solid {factor_color};">
                    <b>{factor["name"]}</b> ({factor["impact"]} Impact) - <span style="color: {factor_color}">{factor["status"]}</span>
                    <div style="margin-top: 5px; background-color: #eee; height: 10px; border-radius: 5px;">
                        <div style="width: {factor['score']}%; background-color: {factor_color}; height: 10px; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Plot credit score history
            st.subheader("Credit Score History")
            
            history_df = pd.DataFrame(credit_data["history"])
            history_df["date"] = pd.to_datetime(history_df["date"])
            
            fig = px.line(
                history_df, 
                x="date", 
                y="score",
                markers=True,
                title="Score Trend Over Time"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Credit Score",
                yaxis=dict(range=[min(history_df["score"]) - 20, 850])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            st.subheader("Improvement Recommendations")
            
            for i, rec in enumerate(credit_data["recommendations"]):
                st.markdown(f"**{i+1}.** {rec}")
    
    # Credit Utilization tab
    with tabs[1]:
        st.subheader("Credit Utilization Analysis")
        
        # Get utilization data
        utilization_data = credit_manager.analyze_credit_utilization()
        
        # Display overall utilization
        overall_util = utilization_data["overall_utilization"]
        util_color = "green" if overall_util < 30 else "orange" if overall_util < 50 else "red"
        
        st.markdown(f"""
        <div style="
            background-color: rgba(0,0,0,0.05); 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            text-align: center;">
            <h3>Overall Utilization: <span style="color: {util_color}">{overall_util:.1f}%</span></h3>
            <p>Total Balance: {format_currency(utilization_data["total_balance"])} of {format_currency(utilization_data["total_limit"])} Total Limit</p>
            <div style="margin-top: 10px; background-color: #eee; height: 20px; border-radius: 10px;">
                <div style="width: {min(overall_util, 100)}%; background-color: {util_color}; height: 20px; border-radius: 10px;"></div>
            </div>
            <p style="margin-top: 10px; font-style: italic;">Status: {utilization_data["status"]} (Target: Below 30%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display individual accounts
        st.subheader("Account Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of account utilization
            account_df = pd.DataFrame(utilization_data["accounts"])
            
            # Add color based on utilization
            account_df["color"] = account_df["utilization"].apply(
                lambda x: "green" if x < 30 else "orange" if x < 50 else "red"
            )
            
            fig = px.bar(
                account_df,
                x="name",
                y="utilization",
                color="color",
                color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
                title="Utilization by Account",
                labels={"name": "Account", "utilization": "Utilization (%)"},
                text="utilization"
            )
            
            fig.update_traces(
                texttemplate="%{y:.1f}%", 
                textposition="outside"
            )
            
            fig.update_layout(
                yaxis=dict(range=[0, max(account_df["utilization"]) * 1.2]),
                showlegend=False
            )
            
            # Add a horizontal line at 30%
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(account_df) - 0.5,
                y0=30,
                y1=30,
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=len(account_df) - 1,
                y=35,
                text="Target: 30%",
                showarrow=False,
                font=dict(color="green")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart of account balances
            fig = px.pie(
                account_df,
                values="balance",
                names="name",
                title="Balance Distribution",
                hole=0.4
            )
            
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:$,.2f} (%{percent})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Account details table
        st.subheader("Account Details")
        
        for account in utilization_data["accounts"]:
            util_color = "green" if account["utilization"] < 30 else "orange" if account["utilization"] < 50 else "red"
            
            st.markdown(f"""
            <div style="
                background-color: rgba(0,0,0,0.05); 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 10px;
                border-left: 5px solid {util_color};">
                <b>{account["name"]}</b>
                <table style="width: 100%; margin-top: 5px;">
                    <tr>
                        <td style="width: 33%;">Balance: {format_currency(account["balance"])}</td>
                        <td style="width: 33%;">Limit: {format_currency(account["limit"])}</td>
                        <td style="width: 33%;">Utilization: <span style="color: {util_color}">{account["utilization"]:.1f}%</span></td>
                    </tr>
                </table>
                <div style="margin-top: 5px; background-color: #eee; height: 10px; border-radius: 5px;">
                    <div style="width: {min(account["utilization"], 100)}%; background-color: {util_color}; height: 10px; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("Optimization Recommendations")
        
        for i, rec in enumerate(utilization_data["recommendations"]):
            st.markdown(f"**{i+1}.** {rec}")
    
    # Payment Optimization tab
    with tabs[2]:
        st.subheader("Payment Optimization")
        
        # Get payment recommendations
        payment_recs = credit_manager.get_payment_recommendations()
        
        # Sort by impact and due date
        payment_recs = sorted(payment_recs, key=lambda x: (
            0 if x["impact"] == "High" else 1 if x["impact"] == "Medium" else 2,
            datetime.datetime.strptime(x["due_date"], "%Y-%m-%d")
        ))
        
        # Display upcoming payments
        st.markdown("### Upcoming Payments")
        
        for rec in payment_recs:
            impact_color = "red" if rec["impact"] == "High" else "orange" if rec["impact"] == "Medium" else "blue"
            
            # Calculate days until due
            due_date = datetime.datetime.strptime(rec["due_date"], "%Y-%m-%d")
            days_until = (due_date - datetime.datetime.now()).days
            
            st.markdown(f"""
            <div style="
                background-color: rgba(0,0,0,0.05); 
                padding: 15px; 
                border-radius: 5px; 
                margin-bottom: 15px;
                border-left: 5px solid {impact_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">{rec["account"]}</h4>
                    <span style="background-color: {impact_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">{rec["impact"]} Impact</span>
                </div>
                <div style="margin-top: 10px;">
                    <p><b>Due Date:</b> {rec["due_date"]} ({days_until} days from now)</p>
                    <p><b>Current Balance:</b> {format_currency(rec["current_balance"])}</p>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; width: 50%;">
                                <b>Minimum Payment:</b> {format_currency(rec["min_payment"])}
                            </td>
                            <td style="padding: 8px; border: 1px solid #ddd; width: 50%; background-color: rgba(0,255,0,0.05);">
                                <b>Recommended Payment:</b> {format_currency(rec["optimal_payment"])}
                            </td>
                        </tr>
                    </table>
                    <p style="margin-top: 10px;"><b>Recommendation:</b> {rec["recommendation"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Payment calendar visualization
        st.subheader("Payment Calendar")
        
        # Create dataframe for visualization
        payment_df = pd.DataFrame([
            {
                "Account": rec["account"],
                "Date": rec["due_date"],
                "Amount": rec["optimal_payment"],
                "Type": "Optimal Payment",
                "Impact": rec["impact"]
            } for rec in payment_recs
        ])
        
        # Add minimum payments for comparison
        payment_df = pd.concat([
            payment_df,
            pd.DataFrame([
                {
                    "Account": rec["account"],
                    "Date": rec["due_date"],
                    "Amount": rec["min_payment"],
                    "Type": "Minimum Payment",
                    "Impact": rec["impact"]
                } for rec in payment_recs
            ])
        ])
        
        # Convert date to datetime
        payment_df["Date"] = pd.to_datetime(payment_df["Date"])
        
        # Create grouped bar chart
        fig = px.bar(
            payment_df,
            x="Date",
            y="Amount",
            color="Type",
            barmode="group",
            title="Payment Schedule",
            color_discrete_map={
                "Optimal Payment": "green",
                "Minimum Payment": "lightgray"
            },
            hover_data=["Account", "Impact"]
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Payment Amount ($)",
            legend_title="Payment Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment strategy summary
        st.subheader("Payment Strategy Summary")
        
        total_min = sum(rec["min_payment"] for rec in payment_recs)
        total_optimal = sum(rec["optimal_payment"] for rec in payment_recs)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Minimum Payments", format_currency(total_min))
        
        with col2:
            st.metric("Total Optimal Payments", format_currency(total_optimal))
        
        with col3:
            st.metric("Additional Recommended", format_currency(total_optimal - total_min), 
                     delta=f"{((total_optimal/total_min) - 1) * 100:.1f}% increase")
        
        # General payment strategies
        st.subheader("General Payment Strategies")
        
        strategies = [
            {
                "name": "Debt Avalanche",
                "description": "Pay minimum on all debts, then put extra money toward the highest interest rate debt first.",
                "best_for": "Minimizing interest costs and paying less overall",
                "impact": "High long-term financial benefit"
            },
            {
                "name": "Debt Snowball",
                "description": "Pay minimum on all debts, then put extra money toward the smallest balance first.",
                "best_for": "Building momentum and motivation with quick wins",
                "impact": "High psychological benefit"
            },
            {
                "name": "Utilization Optimization",
                "description": "Focus payments on keeping all account utilizations below 30% for maximum credit score impact.",
                "best_for": "Improving credit score in the short term",
                "impact": "High credit score benefit"
            }
        ]
        
        for strategy in strategies:
            st.markdown(f"""
            <div style="
                background-color: rgba(0,0,0,0.05); 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 10px;">
                <h4>{strategy["name"]}</h4>
                <p><b>Description:</b> {strategy["description"]}</p>
                <p><b>Best For:</b> {strategy["best_for"]}</p>
                <p><b>Impact:</b> {strategy["impact"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Improvement Plan tab
    with tabs[3]:
        st.subheader("Credit Improvement Plan")
        
        # Get improvement plan
        plan = credit_manager.generate_improvement_plan()
        
        # Display plan overview
        st.markdown(f"""
        <div style="
            background-color: rgba(0,0,0,0.05); 
            padding: 20px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            text-align: center;">
            <h3>6-Month Improvement Plan</h3>
            <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                <div>
                    <p>Current Score</p>
                    <h2>{plan["current_score"]}</h2>
                </div>
                <div style="font-size: 2em; margin-top: 15px;">→</div>
                <div>
                    <p>Target Score</p>
                    <h2>{plan["target_score"]}</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Monthly goals
        st.subheader("Monthly Goals and Actions")
        
        # Create tabs for each month
        month_tabs = st.tabs([goal["month"] for goal in plan["monthly_goals"]])
        
        # Fill each month's tab
        for i, month_tab in enumerate(month_tabs):
            goal = plan["monthly_goals"][i]
            
            with month_tab:
                st.markdown(f"""
                <div style="
                    background-color: rgba(79, 139, 249, 0.1); 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 15px;
                    border-left: 5px solid #4F8BF9;">
                    <h3>{goal["month"]}: {goal["focus"]}</h3>
                    <p><b>Target:</b> {goal["target"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Action Plan")
                
                for j, action in enumerate(goal["actions"]):
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(0,0,0,0.02); 
                        padding: 10px; 
                        border-radius: 5px; 
                        margin-bottom: 10px;
                        border-left: 3px solid #4F8BF9;">
                        <p><b>Action {j+1}:</b> {action}</p>
                    </div>
                    """, unsafe_allow_html=True)