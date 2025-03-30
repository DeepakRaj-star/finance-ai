import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Any, Optional

from utils import format_currency
import db_manager
import ai_agents

class FinancialCoach:
    def __init__(self, user_id: str):
        """
        Initialize the Financial Coach for a specific user.
        
        Args:
            user_id (str): The user ID
        """
        self.user_id = user_id
        self.transactions = db_manager.load_transactions(user_id)
        self.goals = db_manager.load_goals(user_id)
        self.budgets = db_manager.load_budgets(user_id)
        
    def analyze_spending_behavior(self) -> Dict[str, Any]:
        """
        Analyze user's spending behavior patterns and habits.
        
        Returns:
            Dict: Spending behavior analysis
        """
        # Use AI to analyze spending behavior
        behavior_analysis = ai_agents.generate_structured_response(
            f"Analyze spending behavior for user {self.user_id} based on their transaction history",
            system_prompt="You are a financial behavior analyst. Analyze the user's spending patterns, habits, and behavioral trends. Identify strengths, weaknesses, and opportunities for improvement."
        )
        
        # If AI fails, generate structured analysis from transactions
        if not behavior_analysis:
            # Convert transactions to dictionaries
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.transactions]
            
            # Analyze category distribution
            categories = {}
            for txn in txn_dicts:
                if txn.get('type', '').lower() != 'expense':
                    continue
                    
                category = txn.get('category', 'Uncategorized')
                if category not in categories:
                    categories[category] = 0
                categories[category] += txn.get('amount', 0)
            
            # Sort categories by amount
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            top_categories = sorted_categories[:3]
            
            # Analyze transaction frequency
            transactions_by_day = {}
            for txn in txn_dicts:
                date = txn.get('date', '')
                if not date:
                    continue
                    
                day_of_week = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%A")
                if day_of_week not in transactions_by_day:
                    transactions_by_day[day_of_week] = 0
                transactions_by_day[day_of_week] += 1
            
            # Find day with most transactions
            most_active_day = max(transactions_by_day.items(), key=lambda x: x[1])[0] if transactions_by_day else "N/A"
            
            # Analyze transaction sizes
            transaction_amounts = [txn.get('amount', 0) for txn in txn_dicts if txn.get('type', '').lower() == 'expense']
            avg_transaction = sum(transaction_amounts) / len(transaction_amounts) if transaction_amounts else 0
            
            # Count small, medium, and large transactions
            small_txn = sum(1 for a in transaction_amounts if a < 20)
            medium_txn = sum(1 for a in transaction_amounts if 20 <= a < 100)
            large_txn = sum(1 for a in transaction_amounts if a >= 100)
            
            # Identify spending habits
            habits = []
            
            # Check for frequent small purchases
            if small_txn > len(transaction_amounts) * 0.4:
                habits.append({
                    "type": "Frequent small purchases",
                    "description": "You make many small purchases which can add up quickly.",
                    "impact": "Negative",
                    "recommendation": "Try consolidating shopping trips and use a list to avoid impulse purchases."
                })
                
            # Check for weekend spending
            weekend_txn = transactions_by_day.get('Saturday', 0) + transactions_by_day.get('Sunday', 0)
            weekday_txn = sum(v for k, v in transactions_by_day.items() if k not in ['Saturday', 'Sunday'])
            if weekend_txn > weekday_txn * 0.5 and weekend_txn > 0:
                habits.append({
                    "type": "Weekend spender",
                    "description": "A large portion of your spending happens on weekends.",
                    "impact": "Neutral",
                    "recommendation": "Plan weekend activities in advance to control entertainment and dining costs."
                })
                
            # Check for end-of-month spending
            end_month_txn = sum(1 for txn in txn_dicts if 
                              txn.get('type', '').lower() == 'expense' and 
                              txn.get('date', '') and 
                              int(txn.get('date', '').split('-')[2]) > 25)
            if end_month_txn > len(transaction_amounts) * 0.3:
                habits.append({
                    "type": "End-of-month spending",
                    "description": "You tend to spend more at the end of the month.",
                    "impact": "Negative",
                    "recommendation": "Create a monthly budget and track spending throughout the month to avoid end-of-month splurges."
                })
                
            # Check for consistent saving
            income_txn = [txn.get('amount', 0) for txn in txn_dicts if txn.get('type', '').lower() == 'income']
            expense_txn = [txn.get('amount', 0) for txn in txn_dicts if txn.get('type', '').lower() == 'expense']
            
            total_income = sum(income_txn)
            total_expense = sum(expense_txn)
            
            savings_rate = (total_income - total_expense) / total_income if total_income > 0 else 0
            
            if savings_rate > 0.2:
                habits.append({
                    "type": "Strong saver",
                    "description": f"You save approximately {savings_rate*100:.1f}% of your income.",
                    "impact": "Positive",
                    "recommendation": "Consider investing your savings for long-term growth."
                })
            elif savings_rate > 0:
                habits.append({
                    "type": "Moderate saver",
                    "description": f"You save approximately {savings_rate*100:.1f}% of your income.",
                    "impact": "Neutral",
                    "recommendation": "Try to increase your savings rate to at least 20% by identifying non-essential expenses."
                })
            else:
                habits.append({
                    "type": "Spender",
                    "description": "You're currently spending more than you earn.",
                    "impact": "Negative",
                    "recommendation": "Review your budget and identify areas to reduce spending to avoid debt accumulation."
                })
                
            # Prepare behavior analysis
            behavior_analysis = {
                "top_spending_categories": [{"category": cat, "amount": amt} for cat, amt in top_categories],
                "most_active_day": most_active_day,
                "average_transaction": avg_transaction,
                "transaction_distribution": {
                    "small": small_txn,
                    "medium": medium_txn,
                    "large": large_txn
                },
                "spending_habits": habits,
                "savings_rate": savings_rate,
                "overall_assessment": "Healthy" if savings_rate > 0.2 else "Needs Improvement" if savings_rate > 0 else "Concerning"
            }
            
        return behavior_analysis
    
    def get_personalized_tips(self) -> List[Dict[str, str]]:
        """
        Generate personalized financial tips based on user data.
        
        Returns:
            List[Dict]: List of personalized tips
        """
        # Use AI to generate personalized tips
        personalized_tips = ai_agents.generate_structured_response(
            f"Generate personalized financial tips for user {self.user_id} based on their transaction history, goals, and budgets",
            system_prompt="You are a financial coach. Generate actionable, personalized financial tips for the user based on their specific situation."
        )
        
        # If AI fails, generate basic tips
        if not isinstance(personalized_tips, list):
            # Convert transactions to dictionaries
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.transactions]
            
            # Basic analysis for tips
            tips = []
            
            # Calculate income, expenses, and savings
            income = sum(t.get('amount', 0) for t in txn_dicts if t.get('type', '').lower() == 'income')
            expenses = sum(t.get('amount', 0) for t in txn_dicts if t.get('type', '').lower() == 'expense')
            savings = income - expenses
            savings_rate = (savings / income) * 100 if income > 0 else 0
            
            # Get expense categories
            categories = {}
            for t in txn_dicts:
                if t.get('type', '').lower() != 'expense':
                    continue
                    
                category = t.get('category', 'Uncategorized')
                if category not in categories:
                    categories[category] = 0
                categories[category] += t.get('amount', 0)
            
            # Sort categories by amount
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            
            # Generate savings tip
            if savings_rate < 20:
                tips.append({
                    "category": "Savings",
                    "title": "Increase Your Savings Rate",
                    "description": f"Your current savings rate is {savings_rate:.1f}%. Financial experts recommend saving at least 20% of your income for long-term financial security.",
                    "action": "Identify non-essential expenses in your top spending categories and redirect that money to savings or debt repayment."
                })
            else:
                tips.append({
                    "category": "Savings",
                    "title": "Optimize Your Savings",
                    "description": f"You're already saving {savings_rate:.1f}% of your income, which is great!",
                    "action": "Consider diversifying your savings into emergency funds, retirement accounts, and medium-term investment vehicles."
                })
                
            # Generate tip based on top expense category
            if sorted_categories:
                top_category, top_amount = sorted_categories[0]
                category_percentage = (top_amount / expenses) * 100 if expenses > 0 else 0
                
                if category_percentage > 30:
                    tips.append({
                        "category": "Spending",
                        "title": f"Review Your {top_category} Spending",
                        "description": f"{top_category} represents {category_percentage:.1f}% of your total expenses.",
                        "action": f"Look for ways to reduce your {top_category.lower()} costs without sacrificing quality. Consider setting a specific budget for this category."
                    })
                    
            # Generate budget tip
            if not self.budgets:
                tips.append({
                    "category": "Budgeting",
                    "title": "Create a Budget",
                    "description": "You don't currently have any budgets set up.",
                    "action": "Create budgets for your major spending categories to help control expenses and increase savings."
                })
            else:
                budget_dicts = [b.to_dict() if hasattr(b, 'to_dict') else b for b in self.budgets]
                missing_categories = [cat for cat, _ in sorted_categories[:5] 
                                     if not any(b.get('category', '') == cat for b in budget_dicts)]
                
                if missing_categories:
                    tips.append({
                        "category": "Budgeting",
                        "title": "Expand Your Budget Categories",
                        "description": f"You're missing budgets for these important categories: {', '.join(missing_categories)}.",
                        "action": "Create additional budget categories to better track and control your spending in these areas."
                    })
                    
            # Generate goal tip
            if not self.goals:
                tips.append({
                    "category": "Goals",
                    "title": "Set Financial Goals",
                    "description": "You don't currently have any financial goals set up.",
                    "action": "Create specific, measurable financial goals to give your saving and spending purpose and direction."
                })
            else:
                goal_dicts = [g.to_dict() if hasattr(g, 'to_dict') else g for g in self.goals]
                active_goals = [g for g in goal_dicts if not g.get('completed', False)]
                
                if active_goals:
                    nearest_goal = min(active_goals, key=lambda g: 
                                     (datetime.datetime.strptime(g.get('target_date', '2099-12-31'), "%Y-%m-%d") 
                                      - datetime.datetime.now()).days 
                                     if g.get('target_date') else float('inf'))
                    
                    days_left = (datetime.datetime.strptime(nearest_goal.get('target_date', '2099-12-31'), "%Y-%m-%d") 
                                - datetime.datetime.now()).days if nearest_goal.get('target_date') else 0
                    
                    current_amount = nearest_goal.get('current_amount', 0)
                    target_amount = nearest_goal.get('target_amount', 0)
                    remaining = target_amount - current_amount
                    
                    if days_left > 0 and remaining > 0:
                        monthly_needed = (remaining / days_left) * 30
                        
                        tips.append({
                            "category": "Goals",
                            "title": f"Plan for Your {nearest_goal.get('name', 'Goal')}",
                            "description": f"You need {format_currency(remaining)} more in {days_left} days to reach this goal.",
                            "action": f"Set aside approximately {format_currency(monthly_needed)} per month to stay on track."
                        })
            
            # Add general tips
            tips.append({
                "category": "Financial Health",
                "title": "Build an Emergency Fund",
                "description": "An emergency fund is crucial for financial stability and peace of mind.",
                "action": "Aim to save 3-6 months of essential expenses in an easily accessible account."
            })
            
            tips.append({
                "category": "Financial Health",
                "title": "Review Recurring Subscriptions",
                "description": "Subscription services can silently drain your finances over time.",
                "action": "Review all recurring charges and cancel those you no longer use or need."
            })
            
            personalized_tips = tips
            
        return personalized_tips
    
    def track_goal_progress(self) -> List[Dict[str, Any]]:
        """
        Track progress toward financial goals with projections.
        
        Returns:
            List[Dict]: Goal progress data
        """
        # Get goals with progress data
        goal_dicts = [g.to_dict() if hasattr(g, 'to_dict') else g for g in self.goals]
        
        # Track progress for each goal
        for goal in goal_dicts:
            # Get start date and target date
            start_date = datetime.datetime.strptime(goal.get('start_date', '2022-01-01'), "%Y-%m-%d")
            target_date = datetime.datetime.strptime(goal.get('target_date', '2023-01-01'), "%Y-%m-%d")
            current_date = datetime.datetime.now()
            
            # Calculate time progress
            total_days = (target_date - start_date).days
            elapsed_days = (current_date - start_date).days
            days_left = (target_date - current_date).days
            
            time_progress = (elapsed_days / total_days) * 100 if total_days > 0 else 100
            
            # Calculate amount progress
            target_amount = goal.get('target_amount', 0)
            current_amount = goal.get('current_amount', 0)
            
            amount_progress = (current_amount / target_amount) * 100 if target_amount > 0 else 100
            
            # Calculate if on track
            expected_progress = time_progress
            on_track = amount_progress >= expected_progress
            
            # Calculate projected completion
            if elapsed_days > 0 and current_amount > 0:
                daily_rate = current_amount / elapsed_days
                days_to_completion = (target_amount - current_amount) / daily_rate if daily_rate > 0 else float('inf')
                projected_completion = current_date + datetime.timedelta(days=days_to_completion)
                will_complete_on_time = projected_completion <= target_date
            else:
                days_to_completion = 0
                projected_completion = target_date
                will_complete_on_time = True
                
            # Add tracking data to goal
            goal.update({
                "time_progress": time_progress,
                "amount_progress": amount_progress,
                "on_track": on_track,
                "days_left": days_left,
                "projected_completion": projected_completion.strftime("%Y-%m-%d") if days_to_completion != float('inf') else "N/A",
                "will_complete_on_time": will_complete_on_time,
                "amount_remaining": target_amount - current_amount
            })
            
            # Generate recommendations based on progress
            if not on_track and days_left > 0:
                amount_needed = target_amount - current_amount
                daily_needed = amount_needed / days_left
                monthly_needed = daily_needed * 30
                
                goal["recommendation"] = f"To reach your goal on time, you need to save an additional {format_currency(monthly_needed)} per month."
            elif on_track:
                goal["recommendation"] = "You're on track to meet your goal! Keep up the good work."
            else:
                goal["recommendation"] = "This goal is behind schedule. Consider adjusting your target date or increasing your contributions."
        
        return goal_dicts
    
    def get_habit_improvement_plan(self) -> Dict[str, Any]:
        """
        Generate a personalized habit improvement plan based on spending behavior.
        
        Returns:
            Dict: Habit improvement plan
        """
        # Generate improvement plan using AI
        improvement_plan = ai_agents.generate_structured_response(
            f"Generate a personalized financial habit improvement plan for user {self.user_id} based on their spending behavior",
            system_prompt="You are a financial coach specializing in behavior change. Create a structured 30-day plan to help the user improve their financial habits."
        )
        
        # If AI fails, generate a basic plan
        if not improvement_plan:
            # Get spending behavior analysis
            behavior = self.analyze_spending_behavior()
            
            # Determine focus areas based on habits
            focus_areas = []
            
            for habit in behavior.get("spending_habits", []):
                if habit.get("impact") == "Negative":
                    focus_areas.append({
                        "area": habit.get("type"),
                        "reason": habit.get("description"),
                        "goal": habit.get("recommendation")
                    })
                    
            # Add default focus area if none found
            if not focus_areas:
                focus_areas.append({
                    "area": "Spending Awareness",
                    "reason": "Improving awareness is the foundation of better financial habits",
                    "goal": "Track all expenses daily and review weekly"
                })
            
            # Create weekly tasks
            weekly_tasks = [
                {
                    "week": 1,
                    "theme": "Awareness",
                    "tasks": [
                        "Track every expense, no matter how small",
                        "Review your spending at the end of each day",
                        "Identify triggers for impulse purchases",
                        "Create a simple budget for essential categories"
                    ]
                },
                {
                    "week": 2,
                    "theme": "Planning",
                    "tasks": [
                        "Plan meals and shopping lists in advance",
                        "Set specific savings goals for the month",
                        "Identify and cancel unused subscriptions",
                        "Schedule bill payments to avoid late fees"
                    ]
                },
                {
                    "week": 3,
                    "theme": "Implementation",
                    "tasks": [
                        "Implement a 24-hour rule for non-essential purchases",
                        "Use cash for discretionary spending",
                        "Try a no-spend day challenge",
                        "Automate savings transfers"
                    ]
                },
                {
                    "week": 4,
                    "theme": "Reinforcement",
                    "tasks": [
                        "Review your progress and celebrate improvements",
                        "Adjust your budget based on weeks 1-3",
                        "Identify one habit to continue focusing on",
                        "Set up a regular financial review session"
                    ]
                }
            ]
            
            # Create habit improvement plan
            improvement_plan = {
                "focus_areas": focus_areas,
                "duration": "30 days",
                "weekly_tasks": weekly_tasks,
                "daily_habits": [
                    {
                        "habit": "Track expenses",
                        "description": "Record every expense as it happens",
                        "difficulty": "Easy"
                    },
                    {
                        "habit": "Review spending",
                        "description": "Take 5 minutes each evening to review the day's spending",
                        "difficulty": "Easy"
                    },
                    {
                        "habit": "Ask before buying",
                        "description": "Before any non-essential purchase, ask: 'Do I need this? Can I afford it? Is it aligned with my goals?'",
                        "difficulty": "Medium"
                    },
                    {
                        "habit": "Gratitude practice",
                        "description": "Write down one thing you already own that you're grateful for",
                        "difficulty": "Easy"
                    }
                ],
                "success_metrics": [
                    "Reduced spending in top problem categories",
                    "Increased savings rate",
                    "Better awareness of spending patterns",
                    "Reduced financial stress"
                ]
            }
            
        return improvement_plan
        
    def get_educational_resources(self) -> List[Dict[str, str]]:
        """
        Get personalized educational resources based on user's financial situation.
        
        Returns:
            List[Dict]: Educational resources
        """
        # Generate educational resources using AI
        resources = ai_agents.generate_structured_response(
            f"Recommend personalized educational resources for user {self.user_id} based on their financial situation",
            system_prompt="You are a financial education expert. Recommend specific, relevant resources tailored to the user's needs."
        )
        
        # If AI fails, provide default resources
        if not isinstance(resources, list):
            # Basic categories of resources
            resources = [
                {
                    "title": "Budgeting Basics",
                    "description": "Learn the fundamentals of creating and sticking to a budget",
                    "resource_type": "Article",
                    "link": "https://www.investopedia.com/articles/personal-finance/062615/budgeting-basics.asp",
                    "difficulty": "Beginner"
                },
                {
                    "title": "Emergency Fund: What It Is and Why It Matters",
                    "description": "Understanding the importance of an emergency fund and how to build one",
                    "resource_type": "Article",
                    "link": "https://www.nerdwallet.com/article/banking/emergency-fund-why-it-matters",
                    "difficulty": "Beginner"
                },
                {
                    "title": "How to Get Out of Debt",
                    "description": "Step-by-step guide to becoming debt-free",
                    "resource_type": "Guide",
                    "link": "https://www.ramseysolutions.com/debt/how-to-get-out-of-debt",
                    "difficulty": "Intermediate"
                },
                {
                    "title": "Investing for Beginners",
                    "description": "Introduction to investing concepts and strategies",
                    "resource_type": "Course",
                    "link": "https://www.morningstar.com/start-investing",
                    "difficulty": "Beginner"
                },
                {
                    "title": "Retirement Planning: A Step by Step Guide",
                    "description": "Comprehensive guide to planning for retirement",
                    "resource_type": "Guide",
                    "link": "https://www.fidelity.com/retirement/planning/overview",
                    "difficulty": "Intermediate"
                },
                {
                    "title": "Understanding Credit Scores",
                    "description": "Learn how credit scores work and how to improve yours",
                    "resource_type": "Article",
                    "link": "https://www.experian.com/blogs/ask-experian/credit-education/score-basics/what-is-a-good-credit-score/",
                    "difficulty": "Beginner"
                },
                {
                    "title": "Tax-Efficient Investing",
                    "description": "Strategies to minimize tax impact on your investments",
                    "resource_type": "Article",
                    "link": "https://www.schwab.com/resource-center/insights/content/tax-planning-basics",
                    "difficulty": "Advanced"
                },
                {
                    "title": "Behavioral Finance: Why We Make Bad Financial Decisions",
                    "description": "Understanding psychological factors that affect financial decisions",
                    "resource_type": "Video",
                    "link": "https://www.youtube.com/watch?v=SQCPjh_I8tQ",
                    "difficulty": "Intermediate"
                }
            ]
            
        return resources


def financial_coaching_page(user_id: str):
    """
    Display the financial coaching page.
    
    Args:
        user_id (str): The ID of the current user
    """
    st.title("Personalized Financial Coaching")
    
    # Initialize financial coach
    coach = FinancialCoach(user_id)
    
    # Create tabs for different coaching features
    tabs = st.tabs(["Spending Behavior", "Personalized Tips", "Goal Tracking", "Habit Improvement", "Resources"])
    
    # Spending Behavior tab
    with tabs[0]:
        st.subheader("Your Spending Behavior Analysis")
        
        # Get behavior analysis
        behavior = coach.analyze_spending_behavior()
        
        # Display overall assessment
        assessment = behavior.get("overall_assessment", "")
        assessment_color = "green" if assessment == "Healthy" else "orange" if assessment == "Needs Improvement" else "red"
        
        st.markdown(f"""
        <div style="
            background-color: rgba(0,0,0,0.05); 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            text-align: center;
            border: 2px solid {assessment_color};">
            <h3>Overall Assessment: <span style="color: {assessment_color}">{assessment}</span></h3>
            <p>Savings Rate: {behavior.get("savings_rate", 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display top spending categories
        st.subheader("Top Spending Categories")
        
        if behavior.get("top_spending_categories"):
            top_categories = behavior.get("top_spending_categories")
            category_df = pd.DataFrame(top_categories)
            
            fig = px.pie(
                category_df,
                values="amount",
                names="category",
                title="Top Spending Categories",
                hole=0.4
            )
            
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:$,.2f} (%{percent})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display transaction distribution
        st.subheader("Transaction Distribution")
        
        if behavior.get("transaction_distribution"):
            dist = behavior.get("transaction_distribution")
            
            dist_df = pd.DataFrame([
                {"size": "Small (<$20)", "count": dist.get("small", 0)},
                {"size": "Medium ($20-$100)", "count": dist.get("medium", 0)},
                {"size": "Large (>$100)", "count": dist.get("large", 0)}
            ])
            
            fig = px.bar(
                dist_df,
                x="size",
                y="count",
                title="Transaction Size Distribution",
                color="size",
                color_discrete_map={
                    "Small (<$20)": "green",
                    "Medium ($20-$100)": "blue",
                    "Large (>$100)": "purple"
                }
            )
            
            fig.update_layout(showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display average transaction
            avg_txn = behavior.get("average_transaction", 0)
            st.metric("Average Transaction Amount", format_currency(avg_txn))
            
            # Display most active day
            if behavior.get("most_active_day") != "N/A":
                st.info(f"Your most active shopping day is **{behavior.get('most_active_day')}**.")
        
        # Display spending habits
        st.subheader("Spending Habits")
        
        if behavior.get("spending_habits"):
            for habit in behavior.get("spending_habits"):
                habit_type = habit.get("type", "")
                description = habit.get("description", "")
                impact = habit.get("impact", "Neutral")
                recommendation = habit.get("recommendation", "")
                
                impact_color = "green" if impact == "Positive" else "red" if impact == "Negative" else "blue"
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(0,0,0,0.05); 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-bottom: 10px;
                    border-left: 5px solid {impact_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">{habit_type}</h4>
                        <span style="background-color: {impact_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">{impact} Impact</span>
                    </div>
                    <p style="margin-top: 10px;">{description}</p>
                    <p style="margin-top: 5px; font-style: italic;"><b>Recommendation:</b> {recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Personalized Tips tab
    with tabs[1]:
        st.subheader("Personalized Financial Tips")
        
        # Get personalized tips
        tips = coach.get_personalized_tips()
        
        # Group tips by category
        categories = {}
        for tip in tips:
            category = tip.get("category", "General")
            if category not in categories:
                categories[category] = []
            categories[category].append(tip)
            
        # Display tips by category
        for category, category_tips in categories.items():
            st.markdown(f"### {category}")
            
            for tip in category_tips:
                title = tip.get("title", "")
                description = tip.get("description", "")
                action = tip.get("action", "")
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(0,0,0,0.05); 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 15px;
                    border-left: 5px solid #4F8BF9;">
                    <h4>{title}</h4>
                    <p>{description}</p>
                    <p style="margin-top: 10px; font-weight: bold;">Action Step: {action}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Goal Tracking tab
    with tabs[2]:
        st.subheader("Financial Goal Tracking")
        
        # Get goals with progress
        goals = coach.track_goal_progress()
        
        if not goals:
            st.info("You don't have any financial goals set up yet. Add goals in the Goals section to track your progress.")
        else:
            # Display each goal with progress
            for goal in goals:
                name = goal.get("name", "")
                target_amount = goal.get("target_amount", 0)
                current_amount = goal.get("current_amount", 0)
                amount_progress = goal.get("amount_progress", 0)
                time_progress = goal.get("time_progress", 0)
                on_track = goal.get("on_track", False)
                days_left = goal.get("days_left", 0)
                recommendation = goal.get("recommendation", "")
                
                status_color = "green" if on_track else "red"
                progress_color = "green" if amount_progress >= time_progress else "orange" if amount_progress >= time_progress * 0.8 else "red"
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(0,0,0,0.05); 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                    border-left: 5px solid {status_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0;">{name}</h3>
                        <span style="background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                            {on_track and "On Track" or "Behind Schedule"}
                        </span>
                    </div>
                    <div style="margin-top: 15px;">
                        <p><b>Progress:</b> {format_currency(current_amount)} of {format_currency(target_amount)} ({amount_progress:.1f}%)</p>
                        <div style="margin-top: 5px; background-color: #eee; height: 15px; border-radius: 5px; position: relative;">
                            <div style="width: {min(amount_progress, 100)}%; background-color: {progress_color}; height: 15px; border-radius: 5px;"></div>
                            <div style="position: absolute; top: -5px; left: {min(time_progress, 100)}%; width: 2px; height: 25px; background-color: black;"></div>
                        </div>
                        <p style="margin-top: 5px; font-size: 0.8em; text-align: right;">Expected progress: {time_progress:.1f}%</p>
                    </div>
                    <p style="margin-top: 10px;"><b>Days Remaining:</b> {days_left}</p>
                    <p style="margin-top: 5px; font-style: italic;">{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Create comparative goal chart
            st.subheader("Goal Comparison")
            
            goal_df = pd.DataFrame([
                {
                    "Goal": g.get("name", ""),
                    "Current": g.get("current_amount", 0),
                    "Target": g.get("target_amount", 0),
                    "Progress": g.get("amount_progress", 0),
                    "Expected": g.get("time_progress", 0),
                    "Status": "On Track" if g.get("on_track", False) else "Behind"
                } for g in goals
            ])
            
            # Add remaining amount
            goal_df["Remaining"] = goal_df["Target"] - goal_df["Current"]
            
            # Create stacked bar chart
            fig = go.Figure()
            
            # Add current amount bars
            fig.add_trace(go.Bar(
                x=goal_df["Goal"],
                y=goal_df["Current"],
                name="Current Amount",
                marker_color='green'
            ))
            
            # Add remaining amount bars
            fig.add_trace(go.Bar(
                x=goal_df["Goal"],
                y=goal_df["Remaining"],
                name="Remaining",
                marker_color='lightgray'
            ))
            
            # Update layout
            fig.update_layout(
                title="Progress Toward Goals",
                barmode='stack',
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Habit Improvement tab
    with tabs[3]:
        st.subheader("30-Day Financial Habit Improvement Plan")
        
        # Get habit improvement plan
        plan = coach.get_habit_improvement_plan()
        
        # Display focus areas
        st.markdown("### Focus Areas")
        
        for i, focus in enumerate(plan.get("focus_areas", [])):
            st.markdown(f"""
            <div style="
                background-color: rgba(0,0,0,0.05); 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 10px;
                border-left: 5px solid #4F8BF9;">
                <h4>{focus.get("area", "")}</h4>
                <p><b>Why:</b> {focus.get("reason", "")}</p>
                <p><b>Goal:</b> {focus.get("goal", "")}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display weekly plan
        st.markdown("### Weekly Plan")
        
        # Get weekly tasks
        weekly_tasks = plan.get("weekly_tasks", [])
        
        # Create tabs for each week if tasks exist
        if weekly_tasks:
            week_tabs = st.tabs([f"Week {week.get('week', i+1)}: {week.get('theme', '')}" 
                               for i, week in enumerate(weekly_tasks)])
            
            # Fill content for each week
            for i, week_tab in enumerate(week_tabs):
                if i < len(weekly_tasks):
                    week = weekly_tasks[i]
        else:
            st.info("No weekly tasks available in the improvement plan.")
            
            with week_tab:
                for j, task in enumerate(week.get("tasks", [])):
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(0,0,0,0.02); 
                        padding: 10px; 
                        border-radius: 5px; 
                        margin-bottom: 5px;
                        border-left: 3px solid #4F8BF9;">
                        <p><b>Task {j+1}:</b> {task}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display daily habits
        st.markdown("### Daily Habits")
        
        # Create columns for the habits
        cols = st.columns(2)
        
        # Display habits in columns
        for i, habit in enumerate(plan.get("daily_habits", [])):
            with cols[i % 2]:
                difficulty_color = "green" if habit.get("difficulty") == "Easy" else "orange" if habit.get("difficulty") == "Medium" else "red"
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(0,0,0,0.05); 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-bottom: 10px;
                    border-left: 3px solid {difficulty_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">{habit.get("habit", "")}</h4>
                        <span style="background-color: {difficulty_color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.7em;">
                            {habit.get("difficulty", "")}
                        </span>
                    </div>
                    <p style="margin-top: 8px;">{habit.get("description", "")}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display success metrics
        st.markdown("### Success Metrics")
        
        metrics_cols = st.columns(2)
        
        for i, metric in enumerate(plan.get("success_metrics", [])):
            with metrics_cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background-color: rgba(79, 139, 249, 0.1); 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin-bottom: 10px;
                    text-align: center;">
                    <p style="margin: 0;"><b>{metric}</b></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Resources tab
    with tabs[4]:
        st.subheader("Educational Resources")
        
        # Get educational resources
        resources = coach.get_educational_resources()
        
        # Group resources by difficulty
        difficulties = {
            "Beginner": [],
            "Intermediate": [],
            "Advanced": []
        }
        
        for resource in resources:
            difficulty = resource.get("difficulty", "Beginner")
            if difficulty in difficulties:
                difficulties[difficulty].append(resource)
        
        # Create tabs for difficulty levels
        difficulty_tabs = st.tabs(["Beginner", "Intermediate", "Advanced"])
        
        # Fill content for each difficulty level
        for i, (difficulty, diff_resources) in enumerate(difficulties.items()):
            with difficulty_tabs[i]:
                if not diff_resources:
                    st.info(f"No {difficulty.lower()} resources available at the moment.")
                    continue
                    
                # Group by resource type
                resource_types = {}
                for resource in diff_resources:
                    resource_type = resource.get("resource_type", "Other")
                    if resource_type not in resource_types:
                        resource_types[resource_type] = []
                    resource_types[resource_type].append(resource)
                
                # Display resources by type
                for resource_type, type_resources in resource_types.items():
                    st.markdown(f"#### {resource_type}s")
                    
                    for resource in type_resources:
                        title = resource.get("title", "")
                        description = resource.get("description", "")
                        link = resource.get("link", "#")
                        
                        st.markdown(f"""
                        <div style="
                            background-color: rgba(0,0,0,0.05); 
                            padding: 10px; 
                            border-radius: 5px; 
                            margin-bottom: 10px;">
                            <h4><a href="{link}" target="_blank">{title}</a></h4>
                            <p>{description}</p>
                        </div>
                        """, unsafe_allow_html=True)