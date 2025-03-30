import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import random
import streamlit as st
import openai

# Check for API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HAS_API_KEY = bool(OPENAI_API_KEY)

# Import AI functionality
import ai_agents
import vector_processing as vp
from fetch_agents import analyze_transaction, get_financial_insights, answer_query
import db_service as db
from utils import get_date_range, get_recommendations

# Import quantitative finance module
import quant_finance as qf

class FinanceAgent:
    """
    Finance Agent class for AI-powered financial analysis and insights.
    """

    def __init__(self):
        """Initialize the finance agent."""
        self.user_id = None
        self.openai_client = None
        try:
            self.setup_openai()
        except Exception as e:
            print(f"AI service initialization error: {str(e)}")
            st.warning("Some AI features may be unavailable. Please check your OpenAI API key configuration.")

    def setup_openai(self):
        """Set up OpenAI client."""
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            self.openai_client = openai.Client(api_key=OPENAI_API_KEY)
        else:
            raise Exception("OpenAI API key not found")

    def set_user(self, user_id: str):
        """Set the user ID."""
        self.user_id = user_id

    async def analyze_transaction(self, transaction_id: int) -> Dict[str, Any]:
        """
        Analyze a single transaction for insights.

        Args:
            transaction_id: Transaction ID

        Returns:
            Analysis results
        """
        if not self.user_id:
            return {"error": "User not set"}

        # Use Fetch AI agent to analyze
        return await analyze_transaction(self.user_id, transaction_id)

    def categorize_transaction(self, description: str) -> str:
        """
        Categorize a transaction based on its description.

        Args:
            description: Transaction description

        Returns:
            Category string
        """
        try:
            return ai_agents.get_transaction_categorization(description)
        except Exception as e:
            print(f"Error categorizing transaction: {str(e)}")
            # In case of errors, provide a fallback category
            return "Other"

    async def get_insights(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get financial insights.

        Args:
            time_period: Time period (week, month, year, all)

        Returns:
            Insights and recommendations
        """
        if not self.user_id:
            return {"error": "User not set"}

        try:
            # Use Fetch AI agent for insights
            return await get_financial_insights(self.user_id, time_period)
        except Exception as e:
            error_msg = str(e)
            print(f"Error getting insights: {error_msg}")

            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                return {
                    "error": "API_QUOTA_EXCEEDED",
                    "message": "AI insights are currently unavailable due to OpenAI API quota limitations. Please check your OpenAI API plan, billing details, or try again later."
                }
            return {
                "error": "AI_UNAVAILABLE",
                "message": "AI insights are currently unavailable. Please try again later."
            }

    async def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a financial question using AI.

        Args:
            query: Natural language query

        Returns:
            Answer and relevant transactions
        """
        if not self.user_id:
            return {"error": "User not set"}

        try:
            # Use Fetch AI agent to answer
            return await answer_query(self.user_id, query)
        except Exception as e:
            error_msg = str(e)
            print(f"Error answering question: {error_msg}")

            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                return {
                    "error": "API_QUOTA_EXCEEDED",
                    "message": "AI features are currently unavailable due to OpenAI API quota limitations. Please check your OpenAI API plan, billing details, or try again later.",
                    "answer": "I'm sorry, but I can't process your question right now due to API limitations. Please try again later."
                }
            return {
                "error": "AI_UNAVAILABLE",
                "message": "AI features are currently unavailable. Please try again later.",
                "answer": "I'm sorry, but I can't process your question right now. Please try again later."
            }

    def search_transactions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search transactions using semantic similarity.

        Args:
            query: Natural language query
            limit: Maximum number of results

        Returns:
            List of matching transactions
        """
        if not self.user_id:
            return []

        # Use vector search
        return vp.search_transactions(self.user_id, query, limit)

    def get_spending_anomalies(self) -> List[Dict[str, Any]]:
        """
        Identify spending anomalies.

        Returns:
            List of anomalies with transaction details
        """
        if not self.user_id:
            return []

        # Get user transactions
        transactions = db.get_transactions(self.user_id)

        # Convert to dictionary format
        txn_dicts = [t.to_dict() for t in transactions]

        # Get anomalies
        return ai_agents.get_spending_anomalies(txn_dicts)

    def get_budget_status(self) -> List[Dict[str, Any]]:
        """
        Check budget status and provide warnings.

        Returns:
            List of budget statuses with warnings
        """
        if not self.user_id:
            return []

        # Get user budgets
        budgets = db.get_budgets(self.user_id)

        # Get category spending
        start_date, end_date = get_date_range("month")
        spending = db.get_category_spending(self.user_id, start_date, end_date)

        # Convert to dictionary format
        budget_dicts = [b.to_dict() for b in budgets]

        # Check budget status
        return ai_agents.get_budget_status(budget_dicts, spending)

    def get_financial_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive financial summary.

        Returns:
            Summary with metrics and AI-generated text
        """
        if not self.user_id:
            return {
                "total_income": 0,
                "total_expenses": 0,
                "net_cash_flow": 0,
                "top_expense_categories": [],
                "summary_text": "No user selected."
            }

        # Get user transactions
        transactions = db.get_transactions(self.user_id)

        # Convert to dictionary format
        txn_dicts = [t.to_dict() for t in transactions]

        try:
            # Get summary with AI
            return ai_agents.get_financial_summary(txn_dicts)
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating financial summary: {error_msg}")

            # Calculate basic metrics without AI
            total_income = sum(t.get("amount", 0) for t in txn_dicts if t.get("type", "").lower() == "income")
            total_expenses = sum(t.get("amount", 0) for t in txn_dicts if t.get("type", "").lower() == "expense")
            net_cash_flow = total_income - total_expenses

            # Calculate top expense categories
            category_expenses = {}
            for txn in txn_dicts:
                if txn.get("type", "").lower() == "expense":
                    category = txn.get("category", "Other")
                    if category not in category_expenses:
                        category_expenses[category] = 0
                    category_expenses[category] += txn.get("amount", 0)

            top_categories = sorted(
                category_expenses.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5

            # Return basic summary without AI-generated text
            fallback_message = ""
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                fallback_message = "AI-powered insights are currently unavailable due to API quota limitations. Basic financial summary is shown instead."
            else:
                fallback_message = "AI-powered insights are currently unavailable. Basic financial summary is shown instead."

            return {
                "total_income": total_income,
                "total_expenses": total_expenses,
                "net_cash_flow": net_cash_flow,
                "top_expense_categories": [{"category": cat, "amount": amt} for cat, amt in top_categories],
                "summary_text": fallback_message
            }

    def predict_transactions(self) -> List[Dict[str, Any]]:
        """
        Predict future transactions based on historical patterns.

        Returns:
            List of predicted transactions
        """
        if not self.user_id:
            return []

        # Get user transactions
        transactions = db.get_transactions(self.user_id)

        # Convert to dictionary format
        txn_dicts = [t.to_dict() for t in transactions]

        try:
            # Get predictions using AI
            return ai_agents.get_transaction_predictions(txn_dicts)
        except Exception as e:
            error_msg = str(e)
            print(f"Error predicting transactions: {error_msg}")

            # Create a simplified prediction algorithm as fallback
            predictions = []

            if txn_dicts:
                # Find recurring transactions (same description, category and type)
                recurring = {}
                for txn in txn_dicts:
                    desc = txn.get("description", "").lower()
                    amount = txn.get("amount", 0)
                    category = txn.get("category", "Other")
                    txn_type = txn.get("type", "expense")

                    key = f"{desc}-{category}-{txn_type}"

                    if key not in recurring:
                        recurring[key] = {
                            "transactions": [],
                            "description": txn.get("description", ""),
                            "category": category,
                            "type": txn_type,
                            "count": 0
                        }

                    recurring[key]["transactions"].append(txn)
                    recurring[key]["count"] += 1

                # Only use transactions that appear at least twice
                recurring = {k: v for k, v in recurring.items() if v["count"] >= 2}

                # For each recurring transaction, predict the next occurrence
                today = datetime.datetime.now()
                for key, data in recurring.items():
                    txns = data["transactions"]

                    # Only process if we have enough data
                    if len(txns) >= 2:
                        # Sort by date
                        txns = sorted(txns, key=lambda x: x.get("date", ""))

                        # Calculate average amount
                        amounts = [t.get("amount", 0) for t in txns]
                        avg_amount = sum(amounts) / len(amounts)

                        # Calculate average days between transactions
                        dates = []
                        for t in txns:
                            date_str = t.get("date", "")
                            if date_str:
                                try:
                                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                                    dates.append(date_obj)
                                except:
                                    pass

                        if len(dates) >= 2:
                            # Calculate day differences
                            diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                            avg_diff = sum(diffs) / len(diffs)

                            # If seems to be monthly (25-35 days)
                            if 25 <= avg_diff <= 35:
                                next_date = dates[-1] + datetime.timedelta(days=30)

                                # Only add if in the future
                                if next_date > today:
                                    predictions.append({
                                        "description": data["description"],
                                        "category": data["category"],
                                        "type": data["type"],
                                        "amount": avg_amount,
                                        "predicted_date": next_date.strftime("%Y-%m-%d"),
                                        "confidence": 0.6  # Lower confidence for fallback algorithm
                                    })

            return sorted(predictions, key=lambda x: x.get("predicted_date", ""))

    def run_monte_carlo_simulation(self, months_ahead: int = 6, simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for spending trend prediction.

        Args:
            months_ahead: Number of months to predict
            simulations: Number of simulations to run

        Returns:
            Simulation results
        """
        if not self.user_id:
            return {"error": "User not set"}

        # Set user ID on simulator
        qf.monte_carlo.set_user(self.user_id)

        # Run simulation
        return qf.monte_carlo.simulate_spending(months_ahead, simulations)

    def get_financial_risk_assessment(self) -> Dict[str, Any]:
        """
        Perform a comprehensive financial risk assessment.

        Returns:
            Risk assessment results
        """
        if not self.user_id:
            return {"error": "User not set"}

        # Set user ID on risk model
        qf.risk_model.set_user(self.user_id)

        # Run assessment
        return qf.risk_model.assess_financial_health()

    def optimize_budget(self) -> Dict[str, Any]:
        """
        Optimize budget allocation based on spending patterns.

        Returns:
            Optimized budget recommendations
        """
        if not self.user_id:
            return {"error": "User not set"}

        # Set user ID on budget optimizer
        qf.budget_optimizer.set_user(self.user_id)

        # Run optimization
        return qf.budget_optimizer.optimize_budget_allocation()

    def optimize_portfolio(self, risk_tolerance: float = 0.5, time_horizon: int = 5) -> Dict[str, Any]:
        """
        Optimize savings allocation across different instruments.

        Args:
            risk_tolerance: Risk tolerance level (0-1)
            time_horizon: Investment time horizon in years

        Returns:
            Optimized allocation recommendations
        """
        if not self.user_id:
            return {"error": "User not set"}

        # Set user ID on portfolio optimizer
        qf.portfolio_optimizer.set_user(self.user_id)

        # Run optimization
        return qf.portfolio_optimizer.optimize_savings_allocation(risk_tolerance, time_horizon)

    def get_financial_health_score(self) -> Dict[str, Any]:
        """
        Calculate a financial health score.
        Uses both traditional metrics and quantitative risk models.

        Returns:
            Financial health score and breakdown
        """
        if not self.user_id:
            return {"score": 0, "details": {}, "message": "No user selected."}

        try:
            # Try to get advanced risk model assessment
            qf.risk_model.set_user(self.user_id)
            risk_assessment = qf.risk_model.assess_financial_health()

            # If successful, return the more advanced assessment
            if "error" not in risk_assessment:
                return {
                    "score": risk_assessment.get("financial_health_score", 0),
                    "details": risk_assessment.get("metrics", {}),
                    "risk_level": risk_assessment.get("risk_level", "Unknown"),
                    "recommendations": risk_assessment.get("recommendations", []),
                    "message": f"Your financial health score is {risk_assessment.get('financial_health_score', 0):.1f}/100. Risk level: {risk_assessment.get('risk_level', 'Unknown')}."
                }
        except Exception as e:
            print(f"Advanced risk model error: {str(e)}. Falling back to basic scoring.")
            # Continue with basic scoring below

        # Get user data for basic scoring
        transactions = db.get_transactions(self.user_id)
        txn_dicts = [t.to_dict() for t in transactions]

        budgets = db.get_budgets(self.user_id)
        budget_dicts = [b.to_dict() for b in budgets]

        goals = db.get_goals(self.user_id)
        goal_dicts = [g.to_dict() for g in goals]

        # Calculate metrics
        total_income = sum(t.get("amount", 0) for t in txn_dicts if t.get("type", "").lower() == "income")
        total_expense = sum(t.get("amount", 0) for t in txn_dicts if t.get("type", "").lower() == "expense")

        # Calculate scores for different components
        scores = {}

        # 1. Income-to-expense ratio (20 points)
        if total_income > 0:
            income_ratio = min(2.0, total_income / max(1, total_expense))
            scores["income_ratio"] = min(20, int(income_ratio * 10))
        else:
            scores["income_ratio"] = 0

        # 2. Budget adherence (30 points)
        if budget_dicts:
            # Get category spending
            start_date, end_date = get_date_range("month")
            spending = db.get_category_spending(self.user_id, start_date, end_date)

            # Check budgets
            budget_statuses = ai_agents.get_budget_status(budget_dicts, spending)

            # Calculate adherence score
            if budget_statuses:
                over_budget_count = sum(1 for b in budget_statuses if b.get("warning_level") == "critical")
                warning_count = sum(1 for b in budget_statuses if b.get("warning_level") == "warning")

                budget_factor = 1.0 - (over_budget_count * 0.2) - (warning_count * 0.1)
                scores["budget_adherence"] = max(0, min(30, int(budget_factor * 30)))
            else:
                scores["budget_adherence"] = 15  # Neutral if no budgets
        else:
            scores["budget_adherence"] = 10  # Lower score if no budgets set

        # 3. Savings rate (20 points)
        if total_income > 0:
            savings_rate = (total_income - total_expense) / total_income
            scores["savings_rate"] = min(20, int(savings_rate * 100))
        else:
            scores["savings_rate"] = 0

        # 4. Progress towards goals (15 points)
        if goal_dicts:
            goal_progress = sum(g.get("current_amount", 0) / max(1, g.get("target_amount", 1)) for g in goal_dicts)
            avg_progress = goal_progress / len(goal_dicts) if goal_dicts else 0
            scores["goal_progress"] = min(15, int(avg_progress * 15))
        else:
            scores["goal_progress"] = 5  # Lower score if no goals set

        # 5. Transaction consistency (15 points)
        # Simple proxy: check if transactions exist across multiple months
        if txn_dicts:
            months = set()
            for t in txn_dicts:
                date = t.get("date", "")
                if date:
                    month = date[:7]  # YYYY-MM
                    months.add(month)

            consistency_score = min(15, len(months) * 3)
            scores["consistency"] = consistency_score
        else:
            scores["consistency"] = 0

        # Calculate total score
        total_score = sum(scores.values())

        # Generate message based on score
        if total_score >= 80:
            message = "Excellent financial health! You're managing your finances very well."
        elif total_score >= 60:
            message = "Good financial health. There are some areas for improvement."
        elif total_score >= 40:
            message = "Fair financial health. Consider addressing the lower-scoring areas."
        else:
            message = "Needs improvement. Focus on budgeting and increasing income or reducing expenses."

        # Generate recommendations
        recommendations = []
        if scores.get("income_ratio", 0) < 10:
            recommendations.append("Focus on increasing income or reducing expenses to improve your cash flow.")
        if scores.get("budget_adherence", 0) < 15:
            recommendations.append("Review your budget categories and try to stay within your spending limits.")
        if scores.get("savings_rate", 0) < 10:
            recommendations.append("Try to increase your savings rate to at least 10% of your income.")
        if scores.get("goal_progress", 0) < 7:
            recommendations.append("Make regular contributions toward your financial goals.")

        return {
            "score": total_score,
            "details": scores,
            "message": message,
            "recommendations": recommendations
        }

    def generate_assistant_message(self) -> str:
        """
        Generate a personalized assistant message based on user's financial data.

        Returns:
            Personalized message string
        """
        if not self.user_id:
            return "Welcome to AI Finance Tracker. Please log in to get personalized assistance."

        # Get user data
        user = db.get_user_by_id(self.user_id)
        if not user:
            return "Welcome to AI Finance Tracker. Please log in to get personalized assistance."

        # Get current time
        now = datetime.datetime.now()
        hour = now.hour

        # Get time-appropriate greeting
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        # Basic message
        message = f"{greeting}, {user.name}! "

        # Add personalized content based on financial data
        try:
            # Get financial summary
            summary = self.get_financial_summary()

            # Get budget warnings
            budget_status = self.get_budget_status()
            critical_budgets = [b for b in budget_status if b and b.get("warning_level") == "critical"]

            # Get upcoming predicted transactions
            predictions = self.predict_transactions()
            next_week = (now + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            upcoming = []
            for p in predictions:
                pred_date = p.get("predicted_date")
                if pred_date and isinstance(pred_date, str) and pred_date <= next_week:
                    upcoming.append(p)

            # Construct message
            if critical_budgets:
                message += f"You have {len(critical_budgets)} budget categories that are over their limits. "

            if summary.get("net_cash_flow", 0) < 0:
                message += "Your expenses are currently exceeding your income this period. "
            elif summary.get("net_cash_flow", 0) > 0:
                message += "You're keeping expenses below income - great job! "

            if upcoming:
                message += f"You have {len(upcoming)} predicted transactions coming up in the next week. "

            # Add a random tip
            tips = [
                "Remember to regularly check your progress toward financial goals.",
                "Consider reviewing your budget categories to ensure they're aligned with your priorities.",
                "Looking for ways to increase your savings rate can significantly impact long-term wealth.",
                "Tracking everyday expenses helps identify opportunities to reduce spending.",
                "Setting up automatic transfers to savings on payday can help build wealth effortlessly."
            ]

            message += random.choice(tips)

        except Exception as e:
            # Fallback to generic message
            message += "Welcome back to your financial dashboard. Let me know if you need any assistance today."

        return message