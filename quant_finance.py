"""
Quantitative Finance module for financial analysis and optimization.

This module implements various quantitative finance techniques including:
- Monte Carlo simulations for spending trend prediction
- Risk models for financial health assessment
- Time-series forecasting for expenditure prediction
- Optimized budget allocation using portfolio theory concepts
"""

import os
import json
import datetime
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional

# For error handling
import traceback

# AI integration
import ai_agents

# Internal modules
import db_service as db


class MonteCarloSimulator:
    """
    Monte Carlo simulation for spending trend prediction and financial planning.
    Uses historical spending patterns to predict future financial states.
    """
    
    def __init__(self, user_id: str = None):
        """Initialize with optional user ID."""
        self.user_id = user_id
    
    def set_user(self, user_id: str):
        """Set the user ID."""
        self.user_id = user_id
    
    def simulate_spending(self, 
                         months_ahead: int = 6, 
                         simulations: int = 1000) -> Dict[str, Any]:
        """
        Simulate future spending based on historical patterns.
        
        Args:
            months_ahead: Number of months to predict
            simulations: Number of Monte Carlo simulations to run
            
        Returns:
            Dict with simulation results
        """
        if not self.user_id:
            return {"error": "User not set"}
        
        try:
            # Get transaction history
            transactions = db.get_transactions(self.user_id)
            if not transactions:
                return {
                    "error": "Insufficient data",
                    "message": "Not enough transaction history for simulation"
                }
            
            # Convert to list of dicts if needed
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
            
            # Extract income and expense data by month
            monthly_income = {}
            monthly_expenses = {}
            
            for txn in txn_dicts:
                date_str = txn.get('date', '')
                if not date_str:
                    continue
                
                # Parse date
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    month_key = f"{date.year}-{date.month:02d}"
                    
                    amount = txn.get('amount', 0)
                    txn_type = txn.get('type', '').lower()
                    
                    if txn_type == 'income':
                        if month_key not in monthly_income:
                            monthly_income[month_key] = 0
                        monthly_income[month_key] += amount
                    elif txn_type == 'expense':
                        if month_key not in monthly_expenses:
                            monthly_expenses[month_key] = 0
                        monthly_expenses[month_key] += amount
                except:
                    continue
            
            # Calculate statistics for income and expenses
            if not monthly_income or not monthly_expenses:
                return {
                    "error": "Insufficient data",
                    "message": "Not enough income or expense data for simulation"
                }
            
            # Calculate mean and standard deviation of monthly income and expenses
            income_values = list(monthly_income.values())
            expense_values = list(monthly_expenses.values())
            
            income_mean = np.mean(income_values)
            income_std = max(np.std(income_values), income_mean * 0.05)  # Ensure some variability
            
            expense_mean = np.mean(expense_values)
            expense_std = max(np.std(expense_values), expense_mean * 0.05)  # Ensure some variability
            
            # Get current balances (simplified - use the difference between all income and expenses)
            current_balance = sum(income_values) - sum(expense_values)
            
            # Run simulations
            simulation_results = []
            for _ in range(simulations):
                balance = current_balance
                monthly_balances = [balance]
                
                for month in range(months_ahead):
                    # Simulate income (with randomness based on historical variability)
                    simulated_income = max(0, np.random.normal(income_mean, income_std))
                    
                    # Simulate expenses (with randomness based on historical variability)
                    simulated_expense = max(0, np.random.normal(expense_mean, expense_std))
                    
                    # Update balance
                    balance += simulated_income - simulated_expense
                    monthly_balances.append(balance)
                
                simulation_results.append(monthly_balances)
            
            # Calculate percentiles for confidence intervals
            percentiles = {}
            for month in range(months_ahead + 1):
                month_values = [simulation[month] for simulation in simulation_results]
                percentiles[month] = {
                    "median": np.percentile(month_values, 50),
                    "p10": np.percentile(month_values, 10),  # 10th percentile (pessimistic)
                    "p90": np.percentile(month_values, 90)   # 90th percentile (optimistic)
                }
            
            # Generate prediction dates
            today = datetime.datetime.now()
            prediction_dates = []
            for month in range(months_ahead + 1):
                future_date = today + datetime.timedelta(days=30 * month)
                prediction_dates.append(future_date.strftime("%Y-%m-%d"))
            
            # Calculate probability of negative balance
            final_month_values = [simulation[-1] for simulation in simulation_results]
            negative_balance_probability = sum(1 for b in final_month_values if b < 0) / len(final_month_values)
            
            # Create result
            result = {
                "current_balance": current_balance,
                "forecast_months": months_ahead,
                "simulation_count": simulations,
                "prediction_dates": prediction_dates,
                "percentiles": percentiles,
                "negative_balance_probability": negative_balance_probability,
                "statistics": {
                    "avg_monthly_income": income_mean,
                    "avg_monthly_expenses": expense_mean,
                    "income_volatility": income_std / income_mean if income_mean > 0 else 0,
                    "expense_volatility": expense_std / expense_mean if expense_mean > 0 else 0
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {str(e)}")
            traceback.print_exc()
            return {
                "error": "Simulation error",
                "message": str(e)
            }


class FinancialRiskModel:
    """
    Quantitative risk models for financial health assessment.
    """
    
    def __init__(self, user_id: str = None):
        """Initialize with optional user ID."""
        self.user_id = user_id
    
    def set_user(self, user_id: str):
        """Set the user ID."""
        self.user_id = user_id
    
    def assess_financial_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive assessment of financial health.
        
        Returns:
            Dict with financial health metrics and risk scores
        """
        if not self.user_id:
            return {"error": "User not set"}
        
        try:
            # Get transaction history
            transactions = db.get_transactions(self.user_id)
            if not transactions:
                return {
                    "error": "Insufficient data",
                    "message": "Not enough transaction history for assessment"
                }
            
            # Convert to list of dicts if needed
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
            
            # Get budgets
            budgets = db.get_budgets(self.user_id)
            budget_dicts = [b.to_dict() if hasattr(b, 'to_dict') else b for b in budgets]
            
            # Calculate financial metrics
            # 1. Income stability
            income_stability = self._calculate_income_stability(txn_dicts)
            
            # 2. Expense volatility
            expense_volatility = self._calculate_expense_volatility(txn_dicts)
            
            # 3. Savings rate
            savings_rate = self._calculate_savings_rate(txn_dicts)
            
            # 4. Emergency fund ratio
            emergency_fund = self._calculate_emergency_fund_ratio(txn_dicts)
            
            # 5. Debt-to-income ratio
            debt_to_income = self._calculate_debt_to_income(txn_dicts)
            
            # 6. Budget adherence
            budget_adherence = self._calculate_budget_adherence(txn_dicts, budget_dicts)
            
            # 7. Financial stress index
            financial_stress = self._calculate_financial_stress_index(
                income_stability, expense_volatility, savings_rate, 
                emergency_fund, debt_to_income, budget_adherence
            )
            
            # 8. Overall financial health score (0-100)
            financial_health = self._calculate_financial_health_score(
                income_stability, expense_volatility, savings_rate,
                emergency_fund, debt_to_income, budget_adherence
            )
            
            # Determine risk level
            risk_level = "Low"
            if financial_health < 50:
                risk_level = "High"
            elif financial_health < 70:
                risk_level = "Medium"
                
            # Create result
            result = {
                "financial_health_score": financial_health,
                "risk_level": risk_level,
                "metrics": {
                    "income_stability": income_stability,
                    "expense_volatility": expense_volatility,
                    "savings_rate": savings_rate,
                    "emergency_fund_ratio": emergency_fund,
                    "debt_to_income_ratio": debt_to_income,
                    "budget_adherence": budget_adherence,
                    "financial_stress_index": financial_stress
                },
                "recommendations": self._generate_recommendations(
                    income_stability, expense_volatility, savings_rate,
                    emergency_fund, debt_to_income, budget_adherence
                )
            }
            
            return result
            
        except Exception as e:
            print(f"Error in financial health assessment: {str(e)}")
            traceback.print_exc()
            return {
                "error": "Assessment error",
                "message": str(e)
            }
    
    def _calculate_income_stability(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate income stability score (0-1)."""
        # Extract income by month
        monthly_income = {}
        
        for txn in transactions:
            if txn.get('type', '').lower() == 'income':
                date_str = txn.get('date', '')
                if not date_str:
                    continue
                
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    month_key = f"{date.year}-{date.month:02d}"
                    
                    if month_key not in monthly_income:
                        monthly_income[month_key] = 0
                    
                    monthly_income[month_key] += txn.get('amount', 0)
                except:
                    continue
        
        # Insufficient data
        if len(monthly_income) < 2:
            return 0.5  # Neutral score
        
        # Calculate coefficient of variation (lower is better)
        income_values = list(monthly_income.values())
        mean = np.mean(income_values)
        std = np.std(income_values)
        
        # Avoid division by zero
        if mean == 0:
            return 0
            
        cv = std / mean
        
        # Convert to 0-1 scale (higher is better)
        # CV of 0 (perfect stability) = 1.0 score
        # CV of 0.5 or higher = 0.0 score
        stability = max(0, min(1, 1 - (cv / 0.5)))
        
        return stability
    
    def _calculate_expense_volatility(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate expense volatility score (0-1, lower is better)."""
        # Extract expenses by month
        monthly_expenses = {}
        
        for txn in transactions:
            if txn.get('type', '').lower() == 'expense':
                date_str = txn.get('date', '')
                if not date_str:
                    continue
                
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    month_key = f"{date.year}-{date.month:02d}"
                    
                    if month_key not in monthly_expenses:
                        monthly_expenses[month_key] = 0
                    
                    monthly_expenses[month_key] += txn.get('amount', 0)
                except:
                    continue
        
        # Insufficient data
        if len(monthly_expenses) < 2:
            return 0.5  # Neutral score
        
        # Calculate coefficient of variation
        expense_values = list(monthly_expenses.values())
        mean = np.mean(expense_values)
        std = np.std(expense_values)
        
        # Avoid division by zero
        if mean == 0:
            return 0
            
        cv = std / mean
        
        # Convert to 0-1 scale (lower is better)
        # CV of 0 (perfect stability) = 0.0 score
        # CV of 0.5 or higher = 1.0 score
        volatility = min(1, cv / 0.5)
        
        return volatility
    
    def _calculate_savings_rate(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate savings rate (0-1)."""
        total_income = sum(txn.get('amount', 0) for txn in transactions if txn.get('type', '').lower() == 'income')
        total_expenses = sum(txn.get('amount', 0) for txn in transactions if txn.get('type', '').lower() == 'expense')
        
        if total_income == 0:
            return 0
        
        savings_rate = max(0, min(1, (total_income - total_expenses) / total_income))
        return savings_rate
    
    def _calculate_emergency_fund_ratio(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate emergency fund ratio (0-1)."""
        # Estimate average monthly expenses
        monthly_expenses = {}
        
        for txn in transactions:
            if txn.get('type', '').lower() == 'expense':
                date_str = txn.get('date', '')
                if not date_str:
                    continue
                
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    month_key = f"{date.year}-{date.month:02d}"
                    
                    if month_key not in monthly_expenses:
                        monthly_expenses[month_key] = 0
                    
                    monthly_expenses[month_key] += txn.get('amount', 0)
                except:
                    continue
        
        if not monthly_expenses:
            return 0.5  # Neutral score
        
        # Calculate average monthly expense
        avg_monthly_expense = np.mean(list(monthly_expenses.values()))
        
        # Calculate current balance as emergency fund
        total_income = sum(txn.get('amount', 0) for txn in transactions if txn.get('type', '').lower() == 'income')
        total_expenses = sum(txn.get('amount', 0) for txn in transactions if txn.get('type', '').lower() == 'expense')
        current_balance = total_income - total_expenses
        
        if avg_monthly_expense == 0:
            return 1.0  # Best score (no expenses)
        
        # Calculate months of expenses covered by emergency fund
        months_covered = current_balance / avg_monthly_expense
        
        # Convert to 0-1 scale
        # 0 months = 0.0
        # 6+ months = 1.0 (recommended emergency fund size)
        ratio = max(0, min(1, months_covered / 6))
        
        return ratio
    
    def _calculate_debt_to_income(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate debt-to-income ratio (0-1, lower is better)."""
        # In a real implementation, we would need debt information
        # For now, we'll make a simple estimate based on transaction descriptions
        
        monthly_income = {}
        debt_payments = {}
        
        for txn in transactions:
            date_str = txn.get('date', '')
            if not date_str:
                continue
            
            try:
                date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"
                
                # Track income
                if txn.get('type', '').lower() == 'income':
                    if month_key not in monthly_income:
                        monthly_income[month_key] = 0
                    monthly_income[month_key] += txn.get('amount', 0)
                
                # Identify potential debt payments
                if txn.get('type', '').lower() == 'expense':
                    description = txn.get('description', '').lower()
                    category = txn.get('category', '').lower()
                    
                    debt_keywords = ['loan', 'mortgage', 'debt', 'credit card', 'payment']
                    is_debt_payment = any(kw in description or kw in category for kw in debt_keywords)
                    
                    if is_debt_payment:
                        if month_key not in debt_payments:
                            debt_payments[month_key] = 0
                        debt_payments[month_key] += txn.get('amount', 0)
            except:
                continue
        
        # Calculate average DTI
        dti_ratios = []
        for month in monthly_income:
            if month in debt_payments and monthly_income[month] > 0:
                dti = debt_payments[month] / monthly_income[month]
                dti_ratios.append(dti)
        
        if not dti_ratios:
            return 0.1  # Default low value if no debt identified
        
        avg_dti = np.mean(dti_ratios)
        
        # Convert to 0-1 scale (lower is better)
        # DTI of 0% = 0.0 score (best)
        # DTI of 40% or higher = 1.0 score (worst)
        dti_score = min(1, avg_dti / 0.4)
        
        return dti_score
    
    def _calculate_budget_adherence(self, transactions: List[Dict[str, Any]], budgets: List[Dict[str, Any]]) -> float:
        """Calculate budget adherence score (0-1)."""
        if not budgets:
            return 0.5  # Neutral score (no budgets)
        
        # Get all expense categories
        categories = list(set(b.get('category', '') for b in budgets))
        
        # Skip if no budget categories
        if not categories:
            return 0.5
        
        # Calculate spending for each category
        category_spending = {}
        for category in categories:
            category_spending[category] = sum(
                txn.get('amount', 0) 
                for txn in transactions 
                if txn.get('category', '') == category and txn.get('type', '').lower() == 'expense'
            )
        
        # Calculate budget adherence for each category
        adherence_scores = []
        for budget in budgets:
            category = budget.get('category', '')
            limit = budget.get('amount', 0)
            
            if category in category_spending and limit > 0:
                spent = category_spending[category]
                ratio = spent / limit
                
                # Perfect adherence: spending exactly at budget
                # Under budget: slightly less optimal (not using full budget)
                # Over budget: worse (exceeding budget)
                if ratio <= 1:
                    # Under or at budget
                    # Score from 0.8 to 1.0 (best at exactly 100% of budget)
                    score = 0.8 + (0.2 * ratio)
                else:
                    # Over budget
                    # Score from 0.0 to 0.8 (approaches 0 as overspending increases)
                    score = max(0, 0.8 - (0.4 * (ratio - 1)))
                
                adherence_scores.append(score)
        
        if not adherence_scores:
            return 0.5
        
        # Return average adherence score
        return np.mean(adherence_scores)
    
    def _calculate_financial_stress_index(self, 
                                         income_stability: float,
                                         expense_volatility: float,
                                         savings_rate: float,
                                         emergency_fund: float,
                                         debt_to_income: float,
                                         budget_adherence: float) -> float:
        """Calculate financial stress index (0-1, higher is worse)."""
        # Weighted average of metrics (some inverted so higher always = more stress)
        weights = {
            'income_stability': 0.20,
            'expense_volatility': 0.15,
            'savings_rate': 0.20,
            'emergency_fund': 0.15,
            'debt_to_income': 0.20,
            'budget_adherence': 0.10
        }
        
        stress = (
            weights['income_stability'] * (1 - income_stability) +
            weights['expense_volatility'] * expense_volatility +
            weights['savings_rate'] * (1 - savings_rate) +
            weights['emergency_fund'] * (1 - emergency_fund) +
            weights['debt_to_income'] * debt_to_income +
            weights['budget_adherence'] * (1 - budget_adherence)
        )
        
        return stress
    
    def _calculate_financial_health_score(self,
                                         income_stability: float,
                                         expense_volatility: float,
                                         savings_rate: float,
                                         emergency_fund: float,
                                         debt_to_income: float,
                                         budget_adherence: float) -> float:
        """Calculate overall financial health score (0-100)."""
        # Inverse of stress index, scaled to 0-100
        stress = self._calculate_financial_stress_index(
            income_stability, expense_volatility, savings_rate,
            emergency_fund, debt_to_income, budget_adherence
        )
        
        score = (1 - stress) * 100
        return score
    
    def _generate_recommendations(self,
                                 income_stability: float,
                                 expense_volatility: float,
                                 savings_rate: float,
                                 emergency_fund: float,
                                 debt_to_income: float,
                                 budget_adherence: float) -> List[str]:
        """Generate financial recommendations based on metrics."""
        recommendations = []
        
        # Income stability recommendations
        if income_stability < 0.4:
            recommendations.append("Consider diversifying income sources to improve stability")
        
        # Expense volatility recommendations
        if expense_volatility > 0.6:
            recommendations.append("Reduce expense volatility by creating more consistent spending patterns")
        
        # Savings rate recommendations
        if savings_rate < 0.1:
            recommendations.append("Increase your savings rate to at least 10% of income")
        elif savings_rate < 0.2:
            recommendations.append("Consider increasing your savings rate further for better long-term security")
        
        # Emergency fund recommendations
        if emergency_fund < 0.25:
            recommendations.append("Build an emergency fund to cover at least 1-2 months of expenses")
        elif emergency_fund < 0.5:
            recommendations.append("Continue building your emergency fund toward the recommended 3-6 months of expenses")
        
        # Debt-to-income recommendations
        if debt_to_income > 0.7:
            recommendations.append("Prioritize debt reduction - your debt-to-income ratio is high")
        elif debt_to_income > 0.4:
            recommendations.append("Consider a debt reduction strategy to improve your financial flexibility")
        
        # Budget adherence recommendations
        if budget_adherence < 0.6:
            recommendations.append("Review your budget categories and adjust to better match your actual spending patterns")
        
        # If doing well in all areas
        if not recommendations:
            recommendations.append("Your financial health metrics look good! Consider increasing investments and retirement contributions")
        
        return recommendations


class BudgetOptimizer:
    """
    Advanced budget optimization using portfolio theory concepts.
    This applies Modern Portfolio Theory principles to personal finance.
    """
    
    def __init__(self, user_id: str = None):
        """Initialize with optional user ID."""
        self.user_id = user_id
    
    def set_user(self, user_id: str):
        """Set the user ID."""
        self.user_id = user_id
    
    def optimize_budget_allocation(self) -> Dict[str, Any]:
        """
        Optimize budget allocation based on spending patterns and goals.
        Uses portfolio optimization concepts for improving financial outcomes.
        
        Returns:
            Dict with optimized budget recommendations
        """
        if not self.user_id:
            return {"error": "User not set"}
        
        try:
            # Get transaction history and current budgets
            transactions = db.get_transactions(self.user_id)
            budgets = db.get_budgets(self.user_id)
            
            if not transactions:
                return {
                    "error": "Insufficient data",
                    "message": "Not enough transaction history for budget optimization"
                }
            
            # Convert to dicts if needed
            txn_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in transactions]
            budget_dicts = [b.to_dict() if hasattr(b, 'to_dict') else b for b in budgets]
            
            # Analyze current spending by category
            category_analysis = self._analyze_category_spending(txn_dicts)
            
            # Get current budget allocations
            current_budgets = {b.get('category', ''): b.get('amount', 0) for b in budget_dicts}
            
            # Define essentials vs discretionary categories
            essential_categories = [
                'Housing', 'Utilities', 'Transportation', 'Healthcare', 'Groceries', 'Insurance'
            ]
            
            # Calculate current allocation stats
            total_current_budget = sum(current_budgets.values())
            if total_current_budget == 0:
                # No current budgets, create default allocation
                return self._create_initial_budget_recommendation(category_analysis)
            
            # Analyze budget effectiveness using utility function
            budget_effectiveness = self._analyze_budget_effectiveness(
                current_budgets, category_analysis, essential_categories
            )
            
            # Optimize allocation
            optimized_budgets = self._optimize_allocation(
                current_budgets, category_analysis, essential_categories, budget_effectiveness
            )
            
            # Format result
            result = {
                "current_allocation": [
                    {"category": cat, "amount": amt, "percent": (amt / total_current_budget) * 100 if total_current_budget > 0 else 0}
                    for cat, amt in current_budgets.items()
                ],
                "optimized_allocation": [
                    {"category": cat, "amount": amt, "percent": (amt / total_current_budget) * 100 if total_current_budget > 0 else 0}
                    for cat, amt in optimized_budgets.items()
                ],
                "category_insights": [
                    {
                        "category": cat,
                        "avg_monthly": data.get("avg_monthly", 0),
                        "volatility": data.get("volatility", 0),
                        "trend": data.get("trend", 0),
                        "necessity_score": 1.0 if cat in essential_categories else 0.5
                    }
                    for cat, data in category_analysis.items()
                ],
                "effectiveness_improvement": self._calculate_effectiveness_improvement(
                    current_budgets, optimized_budgets, category_analysis, essential_categories
                )
            }
            
            return result
            
        except Exception as e:
            print(f"Error in budget optimization: {str(e)}")
            traceback.print_exc()
            return {
                "error": "Optimization error",
                "message": str(e)
            }
    
    def _analyze_category_spending(self, transactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze spending patterns by category."""
        # Get all categories
        categories = list(set(txn.get('category', 'Other') for txn in transactions 
                          if txn.get('type', '').lower() == 'expense'))
        
        # Monthly spending by category
        monthly_spending = {}
        
        for txn in transactions:
            if txn.get('type', '').lower() != 'expense':
                continue
                
            date_str = txn.get('date', '')
            if not date_str:
                continue
            
            try:
                date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"
                category = txn.get('category', 'Other')
                
                if category not in monthly_spending:
                    monthly_spending[category] = {}
                
                if month_key not in monthly_spending[category]:
                    monthly_spending[category][month_key] = 0
                
                monthly_spending[category][month_key] += txn.get('amount', 0)
            except:
                continue
        
        # Analyze each category
        result = {}
        for category in categories:
            if category not in monthly_spending:
                continue
                
            # Get monthly values
            monthly_values = list(monthly_spending[category].values())
            if not monthly_values:
                continue
                
            # Calculate metrics
            avg_monthly = np.mean(monthly_values)
            volatility = np.std(monthly_values) / avg_monthly if avg_monthly > 0 else 0
            
            # Simplified trend calculation (positive = upward trend, negative = downward trend)
            trend = 0
            if len(monthly_values) >= 2:
                monthly_sorted = sorted(monthly_spending[category].items())
                values_sorted = [v for _, v in monthly_sorted]
                if len(values_sorted) >= 2:
                    # Simple linear regression slope as trend
                    x = np.arange(len(values_sorted))
                    slope, _ = np.polyfit(x, values_sorted, 1)
                    trend = slope / avg_monthly if avg_monthly > 0 else 0
            
            result[category] = {
                "avg_monthly": avg_monthly,
                "volatility": volatility,
                "trend": trend,
                "min_monthly": min(monthly_values),
                "max_monthly": max(monthly_values)
            }
        
        return result
    
    def _analyze_budget_effectiveness(self, 
                                    current_budgets: Dict[str, float],
                                    category_analysis: Dict[str, Dict[str, float]],
                                    essential_categories: List[str]) -> Dict[str, float]:
        """Analyze current budget effectiveness."""
        effectiveness = {}
        
        for category, budget in current_budgets.items():
            if category not in category_analysis or budget <= 0:
                effectiveness[category] = 0.5  # Neutral score
                continue
            
            analysis = category_analysis[category]
            avg_monthly = analysis.get("avg_monthly", 0)
            
            if avg_monthly <= 0:
                effectiveness[category] = 0.5  # Neutral score
                continue
            
            # Calculate budget to spending ratio
            ratio = budget / avg_monthly
            
            # Calculate utility score based on category type
            is_essential = category in essential_categories
            
            if is_essential:
                # Essential categories should be close to 1.0-1.1x average spending
                if ratio < 1.0:
                    # Underbudgeted essential - bad
                    effectiveness[category] = ratio
                elif ratio < 1.1:
                    # Properly budgeted essential - good
                    effectiveness[category] = 1.0
                else:
                    # Overbudgeted essential - diminishing utility
                    effectiveness[category] = 1.0 - (ratio - 1.1) / 10
            else:
                # Discretionary categories have more flexibility
                if ratio < 0.9:
                    # Underbudgeted discretionary - might be restrictive
                    effectiveness[category] = 0.7 * (ratio / 0.9)
                elif ratio < 1.1:
                    # Properly budgeted discretionary - good
                    effectiveness[category] = 0.7 + (0.3 * (ratio - 0.9) / 0.2)
                else:
                    # Overbudgeted discretionary - diminishing utility
                    effectiveness[category] = 1.0 - (ratio - 1.1) / 5
            
            # Ensure within bounds
            effectiveness[category] = max(0, min(1, effectiveness[category]))
        
        return effectiveness
    
    def _optimize_allocation(self,
                           current_budgets: Dict[str, float],
                           category_analysis: Dict[str, Dict[str, float]],
                           essential_categories: List[str],
                           current_effectiveness: Dict[str, float]) -> Dict[str, float]:
        """Optimize budget allocation."""
        # Start with current allocation
        optimized = current_budgets.copy()
        total_budget = sum(current_budgets.values())
        
        if total_budget <= 0:
            return optimized
        
        # Calculate ideal allocations
        ideal_allocations = {}
        
        for category, analysis in category_analysis.items():
            avg_monthly = analysis.get("avg_monthly", 0)
            volatility = analysis.get("volatility", 0)
            
            # Skip categories with no spending
            if avg_monthly <= 0:
                continue
            
            # Calculate ideal budget based on category type
            if category in essential_categories:
                # Essential categories need 100-110% coverage
                ideal_allocations[category] = avg_monthly * (1.05 + (volatility * 0.1))
            else:
                # Discretionary categories are more flexible
                ideal_allocations[category] = avg_monthly * (0.95 - (volatility * 0.1))
        
        # Adjust allocations to match total budget
        ideal_total = sum(ideal_allocations.values())
        
        if ideal_total <= 0:
            return optimized
        
        # Scale to match total budget
        for category in ideal_allocations:
            ideal_allocations[category] = (ideal_allocations[category] / ideal_total) * total_budget
        
        # Blend current and ideal allocations (gradual transition)
        for category in set(list(current_budgets.keys()) + list(ideal_allocations.keys())):
            current_amount = current_budgets.get(category, 0)
            ideal_amount = ideal_allocations.get(category, 0)
            
            # Skip if no data
            if current_amount <= 0 and ideal_amount <= 0:
                continue
            
            # 70% ideal + 30% current as a gradual transition
            optimized[category] = (ideal_amount * 0.7) + (current_amount * 0.3)
        
        # Ensure allocation matches total budget
        optimized_total = sum(optimized.values())
        if optimized_total > 0:
            scaling_factor = total_budget / optimized_total
            for category in optimized:
                optimized[category] *= scaling_factor
        
        return optimized
    
    def _create_initial_budget_recommendation(self, 
                                            category_analysis: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create initial budget recommendation when no budgets exist."""
        # Estimate total monthly expenses
        total_monthly = sum(data.get("avg_monthly", 0) for data in category_analysis.values())
        
        if total_monthly <= 0:
            return {
                "error": "Insufficient data",
                "message": "Unable to create budget recommendation from available data"
            }
        
        # Define essential categories
        essential_categories = [
            'Housing', 'Utilities', 'Transportation', 'Healthcare', 'Groceries', 'Insurance'
        ]
        
        # Create recommended allocation
        recommended = {}
        for category, data in category_analysis.items():
            avg_monthly = data.get("avg_monthly", 0)
            volatility = data.get("volatility", 0)
            
            if avg_monthly <= 0:
                continue
                
            if category in essential_categories:
                # Essential categories need buffer
                recommended[category] = avg_monthly * (1.05 + (volatility * 0.1))
            else:
                # Discretionary categories can be trimmed
                recommended[category] = avg_monthly * (0.95 - (volatility * 0.1))
        
        # Format result
        recommended_list = [
            {"category": cat, "amount": amt, "percent": (amt / total_monthly) * 100}
            for cat, amt in recommended.items()
        ]
        
        # Sort by amount descending
        recommended_list.sort(key=lambda x: x["amount"], reverse=True)
        
        return {
            "recommended_budget": recommended_list,
            "message": "Initial budget recommendation created based on your spending patterns",
            "total_monthly": total_monthly
        }
    
    def _calculate_effectiveness_improvement(self,
                                           current_budgets: Dict[str, float],
                                           optimized_budgets: Dict[str, float],
                                           category_analysis: Dict[str, Dict[str, float]],
                                           essential_categories: List[str]) -> float:
        """Calculate effectiveness improvement from optimization."""
        # Calculate effectiveness scores before and after
        current_effectiveness = self._analyze_budget_effectiveness(
            current_budgets, category_analysis, essential_categories
        )
        
        optimized_effectiveness = self._analyze_budget_effectiveness(
            optimized_budgets, category_analysis, essential_categories
        )
        
        # Calculate average improvement
        categories = set(current_effectiveness.keys()) | set(optimized_effectiveness.keys())
        if not categories:
            return 0
            
        total_improvement = 0
        for category in categories:
            current = current_effectiveness.get(category, 0)
            optimized = optimized_effectiveness.get(category, 0)
            improvement = optimized - current
            total_improvement += improvement
        
        avg_improvement = total_improvement / len(categories)
        return avg_improvement * 100  # Convert to percentage


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory.
    Applies optimization techniques to savings and investments.
    """
    
    def __init__(self, user_id: str = None):
        """Initialize with optional user ID."""
        self.user_id = user_id
    
    def set_user(self, user_id: str):
        """Set the user ID."""
        self.user_id = user_id
    
    def optimize_savings_allocation(self, 
                                   risk_tolerance: float = 0.5, 
                                   time_horizon: int = 5) -> Dict[str, Any]:
        """
        Optimize allocation of savings across different instruments.
        
        Args:
            risk_tolerance: Risk tolerance level (0-1, higher means more risk tolerance)
            time_horizon: Investment time horizon in years
            
        Returns:
            Dict with optimized allocation recommendation
        """
        if not self.user_id:
            return {"error": "User not set"}
        
        try:
            # Get user data
            transactions = db.get_transactions(self.user_id)
            
            # Simple heuristic-based allocation based on risk profile
            # In a real application, this would use actual return data and covariance matrices
            
            # Calculate risk profile
            risk_profile = self._calculate_risk_profile(risk_tolerance, time_horizon)
            
            # Get allocation based on risk profile
            allocation = self._get_allocation_for_risk_profile(risk_profile)
            
            # Create explanations
            explanation = self._generate_allocation_explanation(risk_profile, allocation)
            
            return {
                "risk_profile": risk_profile,
                "allocation": allocation,
                "explanation": explanation,
                "time_horizon": time_horizon,
                "expected_returns": self._calculate_expected_returns(allocation)
            }
            
        except Exception as e:
            print(f"Error in portfolio optimization: {str(e)}")
            traceback.print_exc()
            return {
                "error": "Optimization error",
                "message": str(e)
            }
    
    def _calculate_risk_profile(self, risk_tolerance: float, time_horizon: int) -> str:
        """Calculate risk profile based on tolerance and time horizon."""
        # Adjust risk tolerance based on time horizon
        # Longer horizon allows more risk
        adjusted_tolerance = risk_tolerance * (1 + (time_horizon / 10))
        adjusted_tolerance = min(1.0, adjusted_tolerance)
        
        # Determine risk profile
        if adjusted_tolerance < 0.3:
            return "Conservative"
        elif adjusted_tolerance < 0.6:
            return "Moderate"
        else:
            return "Aggressive"
    
    def _get_allocation_for_risk_profile(self, risk_profile: str) -> List[Dict[str, Any]]:
        """Get recommended allocation based on risk profile."""
        if risk_profile == "Conservative":
            return [
                {"category": "Cash / Money Market", "percent": 20},
                {"category": "Bonds / Fixed Income", "percent": 50},
                {"category": "Stocks / Equities", "percent": 25},
                {"category": "Alternative Investments", "percent": 5}
            ]
        elif risk_profile == "Moderate":
            return [
                {"category": "Cash / Money Market", "percent": 10},
                {"category": "Bonds / Fixed Income", "percent": 40},
                {"category": "Stocks / Equities", "percent": 45},
                {"category": "Alternative Investments", "percent": 5}
            ]
        else:  # Aggressive
            return [
                {"category": "Cash / Money Market", "percent": 5},
                {"category": "Bonds / Fixed Income", "percent": 20},
                {"category": "Stocks / Equities", "percent": 65},
                {"category": "Alternative Investments", "percent": 10}
            ]
    
    def _generate_allocation_explanation(self, risk_profile: str, allocation: List[Dict[str, Any]]) -> str:
        """Generate explanation for the recommended allocation."""
        if risk_profile == "Conservative":
            return (
                "This conservative allocation prioritizes capital preservation and stable income. "
                "The high allocation to bonds and cash provides stability, while the limited exposure "
                "to stocks offers some growth potential with reduced volatility."
            )
        elif risk_profile == "Moderate":
            return (
                "This balanced allocation seeks to provide both income and growth. "
                "The significant allocation to bonds provides stability and income, while "
                "the stocks component offers growth potential. This approach balances "
                "risk and reward for medium-term financial goals."
            )
        else:  # Aggressive
            return (
                "This growth-oriented allocation maximizes long-term potential returns. "
                "The high allocation to stocks increases both growth potential and volatility. "
                "This approach is designed for long-term investors who can tolerate market fluctuations "
                "and have time to recover from potential downturns."
            )
    
    def _calculate_expected_returns(self, allocation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate expected returns and risk metrics for the allocation."""
        # Sample expected returns and standard deviations for asset classes
        # In a real application, these would be based on historical data and forecasts
        expected_asset_returns = {
            "Cash / Money Market": 0.02,  # 2%
            "Bonds / Fixed Income": 0.04,  # 4%
            "Stocks / Equities": 0.08,    # 8%
            "Alternative Investments": 0.06  # 6%
        }
        
        # Calculate portfolio expected return
        portfolio_return = sum(
            item["percent"] / 100 * expected_asset_returns.get(item["category"], 0)
            for item in allocation
        )
        
        # Calculate volatility (simplified)
        portfolio_volatility = sum(
            item["category"] == "Stocks / Equities" for item in allocation
        ) / len(allocation) * 0.15  # Simplified volatility calculation
        
        return {
            "expected_annual_return": portfolio_return * 100,  # Convert to percentage
            "expected_volatility": portfolio_volatility * 100,  # Convert to percentage
            "worst_case_annual_return": (portfolio_return - (portfolio_volatility * 2)) * 100,
            "best_case_annual_return": (portfolio_return + (portfolio_volatility * 2)) * 100
        }


# Global instances
monte_carlo = MonteCarloSimulator()
risk_model = FinancialRiskModel()
budget_optimizer = BudgetOptimizer()
portfolio_optimizer = PortfolioOptimizer()