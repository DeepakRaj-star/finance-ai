import streamlit as st
import random
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Quiz data directory
QUIZ_DATA_DIR = Path("user_data/quiz")

def setup_quiz_system():
    """Ensure the quiz data directory exists."""
    QUIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_user_quiz_file_path(user_id: str) -> Path:
    """Get the path to a user's quiz progress file."""
    return QUIZ_DATA_DIR / f"{user_id}_progress.json"

def load_user_quiz_progress(user_id: str) -> Dict[str, Any]:
    """
    Load quiz progress for a specific user.
    
    Args:
        user_id (str): The user ID
        
    Returns:
        Dict: Quiz progress dictionary
    """
    file_path = get_user_quiz_file_path(user_id)
    
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading quiz progress: {e}")
            return {"completed_quizzes": [], "points": 0, "badges": []}
    else:
        return {"completed_quizzes": [], "points": 0, "badges": []}

def save_user_quiz_progress(user_id: str, progress: Dict[str, Any]) -> bool:
    """
    Save quiz progress for a specific user.
    
    Args:
        user_id (str): The user ID
        progress (Dict): Quiz progress dictionary
        
    Returns:
        bool: Success or failure
    """
    file_path = get_user_quiz_file_path(user_id)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(progress, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving quiz progress: {e}")
        return False

def update_quiz_points(user_id: str, points_to_add: int) -> int:
    """
    Update a user's quiz points.
    
    Args:
        user_id (str): The user ID
        points_to_add (int): Points to add
        
    Returns:
        int: New total points
    """
    progress = load_user_quiz_progress(user_id)
    progress["points"] = progress.get("points", 0) + points_to_add
    
    # Check for new badges
    current_badges = set(progress.get("badges", []))
    
    if progress["points"] >= 100 and "Finance Novice" not in current_badges:
        current_badges.add("Finance Novice")
    if progress["points"] >= 250 and "Finance Apprentice" not in current_badges:
        current_badges.add("Finance Apprentice")
    if progress["points"] >= 500 and "Finance Expert" not in current_badges:
        current_badges.add("Finance Expert")
    if progress["points"] >= 1000 and "Finance Master" not in current_badges:
        current_badges.add("Finance Master")
        
    progress["badges"] = list(current_badges)
    
    save_user_quiz_progress(user_id, progress)
    return progress["points"]

def mark_quiz_completed(user_id: str, quiz_id: str) -> None:
    """
    Mark a quiz as completed for a user.
    
    Args:
        user_id (str): The user ID
        quiz_id (str): The quiz ID
    """
    progress = load_user_quiz_progress(user_id)
    completed = set(progress.get("completed_quizzes", []))
    completed.add(quiz_id)
    progress["completed_quizzes"] = list(completed)
    save_user_quiz_progress(user_id, progress)

def is_quiz_completed(user_id: str, quiz_id: str) -> bool:
    """
    Check if a quiz has been completed by a user.
    
    Args:
        user_id (str): The user ID
        quiz_id (str): The quiz ID
        
    Returns:
        bool: True if completed, False otherwise
    """
    progress = load_user_quiz_progress(user_id)
    return quiz_id in progress.get("completed_quizzes", [])

# Quiz questions
QUIZZES = {
    "basics": {
        "id": "basics",
        "title": "Financial Basics",
        "description": "Test your knowledge of basic financial concepts.",
        "difficulty": "Beginner",
        "points": 50,
        "questions": [
            {
                "question": "What is a budget?",
                "options": [
                    "A mobile app for finances",
                    "A plan for spending and saving money",
                    "A type of credit card",
                    "A financial institution"
                ],
                "correct": 1,
                "explanation": "A budget is a plan that helps you track your income and expenses, ensuring you live within your means and meet financial goals."
            },
            {
                "question": "What is compound interest?",
                "options": [
                    "Interest earned only on the principal amount",
                    "Interest that remains fixed over time",
                    "Interest earned on both principal and accumulated interest",
                    "A type of loan with varying interest rates"
                ],
                "correct": 2,
                "explanation": "Compound interest is when you earn interest not only on your initial principal but also on the interest you've already earned, causing your money to grow exponentially over time."
            },
            {
                "question": "What is an emergency fund?",
                "options": [
                    "Money set aside for vacations",
                    "Money saved for unexpected expenses",
                    "A retirement account",
                    "A type of investment"
                ],
                "correct": 1,
                "explanation": "An emergency fund is money set aside to cover unexpected expenses like medical emergencies or job loss, typically recommended to cover 3-6 months of expenses."
            },
            {
                "question": "What is the difference between a debit card and a credit card?",
                "options": [
                    "There is no difference",
                    "Debit cards offer more rewards",
                    "Debit cards draw from your bank account; credit cards borrow money",
                    "Credit cards can only be used online"
                ],
                "correct": 2,
                "explanation": "A debit card draws money directly from your bank account while a credit card allows you to borrow money up to a limit that you pay back later, often with interest if not paid in full."
            },
            {
                "question": "What is a credit score?",
                "options": [
                    "The amount of credit you have available",
                    "A number indicating your creditworthiness",
                    "The interest rate on your credit card",
                    "The number of credit cards you own"
                ],
                "correct": 1,
                "explanation": "A credit score is a number (typically 300-850) that represents your creditworthiness based on your credit history, payment history, and other factors."
            }
        ]
    },
    "investing": {
        "id": "investing",
        "title": "Investing Fundamentals",
        "description": "Learn about the basics of investing.",
        "difficulty": "Intermediate",
        "points": 75,
        "questions": [
            {
                "question": "What is a stock?",
                "options": [
                    "A type of bond",
                    "A loan to a company",
                    "Ownership in a company",
                    "A guaranteed investment"
                ],
                "correct": 2,
                "explanation": "A stock represents partial ownership (equity) in a company. When you own a stock, you own a piece of that company and may receive dividends and voting rights."
            },
            {
                "question": "What is diversification?",
                "options": [
                    "Investing all your money in one company",
                    "Spreading investments across different assets to reduce risk",
                    "Only investing in bonds",
                    "Cashing out all investments during market downturns"
                ],
                "correct": 1,
                "explanation": "Diversification is the strategy of spreading investments across various asset classes, industries, and geographic regions to reduce risk and minimize the impact of any single investment's poor performance."
            },
            {
                "question": "What is a bond?",
                "options": [
                    "Ownership in a company",
                    "A type of insurance",
                    "A loan you give to a government or corporation",
                    "A savings account with high interest"
                ],
                "correct": 2,
                "explanation": "A bond is essentially an IOU where you lend money to a government or corporation for a specified period at a fixed interest rate, and they promise to pay you back with interest."
            },
            {
                "question": "What is an index fund?",
                "options": [
                    "A high-risk individual stock",
                    "A fund that aims to match the performance of a market index",
                    "A type of cryptocurrency",
                    "A fund managed by the government"
                ],
                "correct": 1,
                "explanation": "An index fund is a type of passive investment that aims to replicate the performance of a specific market index (like the S&P 500), offering broad market exposure with low fees."
            },
            {
                "question": "What is the difference between a bull market and a bear market?",
                "options": [
                    "They refer to markets in different countries",
                    "A bull market is rising; a bear market is declining",
                    "They refer to different stock exchanges",
                    "Bull markets are more volatile than bear markets"
                ],
                "correct": 1,
                "explanation": "A bull market refers to a period when prices are rising or expected to rise, while a bear market refers to a period when prices are falling or expected to fall, typically by 20% or more from recent highs."
            }
        ]
    },
    "budgeting": {
        "id": "budgeting",
        "title": "Budgeting Mastery",
        "description": "Master the art of effective budgeting.",
        "difficulty": "Beginner",
        "points": 60,
        "questions": [
            {
                "question": "What is the 50/30/20 budgeting rule?",
                "options": [
                    "50% needs, 30% wants, 20% savings",
                    "50% savings, 30% needs, 20% wants",
                    "50% wants, 30% savings, 20% needs",
                    "50% investments, 30% needs, 20% wants"
                ],
                "correct": 0,
                "explanation": "The 50/30/20 rule suggests allocating 50% of your income to necessities (housing, food, etc.), 30% to wants (entertainment, dining out), and 20% to savings and debt repayment."
            },
            {
                "question": "What is a zero-based budget?",
                "options": [
                    "A budget where you spend everything you earn",
                    "A budget where you save everything you earn",
                    "A budget where every dollar is assigned a specific purpose",
                    "A budget with zero categories"
                ],
                "correct": 2,
                "explanation": "In a zero-based budget, every dollar of income is assigned a specific purpose (expenses, savings, investments), so your income minus all allocations equals zero."
            },
            {
                "question": "What is a fixed expense?",
                "options": [
                    "An expense that changes each month",
                    "An expense that stays the same each month",
                    "A one-time purchase",
                    "An unnecessary expense"
                ],
                "correct": 1,
                "explanation": "A fixed expense is a cost that remains consistent each month, such as rent/mortgage, car payments, or subscription services with fixed monthly fees."
            },
            {
                "question": "What is the envelope budgeting system?",
                "options": [
                    "Putting all receipts in envelopes",
                    "Sorting bills by size in envelopes",
                    "Allocating cash to different spending categories in envelopes",
                    "Mailing your budget to financial advisors"
                ],
                "correct": 2,
                "explanation": "The envelope system involves allocating cash to different spending categories by physically placing money in labeled envelopes, stopping spending in a category once that envelope is empty."
            },
            {
                "question": "What is 'paying yourself first'?",
                "options": [
                    "Taking all your paycheck as cash",
                    "Prioritizing savings before spending on expenses",
                    "Putting all income towards debt repayment",
                    "Giving yourself a salary from your business"
                ],
                "correct": 1,
                "explanation": "Paying yourself first means automatically setting aside a portion of your income for savings and investments before paying bills or discretionary spending, making saving a priority."
            }
        ]
    },
    "credit": {
        "id": "credit",
        "title": "Credit and Loans",
        "description": "Understand how credit and loans work.",
        "difficulty": "Intermediate",
        "points": 70,
        "questions": [
            {
                "question": "What factors most heavily influence your credit score?",
                "options": [
                    "Your income and education level",
                    "Your payment history and credit utilization",
                    "Your age and location",
                    "Your employment history"
                ],
                "correct": 1,
                "explanation": "Payment history (35%) and credit utilization (30%) are the two most significant factors affecting your FICO score, followed by length of credit history, new credit, and credit mix."
            },
            {
                "question": "What is APR?",
                "options": [
                    "Annual Payment Rate",
                    "Average Percentage Return",
                    "Annual Percentage Rate",
                    "Approved Personal Reserve"
                ],
                "correct": 2,
                "explanation": "APR (Annual Percentage Rate) represents the yearly cost of borrowing money, including interest and fees, expressed as a percentage."
            },
            {
                "question": "What is credit utilization?",
                "options": [
                    "How often you use credit cards",
                    "The ratio of credit card debt to credit limits",
                    "Your total available credit",
                    "How many credit cards you own"
                ],
                "correct": 1,
                "explanation": "Credit utilization is the ratio of your current credit card balances to your credit limits, typically expressed as a percentage. Lower utilization (under 30%) is better for your credit score."
            },
            {
                "question": "What is a secured loan?",
                "options": [
                    "A loan that requires no credit check",
                    "A loan backed by collateral",
                    "A loan with a guaranteed approval",
                    "A loan with a fixed interest rate"
                ],
                "correct": 1,
                "explanation": "A secured loan is backed by collateral (like a car or home) that the lender can take if you fail to repay the loan. Mortgages and auto loans are common examples."
            },
            {
                "question": "What happens when you cosign a loan?",
                "options": [
                    "You get a portion of the loan amount",
                    "You become legally responsible for the debt if the primary borrower doesn't pay",
                    "You act as a reference for the borrower",
                    "You provide your financial advice on the loan terms"
                ],
                "correct": 1,
                "explanation": "When you cosign a loan, you're equally legally responsible for repaying the debt if the primary borrower fails to do so, and any missed payments will affect your credit score."
            }
        ]
    },
    "retirement": {
        "id": "retirement",
        "title": "Retirement Planning",
        "description": "Plan for your financial future.",
        "difficulty": "Advanced",
        "points": 90,
        "questions": [
            {
                "question": "What is a 401(k)?",
                "options": [
                    "A type of health insurance",
                    "A tax form",
                    "An employer-sponsored retirement account",
                    "A government benefit program"
                ],
                "correct": 2,
                "explanation": "A 401(k) is an employer-sponsored retirement plan that allows employees to contribute pre-tax income, often with employer matching contributions."
            },
            {
                "question": "What is an IRA?",
                "options": [
                    "International Revenue Association",
                    "Individual Retirement Arrangement/Account",
                    "Income Reporting Application",
                    "Interest Rate Agreement"
                ],
                "correct": 1,
                "explanation": "An Individual Retirement Account (IRA) is a tax-advantaged account that individuals can set up to save for retirement independently of employer-sponsored plans."
            },
            {
                "question": "What is the 4% rule in retirement planning?",
                "options": [
                    "Withdrawing 4% of your retirement savings each year",
                    "Saving 4% of your income for retirement",
                    "Earning at least 4% annually on investments",
                    "Paying 4% in fees for retirement accounts"
                ],
                "correct": 0,
                "explanation": "The 4% rule suggests withdrawing 4% of your retirement savings in the first year, then adjusting for inflation each year thereafter, aiming for a portfolio that lasts 30 years."
            },
            {
                "question": "What is the difference between a Roth and Traditional IRA?",
                "options": [
                    "Traditional IRAs are government-backed; Roth IRAs are private",
                    "Roth contributions are after-tax with tax-free withdrawals; Traditional contributions are pre-tax with taxable withdrawals",
                    "Roth IRAs are for employed people; Traditional IRAs are for self-employed",
                    "Traditional IRAs earn interest; Roth IRAs do not"
                ],
                "correct": 1,
                "explanation": "With a Roth IRA, you contribute after-tax money and withdrawals in retirement are tax-free. With a Traditional IRA, you contribute pre-tax money (reducing current taxable income) but pay taxes on withdrawals."
            },
            {
                "question": "What is a target-date fund?",
                "options": [
                    "A fund that matures on a specific date",
                    "A fund that aims to beat a specific market index",
                    "A retirement fund that automatically adjusts its asset allocation as you approach retirement",
                    "A fund that requires regular contributions until a target date"
                ],
                "correct": 2,
                "explanation": "A target-date fund automatically adjusts its asset allocation from more aggressive (growth-focused) to more conservative as you approach your retirement (target) date, simplifying long-term investing."
            }
        ]
    },
    "taxes": {
        "id": "taxes",
        "title": "Tax Fundamentals",
        "description": "Understand the basics of taxes and tax planning.",
        "difficulty": "Advanced",
        "points": 85,
        "questions": [
            {
                "question": "What is the difference between a tax deduction and a tax credit?",
                "options": [
                    "They are the same thing",
                    "A deduction reduces taxable income; a credit reduces tax owed",
                    "A credit reduces taxable income; a deduction reduces tax owed",
                    "A deduction is automatic; a credit must be claimed"
                ],
                "correct": 1,
                "explanation": "A tax deduction reduces your taxable income, while a tax credit reduces your tax liability dollar-for-dollar, making credits generally more valuable than deductions of the same amount."
            },
            {
                "question": "What is a progressive tax system?",
                "options": [
                    "A system where everyone pays the same tax rate",
                    "A system where tax rates increase as income increases",
                    "A system where taxes decrease over time",
                    "A modern approach to taxation"
                ],
                "correct": 1,
                "explanation": "A progressive tax system imposes higher tax rates on higher income levels, meaning the percentage of income paid in taxes increases as income increases."
            },
            {
                "question": "What is a capital gains tax?",
                "options": [
                    "A tax on all income",
                    "A tax on profit from selling investments or assets",
                    "A tax on property ownership",
                    "A tax on business revenue"
                ],
                "correct": 1,
                "explanation": "Capital gains tax is levied on the profit from selling investments or assets (like stocks, bonds, or real estate) that have increased in value since purchase."
            },
            {
                "question": "What is tax-loss harvesting?",
                "options": [
                    "Avoiding taxes by hiding income",
                    "A strategy of selling investments at a loss to offset capital gains",
                    "Collecting tax refunds from previous years",
                    "Filing for tax extensions"
                ],
                "correct": 1,
                "explanation": "Tax-loss harvesting involves selling investments at a loss to offset capital gains and potentially reduce taxable income, helping to reduce overall tax liability while maintaining investment strategy."
            },
            {
                "question": "What is the difference between marginal and effective tax rates?",
                "options": [
                    "They are different terms for the same concept",
                    "Marginal rate applies to your next dollar of income; effective rate is your average overall rate",
                    "Marginal rate is for individuals; effective rate is for businesses",
                    "Effective rate applies to your next dollar of income; marginal rate is your average overall rate"
                ],
                "correct": 1,
                "explanation": "Your marginal tax rate is the rate applied to your last dollar of income (or next dollar earned), while your effective tax rate is the average rate paid on all your income after deductions and credits."
            }
        ]
    }
}

def finance_quiz_page(user_id: str):
    """
    Display the finance quiz page.
    
    Args:
        user_id (str): The ID of the current user
    """
    # Initialize quiz system
    setup_quiz_system()
    
    st.title("Finance Education Quiz")
    st.write("Test your financial knowledge and earn points!")
    
    # Get user progress
    progress = load_user_quiz_progress(user_id)
    points = progress.get("points", 0)
    completed_quizzes = progress.get("completed_quizzes", [])
    badges = progress.get("badges", [])
    
    # Display user progress
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Knowledge Points", points)
        st.write(f"Quizzes Completed: {len(completed_quizzes)}/{len(QUIZZES)}")
    
    with col2:
        if badges:
            st.write("Your Badges:")
            for badge in badges:
                st.write(f"🏆 {badge}")
        else:
            st.write("Complete quizzes to earn badges!")
            st.write("- 🏆 Finance Novice (100 points)")
            st.write("- 🏆 Finance Apprentice (250 points)")
            st.write("- 🏆 Finance Expert (500 points)")
            st.write("- 🏆 Finance Master (1000 points)")
    
    st.divider()
    
    # Display available quizzes
    st.subheader("Available Quizzes")
    
    # Sort quizzes by difficulty
    difficulty_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    sorted_quizzes = sorted(QUIZZES.values(), key=lambda q: difficulty_order[q["difficulty"]])
    
    for quiz in sorted_quizzes:
        quiz_id = quiz["id"]
        completed = quiz_id in completed_quizzes
        
        with st.expander(f"{quiz['title']} - {quiz['difficulty']} {' ✓' if completed else ''}"):
            st.write(quiz["description"])
            st.write(f"Points: {quiz['points']}")
            st.write(f"Questions: {len(quiz['questions'])}")
            
            if completed:
                st.success("You've completed this quiz!")
                if st.button(f"Take Again (no points)", key=f"retry_{quiz_id}"):
                    st.session_state.current_quiz = quiz
                    st.session_state.current_question = 0
                    st.session_state.correct_answers = 0
                    st.session_state.quiz_completed = False
                    st.session_state.no_points = True
                    st.rerun()
            else:
                if st.button(f"Start Quiz", key=f"start_{quiz_id}"):
                    st.session_state.current_quiz = quiz
                    st.session_state.current_question = 0
                    st.session_state.correct_answers = 0
                    st.session_state.quiz_completed = False
                    st.session_state.no_points = False
                    st.rerun()
    
    # Active quiz session
    if hasattr(st.session_state, 'current_quiz') and st.session_state.current_quiz:
        quiz = st.session_state.current_quiz
        
        st.divider()
        st.subheader(f"Quiz: {quiz['title']}")
        
        if st.session_state.quiz_completed:
            # Show quiz results
            correct = st.session_state.correct_answers
            total = len(quiz["questions"])
            score_percent = (correct / total) * 100
            
            st.write(f"You answered {correct} out of {total} questions correctly ({score_percent:.1f}%).")
            
            if score_percent >= 80 and not st.session_state.no_points and not is_quiz_completed(user_id, quiz["id"]):
                points_earned = quiz["points"]
                new_total = update_quiz_points(user_id, points_earned)
                mark_quiz_completed(user_id, quiz["id"])
                st.success(f"🎉 Congratulations! You earned {points_earned} points! Your new total is {new_total} points.")
            elif score_percent >= 80 and st.session_state.no_points:
                st.info("No points awarded for retaking the quiz.")
            elif score_percent >= 80 and is_quiz_completed(user_id, quiz["id"]):
                st.info("You've already earned points for this quiz.")
            else:
                st.warning("You need to score at least 80% to earn points. Try again!")
                
            if st.button("Return to Quiz List"):
                # Reset quiz state
                if hasattr(st.session_state, 'current_quiz'):
                    del st.session_state.current_quiz
                if hasattr(st.session_state, 'current_question'):
                    del st.session_state.current_question
                if hasattr(st.session_state, 'correct_answers'):
                    del st.session_state.correct_answers
                if hasattr(st.session_state, 'quiz_completed'):
                    del st.session_state.quiz_completed
                if hasattr(st.session_state, 'no_points'):
                    del st.session_state.no_points
                st.rerun()
        else:
            # Show current question
            questions = quiz["questions"]
            q_idx = st.session_state.current_question
            
            if q_idx < len(questions):
                question = questions[q_idx]
                
                st.write(f"**Question {q_idx + 1} of {len(questions)}**")
                st.write(question["question"])
                
                # Show options with radio buttons
                option = st.radio("Select your answer:", 
                                  question["options"],
                                  key=f"q_{q_idx}")
                
                selected_idx = question["options"].index(option)
                
                if st.button("Submit Answer"):
                    # Check answer
                    if selected_idx == question["correct"]:
                        st.success("Correct! 🎉")
                        st.session_state.correct_answers += 1
                    else:
                        st.error(f"Incorrect. The correct answer is: {question['options'][question['correct']]}")
                    
                    # Show explanation
                    st.info(f"**Explanation**: {question['explanation']}")
                    
                    # Move to next question or finish quiz
                    if q_idx + 1 < len(questions):
                        if st.button("Next Question"):
                            st.session_state.current_question += 1
                            st.rerun()
                    else:
                        if st.button("Finish Quiz"):
                            st.session_state.quiz_completed = True
                            st.rerun()
            else:
                st.session_state.quiz_completed = True
                st.rerun()