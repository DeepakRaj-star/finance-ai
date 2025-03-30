import streamlit as st
import json
import hashlib
import os
import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

def setup_auth_system():
    """Ensure the user data directory and files exist."""
    user_data_path = Path("user_data")
    user_data_path.mkdir(exist_ok=True)
    
    users_file = user_data_path / "users.json"
    
    if not users_file.exists():
        # Create empty users file
        with open(users_file, 'w') as f:
            json.dump({}, f)

def hash_password(password: str) -> str:
    """Create a secure hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> Dict:
    """Load users from the JSON file."""
    users_file = Path("user_data") / "users.json"
    
    if not users_file.exists():
        setup_auth_system()
        return {}
    
    try:
        with open(users_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupt or doesn't exist, create a new one
        setup_auth_system()
        return {}

def save_users(users_data: Dict):
    """Save users to the JSON file."""
    users_file = Path("user_data") / "users.json"
    
    with open(users_file, 'w') as f:
        json.dump(users_data, f)

def register_user(email: str, password: str, name: str) -> Tuple[bool, str]:
    """
    Register a new user.
    
    Args:
        email: User's email
        password: User's password
        name: User's name
        
    Returns:
        Tuple of (success, message)
    """
    users = load_users()
    
    # Check if email already exists
    if email in users:
        return False, "Email already registered"
    
    # Create user record
    user_id = hashlib.md5(email.encode()).hexdigest()
    
    users[email] = {
        "id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Save updated users
    save_users(users)
    
    # Create user data directories
    user_data_path = Path("user_data") / user_id
    user_data_path.mkdir(exist_ok=True)
    
    # Initialize empty data files
    transactions_file = user_data_path / "transactions.json"
    goals_file = user_data_path / "goals.json"
    budgets_file = user_data_path / "budgets.json"
    
    for file_path in [transactions_file, goals_file, budgets_file]:
        if not file_path.exists():
            with open(file_path, 'w') as f:
                json.dump([], f)
    
    return True, "Registration successful"

def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict]]:
    """
    Authenticate a user.
    
    Args:
        email: User's email
        password: User's password
        
    Returns:
        Tuple of (success, user_data)
    """
    users = load_users()
    
    if email not in users:
        return False, None
    
    user = users[email]
    password_hash = hash_password(password)
    
    if user["password_hash"] == password_hash:
        return True, user
    
    return False, None

def get_user_data(user_id: str) -> Optional[Dict]:
    """
    Get user data by ID.
    
    Args:
        user_id: User's ID
        
    Returns:
        User data dict or None if not found
    """
    users = load_users()
    
    for user in users.values():
        if user["id"] == user_id:
            return user
    
    return None

def auth_page():
    """Render the authentication page."""
    setup_auth_system()
    
    # Import custom_style for retro UI elements
    from utils.custom_style import init_retro_ui, inject_custom_css, inject_custom_js, badge, neon_text
    
    # Initialize retro UI
    welcome_html = init_retro_ui()
    st.markdown(welcome_html, unsafe_allow_html=True)
    
    # Use tabs with custom styling
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        # Apply retro login styling
        st.markdown("""
        <div class="login-form">
            <h2 class="glow-text">Access Your Financial Dashboard</h2>
            <div style="margin-bottom: 1.5rem; opacity: 0.8;">
                Enter your credentials to continue
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form content
        email = st.text_input("Email", key="login_email", 
                             placeholder="Enter your email")
        password = st.text_input("Password", type="password", key="login_password",
                               placeholder="Enter your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Enter the Grid", key="login_button", use_container_width=True):
                if email and password:
                    with st.spinner("Authenticating..."):
                        success, user = authenticate_user(email, password)
                        
                        if success:
                            # Set session state for authenticated user
                            st.session_state.user_id = user["id"]
                            st.toast("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password")
                else:
                    st.error("Please enter both email and password")
    
    with tab2:
        st.markdown("""
        <div class="login-form">
            <h2 class="glow-text">Create New Account</h2>
            <div style="margin-bottom: 1.5rem; opacity: 0.8;">
                Join our financial grid network
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        name = st.text_input("Full Name", key="register_name", 
                          placeholder="Enter your full name")
        email = st.text_input("Email", key="register_email",
                           placeholder="Enter your email")
        password = st.text_input("Password", type="password", key="register_password", 
                              placeholder="Create a secure password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password",
                                      placeholder="Confirm your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Initialize Account", key="register_button", use_container_width=True):
                if name and email and password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        with st.spinner("Creating your digital identity..."):
                            success, message = register_user(email, password, name)
                            
                            if success:
                                st.success(message)
                                st.balloons()
                                
                                # Automatically log in the new user
                                success, user = authenticate_user(email, password)
                                if success:
                                    st.session_state.user_id = user["id"]
                                    st.rerun()
                            else:
                                st.error(message)
                else:
                    st.error("Please fill in all fields")
                                
    # Add some cyberpunk/retro design elements
    st.markdown("""
    <style>
    /* Animated background gradient */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            45deg,
            #0D1117 0%,
            #161B22 25%,
            #0D1117 50%,
            #161B22 75%,
            #0D1117 100%
        );
        background-size: 400% 400%;
        z-index: -1;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Grid overlay */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(147, 112, 219, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(147, 112, 219, 0.05) 1px, transparent 1px);
        background-size: 20px 20px;
        z-index: -1;
    }
    
    /* Pulsing effect for login form */
    .login-form {
        animation: pulseBorder 3s infinite alternate;
        border-radius: 15px;
        border: 1px solid #9370DB40;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #161B22 0%, #1d2430 100%);
    }
    
    @keyframes pulseBorder {
        0% {
            box-shadow: 0 0 10px rgba(147, 112, 219, 0.3);
        }
        100% {
            box-shadow: 0 0 20px rgba(147, 112, 219, 0.6);
        }
    }
    
    .glow-text {
        color: #C8A2FF;
        text-shadow: 0 0 10px #9370DB, 0 0 20px #9370DB40;
        animation: textGlow 3s infinite alternate;
    }
    
    @keyframes textGlow {
        0% {
            text-shadow: 0 0 10px #9370DB, 0 0 20px #9370DB40;
        }
        100% {
            text-shadow: 0 0 15px #9370DB, 0 0 30px #9370DB, 0 0 40px #9370DB40;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def is_authenticated():
    """Check if the user is authenticated."""
    return "user_id" in st.session_state and st.session_state.user_id is not None

def get_current_user():
    """Get the current authenticated user's data."""
    if is_authenticated():
        user_id = st.session_state.user_id
        return get_user_data(user_id)
    return None

def require_auth():
    """Require authentication to proceed. Redirects to auth page if not authenticated."""
    if not is_authenticated():
        auth_page()
        st.stop()
    return get_current_user()