import streamlit as st
import hashlib
import datetime
from typing import Dict, Tuple, Optional, Any

import db_service as db

def hash_password(password: str) -> str:
    """Create a secure hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()

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
    # Check if email already exists
    existing_user = db.get_user_by_email(email)
    if existing_user:
        return False, "Email already registered"
    
    try:
        # Create user
        user = db.create_user(email, password, name)
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict]]:
    """
    Authenticate a user.
    
    Args:
        email: User's email
        password: User's password
        
    Returns:
        Tuple of (success, user_data)
    """
    # Get user by email
    user = db.get_user_by_email(email)
    if not user:
        return False, None
    
    # Verify password
    if db.verify_password(user.password_hash, password):
        return True, user.to_dict()
    
    return False, None

def get_user_data(user_id: str) -> Optional[Dict]:
    """
    Get user data by ID.
    
    Args:
        user_id: User's ID
        
    Returns:
        User data dict or None if not found
    """
    user = db.get_user_by_id(user_id)
    if user:
        return user.to_dict()
    
    return None

def auth_page():
    """Render the authentication page."""
    st.title("Welcome to AI Finance Tracker")
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if email and password:
                success, user = authenticate_user(email, password)
                
                if success:
                    # Set session state for authenticated user
                    st.session_state.user_id = user["id"]
                    st.rerun()
                else:
                    st.error("Invalid email or password")
            else:
                st.error("Please enter both email and password")
    
    with tab2:
        st.subheader("Register")
        
        name = st.text_input("Full Name", key="register_name")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register", key="register_button"):
            if name and email and password:
                if password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(email, password, name)
                    
                    if success:
                        st.success(message)
                        # Automatically log in the new user
                        success, user = authenticate_user(email, password)
                        if success:
                            st.session_state.user_id = user["id"]
                            st.rerun()
                    else:
                        st.error(message)
            else:
                st.error("Please fill in all fields")

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