
"""
Custom styling for the finance application with retro/cyberpunk UI elements.
"""
import streamlit as st

def inject_custom_css():
    """Inject custom CSS for retro-style UI."""
    st.markdown("""
        <style>
        /* Hide all HTML/CSS code display */
        .element-container:has(>.stMarkdown) pre {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def inject_custom_js():
    """Inject custom JavaScript for enhanced UI interactions."""
    st.markdown("""
        <script>
        // Basic cursor tracking
        document.addEventListener('mousemove', function(e) {
            const cursor = document.querySelector('.custom-cursor');
            if (cursor) {
                cursor.style.left = e.clientX + 'px';
                cursor.style.top = e.clientY + 'px';
            }
        });
        </script>
    """, unsafe_allow_html=True)

def render_card(title=None, content=None):
    """Render a styled card using Streamlit containers."""
    with st.container():
        if title:
            st.subheader(title)
        if content:
            st.write(content)

def badge(text, badge_type="info"):
    """Create a styled badge using Streamlit."""
    colors = {
        "info": "blue",
        "success": "green",
        "warning": "orange",
        "danger": "red"
    }
    st.markdown(f":{colors.get(badge_type)}[{text}]")

def retro_metric(label, value, delta=None, delta_color="normal"):
    """Create a metric using Streamlit's metric component."""
    st.metric(label=label, value=value, delta=delta)

def animated_progress(percent, label=""):
    """Create a progress bar using Streamlit."""
    st.progress(percent/100)
    if label:
        st.caption(f"{label}: {percent}%")

def format_currency(amount, currency="$"):
    """Format a number as currency."""
    return f"{currency}{amount:,.2f}" if amount >= 0 else f"-{currency}{abs(amount):,.2f}"

def retro_table(data):
    """Display a table using Streamlit's native table component."""
    st.table(data)

def init_retro_ui():
    """Initialize the retro UI style."""
    st.title("AI Finance Tracker")
    st.caption("Your personal finance companion")
    return ""

def neon_text(text, color="#9370DB"):
    """Create neon glowing text."""
    return f"""
    <span style="
        color: {color}; 
        text-shadow: 0 0 5px {color}, 0 0 10px {color}, 0 0 15px {color}; 
        font-weight: bold;
    ">
        {text}
    </span>
    """
