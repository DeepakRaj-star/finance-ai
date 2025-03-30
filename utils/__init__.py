
"""
Utils package for common functions used across the application.
"""

# Import styling functions
from utils.custom_style import (
    inject_custom_css,
    inject_custom_js,
    render_card,
    badge,
    neon_text,
    retro_metric,
    animated_progress,
    format_currency,
    retro_table,
    init_retro_ui
)

# Import from utils_core
from .utils_core import (
    get_date_range,
    get_recommendations,
    create_spending_by_category_chart,
    create_spending_trend_chart,
    create_balance_chart
)
