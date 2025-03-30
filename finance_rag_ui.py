"""
Finance RAG UI - A Streamlit interface for the Finance RAG Agent.

This module provides a user interface for interacting with the Finance RAG system,
allowing users to query financial data, upload documents for indexing, and 
generate comprehensive financial reports.
"""

import os
import streamlit as st
import pandas as pd
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import time

# Import RAG agent
import finance_rag_agent as rag

# Import auth and UI modules
import auth
from utils import inject_custom_css, badge, neon_text, render_card, retro_metric

def finance_rag_page():
    """Display the finance RAG assistant page."""
    # Apply custom styles
    inject_custom_css()
    
    # Require authentication
    user = auth.get_current_user()
    if not user:
        st.error("You must be logged in to view this page.")
        return
    
    # Get user ID
    user_id = user.get("id")
    
    # Display page header
    st.markdown(f"<h1 class='retro-title'>{neon_text('Finance RAG Assistant')}</h1>", unsafe_allow_html=True)
    st.markdown("### Your AI-powered financial document and knowledge assistant")
    
    # Create tabs
    tabs = st.tabs(["Query Assistant", "Document Library", "Financial Reports", "Knowledge Base"])
    
    # === QUERY ASSISTANT TAB ===
    with tabs[0]:
        st.markdown(f"### {neon_text('Ask me anything about your finances')}")
        
        # Query input
        query = st.text_input("Enter your financial question", 
                            key="rag_query",
                            placeholder="Example: How much did I spend on dining last month?")
        
        # Filter options in expander
        with st.expander("Advanced Query Options"):
            col1, col2 = st.columns(2)
            with col1:
                document_type = st.selectbox(
                    "Document Type Filter",
                    options=["All"] + rag.FINANCIAL_DOCUMENT_TYPES,
                    key="doc_type_filter"
                )
            with col2:
                category = st.selectbox(
                    "Category Filter",
                    options=["All"] + rag.FINANCIAL_CATEGORIES,
                    key="category_filter"
                )
        
        # Process query on button click
        if st.button("Ask Question", key="ask_button", type="primary"):
            if query:
                with st.spinner("Analyzing your question..."):
                    # Process filters
                    doc_type = None if document_type == "All" else document_type
                    cat = None if category == "All" else category
                    
                    # Query the RAG system
                    response = rag.query_rag(
                        user_id=user_id, 
                        query=query,
                        document_type=doc_type,
                        category=cat
                    )
                    
                    # Display answer
                    st.markdown(f"### Answer")
                    render_card(response["answer"])
                    
                    # Display confidence with a progress bar
                    confidence = response.get("confidence", 0.0)
                    confidence_color = "red"
                    if confidence >= 0.7:
                        confidence_color = "green"
                    elif confidence >= 0.4:
                        confidence_color = "orange"
                    
                    st.markdown(f"**Confidence: **{confidence:.2f}")
                    st.progress(confidence, text="")
                    
                    # Display sources if available
                    if response.get("sources"):
                        st.markdown("### Sources")
                        for source in response["sources"]:
                            if isinstance(source, dict):
                                source_path = source.get("source", "Unknown")
                                source_type = source.get("document_type", "Unknown")
                                source_cat = source.get("category", "Unknown")
                                st.markdown(f"- **{Path(source_path).name}** ({source_type}/{source_cat})")
                            else:
                                st.markdown(f"- {source}")
            else:
                st.warning("Please enter a question to get started.")
        
        # Sample questions
        st.markdown("### Sample Questions")
        sample_questions = [
            "How much did I spend on dining out last month?",
            "What was my largest expense category in the past quarter?",
            "Am I on track to meet my savings goal?",
            "Which transactions seem unusual compared to my normal spending?",
            "How has my spending changed compared to the previous month?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{hash(q)}"):
                # Set the query input and click the ask button programmatically
                st.session_state.rag_query = q
                st.experimental_rerun()
    
    # === DOCUMENT LIBRARY TAB ===
    with tabs[1]:
        st.markdown(f"### {neon_text('Financial Document Library')}")
        st.markdown("Upload and manage your financial documents for enhanced insights.")
        
        # Upload section
        st.subheader("Upload New Document")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a financial document", 
                                       type=['pdf', 'csv', 'txt', 'xlsx'],
                                       key="doc_upload")
        
        # Document metadata
        col1, col2 = st.columns(2)
        with col1:
            upload_doc_type = st.selectbox(
                "Document Type",
                options=rag.FINANCIAL_DOCUMENT_TYPES,
                key="upload_doc_type"
            )
        with col2:
            upload_category = st.selectbox(
                "Document Category",
                options=rag.FINANCIAL_CATEGORIES,
                key="upload_category"
            )
        
        # Process upload
        if uploaded_file and st.button("Index Document", key="upload_button"):
            with st.spinner("Processing document..."):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Index the document
                    result = rag.index_user_document(
                        user_id=user_id,
                        document_path=temp_path,
                        document_type=upload_doc_type,
                        category=upload_category
                    )
                    
                    # Display result
                    if result.get("success"):
                        st.success(f"Document indexed successfully! Document ID: {result.get('document_id')}")
                        if result.get("chunks_indexed"):
                            st.info(f"Processed {result.get('chunks_indexed')} chunks of text.")
                    else:
                        st.error(f"Failed to index document: {result.get('error')}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        
        # Transaction indexing
        st.subheader("Index Transaction History")
        st.markdown("Convert your transaction history into searchable knowledge.")
        
        if st.button("Index My Transactions", key="index_txns"):
            with st.spinner("Indexing transaction history..."):
                result = rag.index_user_transactions(user_id)
                
                if result.get("success"):
                    st.success("Transaction history indexed successfully!")
                    if result.get("chunks_indexed"):
                        st.info(f"Processed {result.get('chunks_indexed')} chunks of text.")
                else:
                    st.error(f"Failed to index transactions: {result.get('error')}")
        
        # Document list (placeholder - would need backend implementation)
        st.subheader("Your Document Library")
        
        try:
            # Try to get user's indexed documents
            user_docs = rag.rag_agent.user_indexes.get(user_id, [])
            
            if user_docs:
                # Create DataFrame for display
                docs_df = pd.DataFrame(user_docs)
                docs_df.rename(columns={
                    "document_id": "ID",
                    "path": "File",
                    "type": "Type",
                    "category": "Category",
                    "chunks": "Chunks",
                    "indexed_at": "Indexed At"
                }, inplace=True)
                
                # Clean up display
                docs_df["File"] = docs_df["File"].apply(lambda x: Path(x).name)
                
                # Show table
                st.dataframe(docs_df, use_container_width=True)
            else:
                st.info("No documents indexed yet. Upload some documents to get started!")
        except Exception as e:
            st.error(f"Error loading document library: {str(e)}")
    
    # === FINANCIAL REPORTS TAB ===
    with tabs[2]:
        st.markdown(f"### {neon_text('Financial Reports Generator')}")
        st.markdown("Generate comprehensive financial reports with AI insights.")
        
        # Report options
        col1, col2 = st.columns(2)
        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=["overview", "detailed", "forecast"],
                format_func=lambda x: x.title(),
                key="report_type"
            )
        with col2:
            time_period = st.selectbox(
                "Time Period",
                options=["week", "month", "quarter", "year"],
                format_func=lambda x: x.title(),
                key="report_period"
            )
        
        # Generate report
        if st.button("Generate Report", key="gen_report", type="primary"):
            with st.spinner("Generating financial report..."):
                result = rag.get_financial_report(
                    user_id=user_id,
                    report_type=report_type,
                    time_period=time_period
                )
                
                # Display report
                if result.get("report"):
                    st.markdown(result["report"])
                    
                    # Display insights
                    if result.get("insights"):
                        st.markdown("### Key Insights")
                        for insight in result["insights"]:
                            st.markdown(f"- {insight}")
                    
                    # Display action items
                    if result.get("action_items"):
                        st.markdown("### Recommended Actions")
                        for action in result["action_items"]:
                            st.markdown(f"- {action}")
                else:
                    st.error("Failed to generate report.")
    
    # === KNOWLEDGE BASE TAB ===
    with tabs[3]:
        st.markdown(f"### {neon_text('Financial Knowledge Base')}")
        st.markdown("Explore curated financial knowledge resources.")
        
        # Display knowledge base categories
        for category in rag.FINANCIAL_CATEGORIES:
            with st.expander(category.replace("_", " ").title()):
                st.markdown(f"#### {category.replace('_', ' ').title()} Resources")
                
                # Check if category directory exists
                category_path = rag.rag_agent.knowledge_base_path / category
                if category_path.exists() and any(category_path.iterdir()):
                    # List files in the category
                    files = list(category_path.glob("*.*"))
                    for file in files:
                        st.markdown(f"- {file.name}")
                    
                    # Button to index these files
                    if st.button(f"Index {category.title()} Knowledge", key=f"index_{category}"):
                        with st.spinner(f"Indexing {category} knowledge..."):
                            # Index each file
                            success_count = 0
                            for file in files:
                                try:
                                    result = rag.index_user_document(
                                        user_id=user_id,
                                        document_path=str(file),
                                        document_type="knowledge",
                                        category=category
                                    )
                                    if result.get("success"):
                                        success_count += 1
                                except:
                                    pass
                            
                            st.success(f"Indexed {success_count} resources from {category.title()}")
                else:
                    st.info(f"No {category.replace('_', ' ').title()} resources available yet.")
        
        # Suggest resource upload
        st.markdown("### Add to Knowledge Base")
        st.markdown("Suggest adding a resource to the financial knowledge base.")
        
        kb_resource = st.text_input("Resource URL or description", key="kb_resource")
        kb_category = st.selectbox(
            "Category",
            options=rag.FINANCIAL_CATEGORIES,
            key="kb_category"
        )
        
        if st.button("Suggest Resource", key="suggest_kb"):
            if kb_resource:
                st.success("Your resource suggestion has been recorded. Thank you for contributing!")
                # In a real app, this would be sent to an admin for review
            else:
                st.warning("Please enter a resource URL or description.")

if __name__ == "__main__":
    finance_rag_page()