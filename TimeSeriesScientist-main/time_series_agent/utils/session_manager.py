"""
Session Manager for TSci Conversational Agent
Manages user session state, conversation history, and workflow progress.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


class SessionManager:
    """
    Manages Streamlit session state for the conversational time series agent.
    Handles initialization, state transitions, and data persistence.
    """
    
    @staticmethod
    def initialize_session():
        """
        Initialize the session state with default values.
        This should be called at the start of the Streamlit app.
        """
        # Session ID (unique identifier for this session)
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # Current workflow step
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'initial'  # initial, preprocessing, analysis, validation, forecast, report
        
        # Conversation messages
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Dataset information
        if 'data' not in st.session_state:
            st.session_state.data = None
        
        if 'dataset_info' not in st.session_state:
            st.session_state.dataset_info = {
                'name': None,
                'uploaded_at': None,
                'num_rows': None,
                'num_columns': None,
                'columns': None
            }
        
        # Results from each agent
        if 'results' not in st.session_state:
            st.session_state.results = {
                'preprocess': None,
                'analysis': None,
                'validation': None,
                'forecast': None,
                'report': None
            }
        
        # Pending approvals (decisions waiting for user confirmation)
        if 'pending_approval' not in st.session_state:
            st.session_state.pending_approval = None
        
        # Configuration
        if 'config' not in st.session_state:
            st.session_state.config = {
                'horizon': 96,
                'input_length': 512,
                'num_models': 3,
                'confidence_level': 0.95
            }
        
        # Session metadata
        if 'session_metadata' not in st.session_state:
            st.session_state.session_metadata = {
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
    
    @staticmethod
    def update_step(new_step: str):
        """Update the current workflow step."""
        st.session_state.current_step = new_step
        SessionManager.update_activity()
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (e.g., visualizations, data)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        st.session_state.messages.append(message)
        SessionManager.update_activity()
    
    @staticmethod
    def update_activity():
        """Update last activity timestamp."""
        st.session_state.session_metadata['last_activity'] = datetime.now().isoformat()
    
    @staticmethod
    def set_pending_approval(approval_data: Dict[str, Any]):
        """Set a pending approval waiting for user confirmation."""
        st.session_state.pending_approval = approval_data
    
    @staticmethod
    def clear_pending_approval():
        """Clear the pending approval."""
        st.session_state.pending_approval = None
    
    @staticmethod
    def update_config(key: str, value: Any):
        """Update a configuration parameter."""
        st.session_state.config[key] = value
        SessionManager.update_activity()
    
    @staticmethod
    def store_result(agent_name: str, result: Any):
        """Store the result from an agent."""
        st.session_state.results[agent_name] = result
        SessionManager.update_activity()
    
    @staticmethod
    def get_result(agent_name: str) -> Any:
        """Retrieve the result from an agent."""
        return st.session_state.results.get(agent_name)
    
    @staticmethod
    def clear_session():
        """Clear all session data (reset)."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionManager.initialize_session()
    
    @staticmethod
    def get_session_summary() -> Dict[str, Any]:
        """Get a summary of the current session state."""
        return {
            'session_id': st.session_state.session_id,
            'current_step': st.session_state.current_step,
            'num_messages': len(st.session_state.messages),
            'has_data': st.session_state.data is not None,
            'dataset_info': st.session_state.dataset_info,
            'config': st.session_state.config,
            'metadata': st.session_state.session_metadata
        }


def initialize_session():
    """
    Convenience function to initialize the session.
    Can be imported directly in streamlit_app.py
    """
    SessionManager.initialize_session()

