"""
Chat Component for TSci Conversational Agent
Handles the display and interaction of the chat interface.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime


def display_chat_history():
    """
    Display the chat conversation history using Streamlit's chat components.
    Reads messages from st.session_state.messages and displays them.
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display each message in the conversation
    for message in st.session_state.messages:
        role = message.get('role', 'assistant')
        content = message.get('content', '')
        metadata = message.get('metadata', {})
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Display visualizations if present in metadata
            if 'visualization' in metadata:
                st.image(metadata['visualization'])
            
            # Display data tables if present in metadata
            if 'dataframe' in metadata:
                st.dataframe(metadata['dataframe'])


def display_welcome_message():
    """Display the initial welcome message for new sessions."""
    welcome_text = """
👋 **Bonjour ! Je suis TSci-Chat, votre assistant de prévision de séries temporelles.**

Je peux vous aider à :
- 📊 Analyser vos données de séries temporelles
- 🧹 Prétraiter et nettoyer vos données
- 📈 Détecter les tendances et la saisonnalité
- 🤖 Sélectionner et entraîner des modèles de prévision
- 📉 Générer des prévisions avec intervalles de confiance
- 📝 Créer des rapports détaillés

**Pour commencer**, vous pouvez :
1. Uploader un fichier CSV dans la barre latérale
2. Ou me poser des questions comme :
   - "Comment fonctionne le prétraitement ?"
   - "Quels modèles sont disponibles ?"
   - "Montre-moi un exemple"

**Que souhaitez-vous faire ?**
"""
    
    # Add welcome message to session state if not already present
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            'role': 'assistant',
            'content': welcome_text,
            'timestamp': datetime.now().isoformat(),
            'metadata': {}
        })


def add_user_message(content: str):
    """
    Add a user message to the conversation.
    
    Args:
        content: The user's message text
    """
    message = {
        'role': 'user',
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'metadata': {}
    }
    st.session_state.messages.append(message)


def add_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Add an assistant message to the conversation.
    
    Args:
        content: The assistant's response text
        metadata: Optional metadata (visualizations, dataframes, etc.)
    """
    message = {
        'role': 'assistant',
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    st.session_state.messages.append(message)


def display_chat_input():
    """
    Display the chat input box and handle user input.
    Returns the user input if available, None otherwise.
    """
    user_input = st.chat_input("Posez-moi une question ou donnez-moi une instruction...")
    
    if user_input:
        # Add user message to history
        add_user_message(user_input)
        return user_input
    
    return None


def display_suggested_questions():
    """Display suggested questions based on current context."""
    st.markdown("### 💡 Questions suggérées")
    
    # Get current step to provide context-appropriate suggestions
    current_step = st.session_state.get('current_step', 'initial')
    
    if current_step == 'initial':
        suggestions = [
            "Comment uploader mes données ?",
            "Quels formats de fichiers sont supportés ?",
            "Montre-moi un exemple de workflow"
        ]
    elif current_step == 'preprocessing':
        suggestions = [
            "Pourquoi ces valeurs sont des outliers ?",
            "Quelle est la qualité de mes données ?",
            "Montre-moi les statistiques"
        ]
    elif current_step == 'analysis':
        suggestions = [
            "Y a-t-il une tendance dans mes données ?",
            "Détecte-t-on de la saisonnalité ?",
            "Les données sont-elles stationnaires ?"
        ]
    elif current_step == 'validation':
        suggestions = [
            "Pourquoi ces modèles ont été sélectionnés ?",
            "Quels sont les hyperparamètres optimaux ?",
            "Compare les performances des modèles"
        ]
    elif current_step == 'forecast':
        suggestions = [
            "Montre-moi les prévisions",
            "Quelle est la confiance des prévisions ?",
            "Compare les modèles individuels"
        ]
    else:
        suggestions = [
            "Résume les résultats",
            "Quels sont les points clés ?",
            "Export les résultats"
        ]
    
    # Display suggestions as buttons
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                add_user_message(suggestion)
                st.rerun()


def display_pending_approval():
    """Display a pending approval if one exists."""
    if st.session_state.get('pending_approval'):
        approval_data = st.session_state.pending_approval
        
        st.warning("⚠️ **Approbation requise**")
        st.info(approval_data.get('message', 'Une décision nécessite votre approbation.'))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approuver", key="approve_btn"):
                return 'approve'
        with col2:
            if st.button("❌ Rejeter", key="reject_btn"):
                return 'reject'
    
    return None


def clear_chat_history():
    """Clear the entire chat history."""
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        display_welcome_message()
        st.rerun()

