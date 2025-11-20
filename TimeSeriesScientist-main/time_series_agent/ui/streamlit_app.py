"""
TSci-Chat - Conversational Time Series Forecasting Agent
Main Streamlit application file.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import from utils and ui
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.session_manager import SessionManager, initialize_session
from utils.dataset_manager import DatasetManager
from graph.conversational_orchestrator import ConversationalOrchestrator
from ui.components.chat import (
    display_chat_history,
    display_welcome_message,
    display_chat_input,
    display_suggested_questions,
    display_pending_approval,
    clear_chat_history,
    add_user_message,
    add_assistant_message
)


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="TSci-Chat - Time Series Forecasting Agent",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session()
    
    # Add welcome message if this is a new session
    display_welcome_message()
    
    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = ConversationalOrchestrator()
    
    # App header
    st.title("📈 TSci-Chat")
    st.markdown("*Assistant conversationnel pour la prévision de séries temporelles*")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Panneau de Contrôle")
        
        # Session info
        with st.expander("ℹ️ Informations de Session", expanded=False):
            session_info = SessionManager.get_session_summary()
            st.json(session_info)
        
        st.divider()
        
        # Dataset section - ÉTAPE 2
        st.header("📁 Datasets")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Uploader un CSV",
            type=['csv'],
            help="Sélectionnez un fichier CSV contenant vos données de séries temporelles"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            # Check if this is a new file
            if st.session_state.dataset_info.get('name') != uploaded_file.name:
                with st.spinner("Chargement du fichier..."):
                    success = DatasetManager.load_csv_to_session(uploaded_file)
                    
                    if success:
                        # Add message to chat
                        add_assistant_message(
                            f"✅ **Fichier chargé avec succès !**\n\n"
                            f"📊 **{uploaded_file.name}**\n"
                            f"- Lignes : {st.session_state.dataset_info['num_rows']}\n"
                            f"- Colonnes : {st.session_state.dataset_info['num_columns']}\n"
                            f"- Taille : {st.session_state.dataset_info['memory_usage']:.2f} MB\n\n"
                            f"Veuillez maintenant sélectionner les colonnes **date** et **valeur cible**."
                        )
                        st.rerun()
        
        # Display dataset info if loaded
        if st.session_state.data is not None:
            st.success(f"✅ {st.session_state.dataset_info['name']}")
            st.caption(f"{st.session_state.dataset_info['num_rows']} lignes × {st.session_state.dataset_info['num_columns']} colonnes")
            
            # Column selection
            st.subheader("🎯 Sélection des Colonnes")
            
            df = st.session_state.data
            columns = list(df.columns)
            
            # Auto-detect date and target columns
            suggested_date_cols = DatasetManager.detect_date_columns(df)
            suggested_target_cols = DatasetManager.detect_target_columns(df)
            
            # Date column selection
            default_date_idx = 0
            if suggested_date_cols and suggested_date_cols[0] in columns:
                default_date_idx = columns.index(suggested_date_cols[0])
            
            date_col = st.selectbox(
                "📅 Colonne Date/Temps",
                options=columns,
                index=default_date_idx,
                help="Colonne contenant les timestamps ou dates"
            )
            st.session_state.date_col = date_col
            
            # Target column selection
            default_target_idx = 0
            if suggested_target_cols and suggested_target_cols[0] in columns:
                default_target_idx = columns.index(suggested_target_cols[0])
            
            target_col = st.selectbox(
                "🎯 Colonne Valeur Cible",
                options=columns,
                index=default_target_idx,
                help="Colonne contenant les valeurs à prévoir"
            )
            st.session_state.target_col = target_col
            
            # Show preview
            with st.expander("👁️ Aperçu des Données", expanded=False):
                preview_df = DatasetManager.get_dataset_preview(df, n_rows=5)
                st.dataframe(preview_df, use_container_width=True)
            
            # Validation
            validation = DatasetManager.validate_dataset(df)
            
            if validation['warnings']:
                with st.expander("⚠️ Avertissements", expanded=False):
                    for warning in validation['warnings']:
                        st.warning(warning)
            
            if validation['errors']:
                for error in validation['errors']:
                    st.error(error)
            
            # Preprocess button (only show if columns are selected and data is valid)
            if date_col and target_col and validation['is_valid']:
                st.divider()
                
                # Check current step to show appropriate button
                current_step = st.session_state.current_step
                
                if current_step not in ['awaiting_preprocessing_approval', 'preprocessing_complete']:
                    if st.button("🚀 1. Lancer le Pré-traitement", type="primary", use_container_width=True):
                        # Add user message
                        add_user_message("Lancer le pré-traitement")
                        
                        # Call orchestrator
                        with st.spinner("Analyse des données en cours..."):
                            result = st.session_state.orchestrator.handle_command('start_preprocessing')
                        
                        # Add assistant response
                        add_assistant_message(result['message'])
                        
                        st.rerun()
                
                elif current_step == 'awaiting_preprocessing_approval':
                    # Show approval options
                    st.info("⏳ En attente de votre décision sur les stratégies de prétraitement")
                    
                    # Get the analysis result
                    analysis = SessionManager.get_result('preprocess_analysis')
                    
                    if analysis and analysis.get('options'):
                        # Extract default strategies (highest priority)
                        missing_options = [opt for opt in analysis['options'] if opt['type'] == 'missing']
                        outlier_options = [opt for opt in analysis['options'] if opt['type'] == 'outlier']
                        
                        default_missing = missing_options[0]['strategy'] if missing_options else 'interpolate'
                        default_outlier = outlier_options[0]['strategy'] if outlier_options else 'clip'
                        
                        st.markdown("**🎯 Sélectionnez vos stratégies :**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            missing_strategy = st.selectbox(
                                "Valeurs manquantes",
                                options=['interpolate', 'forward_fill', 'backward_fill', 'mean', 'median', 'drop'],
                                index=['interpolate', 'forward_fill', 'backward_fill', 'mean', 'median', 'drop'].index(default_missing)
                            )
                        
                        with col2:
                            outlier_strategy = st.selectbox(
                                "Outliers",
                                options=['clip', 'drop', 'interpolate'],
                                index=['clip', 'drop', 'interpolate'].index(default_outlier)
                            )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("✅ Appliquer", type="primary", use_container_width=True):
                                add_user_message(f"Appliquer : {missing_strategy} + {outlier_strategy}")
                                
                                with st.spinner("Application du prétraitement..."):
                                    result = st.session_state.orchestrator.handle_command(
                                        'apply_preprocessing',
                                        context={
                                            'missing_strategy': missing_strategy,
                                            'outlier_strategy': outlier_strategy
                                        }
                                    )
                                
                                add_assistant_message(result['message'])
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Annuler", use_container_width=True):
                                add_user_message("Annuler le prétraitement")
                                add_assistant_message("Prétraitement annulé. Vous pouvez relancer l'analyse quand vous voulez.")
                                SessionManager.update_step('initial')
                                st.rerun()
                
                elif current_step == 'preprocessing_complete':
                    st.success("✅ Prétraitement terminé")
                    if st.button("📊 2. Lancer l'Analyse Statistique", type="primary", use_container_width=True):
                        add_user_message("Lancer l'analyse statistique")
                        
                        with st.spinner("Analyse statistique en cours..."):
                            result = st.session_state.orchestrator.handle_command('start_analysis')
                        
                        # Prepare metadata with visualizations
                        metadata = {}
                        if 'visualizations' in result and result['visualizations']:
                            metadata['visualizations'] = result['visualizations']
                        
                        add_assistant_message(result['message'], metadata=metadata)
                        st.rerun()
                
                elif current_step == 'analysis_complete':
                    st.success("✅ Analyse terminée")
                    if st.button("🎯 3. Sélection de Modèles", type="primary", use_container_width=True):
                        add_user_message("Lancer la sélection de modèles")
                        result = st.session_state.orchestrator.handle_command('start_validation')
                        add_assistant_message(result['message'])
                        st.rerun()
        
        st.divider()
        
        # Configuration section
        st.header("⚙️ Configuration")
        
        # Seasonal period configuration
        st.subheader("📊 Analyse")
        
        seasonal_options = {
            'Détection automatique': 'auto',
            '7 (Hebdomadaire)': 7,
            '12 (Mensuelle)': 12,
            '24 (Journalière - données horaires)': 24,
            '168 (Hebdomadaire - données horaires)': 168,
            'Personnalisée': 'custom'
        }
        
        seasonal_choice = st.selectbox(
            "Période saisonnière",
            options=list(seasonal_options.keys()),
            index=0,
            help="Période pour l'analyse de saisonnalité"
        )
        
        if seasonal_options[seasonal_choice] == 'custom':
            custom_period = st.number_input(
                "Période personnalisée",
                min_value=2,
                max_value=365,
                value=12,
                help="Entrez une période personnalisée"
            )
            SessionManager.update_config('seasonal_period', custom_period)
        else:
            SessionManager.update_config('seasonal_period', seasonal_options[seasonal_choice])
        
        st.divider()
        
        st.subheader("🔮 Prévision")
        
        horizon = st.number_input(
            "Horizon de prévision",
            min_value=1,
            max_value=500,
            value=st.session_state.config['horizon'],
            help="Nombre de pas de temps à prévoir"
        )
        if horizon != st.session_state.config['horizon']:
            SessionManager.update_config('horizon', horizon)
        
        num_models = st.slider(
            "Nombre de modèles",
            min_value=1,
            max_value=10,
            value=st.session_state.config['num_models'],
            help="Nombre de modèles à sélectionner"
        )
        if num_models != st.session_state.config['num_models']:
            SessionManager.update_config('num_models', num_models)
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Réinitialiser la Session", use_container_width=True):
            SessionManager.clear_session()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Conversation")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Check for pending approvals
        approval_response = display_pending_approval()
        if approval_response:
            st.session_state.messages.append({
                'role': 'user',
                'content': f"{'✅ Approuvé' if approval_response == 'approve' else '❌ Rejeté'}",
                'timestamp': '',
                'metadata': {}
            })
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"Décision enregistrée : {approval_response}",
                'timestamp': '',
                'metadata': {}
            })
            SessionManager.clear_pending_approval()
            st.rerun()
        
        # Chat input
        user_input = display_chat_input()
        
        if user_input:
            # For now, just echo back (will be replaced with orchestrator logic)
            response = f"🤖 Vous avez dit : '{user_input}'\n\n*Note: L'orchestrateur conversationnel sera connecté dans la prochaine étape.*"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'timestamp': '',
                'metadata': {}
            })
            st.rerun()
    
    with col2:
        # Suggested questions panel
        st.subheader("💡 Suggestions")
        display_suggested_questions()
        
        st.divider()
        
        # Current step indicator
        st.subheader("📍 Étape Actuelle")
        current_step = st.session_state.current_step
        
        step_emoji = {
            'initial': '🏁',
            'preprocessing': '🧹',
            'analysis': '🔍',
            'validation': '✅',
            'forecast': '📈',
            'report': '📝'
        }
        
        step_label = {
            'initial': 'Initial',
            'preprocessing': 'Prétraitement',
            'analysis': 'Analyse',
            'validation': 'Validation',
            'forecast': 'Prévision',
            'report': 'Rapport'
        }
        
        st.info(f"{step_emoji.get(current_step, '🔵')} **{step_label.get(current_step, current_step)}**")
        
        st.divider()
        
        # Dataset info (if loaded)
        if st.session_state.data is not None:
            st.subheader("📊 Données Chargées")
            st.success(f"✅ {st.session_state.dataset_info['num_rows']} lignes")
            
            # Show column selections if available
            if 'date_col' in st.session_state and 'target_col' in st.session_state:
                st.caption(f"📅 Date : **{st.session_state.date_col}**")
                st.caption(f"🎯 Cible : **{st.session_state.target_col}**")
        else:
            st.subheader("📊 Données")
            st.warning("Aucune donnée chargée")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        TSci-Chat v0.1 - Agent Conversationnel de Prévision de Séries Temporelles
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

