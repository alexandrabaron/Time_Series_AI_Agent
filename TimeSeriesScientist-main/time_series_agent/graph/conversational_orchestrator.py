"""
Conversational Orchestrator for TSci
Manages the conversation flow and coordinates between UI and agents.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from graph.agent_wrappers import (
    PreprocessAgentWrapper,
    AnalysisAgentWrapper,
    ValidationAgentWrapper,
    ForecastAgentWrapper,
    ReportAgentWrapper
)
from utils.session_manager import SessionManager
from config.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConversationalOrchestrator:
    """
    Orchestrates the conversational flow between user, UI, and agents.
    Manages state transitions and agent invocations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the conversational orchestrator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Initialize agent wrappers
        self.preprocess_wrapper = PreprocessAgentWrapper(self.config)
        self.analysis_wrapper = AnalysisAgentWrapper(self.config)
        self.validation_wrapper = None  # To be implemented
        self.forecast_wrapper = None  # To be implemented
        self.report_wrapper = None  # To be implemented
    
    def handle_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a user command and route to appropriate agent.
        
        Args:
            command: Command string (e.g., 'start_preprocessing', 'start_analysis')
            context: Optional context dictionary
            
        Returns:
            Dictionary with response information
        """
        logger.info(f"Handling command: {command}")
        
        try:
            if command == 'start_preprocessing':
                return self._handle_preprocessing()
            elif command == 'apply_preprocessing':
                return self._handle_apply_preprocessing(context)
            elif command == 'start_analysis':
                return self._handle_analysis()
            elif command == 'start_validation':
                return self._handle_validation()
            elif command == 'start_forecast':
                return self._handle_forecast()
            elif command == 'generate_report':
                return self._handle_report()
            else:
                return {
                    'status': 'error',
                    'message': f"Commande inconnue : {command}"
                }
        
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            return {
                'status': 'error',
                'message': f"Erreur lors de l'exécution : {str(e)}"
            }
    
    def _handle_preprocessing(self) -> Dict[str, Any]:
        """
        Handle the preprocessing command.
        Analyzes data and returns suggestions.
        """
        logger.info("Starting preprocessing analysis...")
        
        # Get data from session state
        if st.session_state.data is None:
            return {
                'status': 'error',
                'message': "❌ Aucune donnée chargée. Veuillez d'abord uploader un fichier CSV."
            }
        
        if 'target_col' not in st.session_state or 'date_col' not in st.session_state:
            return {
                'status': 'error',
                'message': "❌ Veuillez sélectionner les colonnes date et cible."
            }
        
        data = st.session_state.data
        target_col = st.session_state.target_col
        date_col = st.session_state.date_col
        
        # Run preprocessing analysis
        result = self.preprocess_wrapper.run(data, target_col, date_col)
        
        # Store result in session
        SessionManager.store_result('preprocess_analysis', result)
        
        # Format response for chat
        if result['status'] == 'pending_approval':
            # Create formatted message with options
            message = result['summary']
            
            if result['options']:
                message += "\n\n### 📋 Stratégies Recommandées\n"
                
                # Group by type
                missing_options = [opt for opt in result['options'] if opt['type'] == 'missing']
                outlier_options = [opt for opt in result['options'] if opt['type'] == 'outlier']
                
                if missing_options:
                    message += "\n**Valeurs Manquantes** :\n"
                    for i, opt in enumerate(missing_options, 1):
                        message += f"{i}. **{opt['strategy']}** : {opt['reason']}\n"
                
                if outlier_options:
                    message += "\n**Outliers** :\n"
                    for i, opt in enumerate(outlier_options, 1):
                        message += f"{i}. **{opt['strategy']}** : {opt['reason']}\n"
            
            message += "\n\n💬 **Que souhaitez-vous faire ?**"
            
            # Update state
            SessionManager.update_step('awaiting_preprocessing_approval')
            
            return {
                'status': 'success',
                'message': message,
                'requires_approval': True,
                'approval_type': 'preprocessing',
                'options': result['options']
            }
        else:
            return {
                'status': 'error',
                'message': result.get('summary', 'Erreur lors de l\'analyse')
            }
    
    def _handle_apply_preprocessing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the approved preprocessing strategies.
        
        Args:
            context: Dictionary with 'missing_strategy' and 'outlier_strategy'
        """
        logger.info("Applying preprocessing strategies...")
        
        if not context:
            return {
                'status': 'error',
                'message': "❌ Stratégies non spécifiées"
            }
        
        # Get strategies
        missing_strategy = context.get('missing_strategy', 'interpolate')
        outlier_strategy = context.get('outlier_strategy', 'clip')
        
        # Get data
        data = st.session_state.data
        target_col = st.session_state.target_col
        
        # Apply preprocessing
        preprocessed_data = self.preprocess_wrapper.apply_preprocessing(
            data, 
            target_col,
            missing_strategy,
            outlier_strategy
        )
        
        # Store preprocessed data
        st.session_state.preprocessed_data = preprocessed_data
        SessionManager.store_result('preprocess_applied', {
            'missing_strategy': missing_strategy,
            'outlier_strategy': outlier_strategy,
            'original_rows': len(data),
            'processed_rows': len(preprocessed_data)
        })
        
        # Update state
        SessionManager.update_step('preprocessing_complete')
        
        message = (
            f"✅ **Prétraitement appliqué avec succès !**\n\n"
            f"**Configuration** :\n"
            f"- Valeurs manquantes : `{missing_strategy}`\n"
            f"- Outliers : `{outlier_strategy}`\n\n"
            f"**Résultats** :\n"
            f"- Lignes originales : {len(data)}\n"
            f"- Lignes après traitement : {len(preprocessed_data)}\n\n"
            f"🎉 Vos données sont prêtes pour l'analyse !\n\n"
            f"💬 **Prochaine étape** : Voulez-vous lancer l'analyse statistique ?"
        )
        
        return {
            'status': 'success',
            'message': message,
            'next_step': 'analysis'
        }
    
    def _handle_analysis(self) -> Dict[str, Any]:
        """Handle the analysis command."""
        logger.info("Starting statistical analysis...")
        
        # Check if preprocessing is complete
        if 'preprocessed_data' not in st.session_state:
            # Try to use original data if preprocessing was skipped
            if st.session_state.data is None:
                return {
                    'status': 'error',
                    'message': "❌ Aucune donnée disponible. Veuillez d'abord charger et prétraiter les données."
                }
            
            # Use original data
            data = st.session_state.data[[st.session_state.target_col]].copy()
            data.rename(columns={st.session_state.target_col: 'value'}, inplace=True)
        else:
            data = st.session_state.preprocessed_data
        
        # Get configuration
        seasonal_period = st.session_state.config.get('seasonal_period', 'auto')
        
        # Run analysis
        result = self.analysis_wrapper.run(data, seasonal_period=seasonal_period)
        
        # Store result in session
        SessionManager.store_result('analysis', result)
        
        # Update state
        SessionManager.update_step('analysis_complete')
        
        if result['status'] == 'success':
            return {
                'status': 'success',
                'message': result['summary'],
                'visualizations': result.get('visualizations', {}),
                'recommendations': result.get('recommendations', []),
                'next_step': 'validation'
            }
        else:
            return {
                'status': 'error',
                'message': result.get('summary', 'Erreur lors de l\'analyse')
            }
    
    def _handle_validation(self) -> Dict[str, Any]:
        """Handle validation command (to be implemented)."""
        return {
            'status': 'info',
            'message': "🚧 L'agent de validation sera connecté dans la prochaine étape."
        }
    
    def _handle_forecast(self) -> Dict[str, Any]:
        """Handle forecast command (to be implemented)."""
        return {
            'status': 'info',
            'message': "🚧 L'agent de prévision sera connecté dans la prochaine étape."
        }
    
    def _handle_report(self) -> Dict[str, Any]:
        """Handle report generation command (to be implemented)."""
        return {
            'status': 'info',
            'message': "🚧 L'agent de rapport sera connecté dans la prochaine étape."
        }
    
    def answer_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Answer a user question about the current state/data.
        
        Args:
            question: User's question
            context: Optional context
            
        Returns:
            Answer string
        """
        # For now, provide simple answers (LLM integration in future)
        question_lower = question.lower()
        
        if 'qualité' in question_lower or 'quality' in question_lower:
            analysis = SessionManager.get_result('preprocess_analysis')
            if analysis:
                quality = analysis['details']['quality']
                return f"Score de qualité des données : **{quality['score']:.2f}/1.0**"
            return "Veuillez d'abord lancer l'analyse des données."
        
        elif 'manquant' in question_lower or 'missing' in question_lower:
            analysis = SessionManager.get_result('preprocess_analysis')
            if analysis:
                missing = analysis['details']['missing_values']
                return f"Valeurs manquantes : **{missing['count']}** ({missing['percentage']:.2f}%)"
            return "Veuillez d'abord lancer l'analyse des données."
        
        elif 'outlier' in question_lower:
            analysis = SessionManager.get_result('preprocess_analysis')
            if analysis:
                outliers = analysis['details']['outliers']
                return f"Outliers détectés : **{outliers['count']}** ({outliers['percentage']:.2f}%)"
            return "Veuillez d'abord lancer l'analyse des données."
        
        else:
            return "🤔 Je n'ai pas compris votre question. Pouvez-vous reformuler ?"

