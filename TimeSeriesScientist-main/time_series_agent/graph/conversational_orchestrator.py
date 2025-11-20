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
from utils.local_llm import LocalLLM
from utils.conversation_context import ConversationContextBuilder
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
        
        # Initialize Local LLM (Llama3 via Ollama)
        # Using llama3:8b for faster responses (vs llama3:70b)
        self.local_llm = LocalLLM(model="llama3:8b")
        logger.info(f"Local LLM available: {self.local_llm.is_available()}")
    
    def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point for user input.
        Routes to command handler or question answering based on intent.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dictionary with status and message
        """
        logger.info(f"Processing user input: {user_input}")
        
        # Detect intent (command vs question)
        intent = self.local_llm.detect_intent(user_input)
        logger.info(f"Detected intent: {intent}")
        
        # Route based on intent
        if intent == 'command':
            # Try to extract and handle command
            return self._handle_command_from_text(user_input)
        elif intent == 'question':
            # Answer the question using LLM with context
            return self._answer_question(user_input)
        else:
            # Fallback: try both approaches or provide suggestions
            return {
                'status': 'info',
                'message': f"🤔 Je n'ai pas bien compris votre demande : '{user_input}'\n\n💡 Vous pouvez :\n- Poser une question (ex: 'Quelle est la qualité de mes données ?')\n- Donner une commande (ex: 'Lance l'analyse')\n- Utiliser les boutons dans la sidebar"
            }
    
    def _handle_command_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract and handle command from natural language text.
        
        Args:
            text: User's command text
            
        Returns:
            Result dictionary
        """
        text_lower = text.lower()
        
        # Preprocessing commands
        if any(word in text_lower for word in ['prétraitement', 'pretraitement', 'nettoyer', 'nettoyage', 'clean']):
            return self.handle_command('start_preprocessing')
        
        # Analysis commands
        elif any(word in text_lower for word in ['analyse', 'analyser', 'statistical', 'statistique']):
            return self.handle_command('start_analysis')
        
        # Validation commands
        elif any(word in text_lower for word in ['validation', 'valider', 'modèle', 'model']):
            return self.handle_command('start_validation')
        
        # Forecast commands
        elif any(word in text_lower for word in ['prévision', 'prevision', 'forecast', 'prévoir', 'prevoir']):
            return self.handle_command('start_forecast')
        
        # Report commands
        elif any(word in text_lower for word in ['rapport', 'report', 'résumé', 'resume', 'summary']):
            return self.handle_command('generate_report')
        
        else:
            return {
                'status': 'info',
                'message': f"🤔 Commande non reconnue : '{text}'\n\n💡 Utilisez les boutons dans la sidebar pour naviguer."
            }
    
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
    
    def _answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a user question using Local LLM with context from session state.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with status and message
        """
        logger.info(f"Answering question: {question}")
        
        # Build context from session state
        context = ConversationContextBuilder.build_context(st.session_state)
        
        # Check if LLM is available
        if not self.local_llm.is_available():
            # Fallback to hardcoded responses
            return self._fallback_answer(question, context)
        
        # Use LLM to answer with context
        try:
            answer = self.local_llm.ask_with_context(
                question=question,
                context=context,
                role="assistant expert en séries temporelles"
            )
            
            return {
                'status': 'success',
                'message': answer
            }
            
        except Exception as e:
            logger.error(f"Error answering question with LLM: {e}")
            return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide fallback answers when LLM is unavailable.
        
        Args:
            question: User's question
            context: Context dictionary
            
        Returns:
            Dictionary with status and message
        """
        question_lower = question.lower()
        
        # Dataset questions
        if 'qualité' in question_lower or 'quality' in question_lower:
            if context.get('preprocessing', {}).get('quality_score'):
                score = context['preprocessing']['quality_score']
                return {
                    'status': 'success',
                    'message': f"📊 Score de qualité des données : **{score:.2f}/1.0**"
                }
            return {
                'status': 'info',
                'message': "❌ Aucune information de qualité disponible. Veuillez d'abord lancer le prétraitement."
            }
        
        # Missing values
        elif 'manquant' in question_lower or 'missing' in question_lower:
            if context.get('preprocessing', {}).get('missing_values'):
                mv = context['preprocessing']['missing_values']
                return {
                    'status': 'success',
                    'message': f"📊 Valeurs manquantes : **{mv['count']}** ({mv['percentage']:.2f}%)"
                }
            return {
                'status': 'info',
                'message': "❌ Aucune information disponible. Veuillez d'abord lancer le prétraitement."
            }
        
        # Outliers
        elif 'outlier' in question_lower:
            if context.get('preprocessing', {}).get('outliers'):
                outliers = context['preprocessing']['outliers']
                return {
                    'status': 'success',
                    'message': f"📊 Outliers détectés : **{outliers['count']}** ({outliers['percentage']:.2f}%)"
                }
            return {
                'status': 'info',
                'message': "❌ Aucune information disponible. Veuillez d'abord lancer le prétraitement."
            }
        
        # Trend
        elif 'tendance' in question_lower or 'trend' in question_lower:
            if context.get('analysis', {}).get('trend'):
                trend = context['analysis']['trend']
                return {
                    'status': 'success',
                    'message': f"📈 Tendance : **{trend['direction']}** (force: {trend['strength']})"
                }
            return {
                'status': 'info',
                'message': "❌ Aucune information disponible. Veuillez d'abord lancer l'analyse statistique."
            }
        
        # Seasonality
        elif 'saisonnal' in question_lower or 'seasonal' in question_lower:
            if context.get('analysis', {}).get('seasonality'):
                seasonality = context['analysis']['seasonality']
                if seasonality['detected']:
                    return {
                        'status': 'success',
                        'message': f"📊 Saisonnalité détectée : Oui\n- Période : **{seasonality['period']}**\n- Force : **{seasonality['strength_label']}**"
                    }
                else:
                    return {
                        'status': 'success',
                        'message': "📊 Aucune saisonnalité détectée dans vos données."
                    }
            return {
                'status': 'info',
                'message': "❌ Aucune information disponible. Veuillez d'abord lancer l'analyse statistique."
            }
        
        # Stationarity
        elif 'stationnaire' in question_lower or 'stationarity' in question_lower:
            if context.get('analysis', {}).get('stationarity'):
                stationarity = context['analysis']['stationarity']
                return {
                    'status': 'success',
                    'message': f"📊 Stationnarité : **{stationarity['conclusion']}**\n- Différenciation nécessaire : {'Oui' if stationarity['needs_differencing'] else 'Non'}"
                }
            return {
                'status': 'info',
                'message': "❌ Aucune information disponible. Veuillez d'abord lancer l'analyse statistique."
            }
        
        # Model recommendations
        elif 'modèle' in question_lower or 'model' in question_lower or 'recommand' in question_lower:
            if context.get('recommendations'):
                recs = context['recommendations'][:3]
                message = "💡 **Modèles recommandés** :\n\n"
                for i, rec in enumerate(recs, 1):
                    message += f"{i}. **{rec.get('model', 'N/A')}** : {rec.get('reason', 'N/A')}\n"
                return {
                    'status': 'success',
                    'message': message
                }
            return {
                'status': 'info',
                'message': "❌ Aucune recommandation disponible. Veuillez d'abord lancer l'analyse statistique."
            }
        
        # Default
        else:
            return {
                'status': 'info',
                'message': f"🤔 Je n'ai pas pu répondre à votre question : '{question}'\n\n💡 **Ollama n'est pas disponible**. Pour activer les réponses intelligentes, assurez-vous qu'Ollama est en cours d'exécution.\n\nEn attendant, je peux répondre à des questions simples sur :\n- La qualité des données\n- Les valeurs manquantes\n- Les outliers\n- La tendance\n- La saisonnalité\n- La stationnarité\n- Les recommandations de modèles"
            }
    
    def answer_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Legacy method for compatibility. Use _answer_question() instead.
        
        Args:
            question: User's question
            context: Optional context (ignored, uses session state)
            
        Returns:
            Answer string
        """
        result = self._answer_question(question)
        return result.get('message', 'Erreur lors du traitement de la question.')

