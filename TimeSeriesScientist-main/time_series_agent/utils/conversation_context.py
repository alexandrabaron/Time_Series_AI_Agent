"""
Conversation Context Builder for TSci Conversational Agent
Extracts and formats context from session state for LLM consumption.
"""

import logging
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)


class ConversationContextBuilder:
    """
    Builds structured context from session state for conversational LLM.
    """
    
    @staticmethod
    def build_context(session_state: Optional[Any] = None) -> Dict[str, Any]:
        """
        Build a comprehensive context dictionary from session state.
        
        Args:
            session_state: Streamlit session state object (defaults to st.session_state)
            
        Returns:
            Dictionary containing structured context
        """
        if session_state is None:
            session_state = st.session_state
        
        context = {}
        
        # Dataset information
        context['dataset'] = ConversationContextBuilder._extract_dataset_info(session_state)
        
        # Preprocessing results
        context['preprocessing'] = ConversationContextBuilder._extract_preprocessing_info(session_state)
        
        # Analysis results
        context['analysis'] = ConversationContextBuilder._extract_analysis_info(session_state)
        
        # Model recommendations
        context['recommendations'] = ConversationContextBuilder._extract_recommendations(session_state)
        
        # Current workflow step
        context['current_step'] = session_state.get('current_step', 'initial')
        
        # Configuration
        context['config'] = ConversationContextBuilder._extract_config_info(session_state)
        
        return context
    
    @staticmethod
    def _extract_dataset_info(session_state) -> Dict[str, Any]:
        """Extract dataset information."""
        if not hasattr(session_state, 'dataset_info') or not session_state.dataset_info:
            return {}
        
        ds_info = session_state.dataset_info
        
        return {
            'name': ds_info.get('filename', 'Unknown'),
            'num_rows': ds_info.get('num_rows', 0),
            'num_columns': ds_info.get('num_columns', 0),
            'memory_usage': ds_info.get('memory_usage', 0),
            'target_col': session_state.get('target_col', 'N/A'),
            'date_col': session_state.get('date_col', 'N/A'),
        }
    
    @staticmethod
    def _extract_preprocessing_info(session_state) -> Dict[str, Any]:
        """Extract preprocessing results and strategies."""
        prep_info = {}
        
        # Check if preprocessing analysis exists
        if hasattr(session_state, 'results') and 'preprocess_analysis' in session_state.results:
            analysis = session_state.results['preprocess_analysis']
            
            if 'details' in analysis:
                details = analysis['details']
                
                # Missing values info
                if 'missing_values' in details:
                    prep_info['missing_values'] = {
                        'count': details['missing_values'].get('count', 0),
                        'percentage': details['missing_values'].get('percentage', 0)
                    }
                
                # Outliers info
                if 'outliers' in details:
                    prep_info['outliers'] = {
                        'count': details['outliers'].get('count', 0),
                        'percentage': details['outliers'].get('percentage', 0)
                    }
                
                # Data quality
                if 'quality' in details:
                    prep_info['quality_score'] = details['quality'].get('score', 0)
        
        # Check for applied strategies
        if hasattr(session_state, 'preprocess_options') and session_state.preprocess_options:
            options = session_state.preprocess_options
            if 'options' in options:
                # Extract applied strategies from options
                for opt in options['options']:
                    if opt['type'] == 'missing':
                        prep_info['missing_strategy'] = opt['strategy']
                        prep_info['missing_reason'] = opt.get('reason', '')
                    elif opt['type'] == 'outlier':
                        prep_info['outlier_strategy'] = opt['strategy']
                        prep_info['outlier_reason'] = opt.get('reason', '')
        
        return prep_info
    
    @staticmethod
    def _extract_analysis_info(session_state) -> Dict[str, Any]:
        """Extract analysis results."""
        analysis_info = {}
        
        if hasattr(session_state, 'results') and 'analysis_results' in session_state.results:
            analysis = session_state.results['analysis_results']
            
            # Check if results exist
            if 'results' in analysis:
                results = analysis['results']
                
                # Trend analysis
                if 'trend' in results:
                    trend = results['trend']
                    analysis_info['trend'] = {
                        'direction': trend.get('direction', 'Unknown'),
                        'strength': trend.get('strength', 'Unknown'),
                        'slope': trend.get('slope', 0)
                    }
                
                # Seasonality analysis
                if 'seasonality' in results:
                    seasonality = results['seasonality']
                    analysis_info['seasonality'] = {
                        'detected': seasonality.get('detected', False),
                        'period': seasonality.get('period'),
                        'strength': seasonality.get('strength', 0),
                        'strength_label': seasonality.get('strength_label', 'Unknown')
                    }
                
                # Stationarity tests
                if 'stationarity' in results:
                    stationarity = results['stationarity']
                    analysis_info['stationarity'] = {
                        'conclusion': stationarity.get('conclusion', 'Unknown'),
                        'needs_differencing': stationarity.get('needs_differencing', False),
                        'adf_pvalue': stationarity.get('adf_pvalue', 1.0),
                        'kpss_pvalue': stationarity.get('kpss_pvalue', 0.0)
                    }
                
                # Descriptive statistics
                if 'statistics' in results:
                    stats = results['statistics']
                    analysis_info['statistics'] = {
                        'mean': stats.get('mean', 0),
                        'median': stats.get('median', 0),
                        'std': stats.get('std', 0),
                        'min': stats.get('min', 0),
                        'max': stats.get('max', 0)
                    }
        
        return analysis_info
    
    @staticmethod
    def _extract_recommendations(session_state) -> list:
        """Extract model recommendations."""
        if hasattr(session_state, 'results') and 'analysis_results' in session_state.results:
            analysis = session_state.results['analysis_results']
            return analysis.get('recommendations', [])
        
        return []
    
    @staticmethod
    def _extract_config_info(session_state) -> Dict[str, Any]:
        """Extract relevant configuration settings."""
        config_info = {}
        
        if hasattr(session_state, 'config'):
            config = session_state.config
            config_info['horizon'] = config.get('horizon', 12)
            config_info['seasonal_period'] = config.get('seasonal_period', 'auto')
            config_info['num_models'] = config.get('num_models', 3)
        
        return config_info
    
    @staticmethod
    def format_context_for_display(context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a human-readable string for display.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted string
        """
        parts = []
        
        # Dataset
        if context.get('dataset'):
            ds = context['dataset']
            parts.append("📊 **Dataset**")
            parts.append(f"  - Fichier : {ds.get('name', 'N/A')}")
            parts.append(f"  - Taille : {ds.get('num_rows', 0)} lignes × {ds.get('num_columns', 0)} colonnes")
            parts.append(f"  - Cible : {ds.get('target_col', 'N/A')}")
            parts.append("")
        
        # Preprocessing
        if context.get('preprocessing'):
            prep = context['preprocessing']
            parts.append("🧹 **Prétraitement**")
            if 'missing_values' in prep:
                parts.append(f"  - Valeurs manquantes : {prep['missing_values']['count']} ({prep['missing_values']['percentage']:.2f}%)")
            if 'outliers' in prep:
                parts.append(f"  - Outliers : {prep['outliers']['count']} ({prep['outliers']['percentage']:.2f}%)")
            if 'quality_score' in prep:
                parts.append(f"  - Score de qualité : {prep['quality_score']:.2f}/1.0")
            parts.append("")
        
        # Analysis
        if context.get('analysis'):
            anal = context['analysis']
            parts.append("🔍 **Analyse**")
            if 'trend' in anal:
                parts.append(f"  - Tendance : {anal['trend']['direction']} ({anal['trend']['strength']})")
            if 'seasonality' in anal:
                if anal['seasonality']['detected']:
                    parts.append(f"  - Saisonnalité : Oui (période {anal['seasonality']['period']})")
                else:
                    parts.append(f"  - Saisonnalité : Non détectée")
            if 'stationarity' in anal:
                parts.append(f"  - Stationnarité : {anal['stationarity']['conclusion']}")
            parts.append("")
        
        # Recommendations
        if context.get('recommendations'):
            parts.append("💡 **Recommandations**")
            for i, rec in enumerate(context['recommendations'][:3], 1):
                parts.append(f"  {i}. {rec.get('model', 'N/A')}")
            parts.append("")
        
        # Current step
        if context.get('current_step'):
            parts.append(f"📍 **Étape actuelle** : {context['current_step']}")
        
        return "\n".join(parts) if parts else "Aucun contexte disponible."

