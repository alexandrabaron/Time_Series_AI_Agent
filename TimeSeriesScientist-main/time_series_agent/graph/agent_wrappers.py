"""
Agent Wrappers for TSci Conversational Interface
Wraps existing agents to make them conversational and approval-based.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import sys

# Add parent directory to path to import agents
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from agents.preprocess_agent import PreprocessAgent
from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent
from agents.forecast_agent import ForecastAgent
from agents.report_agent import ReportAgent
from config.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class PreprocessAgentWrapper:
    """
    Wrapper for PreprocessAgent that provides conversational interface.
    Analyzes data without modifying it, then asks for user approval.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PreprocessAgent wrapper.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or DEFAULT_CONFIG.copy()
        # We'll initialize the agent when needed to avoid loading at import time
        self.agent = None
    
    def _get_agent(self):
        """Lazy initialization of the agent."""
        if self.agent is None:
            self.agent = PreprocessAgent(
                model=self.config.get('llm_model', 'gpt-4o'),
                config=self.config
            )
        return self.agent
    
    def run(self, data: pd.DataFrame, target_col: str, date_col: str = None) -> Dict[str, Any]:
        """
        Analyze the data for preprocessing needs WITHOUT modifying it.
        Returns suggestions that require user approval.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            date_col: Name of the date column (optional)
            
        Returns:
            Dictionary with analysis results and suggested actions
        """
        try:
            logger.info("PreprocessAgentWrapper: Starting data analysis...")
            
            # Analyze missing values
            missing_analysis = self._analyze_missing_values(data, target_col)
            
            # Analyze outliers
            outlier_analysis = self._analyze_outliers(data, target_col)
            
            # Analyze data quality
            quality_analysis = self._analyze_data_quality(data, target_col)
            
            # Get LLM suggestions for strategies
            suggestions = self._get_llm_suggestions(
                missing_analysis, 
                outlier_analysis, 
                quality_analysis
            )
            
            # Format response
            response = {
                "status": "pending_approval",
                "summary": self._create_summary(missing_analysis, outlier_analysis, quality_analysis),
                "details": {
                    "missing_values": missing_analysis,
                    "outliers": outlier_analysis,
                    "quality": quality_analysis
                },
                "options": suggestions,
                "data_unchanged": True  # Important: data has not been modified
            }
            
            logger.info("PreprocessAgentWrapper: Analysis complete")
            return response
            
        except Exception as e:
            logger.error(f"PreprocessAgentWrapper error: {e}")
            return {
                "status": "error",
                "summary": f"Erreur lors de l'analyse : {str(e)}",
                "options": []
            }
    
    def _analyze_missing_values(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze missing values in the data."""
        total_values = len(data)
        missing_count = data[target_col].isnull().sum()
        missing_pct = (missing_count / total_values) * 100
        
        # Get positions of missing values
        missing_indices = data[target_col].isnull()
        
        # Analyze patterns (are they clustered or scattered?)
        if missing_count > 0:
            missing_positions = data[missing_indices].index.tolist()
            # Check if consecutive
            consecutive_groups = []
            if len(missing_positions) > 1:
                current_group = [missing_positions[0]]
                for i in range(1, len(missing_positions)):
                    if missing_positions[i] == missing_positions[i-1] + 1:
                        current_group.append(missing_positions[i])
                    else:
                        if len(current_group) > 1:
                            consecutive_groups.append(current_group)
                        current_group = [missing_positions[i]]
                if len(current_group) > 1:
                    consecutive_groups.append(current_group)
            
            pattern = "scattered" if len(consecutive_groups) == 0 else "clustered"
        else:
            pattern = "none"
        
        return {
            "count": int(missing_count),
            "percentage": float(missing_pct),
            "pattern": pattern,
            "severity": "high" if missing_pct > 10 else "medium" if missing_pct > 5 else "low"
        }
    
    def _analyze_outliers(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze outliers using IQR method."""
        values = data[target_col].dropna()
        
        if len(values) < 4:
            return {
                "count": 0,
                "percentage": 0.0,
                "method": "iqr",
                "severity": "none"
            }
        
        # IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(values)) * 100
        
        return {
            "count": int(outlier_count),
            "percentage": float(outlier_pct),
            "method": "iqr",
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            },
            "severity": "high" if outlier_pct > 10 else "medium" if outlier_pct > 5 else "low"
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze overall data quality."""
        values = data[target_col].dropna()
        
        quality_score = 1.0
        issues = []
        
        # Check missing values
        missing_pct = (data[target_col].isnull().sum() / len(data)) * 100
        if missing_pct > 0:
            quality_score -= min(missing_pct / 100, 0.3)
            issues.append(f"missing_values_{missing_pct:.1f}%")
        
        # Check variance
        if len(values) > 1:
            std = values.std()
            mean = values.mean()
            cv = std / mean if mean != 0 else 0
            if cv > 1.0:
                issues.append("high_variability")
        
        return {
            "score": float(max(0, quality_score)),
            "issues": issues,
            "data_points": len(data),
            "valid_points": len(values)
        }
    
    def _get_llm_suggestions(
        self, 
        missing_analysis: Dict[str, Any], 
        outlier_analysis: Dict[str, Any],
        quality_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get LLM-based suggestions for handling missing values and outliers.
        For now, this provides rule-based suggestions (LLM integration later).
        """
        suggestions = []
        
        # Missing value strategies
        if missing_analysis['count'] > 0:
            if missing_analysis['pattern'] == 'clustered':
                suggestions.append({
                    "type": "missing",
                    "strategy": "interpolate",
                    "reason": "Les valeurs manquantes sont groupées - l'interpolation linéaire est recommandée pour maintenir la continuité.",
                    "priority": 1
                })
                suggestions.append({
                    "type": "missing",
                    "strategy": "forward_fill",
                    "reason": "Alternative : propager la dernière valeur valide (simple mais peut introduire des plateaux).",
                    "priority": 2
                })
            else:
                suggestions.append({
                    "type": "missing",
                    "strategy": "interpolate",
                    "reason": "Les valeurs manquantes sont dispersées - l'interpolation est la meilleure approche pour les séries temporelles.",
                    "priority": 1
                })
                suggestions.append({
                    "type": "missing",
                    "strategy": "mean",
                    "reason": "Alternative : remplacer par la moyenne (simple mais peut réduire la variance).",
                    "priority": 2
                })
        
        # Outlier strategies
        if outlier_analysis['count'] > 0:
            if outlier_analysis['severity'] == 'high':
                suggestions.append({
                    "type": "outlier",
                    "strategy": "clip",
                    "reason": f"Beaucoup d'outliers détectés ({outlier_analysis['count']}). Le clipping ramène les valeurs aux limites IQR sans perdre de données.",
                    "priority": 1
                })
                suggestions.append({
                    "type": "outlier",
                    "strategy": "drop",
                    "reason": "Alternative : supprimer les outliers (peut créer des gaps dans la série).",
                    "priority": 2
                })
            else:
                suggestions.append({
                    "type": "outlier",
                    "strategy": "clip",
                    "reason": f"Quelques outliers détectés ({outlier_analysis['count']}). Le clipping est une approche conservatrice.",
                    "priority": 1
                })
                suggestions.append({
                    "type": "outlier",
                    "strategy": "interpolate",
                    "reason": "Alternative : traiter comme valeurs manquantes et interpoler.",
                    "priority": 2
                })
        
        # Sort by priority
        suggestions.sort(key=lambda x: x['priority'])
        
        return suggestions
    
    def _create_summary(
        self, 
        missing_analysis: Dict[str, Any], 
        outlier_analysis: Dict[str, Any],
        quality_analysis: Dict[str, Any]
    ) -> str:
        """Create a human-readable summary."""
        parts = []
        
        parts.append(f"📊 **Analyse des données terminée**\n")
        parts.append(f"Score de qualité : **{quality_analysis['score']:.2f}/1.0**\n")
        
        if missing_analysis['count'] > 0:
            parts.append(
                f"\n🔍 **Valeurs manquantes** : {missing_analysis['count']} "
                f"({missing_analysis['percentage']:.2f}%) - Pattern : {missing_analysis['pattern']}"
            )
        else:
            parts.append(f"\n✅ Aucune valeur manquante détectée")
        
        if outlier_analysis['count'] > 0:
            parts.append(
                f"\n⚠️ **Outliers détectés** : {outlier_analysis['count']} "
                f"({outlier_analysis['percentage']:.2f}%) - Méthode IQR"
            )
        else:
            parts.append(f"\n✅ Aucun outlier détecté")
        
        parts.append(f"\n\n💡 Suggestions de stratégies disponibles ci-dessous.")
        
        return "".join(parts)
    
    def apply_preprocessing(
        self, 
        data: pd.DataFrame, 
        target_col: str,
        missing_strategy: str = "interpolate",
        outlier_strategy: str = "clip"
    ) -> pd.DataFrame:
        """
        Actually apply the preprocessing strategies (called after approval).
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            missing_strategy: Strategy for handling missing values
            outlier_strategy: Strategy for handling outliers
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Applying preprocessing: missing={missing_strategy}, outliers={outlier_strategy}")
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Handle missing values
        if missing_strategy == "interpolate":
            df[target_col] = df[target_col].interpolate(method='linear')
        elif missing_strategy == "forward_fill":
            df[target_col] = df[target_col].fillna(method='ffill')
        elif missing_strategy == "backward_fill":
            df[target_col] = df[target_col].fillna(method='bfill')
        elif missing_strategy == "mean":
            df[target_col] = df[target_col].fillna(df[target_col].mean())
        elif missing_strategy == "median":
            df[target_col] = df[target_col].fillna(df[target_col].median())
        elif missing_strategy == "drop":
            df = df.dropna(subset=[target_col])
        
        # Handle outliers
        if outlier_strategy == "clip":
            values = df[target_col].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)
        elif outlier_strategy == "drop":
            values = df[target_col].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        elif outlier_strategy == "interpolate":
            # Replace outliers with NaN then interpolate
            values = df[target_col].copy()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[(df[target_col] < lower_bound) | (df[target_col] > upper_bound), target_col] = np.nan
            df[target_col] = df[target_col].interpolate(method='linear')
        
        logger.info(f"Preprocessing complete: {len(df)} rows remaining")
        return df


class AnalysisAgentWrapper:
    """Wrapper for AnalysisAgent (to be implemented in future steps)."""
    pass


class ValidationAgentWrapper:
    """Wrapper for ValidationAgent (to be implemented in future steps)."""
    pass


class ForecastAgentWrapper:
    """Wrapper for ForecastAgent (to be implemented in future steps)."""
    pass


class ReportAgentWrapper:
    """Wrapper for ReportAgent (to be implemented in future steps)."""
    pass

