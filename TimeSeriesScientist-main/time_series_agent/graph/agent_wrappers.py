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
            Preprocessed DataFrame with standardized 'value' column
        """
        logger.info(f"Applying preprocessing: missing={missing_strategy}, outliers={outlier_strategy}")
        
        # Create a copy with only target column and rename to 'value'
        df = data[[target_col]].copy()
        df.rename(columns={target_col: 'value'}, inplace=True)
        target_col = 'value'  # Use standardized name
        
        # Handle missing values (target_col is now 'value')
        if missing_strategy == "interpolate":
            df['value'] = df['value'].interpolate(method='linear')
        elif missing_strategy == "forward_fill":
            df['value'] = df['value'].fillna(method='ffill')
        elif missing_strategy == "backward_fill":
            df['value'] = df['value'].fillna(method='bfill')
        elif missing_strategy == "mean":
            df['value'] = df['value'].fillna(df['value'].mean())
        elif missing_strategy == "median":
            df['value'] = df['value'].fillna(df['value'].median())
        elif missing_strategy == "drop":
            df = df.dropna(subset=['value'])
        
        # Handle outliers (using standardized 'value' column)
        if outlier_strategy == "clip":
            values = df['value'].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)
        elif outlier_strategy == "drop":
            values = df['value'].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
        elif outlier_strategy == "interpolate":
            # Replace outliers with NaN then interpolate
            values = df['value'].copy()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[(df['value'] < lower_bound) | (df['value'] > upper_bound), 'value'] = np.nan
            df['value'] = df['value'].interpolate(method='linear')
        
        logger.info(f"Preprocessing complete: {len(df)} rows remaining")
        return df


class AnalysisAgentWrapper:
    """
    Wrapper for AnalysisAgent that provides conversational interface.
    Performs comprehensive statistical analysis of time series data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AnalysisAgent wrapper.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.agent = None
    
    def _get_agent(self):
        """Lazy initialization of the agent."""
        if self.agent is None:
            self.agent = AnalysisAgent(
                model=self.config.get('llm_model', 'gpt-4o'),
                config=self.config
            )
        return self.agent
    
    def run(self, data: pd.DataFrame, seasonal_period: Any = 'auto') -> Dict[str, Any]:
        """
        Perform comprehensive analysis of time series data.
        
        Args:
            data: Input DataFrame (preprocessed)
            seasonal_period: Seasonal period ('auto', int, or None)
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            logger.info("AnalysisAgentWrapper: Starting comprehensive analysis...")
            
            # 1. Trend Analysis
            trend_results = self._analyze_trend(data)
            
            # 2. Seasonality Analysis
            seasonality_results = self._analyze_seasonality(data, seasonal_period)
            
            # 3. Stationarity Tests
            stationarity_results = self._test_stationarity(data)
            
            # 4. Autocorrelation Analysis
            acf_pacf_results = self._analyze_autocorrelation(data)
            
            # 5. Decomposition
            decomposition_results = self._decompose_series(data, seasonality_results.get('period'))
            
            # 6. Descriptive Statistics
            stats_results = self._calculate_statistics(data)
            
            # Generate model recommendations based on results
            recommendations = self._generate_model_recommendations(
                trend_results,
                seasonality_results,
                stationarity_results,
                acf_pacf_results
            )
            
            # Create summary message
            summary = self._create_summary(
                trend_results,
                seasonality_results,
                stationarity_results,
                acf_pacf_results,
                stats_results,
                recommendations
            )
            
            response = {
                "status": "success",
                "summary": summary,
                "results": {
                    "trend": trend_results,
                    "seasonality": seasonality_results,
                    "stationarity": stationarity_results,
                    "acf_pacf": acf_pacf_results,
                    "decomposition": decomposition_results,
                    "statistics": stats_results
                },
                "recommendations": recommendations
            }
            
            logger.info("AnalysisAgentWrapper: Analysis complete")
            return response
            
        except Exception as e:
            logger.error(f"AnalysisAgentWrapper error: {e}")
            return {
                "status": "error",
                "summary": f"Erreur lors de l'analyse : {str(e)}",
                "recommendations": []
            }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend in the time series."""
        values = data['value'].values
        x = np.arange(len(values))
        
        # Linear regression
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "croissante"
        else:
            direction = "décroissante"
        
        # Calculate R² to assess trend strength
        y_pred = np.polyval(coefficients, x)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        strength = "forte" if r_squared > 0.7 else "modérée" if r_squared > 0.4 else "faible"
        
        return {
            "direction": direction,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "strength": strength,
            "coefficients": [float(c) for c in coefficients]
        }
    
    def _analyze_seasonality(self, data: pd.DataFrame, period: Any = 'auto') -> Dict[str, Any]:
        """Analyze seasonality in the time series."""
        from scipy import signal
        
        values = data['value'].values
        
        # Auto-detect period if needed
        if period == 'auto' or period is None:
            # Use ACF to detect dominant period
            from statsmodels.tsa.stattools import acf
            
            max_lag = min(len(values) // 2, 100)
            acf_values = acf(values, nlags=max_lag, fft=True)
            
            # Find peaks in ACF
            peaks, _ = signal.find_peaks(acf_values[1:], height=0.3)
            
            if len(peaks) > 0:
                detected_period = peaks[0] + 1
            else:
                detected_period = None
        else:
            detected_period = int(period) if period else None
        
        # Calculate seasonality strength if period detected
        has_seasonality = detected_period is not None and detected_period > 1
        
        if has_seasonality and detected_period < len(values) // 2:
            # Calculate coefficient of variation for seasonal component
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomp = seasonal_decompose(values, model='additive', period=detected_period, extrapolate_trend='freq')
                seasonal_strength = np.std(decomp.seasonal) / np.std(values) * 100
                strength_label = "forte" if seasonal_strength > 15 else "modérée" if seasonal_strength > 5 else "faible"
            except:
                seasonal_strength = 0
                strength_label = "inconnue"
        else:
            seasonal_strength = 0
            strength_label = "absente"
        
        return {
            "detected": has_seasonality,
            "period": int(detected_period) if detected_period else None,
            "strength": float(seasonal_strength),
            "strength_label": strength_label,
            "type": "additive"  # Could be enhanced to detect multiplicative
        }
    
    def _test_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test stationarity using ADF and KPSS tests."""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        values = data['value'].dropna().values
        
        # ADF Test
        try:
            adf_result = adfuller(values, autolag='AIC')
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical = adf_result[4]
            
            # ADF: H0 = non-stationary, reject if p < 0.05
            adf_stationary = adf_pvalue < 0.05
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_stationary = False
            adf_pvalue = 1.0
            adf_statistic = 0.0
            adf_critical = {}
        
        # KPSS Test
        try:
            kpss_result = kpss(values, regression='c', nlags='auto')
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical = kpss_result[3]
            
            # KPSS: H0 = stationary, reject if p < 0.05
            kpss_stationary = kpss_pvalue >= 0.05
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            kpss_stationary = False
            kpss_pvalue = 0.0
            kpss_statistic = 0.0
            kpss_critical = {}
        
        # Overall conclusion
        if adf_stationary and kpss_stationary:
            conclusion = "stationnaire"
            needs_differencing = False
        elif adf_stationary and not kpss_stationary:
            conclusion = "trend-stationary"
            needs_differencing = False
        else:
            conclusion = "non-stationnaire"
            needs_differencing = True
        
        return {
            "adf": {
                "statistic": float(adf_statistic),
                "pvalue": float(adf_pvalue),
                "critical_values": {k: float(v) for k, v in adf_critical.items()},
                "stationary": adf_stationary
            },
            "kpss": {
                "statistic": float(kpss_statistic),
                "pvalue": float(kpss_pvalue),
                "critical_values": {k: float(v) for k, v in kpss_critical.items()},
                "stationary": kpss_stationary
            },
            "conclusion": conclusion,
            "needs_differencing": needs_differencing
        }
    
    def _analyze_autocorrelation(self, data: pd.DataFrame, max_lags: int = 40) -> Dict[str, Any]:
        """Calculate ACF and PACF."""
        from statsmodels.tsa.stattools import acf, pacf
        
        values = data['value'].dropna().values
        max_lags = min(max_lags, len(values) // 2 - 1)
        
        try:
            # ACF
            acf_values = acf(values, nlags=max_lags, fft=True)
            
            # PACF
            pacf_values = pacf(values, nlags=max_lags, method='ywm')
            
            # Find significant lags (beyond 95% confidence interval)
            confidence_interval = 1.96 / np.sqrt(len(values))
            
            significant_acf_lags = [i for i, val in enumerate(acf_values[1:], 1) if abs(val) > confidence_interval]
            significant_pacf_lags = [i for i, val in enumerate(pacf_values[1:], 1) if abs(val) > confidence_interval]
            
            # Suggest ARIMA parameters
            # p: from PACF (number of significant lags)
            # q: from ACF (number of significant lags)
            suggested_p = min(len(significant_pacf_lags[:3]), 3) if significant_pacf_lags else 1
            suggested_q = min(len(significant_acf_lags[:3]), 3) if significant_acf_lags else 1
            
        except Exception as e:
            logger.warning(f"ACF/PACF calculation failed: {e}")
            acf_values = np.array([1.0])
            pacf_values = np.array([1.0])
            significant_acf_lags = []
            significant_pacf_lags = []
            suggested_p = 1
            suggested_q = 1
        
        return {
            "acf": acf_values.tolist(),
            "pacf": pacf_values.tolist(),
            "max_lags": max_lags,
            "significant_acf_lags": significant_acf_lags[:5],  # Top 5
            "significant_pacf_lags": significant_pacf_lags[:5],  # Top 5
            "suggested_p": suggested_p,
            "suggested_q": suggested_q
        }
    
    def _decompose_series(self, data: pd.DataFrame, period: Optional[int] = None) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components."""
        if period is None or period < 2 or period >= len(data) // 2:
            return {
                "decomposed": False,
                "reason": "Période saisonnière non détectée ou invalide"
            }
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            values = data['value'].values
            decomposition = seasonal_decompose(values, model='additive', period=period, extrapolate_trend='freq')
            
            return {
                "decomposed": True,
                "period": period,
                "trend": decomposition.trend.tolist(),
                "seasonal": decomposition.seasonal.tolist(),
                "residual": decomposition.resid.tolist(),
                "observed": decomposition.observed.tolist()
            }
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            return {
                "decomposed": False,
                "reason": f"Échec de la décomposition : {str(e)}"
            }
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        from scipy import stats as scipy_stats
        
        values = data['value'].dropna().values
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "variance": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "skewness": float(scipy_stats.skew(values)),
            "kurtosis": float(scipy_stats.kurtosis(values)),
            "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        }
    
    def _generate_model_recommendations(
        self,
        trend: Dict,
        seasonality: Dict,
        stationarity: Dict,
        acf_pacf: Dict
    ) -> List[Dict[str, Any]]:
        """Generate model recommendations based on analysis results."""
        recommendations = []
        
        has_trend = trend['strength'] != "faible"
        has_seasonality = seasonality['detected']
        is_stationary = stationarity['conclusion'] == "stationnaire"
        needs_diff = stationarity['needs_differencing']
        
        # SARIMA - Best for seasonal data
        if has_seasonality:
            p = acf_pacf['suggested_p']
            q = acf_pacf['suggested_q']
            d = 1 if needs_diff else 0
            period = seasonality['period']
            
            recommendations.append({
                "model": "SARIMA",
                "parameters": f"({p},{d},{q})(1,0,1)[{period}]",
                "reason": f"Saisonnalité détectée (période {period}). SARIMA gère parfaitement les patterns saisonniers.",
                "priority": 1,
                "suitable_for": ["seasonality", "trend"]
            })
        
        # ARIMA - For non-seasonal data
        if not has_seasonality or needs_diff:
            p = acf_pacf['suggested_p']
            q = acf_pacf['suggested_q']
            d = 1 if needs_diff else 0
            
            recommendations.append({
                "model": "ARIMA",
                "parameters": f"({p},{d},{q})",
                "reason": f"{'Non-stationnaire' if needs_diff else 'Stationnaire'} sans saisonnalité marquée. ARIMA classique adapté.",
                "priority": 2 if has_seasonality else 1,
                "suitable_for": ["trend", "autocorrelation"]
            })
        
        # Prophet - Good for seasonal data with trends
        if has_seasonality or has_trend:
            recommendations.append({
                "model": "Prophet",
                "parameters": "default",
                "reason": "Gère très bien la saisonnalité et les tendances. Robuste aux valeurs manquantes.",
                "priority": 2,
                "suitable_for": ["seasonality", "trend", "missing_values"]
            })
        
        # ExponentialSmoothing - For data with trend/seasonality
        if has_trend or has_seasonality:
            trend_type = "add" if has_trend else None
            seasonal_type = "add" if has_seasonality else None
            period = seasonality.get('period') if has_seasonality else None
            
            recommendations.append({
                "model": "ExponentialSmoothing",
                "parameters": f"trend={trend_type}, seasonal={seasonal_type}, period={period}",
                "reason": "Lissage exponentiel adapté aux données avec tendance et/ou saisonnalité.",
                "priority": 3,
                "suitable_for": ["trend", "seasonality"]
            })
        
        # RandomForest / ML models - Always an option
        recommendations.append({
            "model": "RandomForest",
            "parameters": "n_estimators=100",
            "reason": "Modèle ML robuste, capture patterns complexes sans hypothèses statistiques.",
            "priority": 4,
            "suitable_for": ["non_linear", "complex_patterns"]
        })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations[:5]  # Top 5
    
    def _create_summary(
        self,
        trend: Dict,
        seasonality: Dict,
        stationarity: Dict,
        acf_pacf: Dict,
        statistics: Dict,
        recommendations: List[Dict]
    ) -> str:
        """Create human-readable summary."""
        parts = []
        
        parts.append("📊 **Analyse Statistique Terminée !**\n")
        
        # Trend
        parts.append(f"## 📈 Tendance")
        parts.append(f"- Direction : **{trend['direction'].capitalize()}** (pente : {trend['slope']:.6f})")
        parts.append(f"- Force : {trend['strength'].capitalize()} (R² = {trend['r_squared']:.3f})")
        if trend['strength'] == "faible":
            parts.append(f"- ✓ Pas de tendance significative\n")
        else:
            parts.append(f"- ⚠️ Tendance {trend['strength']} détectée\n")
        
        # Seasonality
        parts.append(f"## 🔄 Saisonnalité")
        if seasonality['detected']:
            parts.append(f"- Période détectée : **{seasonality['period']} points**")
            parts.append(f"- Force : **{seasonality['strength_label'].capitalize()}** ({seasonality['strength']:.1f}%)")
            parts.append(f"- Type : {seasonality['type'].capitalize()}\n")
        else:
            parts.append(f"- ✓ Aucune saisonnalité significative détectée\n")
        
        # Stationarity
        parts.append(f"## 📏 Stationnarité")
        adf_symbol = "✅" if stationarity['adf']['stationary'] else "❌"
        kpss_symbol = "✅" if stationarity['kpss']['stationary'] else "❌"
        parts.append(f"- **ADF Test** : {adf_symbol} p-value = {stationarity['adf']['pvalue']:.4f}")
        parts.append(f"- **KPSS Test** : {kpss_symbol} p-value = {stationarity['kpss']['pvalue']:.4f}")
        parts.append(f"- **Conclusion** : Série **{stationarity['conclusion']}**")
        if stationarity['needs_differencing']:
            parts.append(f"- ⚠️ Différenciation recommandée (d=1)\n")
        else:
            parts.append(f"- ✓ Pas de différenciation nécessaire\n")
        
        # ACF/PACF
        parts.append(f"## 🔗 Autocorrélation")
        parts.append(f"- **ACF** : {len(acf_pacf['significant_acf_lags'])} lags significatifs")
        parts.append(f"- **PACF** : {len(acf_pacf['significant_pacf_lags'])} lags significatifs")
        parts.append(f"- **Paramètres ARIMA suggérés** : p={acf_pacf['suggested_p']}, q={acf_pacf['suggested_q']}\n")
        
        # Statistics
        parts.append(f"## 📊 Statistiques Descriptives")
        parts.append(f"- Moyenne : {statistics['mean']:.2f} | Médiane : {statistics['median']:.2f}")
        parts.append(f"- Écart-type : {statistics['std']:.2f} | Variance : {statistics['variance']:.2f}")
        parts.append(f"- Min : {statistics['min']:.2f} | Max : {statistics['max']:.2f}")
        skew_interpretation = "symétrique" if abs(statistics['skewness']) < 0.5 else "asymétrique"
        parts.append(f"- Asymétrie (skewness) : {statistics['skewness']:.2f} ({skew_interpretation})\n")
        
        # Recommendations
        parts.append(f"## 🎯 Modèles Recommandés")
        for i, rec in enumerate(recommendations[:4], 1):
            priority_symbol = "⭐" if i == 1 else "✅"
            parts.append(f"{i}. {priority_symbol} **{rec['model']}** `{rec['parameters']}`")
            parts.append(f"   {rec['reason']}")
        
        parts.append(f"\n📊 **Visualisations** : Graphiques d'analyse générés")
        parts.append(f"\n💬 **Prochaine étape** : Voulez-vous lancer la sélection de modèles ?")
        
        return "\n".join(parts)


class ValidationAgentWrapper:
    """Wrapper for ValidationAgent (to be implemented in future steps)."""
    pass


class ForecastAgentWrapper:
    """Wrapper for ForecastAgent (to be implemented in future steps)."""
    pass


class ReportAgentWrapper:
    """Wrapper for ReportAgent (to be implemented in future steps)."""
    pass

