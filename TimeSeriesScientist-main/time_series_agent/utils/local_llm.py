"""
Local LLM Integration for TSci Conversational Agent
Handles communication with Ollama (Llama3) for conversational responses.
"""

import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Interface for local LLM using Ollama.
    Provides conversational capabilities using Llama3.
    """
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Initialize the LocalLLM client.
        
        Args:
            model: Name of the Ollama model (default: llama3)
            base_url: URL of the Ollama server (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM client."""
        try:
            from langchain_community.llms import Ollama
            
            self._llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=0.7,
                top_p=0.9,
                num_ctx=4096,  # Context window
            )
            
            logger.info(f"LocalLLM initialized with model: {self.model}")
            
        except ImportError as e:
            logger.error("langchain-community not installed. Install with: pip install langchain-community")
            self._llm = None
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            self._llm = None
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and running.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        if self._llm is None:
            return False
        
        try:
            # Try a simple query to check availability
            response = self._llm.invoke("Hello")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def ask(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 30
    ) -> str:
        """
        Send a question to the local LLM.
        
        Args:
            prompt: The user's question or prompt
            system_prompt: Optional system instructions to prepend
            max_retries: Number of retry attempts on failure
            timeout: Timeout in seconds for each attempt
            
        Returns:
            The LLM's response as a string
        """
        if not self.is_available():
            return self._fallback_response("Ollama n'est pas disponible. Veuillez vérifier qu'Ollama est en cours d'exécution.")
        
        # Construct full prompt with system instructions
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prompt to Ollama (attempt {attempt + 1}/{max_retries})")
                
                start_time = time.time()
                response = self._llm.invoke(full_prompt)
                elapsed_time = time.time() - start_time
                
                logger.info(f"Received response from Ollama in {elapsed_time:.2f}s")
                
                return response.strip()
                
            except Exception as e:
                logger.error(f"Ollama request failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    return self._fallback_response(f"Erreur lors de la communication avec Ollama : {str(e)}")
        
        return self._fallback_response("Impossible de contacter Ollama après plusieurs tentatives.")
    
    def ask_with_context(
        self,
        question: str,
        context: Dict[str, Any],
        role: str = "assistant expert en séries temporelles"
    ) -> str:
        """
        Ask a question with structured context.
        
        Args:
            question: The user's question
            context: Dictionary containing context information
            role: The role/persona for the LLM
            
        Returns:
            The LLM's response
        """
        # Build system prompt
        system_prompt = f"""Tu es TSci-Chat, un {role}.
Tu aides les utilisateurs à analyser et prévoir leurs séries temporelles.

RÈGLES IMPORTANTES :
1. Base-toi UNIQUEMENT sur le contexte fourni ci-dessous
2. Si l'information n'est pas dans le contexte, dis-le honnêtement
3. Réponds de manière claire, concise et précise
4. Utilise le markdown pour formater ta réponse
5. Si tu donnes des chiffres, cite-les directement du contexte
"""
        
        # Build context string
        context_str = self._format_context(context)
        
        # Full prompt
        full_prompt = f"""{system_prompt}

CONTEXTE ACTUEL :
{context_str}

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE :"""
        
        return self.ask(full_prompt)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a readable string.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Dataset info
        if 'dataset' in context and context['dataset']:
            ds = context['dataset']
            context_parts.append(f"**DATASET** :")
            context_parts.append(f"  - Nom : {ds.get('name', 'N/A')}")
            context_parts.append(f"  - Lignes : {ds.get('num_rows', 'N/A')}")
            context_parts.append(f"  - Colonnes : {ds.get('num_columns', 'N/A')}")
            context_parts.append(f"  - Colonne cible : {ds.get('target_col', 'N/A')}")
        
        # Preprocessing results
        if 'preprocessing' in context and context['preprocessing']:
            prep = context['preprocessing']
            context_parts.append("\n**PRÉTRAITEMENT** :")
            if 'missing_values' in prep:
                context_parts.append(f"  - Valeurs manquantes : {prep['missing_values'].get('count', 0)} ({prep['missing_values'].get('percentage', 0):.2f}%)")
                context_parts.append(f"  - Stratégie appliquée : {prep.get('missing_strategy', 'N/A')}")
            if 'outliers' in prep:
                context_parts.append(f"  - Outliers : {prep['outliers'].get('count', 0)} ({prep['outliers'].get('percentage', 0):.2f}%)")
                context_parts.append(f"  - Stratégie appliquée : {prep.get('outlier_strategy', 'N/A')}")
        
        # Analysis results
        if 'analysis' in context and context['analysis']:
            anal = context['analysis']
            context_parts.append("\n**ANALYSE STATISTIQUE** :")
            if 'trend' in anal:
                context_parts.append(f"  - Tendance : {anal['trend'].get('direction', 'N/A')} (force: {anal['trend'].get('strength', 'N/A')})")
            if 'seasonality' in anal:
                context_parts.append(f"  - Saisonnalité : {'Détectée' if anal['seasonality'].get('detected') else 'Non détectée'}")
                if anal['seasonality'].get('detected'):
                    context_parts.append(f"    - Période : {anal['seasonality'].get('period', 'N/A')}")
                    context_parts.append(f"    - Force : {anal['seasonality'].get('strength_label', 'N/A')}")
            if 'stationarity' in anal:
                context_parts.append(f"  - Stationnarité : {anal['stationarity'].get('conclusion', 'N/A')}")
        
        # Model recommendations
        if 'recommendations' in context and context['recommendations']:
            context_parts.append("\n**RECOMMANDATIONS DE MODÈLES** :")
            for i, rec in enumerate(context['recommendations'][:3], 1):  # Top 3
                context_parts.append(f"  {i}. {rec.get('model', 'N/A')} - {rec.get('reason', 'N/A')}")
        
        # Current step
        if 'current_step' in context:
            context_parts.append(f"\n**ÉTAPE ACTUELLE** : {context['current_step']}")
        
        return "\n".join(context_parts) if context_parts else "Aucun contexte disponible."
    
    def _fallback_response(self, error_msg: str) -> str:
        """
        Generate a fallback response when LLM is unavailable.
        
        Args:
            error_msg: The error message
            
        Returns:
            A user-friendly fallback response
        """
        return f"""❌ **Erreur** : {error_msg}

💡 **Solutions possibles** :
1. Vérifiez qu'Ollama est en cours d'exécution : `ollama serve`
2. Vérifiez que le modèle {self.model} est installé : `ollama list`
3. Si nécessaire, téléchargez le modèle : `ollama pull {self.model}`

En attendant, vous pouvez utiliser les boutons d'action dans la sidebar pour naviguer dans le workflow.
"""
    
    def detect_intent(self, user_input: str) -> str:
        """
        Detect the intent of user input (command vs question).
        
        Args:
            user_input: The user's input text
            
        Returns:
            'command', 'question', or 'unknown'
        """
        user_input_lower = user_input.lower()
        
        # Command keywords
        command_keywords = [
            'lance', 'lancer', 'démarre', 'démarrer', 'commence', 'commencer',
            'exécute', 'exécuter', 'fait', 'faire', 'applique', 'appliquer',
            'montre', 'montrer', 'affiche', 'afficher', 'génère', 'générer'
        ]
        
        # Question keywords
        question_keywords = [
            'pourquoi', 'comment', 'quoi', 'quel', 'quelle', 'quels', 'quelles',
            'qui', 'où', 'quand', 'combien', 'est-ce que', 'peux-tu',
            'explique', 'expliquer'
        ]
        
        # Check for command keywords
        if any(keyword in user_input_lower for keyword in command_keywords):
            return 'command'
        
        # Check for question keywords or question mark
        if any(keyword in user_input_lower for keyword in question_keywords) or '?' in user_input:
            return 'question'
        
        # If uncertain, use LLM for intent detection (if available)
        if self.is_available():
            try:
                prompt = f"""Classifie l'intention de l'utilisateur en une seule catégorie :
- "command" : l'utilisateur demande une action (lancer, exécuter, appliquer, etc.)
- "question" : l'utilisateur pose une question (pourquoi, comment, combien, etc.)

Réponds UNIQUEMENT avec "command" ou "question", rien d'autre.

Phrase de l'utilisateur : "{user_input}"

Intention :"""
                
                response = self.ask(prompt, max_retries=1).strip().lower()
                
                if 'command' in response:
                    return 'command'
                elif 'question' in response:
                    return 'question'
            except:
                pass
        
        return 'unknown'


# Singleton instance
_local_llm_instance = None


def get_local_llm(model: str = "llama3") -> LocalLLM:
    """
    Get the singleton LocalLLM instance.
    
    Args:
        model: Name of the Ollama model
        
    Returns:
        LocalLLM instance
    """
    global _local_llm_instance
    
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLM(model=model)
    
    return _local_llm_instance

