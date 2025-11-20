# 🤖 Plan d'Implémentation : Chatbot Conversationnel avec Llama3 (Ollama)

## 📋 Objectif

Transformer l'orchestrateur actuel en un **vrai chatbot conversationnel intelligent** qui peut :
- Répondre aux questions de l'utilisateur sur les données, analyses, et résultats
- Expliquer les décisions prises par les agents
- Guider l'utilisateur à travers le workflow
- Utiliser **Llama3 en local via Ollama** (pas d'API externe)

---

## 🔍 État Actuel

### Ce qui fonctionne déjà ✅
1. **Interface de chat Streamlit** : affichage des messages, input utilisateur
2. **Gestion de session** : `SessionManager` maintient l'état et l'historique
3. **Orchestrateur basique** : `ConversationalOrchestrator` gère les commandes prédéfinies
4. **Agents wrappers** : `PreprocessAgentWrapper`, `AnalysisAgentWrapper` retournent des résultats structurés

### Ce qui ne fonctionne PAS ❌
1. **Pas de vraie conversation** : L'input utilisateur fait juste un echo (ligne 358 de `streamlit_app.py`)
2. **`answer_question()` est hardcodée** : Réponses if/else limitées (lignes 300-324 de `conversational_orchestrator.py`)
3. **Pas d'intégration LLM local** : Ollama/Llama3 n'est pas configuré
4. **Pas de contexte dynamique** : Le LLM n'a pas accès aux résultats d'analyse en session

---

## 🛠️ Plan d'Implémentation

### **Étape 1 : Installation et Configuration d'Ollama** 🔧

#### 1.1 Prérequis
- Vérifier qu'Ollama est installé sur votre machine
- Vérifier que le modèle Llama3 est téléchargé : `ollama pull llama3`
- Tester qu'Ollama fonctionne : `ollama run llama3`

#### 1.2 Ajouter les dépendances
Modifier `requirements.txt` pour ajouter :
```txt
langchain-community>=0.0.10
```

**Note** : `langchain-community` contient l'intégration Ollama.

---

### **Étape 2 : Créer un Module LLM Local** 🧠

#### 2.1 Créer `utils/local_llm.py`
Un module dédié pour gérer l'interaction avec Ollama/Llama3.

**Fonctionnalités** :
- Initialiser la connexion Ollama
- Gérer les prompts et réponses
- Retry logic en cas d'erreur
- Fallback vers réponses hardcodées si Ollama est down

**API** :
```python
class LocalLLM:
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        """Initialize Ollama LLM."""
        
    def ask(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Send a question to the local LLM with context."""
        
    def is_available(self) -> bool:
        """Check if Ollama is running."""
```

---

### **Étape 3 : Créer un Système de Contexte Conversationnel** 📚

#### 3.1 Créer `utils/conversation_context.py`
Un module pour construire le contexte dynamique à envoyer au LLM.

**Fonctionnalités** :
- Extraire les informations pertinentes de `st.session_state`
- Formater le contexte en texte lisible pour le LLM
- Inclure : dataset info, résultats d'analyse, recommandations, étape actuelle

**API** :
```python
class ConversationContextBuilder:
    @staticmethod
    def build_context(session_state) -> str:
        """Build a comprehensive context string from session state."""
        # Inclut :
        # - Dataset info (colonnes, taille, qualité)
        # - Résultats de preprocessing (valeurs manquantes, outliers)
        # - Résultats d'analyse (tendance, saisonnalité, stationnarité)
        # - Recommandations de modèles
        # - Étape actuelle du workflow
```

---

### **Étape 4 : Améliorer l'Orchestrateur Conversationnel** 🎯

#### 4.1 Modifier `graph/conversational_orchestrator.py`

**Changements** :
1. Intégrer `LocalLLM` dans `__init__()`
2. Remplacer `answer_question()` hardcodée par un appel au LLM avec contexte
3. Ajouter un système de routing intelligent :
   - Si la question est une commande (ex: "lance l'analyse") → router vers `handle_command()`
   - Si c'est une question (ex: "pourquoi ARIMA ?") → router vers `answer_question()` avec LLM

**Nouveau workflow** :
```python
def handle_user_input(self, user_input: str) -> Dict[str, Any]:
    """
    Main entry point for user input.
    Routes to command handler or question answering.
    """
    # 1. Intent detection (commande vs question)
    intent = self._detect_intent(user_input)
    
    # 2. Route accordingly
    if intent == 'command':
        return self.handle_command(extracted_command, user_input)
    elif intent == 'question':
        return self.answer_question(user_input)
    else:
        return self._fallback_response(user_input)
```

#### 4.2 Améliorer `answer_question()`
```python
def answer_question(self, question: str) -> Dict[str, Any]:
    """Answer user question using local LLM with context."""
    # 1. Build context from session state
    context = ConversationContextBuilder.build_context(st.session_state)
    
    # 2. Create prompt for LLM
    prompt = f"""Tu es TSci-Chat, un assistant expert en analyse de séries temporelles.
    
CONTEXTE ACTUEL :
{context}

QUESTION DE L'UTILISATEUR :
{question}

Réponds de manière claire, concise et précise en te basant sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le honnêtement.
"""
    
    # 3. Get answer from LLM
    answer = self.local_llm.ask(prompt)
    
    # 4. Return formatted response
    return {
        'status': 'success',
        'message': answer,
        'is_question': True
    }
```

---

### **Étape 5 : Intégrer dans l'UI Streamlit** 🖥️

#### 5.1 Modifier `ui/streamlit_app.py`

**Remplacer** la ligne 358 (echo hardcodé) par :
```python
if user_input:
    # Route to orchestrator
    result = st.session_state.orchestrator.handle_user_input(user_input)
    
    # Add response to chat
    add_assistant_message(result['message'])
    st.rerun()
```

---

### **Étape 6 : Améliorer les Prompts pour les Agents** 📝

#### 6.1 Objectif
Permettre au chatbot d'expliquer les décisions des agents (ex: "Pourquoi ARIMA ?", "Pourquoi interpoler ?")

#### 6.2 Approche
- Les wrappers d'agents retournent déjà des `reasons` dans leurs résultats
- L'orchestrateur doit stocker ces justifications dans `st.session_state.results`
- Le contexte conversationnel inclut ces justifications
- Le LLM peut alors les expliquer de manière conversationnelle

**Exemple** :
```
User: "Pourquoi as-tu choisi l'interpolation ?"

Context: {
  "preprocessing": {
    "missing_strategy": "interpolate",
    "reason": "Bon pour préserver les tendances dans les données continues"
  }
}

LLM Response: "J'ai choisi l'interpolation pour les valeurs manquantes car 
elle est particulièrement adaptée aux séries temporelles continues comme la 
vôtre. Elle préserve les tendances naturelles en estimant les valeurs 
manquantes à partir des points voisins."
```

---

## 🧪 Plan de Test

### Test 1 : Ollama fonctionne
```bash
ollama list  # Vérifier que llama3 est installé
ollama run llama3 "Bonjour, es-tu prêt ?"
```

### Test 2 : LocalLLM module
```python
from utils.local_llm import LocalLLM
llm = LocalLLM()
assert llm.is_available()
response = llm.ask("Qu'est-ce qu'ARIMA ?")
print(response)
```

### Test 3 : Contexte conversationnel
```python
from utils.conversation_context import ConversationContextBuilder
context = ConversationContextBuilder.build_context(st.session_state)
print(context)  # Doit afficher dataset info, résultats, etc.
```

### Test 4 : Questions conversationnelles (E2E)
1. Uploader un dataset
2. Lancer le preprocessing
3. Poser des questions :
   - "Quelle est la qualité de mes données ?"
   - "Combien de valeurs manquantes ?"
   - "Pourquoi utiliser l'interpolation ?"
   - "Mes données ont-elles une tendance ?"
   - "Quels modèles recommandes-tu ?"

---

## ⚠️ Risques et Solutions

### Risque 1 : Ollama n'est pas installé ou ne fonctionne pas
**Solution** : Fallback vers réponses hardcodées + message d'erreur clair

### Risque 2 : Llama3 est lent sur l'ordinateur de l'utilisateur
**Solution** : 
- Afficher un spinner pendant le traitement
- Utiliser un modèle plus petit (ex: `llama3:8b` au lieu de `llama3:70b`)
- Limiter la longueur du contexte envoyé au LLM

### Risque 3 : Le LLM donne des réponses incorrectes ou hallucine
**Solution** :
- Prompts très structurés avec instruction claire : "Base-toi UNIQUEMENT sur le contexte fourni"
- Validation des réponses critiques avant affichage
- Permettre à l'utilisateur de signaler des réponses incorrectes

### Risque 4 : L'intent detection rate
**Solution** :
- Utiliser aussi Llama3 pour l'intent detection
- Avoir des mots-clés de secours (ex: "lance", "montre", "affiche" → commande)

---

## 📦 Résumé des Fichiers à Créer/Modifier

### Nouveaux fichiers
1. `utils/local_llm.py` - Intégration Ollama/Llama3
2. `utils/conversation_context.py` - Construction du contexte conversationnel
3. `PLAN_CHATBOT_CONVERSATIONNEL.md` - Ce document

### Fichiers à modifier
1. `requirements.txt` - Ajouter `langchain-community`
2. `graph/conversational_orchestrator.py` - Ajouter `handle_user_input()`, améliorer `answer_question()`
3. `ui/streamlit_app.py` - Connecter l'input utilisateur à l'orchestrateur (ligne 358)
4. `utils/session_manager.py` - Potentiellement ajouter des méthodes pour stocker les justifications

---

## 🚀 Ordre d'Implémentation Recommandé

1. **Vérifier Ollama** : S'assurer qu'Ollama + Llama3 fonctionnent
2. **Créer `local_llm.py`** : Module de base pour communiquer avec Ollama
3. **Tester LocalLLM** : Vérifier que les questions/réponses fonctionnent
4. **Créer `conversation_context.py`** : Builder de contexte
5. **Modifier `conversational_orchestrator.py`** : Intégrer LocalLLM et améliorer `answer_question()`
6. **Modifier `streamlit_app.py`** : Connecter l'input à l'orchestrateur
7. **Tests E2E** : Tester des conversations réelles

---

## 💡 Améliorations Futures (Optionnel)

1. **Mémoire conversationnelle** : Inclure les 5 derniers messages dans le contexte pour des conversations multi-tours
2. **RAG (Retrieval Augmented Generation)** : Stocker les résultats dans une base vectorielle pour récupération intelligente
3. **Multi-modal** : Permettre au LLM de "voir" les graphiques et les commenter
4. **Fine-tuning** : Fine-tuner Llama3 sur des conversations spécifiques aux séries temporelles

---

## ✅ Checklist de Validation

- [ ] Ollama est installé et fonctionne
- [ ] Llama3 est téléchargé localement
- [ ] `LocalLLM` peut communiquer avec Ollama
- [ ] Le contexte conversationnel est construit correctement
- [ ] L'orchestrateur route correctement les questions vs commandes
- [ ] L'UI envoie l'input utilisateur à l'orchestrateur
- [ ] Le chatbot répond de manière pertinente aux questions
- [ ] Le chatbot peut expliquer les décisions des agents
- [ ] Le chatbot fonctionne à toutes les étapes du workflow

---

**Prêt à implémenter ?** 🚀

