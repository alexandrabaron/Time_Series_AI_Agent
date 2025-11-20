# TSci Transformation Analysis: Pipeline Automatique → Agent Conversationnel

## Date: 20 Novembre 2025

---

## 📋 Table des Matières
1. [Résumé Exécutif](#résumé-exécutif)
2. [Architecture Actuelle](#architecture-actuelle)
3. [Architecture Cible](#architecture-cible)
4. [Plan de Transformation](#plan-de-transformation)
5. [Modifications Détaillées](#modifications-détaillées)
6. [Risques et Mitigations](#risques-et-mitigations)
7. [Timeline et Priorités](#timeline-et-priorités)

---

## 📊 Résumé Exécutif

### Objectif
Transformer TSci d'un **pipeline automatique séquentiel** en un **agent conversationnel interactif** permettant aux utilisateurs de :
- Interagir à chaque étape du processus
- Poser des questions et obtenir des explications
- Approuver, modifier ou rejeter les décisions
- Contrôler les hyperparamètres manuellement
- Déclencher des prévisions à la demande
- Uploader et gérer des datasets CSV

### État Actuel vs État Cible

| Aspect | Actuel | Cible |
|--------|--------|-------|
| **Flux** | Séquentiel automatique | Conversationnel à la demande |
| **Interaction** | Aucune | Continue tout au long du processus |
| **Contrôle** | Automatique | Manuel avec suggestions IA |
| **UI** | CLI/Script | Interface web interactive |
| **Questions** | Non supportées | Questions naturelles supportées |
| **Flexibilité** | Pipeline fixe | Modules exécutables indépendamment |

---

## 🏗️ Architecture Actuelle

### Structure des Agents
```
main.py
  └── TimeSeriesAgentGraph (LangGraph orchestrator)
       ├── PreprocessAgent (Curator)
       │   ├── Data loading & validation
       │   ├── Missing value imputation
       │   ├── Outlier detection (IQR)
       │   └── Visualizations generation
       │
       ├── AnalysisAgent (Curator - Analysis)
       │   ├── Trend analysis
       │   ├── Seasonality detection
       │   ├── Stationarity tests (ADF, KPSS)
       │   ├── Seasonal decomposition
       │   └── ACF/PACF analysis
       │
       ├── ValidationAgent (Planner)
       │   ├── Model selection (3-5 best models)
       │   ├── Hyperparameter tuning (grid search)
       │   ├── Cross-validation
       │   └── Model ranking
       │
       ├── ForecastAgent (Forecaster)
       │   ├── Model training
       │   ├── Individual predictions
       │   ├── Ensemble predictions
       │   ├── Confidence intervals
       │   └── Metrics calculation
       │
       └── ReportAgent (Reporter)
           ├── Experiment summary
           ├── Model comparison
           ├── Recommendations
           └── Export (JSON + plots + markdown)
```

### Flux de Données Actuel
```
CSV Input → PreprocessAgent → AnalysisAgent → ValidationAgent → 
ForecastAgent → ReportAgent → Results Output
```

**Problème**: Flux linéaire sans points d'interaction utilisateur.

### Fichiers Clés Existants

#### 1. **graph/agent_graph.py** (426 lignes)
- **Rôle**: Orchestrateur central utilisant LangGraph
- **Problèmes**:
  - `_build_graph()`: Graph rigide avec edges fixes
  - `run()`: Exécution automatique sans pause pour interaction
  - Pas de gestion de session utilisateur
  - Pas de mécanisme de question-réponse

#### 2. **Agents (5 fichiers)**
- `agents/preprocess_agent.py` (1030 lignes)
- `agents/analysis_agent.py` (192 lignes)
- `agents/validation_agent.py` (679 lignes)
- `agents/forecast_agent.py` (844 lignes)
- `agents/report_agent.py` (219 lignes)

**Problèmes communs**:
- Méthodes `run()` exécutent tout d'un coup
- Pas d'interface pour questions utilisateur
- Pas de mécanisme d'approbation/rejet
- Décisions prises automatiquement par LLM

#### 3. **config/default_config.py** (264 lignes)
- Configuration statique
- Pas de gestion de profils utilisateur

#### 4. **main.py** (134 lignes)
- Script d'exécution séquentiel
- Pas d'UI
- Pas de gestion de datasets multiples

---

## 🎯 Architecture Cible

### Nouvelle Structure Proposée

```
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer (Web Interface)                 │
│  - File Upload Component                                    │
│  - Dataset Manager                                          │
│  - Chat Interface                                           │
│  - Visualization Panel                                      │
│  - Control Panel (hyperparameters, model selection, etc.)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Conversational Orchestrator                    │
│  - Intent Recognition (question vs command vs approval)     │
│  - Session Management                                       │
│  - State Tracking (which step, what's pending)             │
│  - Agent Router (which agent to call)                      │
│  - Conversation History                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer (Modular)                    │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ PreprocessAgent  │  │  AnalysisAgent   │               │
│  │ - run()          │  │  - run()         │               │
│  │ - explain()      │  │  - explain()     │               │
│  │ - modify()       │  │  - modify()      │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ ValidationAgent  │  │  ForecastAgent   │               │
│  │ - run()          │  │  - run()         │               │
│  │ - explain()      │  │  - explain()     │               │
│  │ - modify()       │  │  - modify()      │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐                                      │
│  │  ReportAgent     │                                      │
│  │ - run()          │                                      │
│  │ - explain()      │                                      │
│  └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│  - Session Store (Redis / SQLite)                          │
│  - Dataset Store (File system + metadata DB)               │
│  - Results Store (Plots, predictions, reports)             │
│  - Conversation History Store                              │
└─────────────────────────────────────────────────────────────┘
```

### Flux Conversationnel

```
User: "Upload dataset.csv"
  → System: Uploads, validates, shows preview
  → System: "Dataset loaded: 10,000 rows. Ready to preprocess?"

User: "Yes, start preprocessing"
  → System: Runs PreprocessAgent
  → System: "Found 50 outliers (5%). Recommend clipping. Approve?"

User: "Why clipping?"
  → System: Explains IQR method and impact
  → System: Shows visualization of outliers

User: "Use interpolation instead"
  → System: Re-runs with interpolation
  → System: "Preprocessing complete. Start analysis?"

User: "What are the data characteristics?"
  → System: Summarizes trend, seasonality, stationarity
  → System: Shows ACF/PACF plots

User: "Run model selection"
  → System: Runs ValidationAgent
  → System: "Top 3 models: ARIMA, LSTM, Prophet. Continue?"

User: "Why ARIMA?"
  → System: Explains based on data stationarity

User: "Add RandomForest to the list"
  → System: Re-runs validation with 4 models
  → System: "Models ready. Generate forecasts?"

User: "Yes, forecast 96 steps"
  → System: Runs ForecastAgent
  → System: Shows predictions + confidence intervals

User: "Compare model performance"
  → System: Shows MAE, MSE, MAPE for each model
```

---

## 🔄 Plan de Transformation

### Phase 1: Backend Refactoring (Priorité: HAUTE)

#### 1.1 Créer le Conversational Orchestrator
**Fichier**: `graph/conversational_orchestrator.py`

```python
class ConversationalOrchestrator:
    """
    Orchestrateur conversationnel remplaçant le pipeline automatique.
    Gère les sessions utilisateur, les questions, et les approbations.
    """
    def __init__(self):
        self.session_store = SessionStore()
        self.agents = {
            'preprocess': PreprocessAgentWrapper(),
            'analysis': AnalysisAgentWrapper(),
            'validation': ValidationAgentWrapper(),
            'forecast': ForecastAgentWrapper(),
            'report': ReportAgentWrapper()
        }
        self.intent_classifier = IntentClassifier()
        
    def process_user_input(self, session_id: str, user_input: str):
        """Point d'entrée principal pour tout input utilisateur"""
        session = self.session_store.get_session(session_id)
        intent = self.intent_classifier.classify(user_input, session)
        
        if intent['type'] == 'question':
            return self.handle_question(session, intent)
        elif intent['type'] == 'command':
            return self.handle_command(session, intent)
        elif intent['type'] == 'approval':
            return self.handle_approval(session, intent)
        elif intent['type'] == 'modification':
            return self.handle_modification(session, intent)
```

**Méthodes clés**:
- `handle_question()`: Répond aux questions sur données/modèles/résultats
- `handle_command()`: Exécute commandes (preprocess, analyze, forecast, etc.)
- `handle_approval()`: Gère approbations/rejets
- `handle_modification()`: Applique modifications utilisateur

#### 1.2 Wrapper les Agents Existants
**Fichier**: `graph/agent_wrappers.py`

Créer des wrappers pour chaque agent avec méthodes supplémentaires:

```python
class PreprocessAgentWrapper:
    def __init__(self):
        self.agent = PreprocessAgent()
        
    def run(self, data, config, wait_for_approval=True):
        """Exécution avec pause optionnelle pour approbation"""
        result = self.agent.process(data, config)
        if wait_for_approval:
            return {'status': 'pending_approval', 'result': result}
        return result
        
    def explain(self, decision_key: str, context: dict):
        """Explique une décision spécifique"""
        # Ex: "Why clip outliers?", "Why use interpolation?"
        
    def modify(self, modification: dict):
        """Applique une modification utilisateur"""
        # Ex: Change strategy from 'clip' to 'interpolate'
        
    def answer_question(self, question: str, context: dict):
        """Répond à une question sur le preprocessing"""
```

#### 1.3 Gestion de Session
**Fichier**: `utils/session_manager.py`

```python
class Session:
    session_id: str
    dataset_id: str
    current_step: str  # 'idle', 'preprocessing', 'analysis', etc.
    pending_approval: dict  # Décisions en attente
    conversation_history: list
    results: dict  # Résultats de chaque étape
    config: dict  # Configuration actuelle
```

#### 1.4 Intent Classification
**Fichier**: `utils/intent_classifier.py`

Utiliser LLM pour classifier les intents:
```python
class IntentClassifier:
    def classify(self, user_input: str, session: Session):
        """
        Classification des intents:
        - question: "Why?", "What?", "How?", "Show me..."
        - command: "Start preprocessing", "Run forecast", "Generate report"
        - approval: "Yes", "Approve", "Looks good"
        - rejection: "No", "Reject", "Change to..."
        - modification: "Use LSTM instead", "Set horizon to 120"
        """
```

### Phase 2: Dataset Management (Priorité: HAUTE)

#### 2.1 Dataset Manager
**Fichier**: `utils/dataset_manager.py`

```python
class DatasetManager:
    def upload_dataset(self, file, metadata):
        """Upload et validation de CSV"""
        
    def list_datasets(self, user_id):
        """Liste tous les datasets d'un utilisateur"""
        
    def get_dataset_preview(self, dataset_id, n_rows=10):
        """Aperçu du dataset"""
        
    def get_dataset_statistics(self, dataset_id):
        """Statistiques de base"""
        
    def delete_dataset(self, dataset_id):
        """Suppression"""
```

#### 2.2 Dataset Storage
Structure de fichiers:
```
datasets/
  user_{user_id}/
    dataset_{dataset_id}/
      raw_data.csv
      metadata.json
      preview.json
      statistics.json
```

### Phase 3: UI Layer (Priorité: HAUTE)

#### 3.1 Technologie Proposée
**Option 1: Streamlit** (Rapide, Python-natif)
- ✅ Développement rapide
- ✅ Intégration Python directe
- ✅ Components de chat disponibles
- ❌ Moins flexible pour customisation avancée

**Option 2: Gradio** (ML-friendly)
- ✅ Interface ML simple
- ✅ Widgets ML pré-construits
- ✅ Partage facile
- ❌ Moins de contrôle sur layout

**Option 3: FastAPI + React** (Production-ready)
- ✅ Très flexible
- ✅ Performance optimale
- ✅ Architecture moderne
- ❌ Développement plus long

**Recommandation**: **Streamlit** pour MVP, migrer vers FastAPI+React si besoin.

#### 3.2 Layout UI Proposé

```
┌─────────────────────────────────────────────────────────────┐
│  TSci - Time Series Conversational Agent         [User] [⚙]│
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│  📁 Datasets │  💬 Chat Interface                          │
│              │  ┌────────────────────────────────────────┐ │
│  [+ Upload]  │  │ System: Dataset loaded successfully.   │ │
│              │  │ Ready to start preprocessing?          │ │
│  📊 Dataset1 │  │                                        │ │
│  📊 Dataset2 │  │ User: What's the data quality?        │ │
│  📊 Dataset3 │  │                                        │ │
│              │  │ System: Quality score: 0.85           │ │
│  🔍 Preview  │  │ - Missing: 2%                         │ │
│  📈 Stats    │  │ - Outliers: 5%                        │ │
│              │  │ [Show Visualization]                  │ │
│              │  │                                        │ │
│              │  │ [Your message...]                     │ │
│              │  └────────────────────────────────────────┘ │
│              │                                              │
│              │  📊 Visualizations Panel                    │
│              │  [Tabs: Time Series | Distribution |        │
│              │         ACF/PACF | Forecast Results]        │
│              │                                              │
│              │  ⚙️ Control Panel (Collapsible)             │
│              │  Models: [ARIMA] [LSTM] [Prophet]          │
│              │  Horizon: [96] steps                        │
│              │  Confidence: [95]%                          │
│              │  [Generate Forecast]                        │
└──────────────┴──────────────────────────────────────────────┘
```

#### 3.3 Composants UI Clés

**Fichier**: `ui/streamlit_app.py`

```python
import streamlit as st
from graph.conversational_orchestrator import ConversationalOrchestrator

def main():
    st.set_page_config(page_title="TSci Agent", layout="wide")
    
    # Sidebar: Dataset Management
    with st.sidebar:
        render_dataset_panel()
    
    # Main: Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        render_visualization_panel()
        render_control_panel()
```

### Phase 4: Agent Modifications (Priorité: MOYENNE)

#### 4.1 Ajouter Méthodes d'Explication

Chaque agent doit implémenter:

```python
class BaseAgent:
    def explain_decision(self, decision_key: str, context: dict) -> str:
        """
        Explique pourquoi une décision a été prise.
        Ex: "Why ARIMA?", "Why clip outliers?"
        """
        
    def get_alternatives(self, decision_key: str) -> list:
        """
        Retourne des alternatives pour une décision.
        Ex: For "outlier_strategy", return ['clip', 'drop', 'interpolate']
        """
        
    def apply_modification(self, modification: dict) -> dict:
        """
        Applique une modification utilisateur et re-exécute si nécessaire.
        """
```

#### 4.2 Modifications Spécifiques par Agent

**PreprocessAgent**:
- `explain_outlier_detection()`: Pourquoi ces points sont des outliers
- `explain_missing_strategy()`: Pourquoi interpolation vs forward fill
- `show_before_after()`: Visualisation avant/après preprocessing

**AnalysisAgent**:
- `explain_stationarity()`: Tests ADF/KPSS expliqués
- `explain_seasonality()`: Période saisonnière détectée
- `recommend_transformations()`: Suggestions de transformations

**ValidationAgent**:
- `explain_model_selection()`: Pourquoi ces modèles
- `compare_models()`: Comparaison détaillée
- `suggest_hyperparameters()`: Explication des hyperparamètres

**ForecastAgent**:
- `explain_ensemble_weights()`: Pourquoi ces poids
- `show_prediction_intervals()`: Intervalles de confiance
- `compare_individual_models()`: Performance individuelle

**ReportAgent**:
- `generate_custom_report()`: Rapport personnalisé
- `highlight_insights()`: Insights clés
- `export_results()`: Export dans différents formats

### Phase 5: API Layer (Priorité: MOYENNE)

#### 5.1 API Endpoints
**Fichier**: `api/main.py` (FastAPI)

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

# Dataset Management
@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile):
    pass

@app.get("/api/datasets")
async def list_datasets():
    pass

@app.get("/api/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str):
    pass

# Conversational Interface
@app.post("/api/chat/{session_id}/message")
async def send_message(session_id: str, message: str):
    pass

@app.get("/api/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    pass

# Agent Operations
@app.post("/api/agents/preprocess")
async def run_preprocess(session_id: str, config: dict):
    pass

@app.post("/api/agents/analyze")
async def run_analysis(session_id: str):
    pass

@app.post("/api/agents/validate")
async def run_validation(session_id: str, config: dict):
    pass

@app.post("/api/agents/forecast")
async def run_forecast(session_id: str, config: dict):
    pass

# Results and Visualizations
@app.get("/api/results/{session_id}/visualizations")
async def get_visualizations(session_id: str):
    pass

@app.get("/api/results/{session_id}/predictions")
async def get_predictions(session_id: str):
    pass
```

---

## 📝 Modifications Détaillées

### Fichiers à CRÉER (Nouveaux)

| Fichier | Lignes Estimées | Description |
|---------|-----------------|-------------|
| `graph/conversational_orchestrator.py` | ~500 | Orchestrateur conversationnel principal |
| `graph/agent_wrappers.py` | ~400 | Wrappers pour agents existants |
| `utils/session_manager.py` | ~200 | Gestion des sessions utilisateur |
| `utils/intent_classifier.py` | ~150 | Classification des intents utilisateur |
| `utils/dataset_manager.py` | ~300 | Gestion des datasets (upload, list, etc.) |
| `ui/streamlit_app.py` | ~600 | Interface Streamlit principale |
| `ui/components/chat.py` | ~200 | Composant de chat |
| `ui/components/dataset_panel.py` | ~150 | Panel de gestion datasets |
| `ui/components/viz_panel.py` | ~200 | Panel de visualisations |
| `ui/components/control_panel.py` | ~150 | Panel de contrôle |
| `api/main.py` | ~400 | API FastAPI (optionnel pour MVP) |
| `tests/test_orchestrator.py` | ~300 | Tests orchestrateur |
| `tests/test_ui.py` | ~200 | Tests UI |
| **TOTAL** | **~3,750 lignes** | |

### Fichiers à MODIFIER (Existants)

| Fichier | Lignes Actuelles | Modifications |
|---------|------------------|---------------|
| `agents/preprocess_agent.py` | 1030 | + `explain()`, `modify()`, `answer_question()` (~200 lignes) |
| `agents/analysis_agent.py` | 192 | + `explain()`, `modify()`, `answer_question()` (~150 lignes) |
| `agents/validation_agent.py` | 679 | + `explain()`, `modify()`, `answer_question()` (~200 lignes) |
| `agents/forecast_agent.py` | 844 | + `explain()`, `modify()`, `answer_question()` (~200 lignes) |
| `agents/report_agent.py` | 219 | + `explain()`, `generate_custom_report()` (~100 lignes) |
| `config/default_config.py` | 264 | + Session configs, UI configs (~50 lignes) |
| `main.py` | 134 | Refactoring complet pour UI (~100 lignes changées) |
| **TOTAL** | **3,362 lignes** | **~1,000 lignes ajoutées** |

### Fichiers à SUPPRIMER

| Fichier | Raison |
|---------|--------|
| `graph/agent_graph.py` | Remplacé par `conversational_orchestrator.py` |
| (Optionnel: garder pour référence pendant migration) | |

---

## ⚠️ Risques et Mitigations

### Risque 1: Complexité de l'Intent Classification
**Impact**: 🔴 ÉLEVÉ  
**Probabilité**: 🟡 MOYENNE

**Description**: Classifier correctement les intents utilisateur (question vs commande vs modification) peut être difficile.

**Mitigation**:
1. Utiliser un LLM robuste (GPT-4) pour classification
2. Créer un dataset de tests avec exemples couvrant tous les cas
3. Implémenter fallback: demander clarification si incertain
4. Logger tous les intents mal classifiés pour amélioration

### Risque 2: Gestion d'État Complexe
**Impact**: 🟡 MOYEN  
**Probabilité**: 🔴 ÉLEVÉE

**Description**: Suivre l'état de conversation (étape actuelle, décisions en attente, historique) peut devenir complexe.

**Mitigation**:
1. Utiliser une structure `Session` claire et bien documentée
2. Implémenter state machine pour transitions valides
3. Sauvegarder état régulièrement (persistence)
4. Tests exhaustifs de transitions d'état

### Risque 3: Performance UI
**Impact**: 🟡 MOYEN  
**Probabilité**: 🟡 MOYENNE

**Description**: Streamlit peut être lent pour grandes visualisations ou datasets.

**Mitigation**:
1. Caching agressif avec `@st.cache_data`
2. Pagination pour grands datasets
3. Lazy loading des visualisations
4. Option de migration vers FastAPI + React si nécessaire

### Risque 4: Compatibilité avec Code Existant
**Impact**: 🟡 MOYEN  
**Probabilité**: 🟡 MOYENNE

**Description**: Les wrappers doivent rester compatibles avec agents existants.

**Mitigation**:
1. Ne pas modifier la logique interne des agents
2. Wrappers comme couche d'abstraction propre
3. Tests de régression pour chaque agent
4. Maintenir les anciens scripts pour validation

### Risque 5: Rate Limiting OpenAI API
**Impact**: 🟡 MOYEN  
**Probabilité**: 🟡 MOYENNE

**Description**: Interactions fréquentes = plus d'appels API.

**Mitigation**:
1. Implémenter caching intelligent des réponses
2. Batching de requêtes quand possible
3. Fallback vers réponses pré-générées pour questions courantes
4. Monitoring d'utilisation API

### Risque 6: User Experience Confuse
**Impact**: 🔴 ÉLEVÉ  
**Probabilité**: 🟡 MOYENNE

**Description**: Utilisateurs peuvent ne pas savoir quoi faire / quelles questions poser.

**Mitigation**:
1. Suggestions de questions contextuelles
2. Tutorial interactif au premier lancement
3. Exemples de commandes dans l'UI
4. Documentation utilisateur claire
5. Feedback immédiat pour chaque action

---

## 📅 Timeline et Priorités

### Phase 1: Foundation (Semaine 1-2)
**Objectif**: Infrastructure de base fonctionnelle

- [ ] **Jour 1-2**: Créer `ConversationalOrchestrator` (squelette)
- [ ] **Jour 3-4**: Créer `SessionManager` et structures de données
- [ ] **Jour 5-6**: Créer `IntentClassifier` avec tests
- [ ] **Jour 7-8**: Créer wrappers basiques pour 2-3 agents
- [ ] **Jour 9-10**: Tests d'intégration orchestrator + agents

**Livrable**: Backend conversationnel minimal fonctionnel (CLI).

### Phase 2: Dataset Management (Semaine 2-3)
**Objectif**: Upload et gestion de datasets

- [ ] **Jour 1-2**: `DatasetManager` (upload, validation, storage)
- [ ] **Jour 3-4**: Métadonnées et preview
- [ ] **Jour 5**: Tests et edge cases

**Livrable**: API de gestion de datasets complète.

### Phase 3: UI - MVP (Semaine 3-5)
**Objectif**: Interface utilisateur de base

- [ ] **Jour 1-3**: Setup Streamlit, layout de base
- [ ] **Jour 4-6**: Composant de chat fonctionnel
- [ ] **Jour 7-9**: Panel de gestion datasets
- [ ] **Jour 10-12**: Panel de visualisations
- [ ] **Jour 13-14**: Intégration et tests

**Livrable**: UI MVP permettant chat + upload + visualisations.

### Phase 4: Agent Enhancements (Semaine 5-7)
**Objectif**: Fonctionnalités avancées des agents

- [ ] **Semaine 5**: Méthodes `explain()` pour tous agents
- [ ] **Semaine 6**: Méthodes `modify()` et re-exécution
- [ ] **Semaine 7**: Questions contextuelles et suggestions

**Livrable**: Agents entièrement conversationnels.

### Phase 5: Polish & Testing (Semaine 8)
**Objectif**: Stabilisation et tests

- [ ] **Jour 1-3**: Tests end-to-end complets
- [ ] **Jour 4-5**: Corrections bugs
- [ ] **Jour 6-7**: Documentation utilisateur et technique

**Livrable**: Système stable et documenté.

### Phase 6: Advanced Features (Semaine 9-10+)
**Objectif**: Fonctionnalités bonus

- [ ] Multi-utilisateurs et authentification
- [ ] Export avancé (PDF reports, etc.)
- [ ] Comparaison de datasets
- [ ] Templates de workflows
- [ ] API REST complète (FastAPI)

---

## 🔧 Détails d'Implémentation

### Structure de Code Proposée

```
time_series_agent/
├── agents/                      # AGENTS (à modifier)
│   ├── preprocess_agent.py      [MODIFIER: +explain(), +modify()]
│   ├── analysis_agent.py        [MODIFIER: +explain(), +modify()]
│   ├── validation_agent.py      [MODIFIER: +explain(), +modify()]
│   ├── forecast_agent.py        [MODIFIER: +explain(), +modify()]
│   ├── report_agent.py          [MODIFIER: +explain()]
│   └── memory.py                [GARDER]
│
├── graph/                       # ORCHESTRATION (refactoring majeur)
│   ├── conversational_orchestrator.py  [CRÉER]
│   ├── agent_wrappers.py              [CRÉER]
│   └── agent_graph.py                 [GARDER pour référence]
│
├── utils/                       # UTILITAIRES (extensions)
│   ├── session_manager.py       [CRÉER]
│   ├── intent_classifier.py     [CRÉER]
│   ├── dataset_manager.py       [CRÉER]
│   ├── data_utils.py            [GARDER]
│   ├── file_utils.py            [GARDER]
│   ├── model_library.py         [GARDER]
│   └── visualization_utils.py   [GARDER]
│
├── ui/                          # UI LAYER (nouveau)
│   ├── streamlit_app.py         [CRÉER]
│   └── components/
│       ├── chat.py              [CRÉER]
│       ├── dataset_panel.py     [CRÉER]
│       ├── viz_panel.py         [CRÉER]
│       └── control_panel.py     [CRÉER]
│
├── api/                         # API REST (optionnel pour MVP)
│   └── main.py                  [CRÉER - optionnel]
│
├── tests/                       # TESTS
│   ├── test_orchestrator.py    [CRÉER]
│   ├── test_session.py          [CRÉER]
│   ├── test_intent.py           [CRÉER]
│   └── test_ui.py               [CRÉER]
│
├── config/
│   └── default_config.py        [MODIFIER: + session configs]
│
├── main.py                      [REFACTORER complètement]
└── requirements.txt             [AJOUTER: streamlit, fastapi, redis, etc.]
```

### Dépendances Nouvelles

```txt
# UI
streamlit>=1.28.0
streamlit-chat>=0.1.1

# API (optionnel)
fastapi>=0.104.0
uvicorn>=0.24.0

# Session Management
redis>=5.0.0  # ou SQLite pour MVP

# Utilities
python-multipart>=0.0.6  # Pour upload de fichiers
```

---

## 🎯 MVP Definition (Minimum Viable Product)

### Features Essentielles
1. ✅ Upload CSV via UI
2. ✅ Interface de chat conversationnel
3. ✅ Questions basiques ("What's the data quality?", "Show statistics")
4. ✅ Exécution des 5 agents avec confirmation
5. ✅ Visualisations de base (time series, distribution, forecasts)
6. ✅ Modification des hyperparamètres via chat
7. ✅ Export des résultats (JSON, plots)

### Features Non-Essentielles (Post-MVP)
- ❌ Multi-utilisateurs avec authentification
- ❌ Historique des expériences
- ❌ Comparaison de datasets multiples
- ❌ Export PDF avancé
- ❌ Templates de workflows
- ❌ API REST complète

---

## 📊 Estimation Globale

| Catégorie | Effort | Complexité | Risque |
|-----------|--------|------------|--------|
| **Backend Refactoring** | 3 semaines | 🟡 Moyenne | 🟡 Moyen |
| **Dataset Management** | 1 semaine | 🟢 Faible | 🟢 Faible |
| **UI Development** | 2-3 semaines | 🟡 Moyenne | 🟡 Moyen |
| **Agent Enhancements** | 2 semaines | 🟡 Moyenne | 🟢 Faible |
| **Testing & Polish** | 1 semaine | 🟢 Faible | 🟢 Faible |
| **TOTAL MVP** | **8-10 semaines** | | |

---

## 📚 Prochaines Étapes

### Étape 1: Validation du Plan
- [ ] Review ce document avec l'équipe
- [ ] Ajustements priorités
- [ ] Validation timeline

### Étape 2: Setup Environnement
- [ ] Créer nouvelle branche: `feature/conversational-agent`
- [ ] Installer nouvelles dépendances
- [ ] Setup structure de dossiers

### Étape 3: Développement Itératif
- [ ] Commencer par Phase 1 (Foundation)
- [ ] Tests continus
- [ ] Démos régulières

---

## 📞 Questions Ouvertes

1. **Choix UI**: Confirmer Streamlit vs alternatives?
2. **Authentication**: Nécessaire pour MVP ou post-MVP?
3. **Déploiement**: Local seulement ou cloud (Streamlit Cloud, AWS)?
4. **Multi-langue**: Support français/anglais nécessaire?
5. **Historique**: Combien de temps garder sessions/datasets?

---

## ✅ Conclusion

### Faisabilité
**Verdict**: ✅ **FAISABLE** avec effort raisonnable.

### Bénéfices
- ✨ Expérience utilisateur grandement améliorée
- 🎯 Contrôle fin à chaque étape
- 📚 Apprentissage interactif du forecasting
- 🔄 Itération rapide sur modèles/paramètres

### Recommandation
**Commencer par MVP** (8-10 semaines) avec Streamlit, puis évaluer besoin de features avancées.

---

**Document préparé le**: 20 Novembre 2025  
**Auteur**: Claude (AI Assistant)  
**Version**: 1.0

