# TSci-Chat - Conversational Time Series Forecasting Agent

## 🎯 Vue d'Ensemble

TSci-Chat est la nouvelle interface conversationnelle pour le système de prévision de séries temporelles TimeSeriesScientist. Elle transforme le pipeline automatique en un assistant interactif qui vous guide à travers chaque étape du processus d'analyse et de prévision.

## 🚀 Démarrage Rapide

### Installation

1. Installez les dépendances :
```bash
cd time_series_agent
pip install -r requirements.txt
```

2. Configurez votre clé API OpenAI :
```bash
export OPENAI_API_KEY="votre-clé-api"
```

### Lancement

**Mode Conversationnel (Nouveau)** :
```bash
python main.py
```

**Mode Legacy (Pipeline Automatique)** :
```bash
python main.py --legacy
```

## 💬 Interface Conversationnelle

L'interface se compose de plusieurs panneaux :

### 1. **Panneau de Conversation (Centre gauche)**
- Historique des messages
- Input pour poser des questions
- Affichage des visualisations inline

### 2. **Panneau de Suggestions (Droite)**
- Questions suggérées contextuelles
- Indicateur d'étape actuelle
- Informations sur les données chargées

### 3. **Barre Latérale (Gauche)**
- Upload de datasets (à venir)
- Configuration (horizon, nombre de modèles, etc.)
- Informations de session
- Bouton de réinitialisation

## 📋 Fonctionnalités Actuelles (Étape 1)

### ✅ Implémenté
- ✅ Interface Streamlit de base
- ✅ Gestion de session
- ✅ Historique de conversation
- ✅ Questions suggérées contextuelles
- ✅ Configuration des paramètres
- ✅ Indicateur d'étape du workflow
- ✅ Message de bienvenue

### 🚧 En Cours
- 🚧 Upload de datasets CSV
- 🚧 Orchestrateur conversationnel
- 🚧 Intégration avec les agents
- 🚧 Classification des intentions
- 🚧 Génération de visualisations

### 📅 À Venir
- 📅 Approbations/rejets de décisions
- 📅 Modifications d'hyperparamètres
- 📅 Export de résultats
- 📅 Comparaison de modèles
- 📅 Explications détaillées

## 🎨 Exemples d'Utilisation

### Conversation Type

```
User: Bonjour, comment uploader mes données ?
Assistant: Vous pouvez uploader un fichier CSV via le panneau latéral...

User: [Upload dataset.csv]
Assistant: Dataset chargé : 10,000 lignes, 5 colonnes. Prêt à prétraiter ?

User: Quelle est la qualité de mes données ?
Assistant: Score de qualité : 0.85/1.0
            - Valeurs manquantes : 2%
            - Outliers : 5%
            [Visualisation]

User: Commence le prétraitement
Assistant: Prétraitement en cours...
           Stratégie recommandée pour les outliers : clipping
           Approuver ?

User: Pourquoi clipping ?
Assistant: Le clipping est recommandé car...

User: Approuvé
Assistant: Prétraitement terminé. Prêt pour l'analyse ?
```

## 🏗️ Architecture

```
ui/
├── streamlit_app.py          # Application principale
└── components/
    └── chat.py                # Composants de chat

utils/
└── session_manager.py         # Gestion d'état

main.py                        # Point d'entrée
```

## 🔧 Configuration

Les paramètres peuvent être modifiés via :
1. **Interface UI** : Panneau de configuration dans la barre latérale
2. **Session State** : `st.session_state.config`
3. **Code** : `config/default_config.py`

### Paramètres Disponibles

- `horizon` : Nombre de pas de temps à prévoir (défaut: 96)
- `input_length` : Longueur de la séquence d'entrée (défaut: 512)
- `num_models` : Nombre de modèles à sélectionner (défaut: 3)
- `confidence_level` : Niveau de confiance (défaut: 0.95)

## 📊 État de Session

L'état de session conserve :
- `session_id` : Identifiant unique de la session
- `current_step` : Étape actuelle du workflow
- `messages` : Historique de conversation
- `data` : Dataset chargé
- `results` : Résultats des agents
- `config` : Configuration actuelle
- `pending_approval` : Décisions en attente

## 🐛 Débogage

Pour voir l'état complet de la session :
1. Ouvrez l'expander "ℹ️ Informations de Session" dans la barre latérale
2. L'état JSON complet sera affiché

## 🆘 Support

Pour des questions ou des problèmes :
1. Vérifiez que toutes les dépendances sont installées
2. Vérifiez que `OPENAI_API_KEY` est configurée
3. Consultez les logs dans le terminal

## 🔄 Migration depuis le Mode Legacy

Le mode conversationnel ne remplace pas le mode legacy, mais le complète :

**Utiliser Legacy quand** :
- Vous voulez un pipeline automatique
- Vous avez des scripts d'automatisation existants
- Vous ne voulez pas d'interaction

**Utiliser Conversational quand** :
- Vous explorez de nouvelles données
- Vous voulez comprendre les décisions
- Vous voulez un contrôle fin
- Vous apprenez le forecasting

## 📝 Notes de Version

### v0.1 (Étape 1 - Fondation)
- ✨ Interface Streamlit de base
- ✨ Gestion de session
- ✨ Historique de conversation
- ✨ Configuration interactive

### v0.2 (À venir - Étape 2)
- 🚧 Upload de datasets
- 🚧 Preview et statistiques
- 🚧 Orchestrateur conversationnel

## 🎓 En Savoir Plus

- [Documentation complète](../TSCI_TRANSFORMATION_ANALYSIS.md)
- [Architecture du système](../TSCI_TRANSFORMATION_ANALYSIS.md#architecture-cible)
- [Plan de développement](../TSCI_TRANSFORMATION_ANALYSIS.md#timeline-et-priorités)

---

**TSci-Chat v0.1** - Développé avec ❤️ pour rendre le forecasting accessible à tous.

