# 🧪 Guide de Test - Étape 1

## Pré-requis

1. **Python 3.8+** installé
2. **API Key OpenAI** (optionnelle pour cette étape)

## 🚀 Instructions de Test

### Étape 1 : Installation des Dépendances

Ouvrez un terminal dans le dossier `time_series_agent/` et exécutez :

```bash
pip install -r requirements.txt
```

**Note** : Cela peut prendre quelques minutes pour installer toutes les dépendances.

### Étape 2 : Lancement de l'Application

Depuis le dossier `time_series_agent/`, lancez :

```bash
python main.py
```

**Vous devriez voir** :
```
🚀 Launching TSci-Chat conversational interface...
💡 To use the old automated pipeline, run: python main.py --legacy

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

### Étape 3 : Vérification de l'Interface

Une fenêtre de navigateur devrait s'ouvrir automatiquement. Sinon, ouvrez manuellement : http://localhost:8501

#### ✅ Vérifications à Faire

1. **Page chargée** : Vous voyez "📈 TSci-Chat" en haut
2. **Message de bienvenue** : Un message de bienvenue s'affiche dans le chat
3. **Barre latérale** : Vous voyez le panneau de contrôle à gauche avec :
   - Informations de session (expander)
   - Section Datasets
   - Configuration (Horizon, Nombre de modèles)
   - Bouton de réinitialisation

4. **Zone principale** : 
   - Colonne de gauche : Conversation avec le message de bienvenue
   - Colonne de droite : Suggestions et indicateur d'étape

5. **Input de chat** : En bas, vous voyez "Posez-moi une question..."

### Étape 4 : Test des Interactions

#### Test 1 : Envoyer un message
1. Cliquez dans la zone de texte en bas
2. Tapez : "Bonjour"
3. Appuyez sur Entrée

**Résultat attendu** :
- Votre message apparaît dans le chat
- Une réponse automatique s'affiche : "🤖 Vous avez dit : 'Bonjour' ..."

#### Test 2 : Questions suggérées
1. Regardez le panneau de droite "💡 Suggestions"
2. Cliquez sur une des questions suggérées

**Résultat attendu** :
- La question est ajoutée au chat
- Une réponse s'affiche

#### Test 3 : Modifier la configuration
1. Dans la barre latérale, changez "Horizon de prévision" à 120
2. Notez que la valeur change immédiatement

**Résultat attendu** :
- La configuration est mise à jour (visible dans l'état de session)

#### Test 4 : Informations de session
1. Dans la barre latérale, cliquez sur "ℹ️ Informations de Session"
2. L'expander se déploie

**Résultat attendu** :
- Vous voyez un JSON avec :
  - `session_id` (un UUID)
  - `current_step: "initial"`
  - `num_messages` (nombre de messages)
  - `has_data: false`
  - `config` (horizon, num_models, etc.)

#### Test 5 : Réinitialisation
1. Cliquez sur "🔄 Réinitialiser la Session"
2. Confirmez

**Résultat attendu** :
- L'historique de chat est effacé
- Le message de bienvenue réapparaît
- Un nouveau `session_id` est généré

### Étape 5 : Test de l'Indicateur d'Étape

1. L'indicateur d'étape devrait afficher : "🏁 Initial"
2. C'est normal, les autres étapes seront implémentées dans les prochaines étapes

## ✅ Critères de Succès

L'Étape 1 est réussie si :

- [x] ✅ L'application Streamlit se lance sans erreur
- [x] ✅ Le message de bienvenue s'affiche
- [x] ✅ Vous pouvez envoyer des messages et voir des réponses
- [x] ✅ Les questions suggérées fonctionnent
- [x] ✅ La configuration peut être modifiée
- [x] ✅ L'état de session s'affiche correctement
- [x] ✅ La réinitialisation fonctionne
- [x] ✅ Aucune erreur dans la console/terminal

## 🐛 Dépannage

### Erreur : "ModuleNotFoundError: No module named 'streamlit'"
**Solution** : Réinstallez les dépendances
```bash
pip install -r requirements.txt
```

### Erreur : "Address already in use"
**Solution** : Un autre processus utilise le port 8501
```bash
# Arrêtez l'ancien processus ou utilisez un autre port
streamlit run ui/streamlit_app.py --server.port 8502
```

### L'application ne s'ouvre pas automatiquement
**Solution** : Ouvrez manuellement http://localhost:8501 dans votre navigateur

### Erreur d'import dans Python
**Solution** : Assurez-vous d'être dans le bon dossier
```bash
cd time_series_agent/
python main.py
```

### Les chemins de fichiers ne fonctionnent pas
**Solution** : Vérifiez la structure des dossiers :
```
time_series_agent/
├── main.py
├── requirements.txt
├── utils/
│   └── session_manager.py
└── ui/
    ├── streamlit_app.py
    └── components/
        └── chat.py
```

## 📸 Captures d'Écran Attendues

### Vue principale
```
┌─────────────────────────────────────────────────────────────┐
│  📈 TSci-Chat                                    [User] [⚙] │
│  Assistant conversationnel pour la prévision...             │
├──────────────┬──────────────────────────────────────────────┤
│  🎛️ Panneau │  💬 Conversation                             │
│              │  ┌────────────────────────────────────────┐  │
│  ℹ️ Session  │  │ 🤖 Bonjour ! Je suis TSci-Chat...     │  │
│              │  │                                        │  │
│  📁 Datasets │  │ 👤 Bonjour                            │  │
│  [Upload...]│  │                                        │  │
│              │  │ 🤖 Vous avez dit : 'Bonjour'          │  │
│  ⚙️ Config   │  │ Note: L'orchestrateur sera...        │  │
│  Horizon: 96 │  │                                        │  │
│  Models: 3   │  │ [Posez-moi une question...]           │  │
│              │  └────────────────────────────────────────┘  │
│  🔄 Reset    │                                              │
│              │  💡 Suggestions                              │
│              │  [Comment uploader mes données ?]            │
│              │  [Quels formats sont supportés ?]            │
│              │                                              │
│              │  📍 Étape Actuelle                           │
│              │  🏁 Initial                                  │
└──────────────┴──────────────────────────────────────────────┘
```

## 📋 Checklist de Test

Cochez au fur et à mesure :

### Installation
- [ ] Dépendances installées sans erreur
- [ ] Aucun warning critique

### Lancement
- [ ] Application démarre avec `python main.py`
- [ ] Page web s'ouvre (auto ou manuel)
- [ ] Aucune erreur dans le terminal

### Interface
- [ ] Titre visible : "📈 TSci-Chat"
- [ ] Message de bienvenue affiché
- [ ] Barre latérale visible et fonctionnelle
- [ ] Zones de conversation et suggestions visibles

### Interactions
- [ ] Input de message fonctionne
- [ ] Messages s'ajoutent à l'historique
- [ ] Réponses s'affichent
- [ ] Questions suggérées cliquables
- [ ] Boutons de configuration réactifs

### État et Persistance
- [ ] Session ID généré et affiché
- [ ] État JSON visible dans l'expander
- [ ] Configuration modifiable
- [ ] Réinitialisation fonctionne

### Performance
- [ ] Temps de chargement < 5 secondes
- [ ] Aucun lag lors de l'envoi de messages
- [ ] Interface responsive

## 🎉 Prochaines Étapes

Si tous les tests passent, vous êtes prêt pour **l'Étape 2** :
- Upload de datasets CSV
- Preview et statistiques
- Validation des données

---

**Bonne chance pour les tests !** 🚀

Si vous rencontrez des problèmes, consultez la section Dépannage ou créez une issue.

