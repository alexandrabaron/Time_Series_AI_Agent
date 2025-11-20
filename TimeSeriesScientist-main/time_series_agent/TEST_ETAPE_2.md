# 🧪 Guide de Test - Étape 2 : Chargement des Données

## 🎯 Objectifs de l'Étape 2

Permettre à l'utilisateur de :
1. ✅ Uploader un fichier CSV
2. ✅ Voir les informations du dataset (lignes, colonnes, taille)
3. ✅ Sélectionner la colonne date
4. ✅ Sélectionner la colonne valeur cible
5. ✅ Voir un aperçu des données
6. ✅ Voir les avertissements de validation
7. ✅ Cliquer sur "Lancer le Pré-traitement"

## 📋 Pré-requis

- ✅ Étape 1 complétée et testée
- ✅ Application fonctionne : `python main.py`
- 📄 Un fichier CSV de test (ou utilisez l'exemple fourni ci-dessous)

## 🗂️ Fichier CSV de Test

Créez un fichier `test_data.csv` avec ce contenu :

```csv
date,temperature,humidity,sales
2023-01-01,15.2,65,100
2023-01-02,16.1,68,120
2023-01-03,14.8,70,95
2023-01-04,15.9,67,110
2023-01-05,17.2,65,130
2023-01-06,18.5,62,145
2023-01-07,19.1,60,160
2023-01-08,18.7,61,155
2023-01-09,17.3,64,140
2023-01-10,16.8,66,125
```

Ou utilisez le dataset ETTh1.csv déjà présent dans le projet :
```
TimeSeriesScientist-main/dataset/ETTh1.csv
```

## 🚀 Instructions de Test

### Test 1 : Upload de Fichier

1. **Lancez l'application** (si pas déjà lancée)
   ```bash
   python main.py
   ```

2. **Dans la barre latérale**, trouvez la section "📁 Datasets"

3. **Cliquez sur "Browse files"** ou glissez-déposez votre CSV

4. **Sélectionnez votre fichier** `test_data.csv` ou `ETTh1.csv`

**✅ Résultat attendu :**
- Un spinner "Chargement du fichier..." apparaît brièvement
- Un message de succès s'affiche dans le chat :
  ```
  ✅ Fichier chargé avec succès !
  📊 test_data.csv
  - Lignes : 10
  - Colonnes : 4
  - Taille : 0.01 MB
  Veuillez maintenant sélectionner les colonnes date et valeur cible.
  ```
- Dans la barre latérale, vous voyez maintenant :
  ```
  ✅ test_data.csv
  10 lignes × 4 colonnes
  ```

### Test 2 : Sélection des Colonnes

1. **Dans la section "🎯 Sélection des Colonnes"**, vous devriez voir 2 menus déroulants

2. **Premier menu : "📅 Colonne Date/Temps"**
   - Devrait afficher toutes les colonnes disponibles
   - La colonne "date" devrait être pré-sélectionnée (auto-détection)

3. **Deuxième menu : "🎯 Colonne Valeur Cible"**
   - Devrait afficher toutes les colonnes disponibles
   - Une colonne numérique devrait être pré-sélectionnée

4. **Testez la sélection** :
   - Changez la colonne date à "date"
   - Changez la colonne cible à "sales" (ou "OT" pour ETTh1.csv)

**✅ Résultat attendu :**
- Les sélections changent immédiatement
- Dans le panneau de droite "📊 Données Chargées", vous voyez :
  ```
  ✅ 10 lignes
  📅 Date : date
  🎯 Cible : sales
  ```

### Test 3 : Aperçu des Données

1. **Cliquez sur l'expander "👁️ Aperçu des Données"**

2. **Un tableau devrait s'afficher** montrant les 5 premières lignes

**✅ Résultat attendu :**
- Tableau interactif avec 5 lignes
- Toutes les colonnes visibles
- Données bien formatées

### Test 4 : Avertissements de Validation

Si votre dataset a des problèmes (valeurs manquantes, duplicatas, etc.) :

1. **Un expander "⚠️ Avertissements"** apparaît

2. **Cliquez dessus** pour voir les détails

**✅ Résultat attendu :**
- Liste des avertissements si applicable
- Ex: "Valeurs manquantes détectées : 3 (5.00%)"
- Ex: "Lignes dupliquées détectées : 2"

### Test 5 : Bouton Pré-traitement

1. **Après avoir sélectionné les colonnes**, un bouton apparaît :
   ```
   🚀 1. Lancer le Pré-traitement
   ```

2. **Cliquez sur ce bouton**

**✅ Résultat attendu :**
- Un message utilisateur s'ajoute au chat : "Lancer le pré-traitement"
- Une réponse de l'assistant s'affiche :
  ```
  🔄 Pré-traitement lancé...
  
  Configuration :
  - Date : date
  - Valeur cible : sales
  
  Note : L'agent de prétraitement sera connecté dans la prochaine étape.
  ```
- L'indicateur d'étape change : 🏁 Initial → 🧹 Prétraitement
- Les questions suggérées changent (contextuelles à l'étape preprocessing)

### Test 6 : Questions Suggérées Contextuelles

1. **Regardez le panneau "💡 Suggestions"**

2. **Les questions devraient avoir changé** après avoir cliqué sur "Lancer le Pré-traitement" :
   - "Pourquoi ces valeurs sont des outliers ?"
   - "Quelle est la qualité de mes données ?"
   - "Montre-moi les statistiques"

3. **Cliquez sur une question suggérée**

**✅ Résultat attendu :**
- La question est ajoutée au chat automatiquement
- Une réponse générique s'affiche (pour l'instant)

### Test 7 : Charger un Nouveau Fichier

1. **Cliquez à nouveau sur "Browse files"** dans la barre latérale

2. **Sélectionnez un autre fichier CSV** (ou le même)

**✅ Résultat attendu :**
- Le nouveau fichier est chargé
- L'ancien est remplacé
- Les sélections de colonnes sont réinitialisées
- Un nouveau message de succès apparaît dans le chat
- L'étape revient à "Initial"

### Test 8 : Validation avec Dataset Invalide

1. **Créez un fichier CSV vide** `empty.csv` avec juste une ligne :
   ```csv
   date,value
   ```

2. **Uploadez ce fichier**

**✅ Résultat attendu :**
- Le fichier se charge
- Des avertissements apparaissent :
  - "Dataset très petit (0 lignes). Recommandé : au moins 100 lignes."
- Le bouton de prétraitement peut ne pas apparaître (si validation échoue)

## 🎯 Fonctionnalités Testées

### ✅ Upload et Stockage
- [x] Upload de fichier CSV
- [x] Parsing du CSV avec pandas
- [x] Stockage dans `st.session_state.data`
- [x] Métadonnées extraites et stockées

### ✅ Auto-détection
- [x] Détection automatique des colonnes date (par nom ou type)
- [x] Détection automatique des colonnes numériques (target)
- [x] Pré-sélection intelligente dans les menus

### ✅ Sélection Interactive
- [x] Menu déroulant pour colonne date
- [x] Menu déroulant pour colonne cible
- [x] Stockage des sélections dans session_state
- [x] Affichage des sélections dans le panneau de droite

### ✅ Validation
- [x] Détection des valeurs manquantes
- [x] Détection des duplicatas
- [x] Vérification de la taille du dataset
- [x] Affichage des avertissements

### ✅ Aperçu
- [x] Affichage des premières lignes
- [x] Format tableau interactif
- [x] Expander collapsible

### ✅ Workflow
- [x] Bouton "Lancer le Pré-traitement" conditionnel
- [x] Message ajouté au chat lors du clic
- [x] Changement d'étape (initial → preprocessing)
- [x] Questions suggérées mises à jour

## 🐛 Cas d'Erreur à Tester

### Erreur 1 : Fichier Non-CSV
**Action** : Essayez d'uploader un fichier .txt ou .xlsx

**Résultat attendu** : 
- Le file uploader ne permet pas la sélection
- Seuls les .csv sont acceptés

### Erreur 2 : CSV Mal Formaté
**Action** : Créez un fichier avec des données incohérentes

**Résultat attendu** :
- Message d'erreur : "❌ Erreur lors du chargement du fichier : ..."
- Le dataset n'est pas chargé

### Erreur 3 : Fichier Trop Grand
**Action** : Essayez avec un fichier > 200 MB (limite Streamlit par défaut)

**Résultat attendu** :
- Streamlit affiche une erreur de taille
- Le fichier n'est pas chargé

## 📊 État de Session après Étape 2

Vérifiez dans l'expander "ℹ️ Informations de Session" :

```json
{
  "session_id": "...",
  "current_step": "preprocessing",  // ou "initial" selon où vous en êtes
  "num_messages": 3,  // ou plus
  "has_data": true,  // IMPORTANT : doit être true
  "dataset_info": {
    "name": "test_data.csv",
    "num_rows": 10,
    "num_columns": 4,
    "columns": ["date", "temperature", "humidity", "sales"]
  },
  "config": {
    "horizon": 96,
    "num_models": 3,
    ...
  }
}
```

Et dans `st.session_state` (non visible directement, mais vérifié en interne) :
- `st.session_state.data` : DataFrame pandas
- `st.session_state.date_col` : "date"
- `st.session_state.target_col` : "sales"

## ✅ Critères de Succès Globaux

L'Étape 2 est réussie si :

- [x] ✅ Un fichier CSV peut être uploadé sans erreur
- [x] ✅ Les informations du dataset s'affichent correctement
- [x] ✅ Les menus déroulants montrent toutes les colonnes
- [x] ✅ L'auto-détection fonctionne (colonnes pré-sélectionnées intelligemment)
- [x] ✅ Les sélections de colonnes sont stockées et affichées
- [x] ✅ L'aperçu des données fonctionne
- [x] ✅ La validation détecte les problèmes
- [x] ✅ Le bouton "Lancer le Pré-traitement" apparaît
- [x] ✅ Cliquer sur le bouton change l'étape et ajoute un message
- [x] ✅ Les questions suggérées changent selon l'étape
- [x] ✅ Aucune erreur dans la console/terminal

## 🎨 Capture d'Écran Attendue

### Après Upload et Sélection

```
┌─────────────────────────────────────────────────────────────┐
│  📈 TSci-Chat                                                │
├──────────────┬──────────────────────────────────────────────┤
│ 📁 Datasets  │  💬 Conversation                             │
│              │  ┌────────────────────────────────────────┐  │
│ [Browse...]  │  │ 🤖 Bonjour ! Je suis TSci-Chat...     │  │
│              │  │                                        │  │
│ ✅ test.csv  │  │ 🤖 ✅ Fichier chargé avec succès !   │  │
│ 10 × 4       │  │    📊 test_data.csv                   │  │
│              │  │    - Lignes : 10                      │  │
│ 🎯 Sélection │  │    - Colonnes : 4                     │  │
│              │  │    - Taille : 0.01 MB                 │  │
│ 📅 Date:     │  │    Sélectionnez les colonnes...      │  │
│ [date ▼]     │  │                                        │  │
│              │  │ [Votre message...]                    │  │
│ 🎯 Cible:    │  └────────────────────────────────────────┘  │
│ [sales ▼]    │                                              │
│              │  💡 Suggestions                              │
│ 👁️ Aperçu   │  [Comment uploader mes données ?]            │
│ [5 lignes]   │  ...                                         │
│              │                                              │
│ 🚀 1. Lancer │  📍 Étape Actuelle                           │
│ Prétraitement│  🏁 Initial                                  │
│              │                                              │
│ ⚙️ Config    │  📊 Données Chargées                         │
│ Horizon: 96  │  ✅ 10 lignes                                │
│              │  📅 Date : date                              │
│              │  🎯 Cible : sales                            │
└──────────────┴──────────────────────────────────────────────┘
```

## 🎉 Prochaine Étape

Si tous les tests passent, vous êtes prêt pour **l'Étape 3** :
- Orchestrateur conversationnel
- Classification des intentions
- Intégration avec PreprocessAgent

---

**Bonne chance pour les tests de l'Étape 2 !** 🚀

Si vous rencontez des problèmes, vérifiez :
1. Que `utils/dataset_manager.py` existe
2. Que le fichier CSV est bien formaté
3. Les logs dans le terminal

