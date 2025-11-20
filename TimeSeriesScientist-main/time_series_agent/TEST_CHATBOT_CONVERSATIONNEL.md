# 🧪 Test Guide : Chatbot Conversationnel avec Llama3

## ✅ Prérequis

Avant de tester, vérifiez que :
1. **Ollama est installé et en cours d'exécution**
2. **Llama3 est téléchargé**

### Vérification Ollama

```bash
# Vérifier qu'Ollama fonctionne
ollama list

# Vous devriez voir :
# NAME         ID           SIZE    MODIFIED
# llama3:...   ...          4.7GB   ...

# Si llama3 n'est pas installé :
ollama pull llama3

# Démarrer Ollama (si nécessaire)
ollama serve
```

---

## 📦 Installation des Dépendances

```bash
cd TimeSeriesScientist-main/time_series_agent
pip install langchain-community
```

---

## 🚀 Lancement de l'Application

```bash
python main.py
```

L'application devrait s'ouvrir dans votre navigateur à `http://localhost:8501`.

---

## 🧪 Tests à Effectuer

### **Test 1 : Vérification que le LLM est disponible**

**Objectif** : S'assurer qu'Ollama est correctement connecté.

**Procédure** :
1. Ouvrez l'application
2. Dans le chat, tapez : `Bonjour, es-tu là ?`
3. **Résultat attendu** : Le bot répond de manière conversationnelle (ex: "Bonjour ! Oui, je suis là pour vous aider...")

**Si ça ne fonctionne pas** :
- Vérifiez qu'Ollama est en cours d'exécution : `ollama serve`
- Vérifiez les logs dans le terminal pour voir les erreurs

---

### **Test 2 : Questions sur le Dataset (sans données chargées)**

**Objectif** : Tester les réponses quand aucune donnée n'est disponible.

**Procédure** :
1. Sans uploader de fichier, posez les questions suivantes :
   - `Quelle est la qualité de mes données ?`
   - `Combien de valeurs manquantes ?`
   - `Mes données ont-elles une tendance ?`

**Résultat attendu** :
- Le bot répond qu'aucune donnée n'est disponible et invite à uploader un fichier

---

### **Test 3 : Workflow Complet avec Questions**

**Objectif** : Tester le chatbot tout au long du workflow.

#### Étape 1 : Upload du Dataset
1. Uploadez `ETTh1.csv` (ou votre fichier)
2. Sélectionnez `date` et `OT` (ou votre colonne cible)

#### Étape 2 : Questions Préliminaires
Posez ces questions :
- `Combien de lignes dans mon dataset ?`
- `Quelle est la colonne cible ?`
- `Résume-moi les informations du dataset`

**Résultat attendu** :
- Le bot répond avec les informations exactes du dataset

#### Étape 3 : Prétraitement
1. Cliquez sur **"🚀 1. Lancer le Pré-traitement"**
2. Attendez l'analyse
3. Appliquez les stratégies recommandées

#### Étape 4 : Questions sur le Prétraitement
Posez ces questions :
- `Pourquoi utiliser l'interpolation pour les valeurs manquantes ?`
- `Combien d'outliers ont été détectés ?`
- `Quelle est la qualité de mes données ?`

**Résultat attendu** :
- Le bot explique les choix de preprocessing en se basant sur le contexte
- Il cite les chiffres exacts (nombre d'outliers, pourcentage, etc.)

#### Étape 5 : Analyse Statistique
1. Cliquez sur **"📊 2. Lancer l'Analyse"**
2. Sélectionnez la période saisonnière (ex: 168 pour hebdomadaire)
3. Attendez les résultats

#### Étape 6 : Questions sur l'Analyse
Posez ces questions :
- `Mes données ont-elles une tendance ?`
- `Y a-t-il de la saisonnalité ?`
- `Les données sont-elles stationnaires ?`
- `Quels modèles recommandes-tu ?`
- `Pourquoi ARIMA ?`

**Résultat attendu** :
- Le bot répond avec les résultats de l'analyse
- Il explique les recommandations de modèles
- Il justifie pourquoi certains modèles sont adaptés

---

### **Test 4 : Commandes en Langage Naturel**

**Objectif** : Tester la détection d'intent et l'extraction de commandes.

**Procédure** :
Essayez de donner des commandes en langage naturel :
- `Lance l'analyse statistique`
- `Peux-tu analyser mes données ?`
- `Montre-moi les prévisions` (si implémenté)
- `Génère un rapport`

**Résultat attendu** :
- Le bot détecte qu'il s'agit d'une commande
- Il exécute la commande appropriée (ex: lance l'analyse)
- Ou indique que la fonctionnalité n'est pas encore implémentée

---

### **Test 5 : Questions Multi-Tours**

**Objectif** : Tester que le bot peut répondre à plusieurs questions successives.

**Procédure** :
Posez une série de questions :
1. `Quelle est la tendance ?`
2. `Et la saisonnalité ?`
3. `Pourquoi cette période ?`
4. `Quels modèles pour ce type de données ?`

**Résultat attendu** :
- Le bot répond à chaque question en se basant sur le contexte actuel
- Les réponses sont cohérentes et liées

---

### **Test 6 : Fallback quand Ollama est Down**

**Objectif** : Tester le comportement quand Ollama n'est pas disponible.

**Procédure** :
1. **Arrêtez Ollama** : Fermez le processus Ollama
2. Rechargez l'application Streamlit
3. Posez une question : `Quelle est la qualité de mes données ?`

**Résultat attendu** :
- Le bot indique qu'Ollama n'est pas disponible
- Il fournit des réponses fallback basiques (hardcodées) pour les questions simples
- Il suggère de redémarrer Ollama

---

### **Test 7 : Questions Hors Contexte**

**Objectif** : Tester que le bot ne "hallucine" pas et reste dans le contexte.

**Procédure** :
Posez des questions sans rapport avec vos données :
- `Quelle est la capitale de la France ?`
- `Comment faire un gâteau ?`
- `Explique-moi la mécanique quantique`

**Résultat attendu** :
- Le bot indique qu'il ne peut répondre qu'à des questions liées à l'analyse de séries temporelles
- Ou redirige vers des questions pertinentes

---

## 🐛 Problèmes Fréquents et Solutions

### Problème 1 : `ModuleNotFoundError: No module named 'langchain_community'`

**Solution** :
```bash
pip install langchain-community
```

### Problème 2 : "Ollama n'est pas disponible"

**Solution** :
```bash
# Démarrez Ollama
ollama serve

# Dans un autre terminal, testez
ollama run llama3 "Hello"
```

### Problème 3 : Le bot est très lent

**Cause** : Llama3 peut être lent sur certains ordinateurs

**Solutions** :
- Utilisez un modèle plus petit : `llama3:8b` au lieu de `llama3:70b`
- Dans `utils/local_llm.py`, modifiez :
  ```python
  self._llm = Ollama(
      model="llama3:8b",  # Plus rapide
      ...
  )
  ```

### Problème 4 : Réponses incohérentes

**Cause** : Le contexte n'est pas bien construit ou le prompt n'est pas assez précis

**Solution** :
- Vérifiez que les résultats sont bien stockés dans `st.session_state.results`
- Ajoutez des logs pour voir le contexte envoyé au LLM :
  ```python
  logger.info(f"Context: {context}")
  ```

---

## 📊 Critères de Succès

L'implémentation est considérée comme réussie si :

- [ ] Ollama se connecte correctement
- [ ] Le bot répond aux questions simples (qualité, valeurs manquantes, outliers)
- [ ] Le bot répond aux questions d'analyse (tendance, saisonnalité, stationnarité)
- [ ] Le bot explique les recommandations de modèles
- [ ] Les commandes en langage naturel sont détectées et exécutées
- [ ] Le fallback fonctionne quand Ollama est down
- [ ] Le bot reste dans le contexte (pas d'hallucinations)

---

## 📝 Rapport de Test

Après avoir effectué les tests, remplissez ce rapport :

### ✅ Tests Réussis
- Test 1 : ☐
- Test 2 : ☐
- Test 3 : ☐
- Test 4 : ☐
- Test 5 : ☐
- Test 6 : ☐
- Test 7 : ☐

### ❌ Tests Échoués
- (Listez les tests qui ont échoué et les erreurs observées)

### 🐛 Bugs Identifiés
- (Listez les bugs rencontrés)

### 💡 Améliorations Suggérées
- (Suggestions pour améliorer le chatbot)

---

## 🚀 Prochaines Étapes

Si tous les tests passent, vous pouvez :
1. **Améliorer les prompts** : Affiner les instructions données au LLM
2. **Ajouter la mémoire conversationnelle** : Inclure les derniers messages dans le contexte
3. **Intégrer les agents restants** : ValidationAgent, ForecastAgent, ReportAgent
4. **Optimiser les performances** : Réduire le temps de réponse

---

**Bon test ! 🎯**

