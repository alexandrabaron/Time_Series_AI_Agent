# 🧪 Guide de Test - Étape 5 : Analysis Agent

## 🎯 Objectifs de l'Étape 5

Permettre à l'utilisateur de :
1. ✅ Configurer la période saisonnière
2. ✅ Lancer l'analyse statistique complète
3. ✅ Voir les résultats d'analyse dans le chat
4. ✅ Obtenir des recommandations de modèles
5. ✅ Continuer vers la sélection de modèles

## 📋 Pré-requis

- ✅ Étapes 1-4 complétées
- ✅ Données prétraitées (Étape 4 complétée)
- ✅ Application en cours d'exécution

## 🚀 Instructions de Test

### Test 1 : Configuration de la Période Saisonnière

**Où** : Barre latérale, section "⚙️ Configuration" → "📊 Analyse"

1. **Trouvez le menu déroulant "Période saisonnière"**

2. **Vérifiez les options disponibles** :
   - Détection automatique (sélectionné par défaut)
   - 7 (Hebdomadaire)
   - 12 (Mensuelle)
   - 24 (Journalière - données horaires)
   - 168 (Hebdomadaire - données horaires)
   - Personnalisée

3. **Testez chaque option** :
   - Sélectionnez "24 (Journalière - données horaires)"
   - Vérifiez que la sélection change

4. **Testez l'option personnalisée** :
   - Sélectionnez "Personnalisée"
   - Un input numérique devrait apparaître
   - Entrez "30"
   - La valeur devrait être acceptée

5. **Remettez sur "Détection automatique"** pour les tests suivants

**✅ Résultat attendu** :
- Toutes les options sont disponibles
- L'option personnalisée fonctionne
- La configuration est mise à jour dans session_state

---

### Test 2 : Lancer l'Analyse Statistique

**Où** : Barre latérale, section "Datasets"

**État requis** : current_step = 'preprocessing_complete'

1. **Vérifiez que vous voyez** :
   ```
   ✅ Prétraitement terminé
   [📊 2. Lancer l'Analyse Statistique]
   ```

2. **Cliquez sur "📊 2. Lancer l'Analyse Statistique"**

3. **Observez** :
   - Spinner "Analyse statistique en cours..." s'affiche
   - Message utilisateur ajouté au chat : "Lancer l'analyse statistique"
   - Analyse s'exécute (peut prendre 5-10 secondes)

**✅ Résultat attendu** :
- Pas d'erreur
- Spinner disparaît après quelques secondes
- Message complet apparaît dans le chat

---

### Test 3 : Vérifier les Résultats dans le Chat

**Contenu attendu du message** :

```markdown
📊 Analyse Statistique Terminée !

## 📈 Tendance
- Direction : Stable (pente : 0.XXXXXX)
- Force : Faible/Modérée/Forte (R² = X.XXX)
- ✓ Pas de tendance significative
  OU
- ⚠️ Tendance forte/modérée détectée

## 🔄 Saisonnalité
- Période détectée : XX points
- Force : Forte/Modérée/Faible (XX.X%)
- Type : Additive
  OU
- ✓ Aucune saisonnalité significative détectée

## 📏 Stationnarité
- ADF Test : ✅/❌ p-value = X.XXXX
- KPSS Test : ✅/❌ p-value = X.XXXX
- Conclusion : Série stationnaire/non-stationnaire
- ✓ Pas de différenciation nécessaire
  OU
- ⚠️ Différenciation recommandée (d=1)

## 🔗 Autocorrélation
- ACF : X lags significatifs
- PACF : X lags significatifs
- Paramètres ARIMA suggérés : p=X, q=X

## 📊 Statistiques Descriptives
- Moyenne : XX.XX | Médiane : XX.XX
- Écart-type : XX.XX | Variance : XX.XX
- Min : XX.XX | Max : XX.XX
- Asymétrie (skewness) : X.XX (symétrique/asymétrique)

## 🎯 Modèles Recommandés
1. ⭐ SARIMA (X,X,X)(X,X,X)[XX]
   Raison...
2. ✅ ARIMA (X,X,X)
   Raison...
3. ✅ Prophet default
   Raison...
4. ✅ ExponentialSmoothing trend=X, seasonal=X
   Raison...

📊 Visualisations : Graphiques d'analyse générés

💬 Prochaine étape : Voulez-vous lancer la sélection de modèles ?
```

**Vérifications** :
- [ ] Toutes les sections sont présentes
- [ ] Les valeurs numériques sont affichées
- [ ] Les symboles ✅/❌ sont corrects
- [ ] Les recommandations de modèles sont pertinentes
- [ ] Le message est bien formaté

---

### Test 4 : Vérifier le Changement d'État

1. **Dans la barre latérale**, vérifiez que le statut a changé :
   ```
   ✅ Analyse terminée
   [🎯 3. Sélection de Modèles]
   ```

2. **Dans le panneau de droite "📍 Étape Actuelle"** :
   - Devrait afficher : 🔍 Analyse (ou équivalent)

3. **Dans "ℹ️ Informations de Session"** (expander) :
   ```json
   {
     "current_step": "analysis_complete",
     "results": {
       "preprocess_analysis": {...},
       "preprocess_applied": {...},
       "analysis": {...}  ← NOUVEAU
     }
   }
   ```

**✅ Résultat attendu** :
- État correctement mis à jour
- Nouveau bouton "Sélection de Modèles" visible
- Résultats stockés dans session_state

---

### Test 5 : Vérifier les Résultats Stockés

**Dans l'expander "ℹ️ Informations de Session"**, vérifiez que `results.analysis` contient :

```json
{
  "status": "success",
  "summary": "...",
  "results": {
    "trend": {
      "direction": "...",
      "slope": 0.xxx,
      "r_squared": 0.xxx,
      "strength": "..."
    },
    "seasonality": {
      "detected": true/false,
      "period": XX,
      "strength": XX.X,
      "strength_label": "..."
    },
    "stationarity": {
      "adf": {...},
      "kpss": {...},
      "conclusion": "...",
      "needs_differencing": true/false
    },
    "acf_pacf": {
      "suggested_p": X,
      "suggested_q": X,
      "significant_acf_lags": [...],
      "significant_pacf_lags": [...]
    },
    "decomposition": {...},
    "statistics": {
      "mean": XX.XX,
      "std": XX.XX,
      ...
    }
  },
  "recommendations": [...]
}
```

---

### Test 6 : Questions sur l'Analyse

**Testez ces questions dans le chat** :

1. **"Pourquoi la série est-elle stationnaire ?"**
   - Devrait répondre avec référence aux tests ADF/KPSS

2. **"Quelle est la tendance ?"**
   - Devrait répondre avec direction et pente

3. **"Y a-t-il de la saisonnalité ?"**
   - Devrait répondre avec période détectée (ou absence)

4. **"Pourquoi recommander SARIMA ?"**
   - Devrait expliquer basé sur saisonnalité détectée

**✅ Résultat attendu** :
- Réponses pertinentes (même si simples pour le moment)
- Pas d'erreur

---

### Test 7 : Tester Avec Différentes Périodes

**Recommencez le preprocessing puis l'analyse avec différentes configurations** :

1. **Période = 7** :
   - Relancez l'analyse
   - Vérifiez que la période 7 est utilisée
   - Les résultats changent

2. **Période = 24** :
   - Relancez l'analyse
   - Vérifiez que la période 24 est utilisée
   - Les résultats changent

3. **Période = Auto** :
   - Relancez l'analyse
   - Vérifiez que la période est auto-détectée
   - Comparez avec les résultats précédents

**✅ Résultat attendu** :
- La configuration est bien prise en compte
- Les résultats varient selon la période
- Pas d'erreur pour aucune configuration

---

### Test 8 : Tester sur Données Sans Saisonnalité

**Si vous avez un dataset sans saisonnalité** :

1. Uploadez-le
2. Prétraitez-le
3. Lancez l'analyse

**✅ Résultat attendu** :
- Message : "✓ Aucune saisonnalité significative détectée"
- Recommandations n'incluent pas SARIMA en priorité 1
- ARIMA simple recommandé à la place

---

### Test 9 : Tester sur Données Non-Stationnaires

**Si vous avez un dataset avec forte tendance** :

1. Uploadez-le
2. Prétraitez-le
3. Lancez l'analyse

**✅ Résultat attendu** :
- ADF Test : ❌ (p-value > 0.05)
- Message : "⚠️ Différenciation recommandée (d=1)"
- ARIMA(p,1,q) recommandé (avec d=1)

---

## 🐛 Cas d'Erreur à Tester

### Erreur 1 : Données Trop Courtes

**Si vous avez un dataset < 50 points** :

**Résultat attendu** :
- Analyse devrait fonctionner mais avec warnings
- Certains tests peuvent échouer gracieusement
- Message d'erreur clair si échec total

---

### Erreur 2 : Données avec Valeurs Constantes

**Si toutes les valeurs sont identiques** :

**Résultat attendu** :
- Message : "Tendance : Stable"
- Pas de saisonnalité détectée
- Variance = 0

---

### Erreur 3 : Période Invalide

**Testez avec période personnalisée = 1** :

**Résultat attendu** :
- Devrait refuser (min = 2)
- OU gérer gracieusement

---

## 📊 Analyses Effectuées

### ✅ Vérifiez que Toutes Ces Analyses Sont Faites :

- [x] **Tendance**
  - Direction calculée (croissante/décroissante/stable)
  - Pente calculée
  - R² calculé
  - Force évaluée

- [x] **Saisonnalité**
  - Période détectée (ou absence)
  - Force calculée
  - Type identifié (additive)

- [x] **Stationnarité**
  - Test ADF exécuté
  - Test KPSS exécuté
  - Conclusion donnée
  - Recommandation de différenciation si nécessaire

- [x] **Autocorrélation**
  - ACF calculée
  - PACF calculée
  - Lags significatifs identifiés
  - Paramètres p et q suggérés

- [x] **Décomposition**
  - Trend extrait
  - Seasonal extrait
  - Residual calculé
  - (Seulement si saisonnalité détectée)

- [x] **Statistiques Descriptives**
  - Moyenne, médiane, écart-type
  - Min, max, range
  - Quartiles
  - Skewness, kurtosis

- [x] **Recommandations de Modèles**
  - Au moins 3 modèles suggérés
  - Priorités assignées
  - Raisons expliquées

---

## ✅ Critères de Succès Globaux

L'Étape 5 est réussie si :

- [x] ✅ Configuration de période saisonnière fonctionne
- [x] ✅ Analyse statistique s'exécute sans erreur
- [x] ✅ Message complet et formaté dans le chat
- [x] ✅ Toutes les 6 analyses sont effectuées
- [x] ✅ Recommandations de modèles pertinentes
- [x] ✅ État changé vers 'analysis_complete'
- [x] ✅ Bouton "Sélection de Modèles" visible
- [x] ✅ Résultats stockés dans session_state
- [x] ✅ Aucune erreur dans console/terminal

---

## 🎨 Capture d'Écran Attendue

### Après Analyse Complète

```
┌─────────────────────────────────────────────────────────┐
│ 📁 Datasets                                             │
├─────────────────────────────────────────────────────────┤
│ ✅ ETTh1.csv                                            │
│ 17420 lignes × 8 colonnes                              │
│                                                         │
│ 🎯 Sélection des Colonnes                              │
│ ...                                                     │
│                                                         │
│ ✅ Analyse terminée                                     │
│ [🎯 3. Sélection de Modèles]                           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ ⚙️ Configuration                                        │
├─────────────────────────────────────────────────────────┤
│ 📊 Analyse                                              │
│                                                         │
│ Période saisonnière :                                   │
│ [Détection automatique ▼]                               │
│                                                         │
│ ────────────────                                        │
│                                                         │
│ 🔮 Prévision                                            │
│ Horizon : [96]                                          │
│ Modèles : ━━━━━●━━ 3                                    │
└─────────────────────────────────────────────────────────┘
```

### Dans le Chat

```
💬 User: Lancer l'analyse statistique

🤖 Assistant:

📊 Analyse Statistique Terminée !

## 📈 Tendance
- Direction : Stable (pente : 0.000012)
...

[Reste du message complet]

💬 Prochaine étape : Voulez-vous lancer la sélection de modèles ?
```

---

## 🎉 Prochaine Étape

Si tous les tests passent, vous êtes prêt pour **l'Étape 6** :
- ValidationAgent wrapper (sélection de modèles)
- Optimisation des hyperparamètres
- Comparaison des modèles

---

**Bonne chance pour les tests de l'Étape 5 !** 🚀

Si vous rencontrez des problèmes :
1. Vérifiez les logs dans le terminal
2. Consultez l'état de session (expander)
3. Vérifiez que scipy et statsmodels sont installés

