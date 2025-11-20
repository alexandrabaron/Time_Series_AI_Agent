# 📊 Analysis Agent - Guide Complet & Proposition UI

## Date: 20 Novembre 2025

---

## 🎯 Résumé Exécutif

L'**AnalysisAgent** effectue une analyse statistique approfondie des séries temporelles pour :
1. Comprendre les caractéristiques des données
2. Identifier les patterns (tendance, saisonnalité)
3. Tester les propriétés statistiques (stationnarité)
4. Guider la sélection des modèles de prévision

---

## 📋 Table des Analyses Disponibles

| Analyse | Description | Obligatoire | Coût Computation |
|---------|-------------|-------------|------------------|
| **Tendance** | Direction et force | ✅ Oui | 🟢 Faible |
| **Saisonnalité** | Patterns périodiques | ✅ Oui | 🟡 Moyen |
| **Stationnarité** | ADF + KPSS tests | ✅ Oui | 🟢 Faible |
| **Autocorrélation** | ACF + PACF | ✅ Oui | 🟢 Faible |
| **Décomposition** | Trend/Season/Residual | ✅ Oui | 🟡 Moyen |
| **Points de rupture** | Change points | 🔵 Optionnel | 🟡 Moyen |
| **Volatilité** | Clustering | 🔵 Optionnel | 🟡 Moyen |
| **Statistiques** | Descriptives | ✅ Oui | 🟢 Faible |

---

## 🎨 Proposition d'Interface UI

### Option A : Interface Simple (Recommandée pour MVP)

```
┌─────────────────────────────────────────────────────────┐
│ 📊 2. Analyse Statistique                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Les analyses suivantes seront effectuées :             │
│                                                         │
│ ✓ Analyse de tendance                                  │
│ ✓ Détection de saisonnalité                            │
│ ✓ Tests de stationnarité (ADF, KPSS)                  │
│ ✓ Autocorrélation (ACF/PACF)                           │
│ ✓ Décomposition de la série                            │
│ ✓ Statistiques descriptives                            │
│                                                         │
│ ⚙️ Configuration :                                      │
│                                                         │
│ Période saisonnière :                                   │
│ [Détection automatique ▼]                               │
│   - Automatique (recommandé)                            │
│   - 7 (Hebdomadaire)                                   │
│   - 12 (Mensuelle)                                     │
│   - 24 (Journalière)                                   │
│   - Personnalisée : [___]                               │
│                                                         │
│ [🚀 Lancer l'Analyse Complète]                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Avantages** :
- Simple et rapide
- Pas de choix à faire → moins de confusion
- Toutes les analyses importantes sont faites
- 1 seul clic pour lancer

**Inconvénients** :
- Moins de contrôle pour utilisateur avancé
- Temps de calcul plus long (tout est fait)

---

### Option B : Interface Modulaire (Pour utilisateurs avancés)

```
┌─────────────────────────────────────────────────────────┐
│ 📊 2. Analyse Statistique                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🎯 Sélectionnez les analyses à effectuer :             │
│                                                         │
│ Analyses de base :                                      │
│ [x] Tendance                                            │
│ [x] Saisonnalité                                        │
│ [x] Stationnarité                                       │
│ [x] Autocorrélation                                     │
│                                                         │
│ Analyses avancées :                                     │
│ [ ] Points de rupture (change points)                   │
│ [ ] Analyse de volatilité                               │
│ [ ] Détection de cycles                                 │
│                                                         │
│ ────────────────────────────                            │
│                                                         │
│ ⚙️ Paramètres :                                         │
│                                                         │
│ └► Saisonnalité                                         │
│    Période : [Auto ▼] Lags ACF : [40]                  │
│                                                         │
│ └► Décomposition                                        │
│    Type : [STL (robuste) ▼]                            │
│                                                         │
│ └► Visualisations extra                                │
│    [ ] Rolling statistics                               │
│    [ ] Periodogram (fréquences)                        │
│    [ ] Seasonal plot                                   │
│                                                         │
│ [🚀 Lancer l'Analyse Personnalisée]                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Avantages** :
- Contrôle fin pour utilisateurs experts
- Peut désactiver analyses coûteuses
- Personnalisation des paramètres
- Options de visualisation

**Inconvénients** :
- Plus complexe
- Risque de confusion pour débutants
- Plus d'espace UI nécessaire

---

### Option C : Interface Hybride (Ma Recommandation) 🌟

```
┌─────────────────────────────────────────────────────────┐
│ 📊 2. Analyse Statistique                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Mode : (•) Standard    ( ) Avancé                       │
│                                                         │
│ ═══════════ MODE STANDARD ═══════════                   │
│                                                         │
│ ✓ Toutes les analyses de base incluses                 │
│                                                         │
│ Période saisonnière :                                   │
│ [Détection automatique ▼]                               │
│                                                         │
│ [🚀 Lancer l'Analyse]                                   │
│                                                         │
│ [⚙️ Options avancées ▼]  ←  Expander                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Quand l'utilisateur clique sur "Options avancées" :

```
┌─────────────────────────────────────────────────────────┐
│ ⚙️ Options Avancées                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Analyses supplémentaires :                              │
│ [ ] Détection de points de rupture                      │
│ [ ] Analyse de volatilité                               │
│                                                         │
│ Paramètres de décomposition :                           │
│ Type : [STL ▼]  Robuste : [x]                          │
│                                                         │
│ ACF/PACF :                                              │
│ Nombre de lags : [40] (recommandé: 20-50)              │
│                                                         │
│ Visualisations supplémentaires :                        │
│ [ ] Rolling statistics (fenêtre mobile)                 │
│ [ ] Periodogram (analyse fréquentielle)                │
│ [ ] Seasonal plot (comparaison périodes)               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Avantages** :
- ✅ Simple par défaut (bon pour débutants)
- ✅ Puissant quand nécessaire (bon pour experts)
- ✅ Progressive disclosure (complexité masquée)
- ✅ Garde l'UI propre

**Recommandation** : **Option C (Hybride)** pour équilibrer simplicité et puissance.

---

## 📊 Résultats Attendus dans le Chat

### Format de Sortie

```markdown
🤖 Assistant :

📊 **Analyse Statistique Terminée !**

## 📈 Tendance
- Direction : **Stable** (pente : 0.0012)
- Type : Linéaire
- Conclusion : Pas de tendance significative détectée

## 🔄 Saisonnalité
- Période détectée : **24 heures** (forte saisonnalité)
- Force : **Élevée** (amplitude : 15.3%)
- Type : Additive

## 📏 Stationnarité
- **ADF Test** : p-value = 0.003 → ✅ **Stationnaire**
- **KPSS Test** : p-value = 0.12 → ✅ **Stationnaire**
- Conclusion : Série stationnaire, pas de différenciation nécessaire

## 🔗 Autocorrélation
- **ACF** : Décroissance lente → dépendance temporelle forte
- **PACF** : Pic significatif à lag 1 → AR(1) suggéré
- Conclusion : Modèles ARIMA(1,0,0) ou SARIMA recommandés

## 📊 Statistiques Descriptives
- Moyenne : 45.67
- Écart-type : 12.34
- Skewness : 0.23 (légèrement asymétrique à droite)
- Kurtosis : 2.98 (distribution normale)

## 🎯 Recommandations pour le Modèle
1. ✅ **SARIMA(1,0,0)(1,0,0)[24]** - Recommandé (saisonnalité)
2. ✅ **ExponentialSmoothing** - Bonne alternative
3. ✅ **Prophet** - Gère bien la saisonnalité
4. ⚠️ **ARIMA simple** - Peut ne pas capturer la saisonnalité

📊 **Visualisations générées** : 6 graphiques disponibles

💬 **Prochaine étape** : Voulez-vous lancer la sélection de modèles ?
```

---

## 🖼️ Visualisations Générées

### 1. **Vue d'Ensemble** (4 sous-graphiques)

```
┌─────────────────┬─────────────────┐
│ Série Temporelle│ Décomposition   │
│ + Tendance      │ - Trend         │
│                 │ - Seasonal      │
│                 │ - Residual      │
├─────────────────┼─────────────────┤
│ ACF             │ PACF            │
│                 │                 │
└─────────────────┴─────────────────┘
```

### 2. **Distribution** (4 sous-graphiques)

```
┌─────────────────┬─────────────────┐
│ Histogramme     │ Box Plot        │
│ + KDE           │                 │
├─────────────────┼─────────────────┤
│ Q-Q Plot        │ Statistiques    │
│ (normalité)     │ (tableau)       │
└─────────────────┴─────────────────┘
```

### 3. **Seasonal Plot** (si demandé)

```
┌─────────────────────────────────────┐
│ Plusieurs périodes superposées      │
│ (ex: 12 mois sur 3 ans)            │
└─────────────────────────────────────┘
```

### 4. **Periodogram** (si demandé)

```
┌─────────────────────────────────────┐
│ Analyse fréquentielle (FFT)         │
│ Pics = périodes dominantes          │
└─────────────────────────────────────┘
```

---

## ⚙️ Paramètres Configurables

### 1. **Période Saisonnière**

```python
OPTIONS = {
    'auto': None,  # Détection automatique
    'hourly': 24,
    'daily': 7,
    'weekly': 52,
    'monthly': 12,
    'quarterly': 4,
    'custom': user_input
}
```

**Recommandation** : Toujours proposer "Auto" par défaut.

---

### 2. **Type de Décomposition**

```python
DECOMPOSITION_TYPES = {
    'additive': 'Y = T + S + R',      # Variations constantes
    'multiplicative': 'Y = T × S × R',# Variations proportionnelles
    'stl': 'Seasonal-Trend Loess',    # Robuste (recommandé)
}
```

**Recommandation** : STL par défaut (robuste aux outliers).

---

### 3. **Lags pour ACF/PACF**

```python
max_lag = min(len(data) // 2, 40)  # Généralement 20-40
```

**Recommandation** : 40 par défaut, ajustable si données courtes.

---

### 4. **Niveau de Confiance (Tests)**

```python
CONFIDENCE_LEVELS = {
    0.90: 0.10,  # 90% confidence
    0.95: 0.05,  # 95% confidence (défaut)
    0.99: 0.01,  # 99% confidence
}
```

**Recommandation** : 95% par défaut (standard).

---

## 🎯 Workflow Conversationnel Détaillé

### Scénario 1 : Analyse Standard (Pas d'options)

```
1. User: [Clic sur "🚀 Lancer l'Analyse"]
   
2. System: 
   - Spinner: "Analyse en cours..."
   - Progress: 
     ✓ Calcul de la tendance...
     ✓ Détection de saisonnalité...
     ✓ Tests de stationnarité...
     ✓ Calcul ACF/PACF...
     ✓ Génération visualisations...
   
3. Assistant:
   - Message complet avec résultats
   - 6 visualisations disponibles
   - Recommandations de modèles
   
4. User peut :
   - Poser des questions : "Pourquoi SARIMA ?"
   - Voir les visualisations (tabs)
   - Continuer vers sélection de modèles
```

---

### Scénario 2 : Analyse Avec Options

```
1. User: [Ouvre "Options avancées"]
   User: [Coche "Détection de points de rupture"]
   User: [Change période à "7"]
   User: [Coche "Periodogram"]
   User: [Clic "Lancer l'Analyse"]
   
2. System:
   - Spinner avec étapes supplémentaires
   - "Détection de change points..."
   - "Génération du periodogram..."
   
3. Assistant:
   - Message complet PLUS
   - Section "Points de Rupture" :
     "2 points de rupture détectés aux indices 2341, 5678"
   - Visualisation periodogram ajoutée
   
4. User peut demander :
   - "Montre-moi les points de rupture"
   - "Pourquoi ces pics dans le periodogram ?"
```

---

### Scénario 3 : Questions Utilisateur

```
User: "Pourquoi la série est-elle stationnaire ?"
