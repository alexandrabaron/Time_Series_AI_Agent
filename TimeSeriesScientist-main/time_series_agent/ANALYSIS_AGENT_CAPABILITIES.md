# 📊 Analysis Agent - Capacités & Options UI

## Date: 20 Novembre 2025

---

## 🎯 Vue d'Ensemble

L'**AnalysisAgent** est responsable de l'analyse statistique approfondie des séries temporelles. C'est l'étape qui détermine les caractéristiques des données et guide la sélection des modèles.

---

## 📋 Analyses Disponibles

### 1. **Analyse de Tendance (Trend Analysis)** 📈

#### Ce qui peut être détecté :
- **Direction** : Tendance croissante, décroissante, ou stable
- **Force** : Pente de la tendance (régression linéaire)
- **Type** : Linéaire, exponentielle, polynomiale
- **Changements de régime** : Points où la tendance change

#### Méthodes disponibles :
- Régression linéaire simple
- Régression polynomiale (degré 2, 3)
- Moving average (moyenne mobile)
- Exponential smoothing
- Décomposition STL

#### Options UI possibles :
```
🎛️ Type de détection de tendance :
[ ] Linéaire (simple, rapide)
[ ] Polynomiale (degré 2-3)
[ ] Moyenne mobile (fenêtre: [7, 14, 30] jours)
[x] Automatique (recommandé)
```

---

### 2. **Analyse de Saisonnalité (Seasonality)** 🔄

#### Ce qui peut être détecté :
- **Périodes saisonnières** : Quotidienne, hebdomadaire, mensuelle, annuelle
- **Force de la saisonnalité** : Amplitude des variations
- **Composantes multiples** : Plusieurs périodes simultanées
- **Type** : Additive ou multiplicative

#### Méthodes disponibles :
- Décomposition saisonnière (seasonal_decompose)
- Analyse de Fourier (FFT)
- Autocorrélation (ACF)
- Tests statistiques (Kruskal-Wallis, Friedman)

#### Périodes à tester :
```python
seasonal_periods = [
    3,   # Tri-horaire (pour données horaires)
    7,   # Hebdomadaire
    12,  # Mensuelle (si 12 mois)
    24,  # Journalière (si données horaires)
    168, # Hebdomadaire (si données horaires)
    365, # Annuelle (si données journalières)
]
```

#### Options UI possibles :
```
🔄 Périodes saisonnières à analyser :
[x] Automatique (détection automatique)
[ ] Personnalisée :
    Période 1: [____] points
    Période 2: [____] points
    
🎯 Type de décomposition :
[ ] Additive (variations constantes)
[ ] Multiplicative (variations proportionnelles)
[x] Automatique
```

---

### 3. **Tests de Stationnarité** 📏

#### Tests disponibles :

##### A. **Augmented Dickey-Fuller (ADF)**
- Test pour racine unitaire
- H0 : Série non-stationnaire
- p-value < 0.05 → Série stationnaire

##### B. **KPSS Test**
- Test de stationnarité autour d'une tendance déterministe
- H0 : Série stationnaire
- p-value < 0.05 → Série non-stationnaire

##### C. **Phillips-Perron (PP)**
- Alternative robuste à ADF
- Gère mieux l'hétéroscédasticité

#### Résultats :
- **Stationnaire** : Variance et moyenne constantes → Bon pour ARIMA/ARMA
- **Non-stationnaire** : Nécessite différenciation → ARIMA avec d>0
- **Trend-stationary** : Stationnaire après suppression de tendance

#### Options UI possibles :
```
📏 Tests de stationnarité :
[x] ADF (Augmented Dickey-Fuller)
[x] KPSS 
[ ] Phillips-Perron (PP)

🎯 Niveau de confiance :
( ) 90% (p < 0.10)
(x) 95% (p < 0.05)
( ) 99% (p < 0.01)
```

---

### 4. **Analyse d'Autocorrélation** 🔗

#### Graphiques disponibles :

##### A. **ACF (Autocorrelation Function)**
- Corrélation entre t et t-k
- Identifie le paramètre q de ARIMA
- Détecte la saisonnalité

##### B. **PACF (Partial Autocorrelation Function)**
- Corrélation partielle (contrôle des lags intermédiaires)
- Identifie le paramètre p de ARIMA

#### Nombre de lags :
```python
max_lag = min(len(data) // 2, 40)  # Généralement 20-40
```

#### Options UI possibles :
```
🔗 Analyse d'autocorrélation :
[x] ACF (Autocorrelation)
[x] PACF (Partial Autocorrelation)

Nombre de lags : [40] (max recommandé: 40)

🎯 Affichage :
[x] Bandes de confiance 95%
[ ] Valeurs numériques
[x] Graphiques
```

---

### 5. **Décomposition de Série Temporelle** 🧩

#### Composantes décomposées :
- **Tendance (Trend)** : Mouvement à long terme
- **Saisonnalité (Seasonal)** : Fluctuations périodiques
- **Résidus (Residual)** : Bruit aléatoire

#### Méthodes disponibles :
- **Classique** : Moyenne mobile
- **STL** : Seasonal and Trend decomposition using Loess (plus robuste)
- **X-11/X-13** : Pour séries économiques

#### Options UI possibles :
```
🧩 Décomposition :
( ) Classique (moyenne mobile)
(x) STL (recommandé - robuste)
( ) X-13 (données économiques)

🎯 Paramètres STL :
Période : [Automatique ▼] ou [12]
Robuste : [x] (ignore outliers)
```

---

### 6. **Statistiques Descriptives** 📊

#### Mesures disponibles :

##### Tendance Centrale :
- Moyenne, médiane, mode
- Moyenne tronquée (trimmed mean)

##### Dispersion :
- Variance, écart-type
- Intervalle interquartile (IQR)
- Min, Max, Range
- Coefficient de variation (CV)

##### Forme de Distribution :
- **Skewness** (asymétrie) : 
  - Négatif : Queue à gauche
  - Positif : Queue à droite
  - 0 : Symétrique
- **Kurtosis** (aplatissement) :
  - < 3 : Platykurtique (aplatie)
  - = 3 : Normale
  - > 3 : Leptokurtique (pointue)

##### Tests de Normalité :
- Shapiro-Wilk test
- Kolmogorov-Smirnov test
- Jarque-Bera test

#### Options UI possibles :
```
📊 Statistiques à calculer :
[x] Toutes (recommandé)
[ ] Personnalisées :
    [x] Tendance centrale
    [x] Dispersion
    [x] Forme (skewness, kurtosis)
    [ ] Tests de normalité
```

---

### 7. **Détection d'Anomalies & Points de Rupture** 🚨

#### Anomalies détectables :
- **Outliers** (déjà fait au preprocessing)
- **Change points** : Changements de moyenne/variance
- **Structural breaks** : Changements de régime
- **Spikes** : Pics isolés
- **Level shifts** : Changements permanents de niveau

#### Méthodes disponibles :
- CUSUM (Cumulative Sum)
- PELT (Pruned Exact Linear Time)
- Binary Segmentation
- Bayesian change point detection

#### Options UI possibles :
```
🚨 Détection de points de rupture :
[x] Activer la détection
Sensibilité : [Moyenne ▼]
            (Faible / Moyenne / Élevée)
            
Type de rupture à détecter :
[x] Changement de moyenne
[x] Changement de variance
[ ] Changement de tendance
```

---

### 8. **Analyse de Patterns Spécifiques** 🔍

#### Patterns détectables :

##### A. **Cycles**
- Mouvements non-périodiques
- Différents de la saisonnalité (pas de période fixe)

##### B. **Volatilité**
- Clustering de volatilité (périodes calmes/agitées)
- Effet ARCH/GARCH

##### C. **Long-term dependencies**
- Effets de mémoire longue
- Tests de Hurst exponent

##### D. **Patterns répétitifs**
- Motifs qui se répètent sans période fixe

#### Options UI possibles :
```
🔍 Patterns avancés :
[ ] Détection de cycles
[x] Analyse de volatilité
[ ] Mémoire longue (Hurst)
[ ] Motifs répétitifs
```

---

## 🎨 Visualisations Proposées

### Visualisations Obligatoires (Toujours générées) :
1. **Série temporelle brute** avec tendance superposée
2. **Décomposition** (Trend + Seasonal + Residual)
3. **ACF / PACF** (côte à côte)
4. **Distribution** (histogramme + boxplot + Q-Q plot)

### Visualisations Optionnelles (Sélectionnables) :
5. **Rolling statistics** (moyenne et écart-type mobiles)
6. **Seasonal plot** (plusieurs années superposées)
7. **Lag plot** (scatter plot t vs t-1)
8. **Periodogram** (analyse fréquentielle)
9. **Heatmap de corrélations** (entre différents lags)

#### Options UI possibles :
```
📊 Visualisations :
✓ Obligatoires (4) déjà incluses

📈 Visualisations supplémentaires :
[ ] Rolling statistics (fenêtre mobile)
[ ] Seasonal plot (comparaison périodes)
[x] Periodogram (analyse fréquentielle)
[ ] Lag plot
[ ] Heatmap corrélations
```

---

## 🎯 Proposition d'Interface UI

### Layout Proposé dans la Sidebar :

```
┌────────────────────────────────────────┐
│ 📊 2. Analyse Statistique              │
├────────────────────────────────────────┤
│                                        │
│ 🎯 Analyses à Effectuer :              │
│                                        │
│ Analyse de base :                      │
│ [x] Tendance                           │
│ [x] Saisonnalité                       │
│ [x] Stationnarité (ADF + KPSS)        │
│ [x] Autocorrélation (ACF/PACF)        │
│                                        │
│ ────────────────────────────           │
│                                        │
│ Analyses avancées :                    │
│ [ ] Points de rupture                  │
│ [ ] Cycles économiques                 │
│ [ ] Volatilité clustering              │
│                                        │
│ ────────────────────────────           │
│                                        │
│ ⚙️ Paramètres :                        │
│                                        │
│ Période saisonnière :                  │
│ (•) Auto-détection                     │
│ ( ) Manuel : [____]                    │
│                                        │
│ Lags ACF/PACF : [40]                   │
│                                        │
│ Type décomposition :                   │
│ [STL (robuste) ▼]                      │
│                                        │
│ ────────────────────────────           │
│                                        │
│ 📊 Visualisations Extra :              │
│ [ ] Rolling statistics                 │
│ [ ] Periodogram                        │
│ [ ] Seasonal plot                      │
│                                        │
│ ────────────────────────────           │
│                                        │
│ [🚀 Lancer l'Analyse]                  │
│                                        │
└────────────────────────────────────────┘
```

---

## 💬 Workflow Conversationnel Proposé

### Étape 1 : Lancer l'Analyse

```
User: [Clic sur "Lancer l'Analyse"]
