# 📊 Analysis Agent - Résumé des Options pour l'UI

## 🎯 Ma Recommandation : **Interface Hybride**

---

## ✅ Ce que l'Analysis Agent PEUT Faire

| Analyse | Description | Utile pour | Coût |
|---------|-------------|------------|------|
| **Tendance** | Croissante/Décroissante/Stable | Choix de modèles | 🟢 |
| **Saisonnalité** | Périodes répétitives (7j, 24h, etc.) | SARIMA vs ARIMA | 🟡 |
| **Stationnarité** | Tests ADF & KPSS | Différenciation nécessaire? | 🟢 |
| **Autocorrélation** | ACF/PACF (paramètres p,q) | Ordre ARIMA | 🟢 |
| **Décomposition** | Trend + Seasonal + Residual | Comprendre la structure | 🟡 |
| **Statistiques** | Moyenne, écart-type, skewness, etc. | Vue d'ensemble | 🟢 |
| **Points de rupture** | Changements de régime | Optionnel, détection anomalies | 🟡 |

---

## 🎨 UI Proposée (Mode Simple par Défaut)

```
┌───────────────────────────────────────────────┐
│ 📊 2. Analyse Statistique                     │
├───────────────────────────────────────────────┤
│                                               │
│ Les analyses suivantes seront effectuées :   │
│ ✓ Tendance                                    │
│ ✓ Saisonnalité                                │
│ ✓ Stationnarité (ADF, KPSS)                  │
│ ✓ Autocorrélation (ACF/PACF)                 │
│ ✓ Décomposition                               │
│                                               │
│ ⚙️ Configuration :                            │
│                                               │
│ Période saisonnière :                         │
│ [Détection automatique ▼]                     │
│   - Automatique (recommandé)                  │
│   - 7 (hebdomadaire)                          │
│   - 12 (mensuelle)                            │
│   - 24 (journalière - données horaires)      │
│   - Personnalisée                             │
│                                               │
│ [⚙️ Options avancées ▼]  ← Expander          │
│                                               │
│ [🚀 Lancer l'Analyse Complète]               │
│                                               │
└───────────────────────────────────────────────┘
```

### Si l'utilisateur ouvre "Options avancées" :

```
┌───────────────────────────────────────────────┐
│ ⚙️ Options Avancées                           │
├───────────────────────────────────────────────┤
│                                               │
│ Analyses supplémentaires :                    │
│ [ ] Détection de points de rupture            │
│     (change points dans la série)             │
│                                               │
│ Paramètres :                                  │
│ Lags ACF/PACF : [40] (recommandé: 20-50)     │
│                                               │
│ Type décomposition :                          │
│ [STL - Robuste ▼]                             │
│   - STL (recommandé - ignore outliers)        │
│   - Classique (moyenne mobile)                │
│                                               │
│ Visualisations supplémentaires :              │
│ [ ] Rolling statistics (fenêtre mobile)       │
│ [ ] Periodogram (analyse fréquentielle)       │
│ [ ] Seasonal plot (comparaison périodes)      │
│                                               │
└───────────────────────────────────────────────┘
```

---

## 💬 Résultat dans le Chat

```
🤖 Assistant :

📊 **Analyse Statistique Terminée !**

## 📈 Tendance
Direction : **Stable** (pente : 0.0012)
Type : Linéaire
✓ Pas de tendance significative

## 🔄 Saisonnalité
Période détectée : **24 heures**
Force : **Élevée** (15.3% d'amplitude)
Type : Additive

## 📏 Stationnarité
✅ **ADF Test** : p=0.003 → Stationnaire
✅ **KPSS Test** : p=0.12 → Stationnaire
✓ Pas de différenciation nécessaire

## 🔗 Autocorrélation
ACF : Décroissance lente
PACF : Pic à lag 1
→ Suggère AR(1)

## 🎯 Modèles Recommandés
1. **SARIMA(1,0,0)(1,0,0)[24]** ⭐ Meilleur choix
2. **ExponentialSmoothing** (alternative)
3. **Prophet** (gère bien la saisonnalité)

📊 6 visualisations générées

💬 Voulez-vous lancer la sélection de modèles ?
```

---

## 🖼️ Visualisations Générées

### Obligatoires (toujours créées) :

1. **Vue d'ensemble** (4 subplots)
   - Série temporelle + tendance
   - ACF
   - PACF  
   - Distribution (histogramme)

2. **Décomposition** (3 subplots)
   - Trend
   - Seasonal
   - Residual

### Optionnelles (si activées) :

3. **Rolling statistics** (moyenne & std mobiles)
4. **Periodogram** (analyse fréquentielle FFT)
5. **Seasonal plot** (périodes superposées)

---

## 🎯 Questions Utilisateur Supportées

| Question | Réponse Automatique |
|----------|---------------------|
| "Pourquoi SARIMA ?" | "Car saisonnalité de période 24 détectée + stationnarité" |
| "C'est quoi la stationnarité ?" | Explication + lien vers graphiques |
| "Montre-moi l'ACF" | Affiche la visualisation ACF |
| "Les données sont-elles normales ?" | Réfère au Q-Q plot + Shapiro test |
| "Quelle est la tendance ?" | "Stable, pente 0.0012" |

---

## 🔧 Implémentation Technique

### Wrapper à créer :

```python
class AnalysisAgentWrapper:
    def run(self, data, config):
        """
        Effectue l'analyse complète.
        
        Args:
            data: DataFrame preprocessé
            config: {
                'seasonal_period': int | 'auto',
                'acf_lags': int (default: 40),
                'decomposition_type': 'stl' | 'classic',
                'detect_changepoints': bool,
                'extra_viz': ['rolling', 'periodogram', 'seasonal']
            }
        
        Returns:
            {
                'status': 'success',
                'results': {
                    'trend': {...},
                    'seasonality': {...},
                    'stationarity': {...},
                    'acf_pacf': {...},
                    'decomposition': {...},
                    'statistics': {...},
                    'change_points': {...} (si activé)
                },
                'visualizations': {
                    'overview': 'path/to/overview.png',
                    'decomposition': 'path/to/decomp.png',
                    ...
                },
                'recommendations': [
                    {'model': 'SARIMA', 'reason': '...', 'priority': 1},
                    {'model': 'Prophet', 'reason': '...', 'priority': 2}
                ]
            }
        ```

---

## ✅ Checklist pour Implémentation

### Analyses :
- [ ] Détection de tendance (régression linéaire)
- [ ] Détection de saisonnalité (autocorrélation)
- [ ] Test ADF (stationnarité)
- [ ] Test KPSS (stationnarité)
- [ ] Calcul ACF
- [ ] Calcul PACF
- [ ] Décomposition STL
- [ ] Statistiques descriptives
- [ ] (Optionnel) Points de rupture

### Visualisations :
- [ ] Série temporelle + tendance
- [ ] ACF plot
- [ ] PACF plot
- [ ] Distribution (histogram + Q-Q)
- [ ] Décomposition (3 subplots)
- [ ] (Optionnel) Rolling stats
- [ ] (Optionnel) Periodogram
- [ ] (Optionnel) Seasonal plot

### UI :
- [ ] Dropdown période saisonnière
- [ ] Expander "Options avancées"
- [ ] Checkboxes analyses optionnelles
- [ ] Slider/Input lags ACF
- [ ] Dropdown type décomposition
- [ ] Checkboxes visualisations extra
- [ ] Bouton "Lancer l'Analyse"

### Workflow :
- [ ] Appel orchestrator.handle_command('start_analysis')
- [ ] Progress spinner avec étapes
- [ ] Message formaté dans chat
- [ ] Stockage résultats dans session
- [ ] Changement d'étape → 'analysis_complete'
- [ ] Bouton "3. Sélection de Modèles" apparaît

---

## 🚀 Prochaines Étapes

**Si vous êtes d'accord avec cette proposition** :

1. Je crée l'`AnalysisAgentWrapper` avec mode simple par défaut
2. J'implémente les analyses de base (tendance, saisonnalité, stationnarité, ACF/PACF)
3. Je génère les visualisations obligatoires
4. J'intègre dans l'UI (dropdown + bouton)
5. Je connecte à l'orchestrateur
6. On teste !

Les **options avancées** peuvent être ajoutées plus tard si nécessaire.

---

## 💡 Ma Recommandation Finale

**Pour le MVP (Étape 5)**, implémentons :

### ✅ INCLURE (Obligatoire) :
- Tendance (simple régression)
- Saisonnalité (autocorrélation + période dominante)
- Tests stationnarité (ADF + KPSS)
- ACF/PACF (graphiques)
- Décomposition STL
- Statistiques descriptives
- 2 visualisations (overview + décomposition)

### 🔵 OPTIONNEL (Post-MVP) :
- Points de rupture
- Rolling statistics
- Periodogram
- Seasonal plot
- Mode avancé complet

### ⚙️ UI Minimale :
- 1 dropdown : Période saisonnière (Auto par défaut)
- 1 bouton : "🚀 Lancer l'Analyse"
- Options avancées dans expander (pour plus tard)

**Ça vous convient ?** 🤔

Si oui, je commence l'implémentation immédiatement ! 🚀

