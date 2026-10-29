# AUDIT — volatility-surface-svi (Phase 0)

| | |
|---|---|
| Repo audité | `tancredeg/volatility-surface-svi`, commit `3519491` (« Embed figures in README ») |
| Date | 2026-10-13 |
| Méthode | Lecture intégrale (`iv_surface.py` 2041 l., `ssvi.py` 517 l., `fetch_data.py` 180 l., `README.md`) + exécution réelle du code non modifié sur les deux snapshots |
| Environnement d'exécution | CPython 3.13.16, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, Linux |
| Reproduction de l'audit | Code v0 : tag `v0-published` (commit `3519491`). `git worktree add ../v0 v0-published`, puis `python scripts/audit/reproduce_published_run.py ../v0 data/raw/spy_chains_20260604.csv 2026-06-04 752.18 out/replay_0604` et `python scripts/audit/audit_diagnostics.py data/raw out/diagnostics` (chaînes brutes Yahoo locales, non redistribuées) |

Chaque chiffre de ce document sort de l'une de ces deux commandes (fichiers `out/replay_0604/replay_results.json`, `out/diagnostics/diagnostics.json`) ou d'une exécution citée explicitement.

---

## 1. Verdict

1. **Les résultats publiés reposent à 100 % sur des derniers prix de transaction périmés**, pas sur des cotations. La rejoue du code non modifié sur `spy_chains_20260604.csv` (98,5 % de lignes bid = ask = 0) reproduit exactement les paramètres SSVI du README (ρ = −0,4942, η = 0,8317, γ = 0,5816, θ_t identiques à 5 décimales). Les deux arguments centraux du récit d'entretien en dépendent.
2. **Le compteur « 152 violations calendaires » est un artefact d'extrapolation** : toutes les violations sont à k ≥ 0,224, alors que les données d'aile droite s'arrêtent à k ≤ 0,10–0,18.
3. **La contrainte de butterfly imposée à SVI est inopérante** (borne 4/T mal transposée en variance totale, jamais active). Aucune contrainte effective n'existe côté SVI : le résultat Phase 4 n° 2 (« violations avant/après contraintes ») n'a aujourd'hui pas de « après ».
4. **Le repo n'est pas reproductible** : T et le spot dépendent de la date et du réseau au moment de l'exécution ; relancé aujourd'hui sur ses propres données, il affiche `PIPELINE COMPLETE` avec une seule maturité, T faux, sans erreur. Les dépendances épinglées ne s'installent pas sous Python 3.13.
5. **Ce qui est juste et à conserver** : formule de Durrleman g(k) et dérivées analytiques SVI, formule SSVI et passage SSVI → SVI brut, conditions de butterfly SSVI (Gatheral–Jacquier 2014, th. 4.2), sélection OTM, inversion NR + Brent avec bracketing, échec explicite sur spot manquant.

Conséquence : **corriger les données avant d'écrire le moindre test de régression** (sinon on fige des chiffres faux). La Phase 1 inclut donc les corrections C1–C3.

---

## 2. Diagnostics chiffrés

### 2.1 Snapshots

| Fichier | Lignes | bid = ask = 0 | `underlying_spot` | Dernière transaction (UTC) | Statut |
|---|---|---|---|---|---|
| `spy_chains_20260603.csv` | 1215 | 0,58 % | absent | 2026-06-03 14:35:53 | cotations exploitables, séance en cours |
| `spy_chains_20260604.csv` | 1215 | 98,5 % | 751,765 | 2026-06-03 20:14:55 | marché fermé : uniquement `lastPrice` |

Échéances identiques : 2026-07-02, 2026-07-31, 2026-08-31, 2026-11-30.

### 2.2 Rejoue du run publié (code non modifié, date figée 2026-06-04, spot 752,18 du README)

| T (j) | Points OTM | k min | k max | a | b | ρ | RMSE (vol) | b(1+\|ρ\|) | 4/T | g min (Durrleman) |
|---|---|---|---|---|---|---|---|---|---|---|
| 28 | 139 | −0,344 | 0,101 | −0,0170 | 0,0662 | −0,2615 | 0,0037 | 0,083 | 52,1 | 0,0018 |
| 57 | 204 | −0,347 | 0,104 | −0,0095 | 0,0624 | −0,5157 | 0,0038 | 0,095 | 25,6 | 0,0860 |
| 88 | 201 | −0,341 | 0,124 | 0,0031 | 0,0458 | **−0,9990** | 0,0085 | 0,092 | 16,6 | 0,2058 |
| 179 | 98 | −0,333 | 0,176 | 0,0009 | 0,0740 | −0,6562 | 0,0037 | 0,122 | 8,2 | 0,2742 |

- Source de prix des points OTM : `lastPrice` pour **100 %** des 8 jeux (4 échéances × call/put). Âge maximal de transaction : 28,4 jours (puts 88 j).
- Le nuage aberrant à 88 j, k ≈ −0,20 (vol 30–32 %, visible fig0/fig1/fig5) correspond aux strikes 621, 623, 626, 632, 633, 634, tradés entre le 6 et le 19 mai ; les strikes voisins tradés du 1er au 3 juin donnent 26–28 %.
- « Parité » call/put du repo : 28 j : 69/76 violations, max 2034 pb ; 57 j : 133/145, max 4006 pb ; 88 j : 102/120, max 3871 pb ; 179 j : 16/24, max 431 pb.
- Calendrier SVI (grille du repo |k| ≤ 0,35) : 28→57 j : 91 points, k ∈ [0,224 ; 0,35] ; 57→88 j : 70 points, k ∈ [0,253 ; 0,35] ; 88→179 j : 0. Total 161 (README : 152).
- SSVI : RMSE 0,0158 / 0,0099 / 0,0113 / 0,0053 ; 0 violation calendaire ; g min > 0 sur les 4 tranches.
- Temps (conteneur cloud, 1 exécution, non représentatif d'un benchmark) : SVI 4 tranches × 8 départs 0,65 s ; SSVI joint 12 départs 0,57 s ; pipeline SVI complet (IV + 5 figures) 4,5 s.
- Concordance avec le dépôt : figures identiques à 3 décimales (RMSE affichés fig5) ; écarts en 4e décimale avec le tableau README (0,0039/0,0040/0,0086/0,0037) et 161 vs 152 violations. Cause non tranchable sans l'environnement d'origine (version de scipy/SLSQP ou version antérieure du code).

### 2.3 Snapshot du 3 juin traité correctement (Black-76 sur forward de parité)

| Échéance | T réel (j) | F parité | Dispersion F (p90−p10) | D OLS | r implicite OLS | Écart IV call/put \|k\| ≤ 3 % (médiane / max, pb) | k max OTM coté |
|---|---|---|---|---|---|---|---|
| 2026-07-02 | 29,2 | 757,15 | 0,96 $ | 1,0050 | −6,2 % | 22 / 206 | 0,173 |
| 2026-07-31 | 58,2 | 759,30 | 0,68 $ | 1,0166 | −10,4 % | 21 / 113 | 0,101 |
| 2026-08-31 | 89,2 | 761,76 | 0,57 $ | 1,0223 | −9,0 % | 11 / 108 | 0,121 |
| 2026-11-30 | 180,3 | 767,12 | 1,03 $ | 1,0118 | −2,4 % | 17 / 58 | 0,176 |

- D est fixé à exp(−r T) avec r = 3,64 % (adjudication T-bill 3 mois du 2026-06-15, taux d'escompte) ; F = médiane de K + (C − P)/D sur |K/F − 1| ≤ 3 %.
- Portage implicite entre échéances (indépendant du spot) : 3,57 %, 3,80 %, 2,82 % ; le repo suppose r − q = 4,00 % plat et r = 5,30 %.
- La régression OLS de C − P sur K donne D > 1 sur les 4 échéances : **le facteur d'actualisation n'est pas identifiable par parité européenne sur ces données** (causes plausibles : prime d'exercice anticipé des puts américains ITM, asynchronisme des cotations Yahoo ; non départagées).
- Colonne `impliedVolatility` de Yahoo vs Black-76 recalculé (n = 1188) : médiane −0,52 pt de vol, p5 −2,03, p95 +8,94. Inutilisable comme référence ; le repo ne l'utilise pas pour les calculs (correct).

### 2.4 Autres mesures

- Convention de temps du run publié : T = jours entiers depuis le 2026-06-04 alors que les prix datent du 3 juin à la clôture ; T réel 28,99 / 57,99 / 88,99 / 180,03 j. Biais sur la vol : +1,75 % / +0,86 % / +0,56 % / +0,29 % en relatif (≈ +0,24 pt à 28 j).
- k = 0,10 (RR/BF du repo) exprimé en écarts-types σ√T : 2,62 (28 j), 1,69 (57 j), 1,32 (88 j), 0,86 (179 j).
- SSVI publié (γ = 0,582 > 1/2) : la condition suffisante θφ(θ)²(1+|ρ|) ≤ 4 échoue pour θ < 2,6·10⁻⁴, soit environ 5,7 jours à 12,9 % de vol ATM. Vérification directe : g min reste ≥ +0,215 jusqu'à θ = 10⁻⁵ sur k ∈ [−1,5 ; 1,5]. Pas de violation mesurée, mais garantie théorique absente en deçà de ~6 jours.
- Relance aujourd'hui (2026-10-13) de `_run_pipeline` sur `spy_chains_20260604.csv` : 3 échéances « dans le passé » ignorées, une tranche unique T = 48 j (vraie maturité ≈ 180 j), RMSE 0,0242, ρ = −0,999, calendrier « PASS », `PIPELINE COMPLETE`, figures réécrites.
- `pip download --only-binary=:all: numpy==1.26.4` et `pandas==2.2.2` échouent sous CPython 3.13 (aucune roue disponible).

---

## 3. Problèmes classés par gravité

Gravité : **Critique** = invalide un résultat publié ; **Majeur** = fausse une conclusion ou bloque le standard ; **Mineur** = qualité, robustesse ou cohérence ; références `fichier:lignes` sur le commit audité.

### Critique

**C1 — Prix périmés utilisés comme prix de marché.**
`iv_surface.py:464-483` bascule sur `lastPrice` dès que bid/ask manquent ; `:508-521` dispense ces lignes du filtre de spread ; aucun filtre sur `lastTradeDate`. Le run publié est 100 % `lastPrice`, transactions jusqu'à 28 jours d'âge, asynchrones entre strikes (§2.2).
Impact : toutes les figures, tous les tableaux du README, et les « violations de parité » (jusqu'à 4006 pb) qui servent de justification au filtre OTM.
Correction : exiger une cotation bid/ask > 0 ; rejeter le fichier (erreur explicite) si la part de cotations valides est sous un seuil ; utiliser le snapshot du 3 juin comme échantillon figé. Phase 1.

**C2 — Résultats dépendants de la date et du réseau au moment de l'exécution.**
`iv_surface.py:1760` et `ssvi.py:349` : T calculé depuis `date.today()`. `iv_surface.py:1995-2004` : spot live prioritaire sur le spot du fichier (README 752,18 ≠ fichier 751,765). `.gitignore` exclut `data/*.csv` : aucun recruteur ne peut rejouer. `README.md:38` : `cd iv-surface-svi` (mauvais nom de dossier). Pins incompatibles Python 3.13 (§2.4). Relancé aujourd'hui, le code produit en silence une « surface » d'une tranche à T faux (§2.4), en contradiction avec le principe « échouer bruyamment » du projet.
Correction : date de valorisation, horodatage et forward lus dans l'échantillon figé ; plus aucun `today()` ni appel réseau dans le cœur de calcul ; `pyproject.toml` avec `requires-python` et pins installables 3.11–3.13. Phase 1.

**C3 — Forward, taux et temps mal spécifiés.**
`iv_surface.py:69-70` : r = 5,3 % (taux T-bill de 2023–24 ; 3,64 % en juin 2026), q = 1,3 % continu alors que SPY verse des dividendes discrets (le portage implicite chute à 2,82 % sur l'intervalle août–novembre, cohérent avec un détachement, à confirmer sur le calendrier de distribution). T en jours entiers depuis une date erronée (biais de vol +1,75 % relatif à 28 j). Les effets se compensent partiellement sur r − q (4,00 % supposé, 3,57–3,80 % implicite hors dividende), ce qui masque l'erreur.
Correction : Black-76 sur forward de parité par échéance (le spot, r et q disparaissent du calcul d'IV) ; D depuis un taux public figé avec l'échantillon ; T en ACT/365 de l'horodatage du snapshot à 16:00 New York. Phase 1.

### Majeur

**M1 — Compteur de violations calendaires mesuré dans la zone extrapolée.**
`iv_surface.py:1210-1216` affirme que |k| ≤ 0,35 est « la plage de calibration » : faux côté calls (données jusqu'à k = 0,10–0,18). Les 161 violations (152 publiées) sont toutes à k ≥ 0,224. Le récit « SVI par tranche révèle de l'arbitrage calendaire, SSVI l'élimine » repose sur ce chiffre.
Correction : compter séparément sur la plage commune supportée par les données de chaque paire et sur la zone extrapolée ; publier les deux. Phase 2 (fonction) et 4 (résultat).

**M2 — Contrainte de butterfly SVI inopérante et mal transposée.**
`iv_surface.py:879-902`, `:1043-1048`. Avec w = σ²T (variance totale), la formule des moments de Lee impose une pente d'aile de w au plus égale à 2, soit b(1+|ρ|) ≤ 2 ; la borne 4/T provient d'une paramétrisation en variance σ². Ici 4/T vaut 8 à 52 et b(1+|ρ|) vaut 0,08 à 0,12 : la contrainte n'est jamais active. Le « butterfly = ✓ » de fig1 ne reflète que ce test. g(k) n'est vérifié qu'a posteriori, jamais imposé.
Correction : SVI contraint (g(k) ≥ 0 sur grille dense + non-croisement avec la tranche précédente), version non contrainte conservée pour la comparaison avant/après. Phase 2 et 4.

**M3 — Fonction objectif différente entre SVI et SSVI.**
SVI : moindres carrés non pondérés sur w (`iv_surface.py:1037-1040`) ; SSVI : moindres carrés sur la vol (`ssvi.py:193-200`). Les deux RMSE sont rapportés en vol : la comparaison du résultat Phase 4 n° 1 mélange deux estimateurs.
Correction : même perte pour tous les modèles (vol, ou prix pondéré par la vega). Phase 1.

**M4 — Paramètre en butée.** ρ = −0,999 à 88 j (borne `iv_surface.py:1032`), d'où l'aile droite plate puis décroissante en vol (fig1). Probablement induit par les données périmées ; à surveiller dans le test de stabilité.
Correction : signaler toute butée dans les sorties ; tester la dispersion sous bruit. Phase 2.

**M5 — SSVI rate l'ATM court.** ATM 28 j : SSVI 12,9 % contre SVI 13,8 % (README), RMSE SSVI 1,58 pt de vol à 28 j. Bloquant pour l'extension vol locale → barrière, dont le prix dépend d'abord du niveau ATM.
Correction (décision D4) : θ_t ancré sur la variance totale ATM de marché puis (ρ, η, γ) globaux (approche en deux temps de Gatheral–Jacquier), ou eSSVI à ρ(θ). Phase 1 ou 4.

**M6 — Aucun test.** Le chemin synthétique (`iv_surface.py:1646-1739`) n'est pas un test : aucune assertion, paramètres irréalistes (vol ATM 30–60 %, reconnu dans le README), aucune comparaison paramètres vrais / recalibrés. Phase 2.

**M7 — Figures et métriques présentant de l'extrapolation comme de l'information.**
fig2 et fig6 tracées sur |k| ≤ 0,35 ; à 28 j l'aile droite SVI monte à 30,4 % de vol à k = 0,35 sans aucune donnée au-delà de 0,10 (57 j : 18,9 % ; 88 j : 12,1 % — c'est ce croisement extrapolé qui produit M1). Interpolation linéaire des 5 paramètres SVI en T (`:1289-1315`), non garantie sans arbitrage. RR/BF à k = ±0,10 fixe (`:1505-1519`) : 2,6 écarts-types à 28 j contre 0,86 à 179 j, et k = +0,10 est en bord de données à 28 et 57 j. La conclusion du README « le skew s'aplatit avec la maturité » est en partie mécanique.
Correction : tracer sur la plage supportée (zone extrapolée hachurée ou omise) ; RR/BF 25Δ par inversion delta-strike sur la tranche calibrée. Phase 4.

**M8 — Extension Phase 4 absente.** Ni vol locale de Dupire ni pricing barrière. Avec γ = 0,582 > 1/2, la garantie de butterfly SSVI n'est pas démontrée pour T < ~6 jours (§2.4) alors que Dupire et la barrière exigent la surface de 0 à T : vérification numérique de g(k) et de la positivité de la variance locale obligatoire sur [0, T]. Phase 4.

### Mineur

- **m1** Le facteur d'actualisation n'est pas identifiable sur Yahoo (§2.3) : taux public figé avec l'échantillon (source, date, valeur). Décision D2.
- **m2** `validate_option_chain` exige `impliedVolatility` (`:114-115`), colonne inutilisée et non fiable (§2.3) : couplage inutile.
- **m3** « ATM brut » = un seul point OTM le plus proche de k = 0 (`:1906-1908`), comparé au modèle en k = 0 et non au k de ce point : l'écart affiché fig3 mélange bruit et décalage de moneyness.
- **m4** Effets de bord à l'import : `mkdir` (`:58-59`), `logging.basicConfig` (`:47-51`), état global `DATA_SOURCE` muté ; boucles `iterrows` pour l'IV ; fichier monolithique de 2041 lignes ; `Author: <your name>` (`:12`).
- **m5** README incohérent avec le code corrigé : Q3 (`README.md:368`) cite encore « Roper (2010) sufficient condition » et présente la contrainte de Lee comme imposée ; « arbitrage-free by construction » sans la réserve petite maturité ; marge g min 28 j annoncée +0,005 (rejoue : +0,0018).
- **m6** Plancher de prix absolu 0,05 $ et spread relatif 20 % sans justification ; à 29 j, 10,6 % des cotations OTM ont un spread > 20 % (§2.3) : seuils à documenter et à tester en sensibilité.
- **m7** Redistribution de données Yahoo dans un dépôt public : conditions d'utilisation restrictives. Pas un avis juridique ; option prudente : ne publier que les colonnes nécessaires, avec source et date, et le script de collecte. Décision D3.
- **m8** Noms de fichiers : le code écrit `spy_chains_YYYY-MM-DD.csv`, les fichiers fournis sont `spy_chains_YYYYMMDD.csv` ; le chargement « fichier le plus récent » par tri lexical est fragile.

### Vérifié correct (à conserver tel quel)

- g(k) de Durrleman et dérivées analytiques SVI (`:934-979`, `:905-931`).
- SSVI et correspondance SVI brut (`ssvi.py:77-131`), conformes à Gatheral–Jacquier (2014).
- Conditions de butterfly SSVI imposées sur grille de θ (`ssvi.py:208-219`).
- Calendrier SSVI : avec φ en loi puissance, ∂_θ(θφ)/φ = (1−γ)/(1+θ) ∈ (0, 1), toujours inférieur à (1+√(1−ρ²))/ρ² ; avec θ_t croissant (contrainte présente), la condition du théorème 4.1 est satisfaite pour tout γ ∈ [0, 1). À couvrir par un test numérique en Phase 2.

---

## 4. Décisions à valider avant la Phase 1

| # | Décision | Recommandation | Conséquence |
|---|---|---|---|
| D1 | Échantillon figé | `spy_chains_20260603.csv` seul en échantillon de calibration ; `20260604` conservé uniquement comme cas de test « fichier sans cotations → erreur » | Tous les chiffres publiés changent ; le récit d'entretien sera réécrit sur les nouveaux chiffres |
| D2 | Actualisation | Taux public figé avec l'échantillon (T-bill 3 mois, source et date dans `data/sample/README.md`) ; forward par parité | Spot, r, q sortent du calcul d'IV |
| D3 | Données publiques | Colonnes minimales (échéance, type, strike, bid, ask, horodatage), source citée | Risque de conditions d'utilisation réduit, pas éliminé |
| D4 | SSVI court terme | θ_t ancré sur l'ATM de marché ; eSSVI seulement si l'erreur ATM reste > 0,25 pt | Prérequis de l'extension barrière |
| D5 | Produit barrière | Down-and-out call, 3 mois, strike ATM forward, barrière 90 % du spot, monitoring continu ; Monte-Carlo vol locale (seed fixée, correction de pont brownien) vs BS à vol plate = vol implicite au strike | Le résultat « structuring » est l'écart de prix en % et en équivalent vol |
| D6 | Code | Reconstruction en package autour des fonctions vérifiées correctes, pas de patch du script monolithique | Phase 1 plus longue, Phases 2–3 plus simples |

---

## 5. Plan des phases 1–6 adapté à ce repo

Estimation : 9 à 11 séances de 2 h, soit 3 à 4 semaines au rythme de 6–8 h/semaine. Hypothèse, à recaler après la Phase 1.

### Phase 1 — Structure et données (2 séances)
- `src/volsurf/` : `data.py` (chargement de l'échantillon, filtres documentés, rejet sans cotation), `forward.py` (parité, actualisation), `iv.py` (Black-76 vectorisé, Brent), `svi.py` (SVI brut, dérivées, g, calibration avec et sans contraintes), `ssvi.py`, `arbitrage.py` (butterfly sur grille, calendrier sur plage supportée et zone extrapolée), `plots.py` ; `localvol.py` et `barrier.py` créés en Phase 4.
- `data/sample/` : snapshot figé + `README.md` (source, horodatage, taux retenu, schéma, filtres et compte de lignes à chaque filtre).
- Corrections C1, C2, C3, M3 ; `fetch_data.py` → `scripts/fetch_data.py` (dépendance optionnelle `yfinance`, hors cœur).
- `pyproject.toml` (`requires-python >= 3.11`, pins installables 3.11–3.13), `pip install -e .[dev]`.

### Phase 2 — Tests pytest (2 séances)
- Inversion : prix → IV → prix (calls, puts, ailes), identité de parité Black-76.
- Forward : chaîne synthétique de forward connu, forward retrouvé à 1e-6 près.
- Récupération SVI : surface SVI réaliste connue (paramètres de l'ordre de ceux calibrés), recalibration ; sans bruit : erreur sur w < 1e-8 ; avec bruit 0,2 pt de vol : RMSE ≤ 1,5 × bruit.
- Butterfly : tranche connue sans arbitrage → g ≥ 0 ; contre-exemple de Vogt repris par Gatheral–Jacquier (2014, ex. 3.1) → g < 0 détecté (vérifié avec la fonction du repo : g min = −0,033 en k ≈ 0,88) ; recoupement par densité ∂²C/∂K² ≥ 0 en différences finies sur prix Black-76.
- Calendrier : SSVI → 0 violation ; deux tranches croisées → violation détectée, localisée dans la bonne zone.
- Stabilité : 200 recalibrations bruitées (seed fixée), dispersion des paramètres, fréquence des butées.
- Régression sur `data/sample/` (RMSE, θ_t, compteurs de violations, tolérances explicites) ; couverture ≥ 80 % sur `src/volsurf`.

### Phase 3 — Qualité et CI (1 séance)
- ruff (lint + format), mypy sur `src/` ; GitHub Actions matrice Python 3.11–3.13 ; badge.

### Phase 4 — Résultats chiffrés (3 séances)
`scripts/run_results.py` régénère toutes les tables et figures du README.
1. RMSE en points de vol par maturité : SVI non contraint, SVI contraint, SSVI — même perte, même données.
2. Violations d'arbitrage avant/après contraintes : butterfly (points où g < 0) et calendrier, sur plage supportée et en zone extrapolée, pour les trois modèles.
3. Temps de calibration : médiane sur 20 exécutions, par tranche et joint, machine déclarée.
4. Extension : vol locale de Dupire tirée de SSVI (formule en variance totale, dérivées analytiques en k et en t), contrôle de positivité sur [0, T] ; prix du produit D5 en vol locale vs BS à vol plate ; écart en % du prix et en équivalent vol, avec erreur standard Monte-Carlo.
Chaque résultat interprété en une phrase.

### Phase 5 — README (anglais, ordre imposé) (1 séance)
Une phrase ; résultats (tableau + 1–2 figures) ; méthodologie et limites assumées (données différées Yahoo, options américaines traitées en européennes hors OTM, un seul snapshot, taux plat) ; reproduire en 2 commandes ; structure ; références.

### Phase 6 — Livraison (1 séance)
Notebook de démonstration < 2 min ; tag v1.0 ; épinglage ; post LinkedIn 5 lignes.

---

## 6. Hypothèses et incertitudes

- Horodatage du snapshot du 3 juin pris égal à la dernière transaction du fichier ; cotations Yahoo différées, erreur sur T inférieure à 0,1 % à 29 j.
- Taux d'actualisation 3,64 % : adjudication du 2026-06-15 (12 jours après le snapshot), taux d'escompte non converti ; effet sur l'IV via D seulement, du second ordre.
- Le détachement de dividende entre août et novembre est inféré du portage implicite, non vérifié sur le calendrier de distribution.
- Les écarts résiduels call/put (médiane 11–22 pb) incluent la prime d'exercice anticipé des options américaines ITM ; non corrigés ici.
- L'écart 161 vs 152 violations et les écarts en 4e décimale de RMSE avec le README ne sont pas attribués avec certitude.
