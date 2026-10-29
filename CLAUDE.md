# CLAUDE.md — volatility-surface-svi

Instructions permanentes pour Claude Code dans ce repo. À relire en début de séance avec `AUDIT.md` et `PROGRESS.md`.

## Objectif
Un recruteur quant/trading comprend en 90 s ce que fait le repo et ce qu'il prouve, et relance tout en 2 commandes.
Sujet : surface de volatilité implicite SPY, SVI par tranche (Gatheral 2004) vs SSVI (Gatheral & Jacquier 2014), contrôle automatique de l'arbitrage, extension vol locale de Dupire → option barrière vs BS à vol plate.

## Méthode de travail (standard « repo recruteur »)
- Une phase par séance. Ordre : 0 Audit → 1 Structure → 2 Tests → 3 Qualité/CI → 4 Résultats → 5 README → 6 Livraison.
- Avant chaque séance : lire `AUDIT.md` et `PROGRESS.md`. Après : mettre à jour `PROGRESS.md` (fait, chiffres obtenus, reste à faire, décisions).
- Plan validé par Tancrède avant toute refonte importante. Commits atomiques, messages explicites.
- Ne jamais inventer un résultat : chaque chiffre du README sort d'un script exécuté (`scripts/run_results.py`).
- Toute affirmation de correction est appuyée par une exécution réelle, jamais par lecture seule du code.
- Livrer des fichiers complets (pas de diffs partiels). Blocs de commandes : uniquement des commandes exécutables telles quelles sous PowerShell (Windows, VS Code), sans `<placeholder>`.
- Données publiques uniquement. Aucune donnée, méthode ou information liée au stage en cours.
- Signaler hypothèses et incertitudes. Réponses denses, sans remplissage.

## Invariants de calcul (issus de l'audit, ne pas régresser)
1. **Aucun `date.today()` ni appel réseau dans le cœur de calcul.** Date de valorisation et horodatage lus dans les métadonnées de l'échantillon figé.
2. **Prix = mid de cotations bid/ask > 0.** Jamais `lastPrice` comme prix de marché. Un fichier sans cotations exploitables lève une erreur explicite.
3. **Forward par parité call/put par échéance**, IV par Black-76 sur ce forward. Facteur d'actualisation depuis un taux public figé avec l'échantillon (non identifiable par parité sur Yahoo : voir AUDIT §2.3).
4. **T en ACT/365** de l'horodatage du snapshot à l'expiration 16:00 America/New_York.
5. **Smile de calibration OTM uniquement** (calls K > F, puts K ≤ F).
6. **Même fonction de perte** pour tous les modèles comparés (vol ou prix pondéré vega), RMSE rapporté en points de vol.
7. **Butterfly** : g(k) de Durrleman ≥ 0 (Gatheral–Jacquier 2014, section 2). Borne de Lee en variance totale : b(1+|ρ|) ≤ 2 (et non 4/T).
8. **Calendrier** : w(k, T) croissant en T, compté séparément sur la plage supportée par les données de chaque paire et en zone extrapolée.
9. **SSVI** : conditions du th. 4.2 (butterfly) et 4.1 (calendrier) de Gatheral–Jacquier ; avec γ > 1/2 la garantie de butterfly ne couvre pas θ → 0 : vérifier g(k) numériquement sur [0, T] avant Dupire.
10. Seeds fixées partout ; aucun chemin absolu ; fonctions pures, type hints, docstrings avec formule et référence.

## Commandes
- Installation : `pip install -e .[dev,crosscheck]`
- Tests : `pytest`
- Lint/format/types : `ruff check .` ; `ruff format .` ; `mypy src`
- Résultats sur l'échantillon synthétique : `python scripts/run_results.py` ; résultats du README (chaîne brute requise dans `data/raw/`) : `python scripts/run_results.py --snapshot data/snapshots/spy_20260603.json --update-readme`
- Rejoue du run publié v0 (audit) : `git worktree add ../v0 v0-published` puis `python scripts/audit/reproduce_published_run.py ../v0 data/raw/spy_chains_20260604.csv 2026-06-04 752.18 out/replay_0604`
- Diagnostics d'audit : `python scripts/audit/audit_diagnostics.py data/raw out/diagnostics`

## Arborescence cible (Phase 1)
```
src/volsurf/      data.py forward.py iv.py svi.py ssvi.py arbitrage.py localvol.py barrier.py pipeline.py report.py plots.py
tests/            un fichier par module + test_pipeline.py (régression échantillon) + test_readme_and_real.py
data/snapshots/   métadonnées du snapshot réel (horodatage, taux, dividendes, SHA-256) ; data/raw/ local, non versionné (CGU Yahoo)
data/sample/      chaîne synthétique SPY-like versionnée (scripts/make_sample.py) ; data/README.md : source, schéma, filtres
scripts/          run_results.py, fetch_data.py (optionnel, yfinance), audit/
notebooks/        demo.ipynb (< 2 min)
docs/             notes de méthode
```

## Références
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parametrization.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance 14(1).
- Lee, R. (2004). The moment formula for implied volatility at extreme strikes. Mathematical Finance 14(3).
- Dupire, B. (1994). Pricing with a smile. Risk 7(1).
