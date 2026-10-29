# PROGRESS — volatility-surface-svi

## Séance 1 — 2026-10-13 — Phase 0 (Audit) : terminée

Fait
- Lecture intégrale du code au commit `3519491` et exécution réelle du code non modifié.
- `AUDIT.md` : 3 critiques, 8 majeurs, 8 mineurs ; plan des phases 1–6 ; 6 décisions à valider.
- `scripts/audit/reproduce_published_run.py` : rejoue le run publié à date et spot figés.
- `scripts/audit/audit_diagnostics.py` : diagnostics sur les snapshots (forward de parité, écarts call/put, couverture, convention de temps, SSVI petite maturité).

Chiffres clés établis
- Run publié = 100 % `lastPrice` (fichier du 4 juin, 98,5 % de bid = ask = 0) ; paramètres SSVI publiés reproduits exactement.
- 161 violations calendaires SVI en rejoue (152 publiées), toutes à k ≥ 0,224 alors que les données d'aile droite s'arrêtent à k ≤ 0,18.
- Contrainte de butterfly SVI jamais active : b(1+|ρ|) ∈ [0,08 ; 0,12] contre une borne codée 4/T ∈ [8 ; 52].
- Snapshot du 3 juin traité en Black-76 sur forward de parité : écart IV call/put médian 11–22 pb près de l'ATM.

## Séances 2–3 — 2026-10-28/29 — Phases 1 à 6 : terminées (travail autonome, sans validation intermédiaire)

Décisions prises par défaut (Tancrède a délégué la suite sans arbitrer D1–D6)
- D1 : calibration sur le snapshot du 3 juin uniquement ; celui du 4 juin est rejeté par construction (`StaleQuotesError`) et ne sert plus qu'à l'audit.
- D2 : actualisation au DTB3 du 2026-06-03 (3,63 % escompte → 3,697 % continu) ; forward par parité.
- D3 : **écart au cahier des charges.** Les CGU Yahoo interdisent la redistribution (§2.8, page « data providers ») : les cotations brutes restent locales (`data/raw/`, ignoré par git) et seules leurs métadonnées + SHA-256 sont versionnées (`data/snapshots/`). L'échantillon versionné `data/sample/` est une chaîne synthétique « SPY-like » générée à partir des ajustements réels. Le README publie les chiffres réels ; les 2 commandes reproduisent tout le pipeline sur l'échantillon.
- D4 : θ_t ancré sur la variance totale ATM de marché ; γ ≤ 1/2 et borne exacte du théorème 4.2 sur η (garantie sans arbitrage à toute maturité, y compris t → 0). Pas d'eSSVI.
- D5 : produit barrière retenu = put down-and-in (le risque vendu dans un reverse convertible) et call down-and-out, strike 100 %, barrières 75–95 %, échéance 2026-08-31 (89 j), monitoring continu (pont brownien), dividendes cash SPY (18 juin, 18 sept.).
- D6 : reconstruction en package `volsurf` ; code v0 retiré de `main`, toujours accessible au commit `3519491` (historique de `main`).

Fait
- Phase 1 : package `src/volsurf/` (data, forward, iv, svi, ssvi, arbitrage, localvol, barrier, pipeline, report, plots), `pyproject.toml` épinglé (Python 3.11–3.13), métadonnées de snapshot, `scripts/fetch_data.py` (refuse une chaîne marché fermé), `scripts/make_sample.py`.
- Phase 2 : 72 tests pytest, couverture 98 % ; recoupement QuantLib des formules fermées barrière (32 combinaisons, écart < 1e-10) ; test de régression sur l'échantillon ; test qui vérifie que les tableaux du README sont exactement ceux générés ; test de régression sur données réelles (sauté si la chaîne brute est absente).
- Phase 3 : ruff (lint + format), mypy strict sur `src/`, GitHub Actions (lint + tests 3.11/3.12/3.13), badge.
- Phase 4 : `scripts/run_results.py` régénère toutes les tables et figures (`results/{real,sample}/`, `figures/{real,sample}/`).
- Phase 5 : README anglais dans l'ordre imposé, tableaux injectés automatiquement depuis `results/real/tables.md`.
- Phase 6 : notebook `notebooks/demo.ipynb` exécuté de bout en bout en 17 s, notes de version `docs/release_notes_v1.0.md`. Tag `v1.0` non publié : le proxy de la session refuse le push de tags (HTTP 403).

Bug trouvé et corrigé pendant la Phase 4 (par exécution, pas par lecture)
- Le SVI contraint (SLSQP seul) finissait parfois dans un point faisable à 3,2 points de vol de RMSE au lieu de 0,2 : écart-type de la vol ATM 0,44 point sous un bruit de 0,1 point (contre 0,012 en SVI brut). Correctif : départ par continuation de pénalité (moindres carrés pénalisés, λ = 10 → 10⁵) avant SLSQP. Après correctif, sur 50 tirages : écart-type ATM 0,012 point, surcoût des contraintes ≤ 0,062 point, 0 échec. Test de non-régression ajouté.

Chiffres clés (snapshot réel du 3 juin, `results/real/`)
- RMSE (points de vol) par échéance : SVI brut 0,06–0,14 ; SVI contraint 0,09–0,21 ; SSVI 0,73–1,42.
- SVI brut : arbitrage papillon sur 3 échéances sur 4 (g min −0,44 à k ≈ 1,2) et croisement calendaire 89→180 j pour k ≥ 0,27 ; aucune violation dans la zone cotée (k ≤ 0,18). SVI contraint et SSVI : aucune violation (deux contrôles indépendants).
- Temps médian d'une surface complète (10 exécutions, 1 cœur) : SSVI 0,05 s, SVI brut 1,8 s, SVI contraint 3,4 s.
- Barrière 90 %, 89 j : put down-and-in 2,35 % du nominal en vol locale contre 1,57 % en BS à la vol du strike (15,5 %), soit 0,78 % de nominal sous-facturé ≈ 3,2 % de coupon annuel surestimé sur un reverse convertible.
- Contrôle du moteur : le Monte-Carlo vol locale reprice les vanilles SSVI à 0,05 point de vol près (≤ 2,3 erreurs-types) ; SSVI est à 1,2 point du marché au strike 90 %.

Reste à faire (hors périmètre ou nécessite Tancrède)
- Épingler le repo sur le profil GitHub (réglage de profil, non faisable par l'API).
- Créer et pousser le tag `v1.0` depuis le poste local (commandes dans le compte rendu de session), puis éventuellement une Release GitHub avec `docs/release_notes_v1.0.md`.
- CI : verte sur le commit `f914e37` (lint, tests 3.11/3.12/3.13).
- Option : eSSVI (ρ dépendant de θ) pour corriger le biais de skew court de SSVI qui borne la précision des prix barrière.
- Option : si des cotations réelles doivent être versionnées, utiliser une source dont la licence autorise la redistribution (ce n'est pas le cas de Yahoo).
