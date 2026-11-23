# Fraud Monitor

Un détecteur de fraude en temps réel que l'on attaque avec son propre bot
fraudeur adaptatif, puis que l'on durcit par ré-entraînement. Les règles
constituent la baseline ; le cœur du travail, c'est le ML, le seuil d'alerte
optimisé par les coûts, et la course à l'armement attaquant/défenseur.

**Résultats visés :** le seuil d'alerte qui minimise le coût attendu (avec
intervalle de confiance), et la preuve que le ré-entraînement bat un attaquant
adaptatif.

---

## État d'avancement

| Phase | Contenu | État |
|------|---------|------|
| 1 | Données + baseline de règles | ✅ terminé |
| 2 | Détecteur ML (LogReg, XGBoost, Isolation Forest) | ⬜ à suivre |
| 3 | Couche quant (seuil minimisant le coût, IC bootstrap, SHAP) | ⬜ |
| 4 | Course à l'armement adverse (attaquant vs. ré-entraînement) | ⬜ |
| 5 | Mise en production (flux, dashboard, write-up) | ⬜ |

## Résultat Phase 1 — la baseline que tout modèle devra battre

Données synthétiques : ~262k transactions, **0,18 % de fraude** (comparable au
0,172 % du jeu Kaggle), découpées **par le temps** (jamais aléatoirement) en
75 % entraînement / 25 % test.

Moteur de règles sur le jeu de test tenu à l'écart et ordonné dans le temps :

| règle | précision | rappel |
|-------|-----------|--------|
| vélocité | 1,00 | 0,23 |
| voyage impossible | 1,00 | 0,01 |
| pic de montant | 0,03 | 0,01 |
| **alerte combinée** | **0,64** | **0,24** |

![Baseline Phase 1](reports/phase1_rule_baseline.png)

**Lecture :** les règles attrapent ~un quart des fraudes avec 64 % de précision.
Elles sont fortes sur les rafales de vélocité et précises sur le voyage
impossible, mais **ratent presque entièrement le card-testing (11 % de rappel)**
— les petites transactions de test en ligne ressemblent à du trafic normal pour
une règle. C'est précisément ce trou que le détecteur ML devra combler en Phase 2.

> Mise en garde sur les petits échantillons : `voyage impossible` (2/4) et
> `pic de montant` (1/3) ont peu de positifs dans la fenêtre de test car chaque
> épisode ne fait qu'1–2 lignes. Augmenter leur part dans `GenConfig.pattern_mix`
> pour des chiffres par pattern plus stables.

## Décisions de conception

- **Les données synthétiques sont la colonne vertébrale**, Kaggle est un benchmark
  séparé de la Phase 2. Les features de Kaggle sont anonymisées par PCA
  (`V1..V28`), elles ne peuvent donc pas supporter la vélocité par carte, le
  voyage impossible géographique, ni un attaquant qui manipule de vrais leviers.
  Notre générateur expose exactement ces leviers (`src/schema.py` est le contrat
  partagé).
- **Aucune fuite de données, garantie par un test.** Toutes les règles (et plus
  tard, toutes les features) n'utilisent que les transactions d'instant `< t`.
  `test_rules_are_causal` prouve qu'une décision ne change pas quand on masque le
  futur.
- **Pas de métrique d'accuracy.** À 0,18 % de fraude, « 99,8 % correct » ne veut
  rien dire. On rapporte précision / rappel / F1 (et la PR-AUC dès la Phase 2).

## Arborescence

```
src/
  schema.py        # le contrat partagé (colonnes, catégories, patterns de fraude)
  generator.py     # transactions synthétiques + 4 patterns de fraude injectables
  rules.py         # vélocité / voyage-impossible / pic-de-montant (toutes causales)
  split.py         # découpage train/test ordonné dans le temps
  metrics.py       # précision/rappel/F1, confusion, rappel par pattern
  run_baseline.py  # point d'entrée Phase 1 -> affiche les chiffres de la baseline
  make_chart.py    # génère reports/phase1_rule_baseline.png
  test_baseline.py # schéma, ordre temporel, et la garantie d'absence de fuite
notebooks/
data/              # artefacts générés (ignoré par git)
reports/           # graphiques de résultats (un par phase)
```

## Lancer le projet

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_baseline.py     # affiche la baseline Phase 1
python src/test_baseline.py    # lance les tests (dont la garantie anti-fuite)
python src/make_chart.py       # régénère le graphique de résultats
# ou : make run
```
