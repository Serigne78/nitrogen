# ADR 0001 — Un seul produit, un monorepo

- Statut : accepté
- Date : 2026-10-10
- Décideur : Mike

## Contexte
Nitrogen couvre trois moments (scan, admission, runtime) et deux publics (utilisateur local, entreprise sur Kubernetes). Faut-il deux produits (statique / runtime) ou un seul ?

## Décision
Un seul produit `nitrogen`, dans un monorepo (`scanner/`, `webhook/`, `runtime/`), avec une seule CLI à sous-commandes. Le scanner reste utilisable seul, sans Kubernetes ni signature.

## Conséquences
- La valeur différenciante (la chaîne scan → attestation → admission → runtime) reste dans un seul produit.
- Une seule marque et une seule communauté à faire vivre.
- L'utilisateur local et l'entreprise utilisent la même commande `nitrogen scan` ; seul l'usage du résultat change.

## Options écartées
- Deux produits séparés : double effort de visibilité, et la chaîne perd sa cohérence.
