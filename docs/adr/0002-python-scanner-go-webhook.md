# ADR 0002 — Python pour le scanner, Go pour le webhook

- Statut : accepté
- Date : 2026-10-10
- Décideur : Mike

## Contexte
Le scanner analyse des formats de modèles. Le webhook tourne dans le chemin de déploiement Kubernetes. Faut-il un seul langage ?

## Décision
- Scanner en Python : `pickletools` (stdlib), écosystème ML (`safetensors`, `onnx`, `h5py`), langage maîtrisé.
- Webhook en Go (v0.2) : écosystème Kubernetes natif (`client-go`, `controller-runtime`), `sigstore-go`, binaire statique léger.
- Le webhook ne parse jamais de modèle : il vérifie seulement une attestation signée liée au digest.

## Conséquences
- Le code d'analyse n'existe qu'à un seul endroit (Python, en CI ou en local).
- Deux langages dans le repo, mais sans duplication de logique.
- Si un scan dans le cluster devient nécessaire, il tournera dans un Job Kubernetes avec le scanner Python.

## Options écartées
- Tout en Go : ralentit la v0.1 sur la partie la plus difficile (pickle).
- Tout en Python : webhook moins adapté (latence, distribution, écosystème Kubernetes).
