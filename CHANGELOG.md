# Changelog

## Non publié (v0.1.1 en cours)

### Étape 8 — sévérités risk / warning / info (ADR 0008)
- Trois niveaux de finding portés par le `Finding` : `risk` (bloquant, code 1), `warning` (affiché, code 0 par défaut, dans l'AI-BOM), `info`.
- Un membre d'archive chiffré est désormais un `warning` (`archive.encrypted`), plus un risque : aucun chargeur ne l'ouvre sans mot de passe. Supprime le faux positif sur un zip chiffré bénin.
- Policy : `fail_on` (ex. `[warning]` rend les warnings bloquants) et `allow` (neutralise un finding précis).
- L'AI-BOM mappe les sévérités vers CycloneDX et porte le verdict `compliant` / `compliant-with-warnings` / `risk`.

## v0.1.0 — 2026-10-21

Première pré-release. Scanner statique autonome ; l'admission Kubernetes (v0.2) et le runtime (v0.3) ne sont pas inclus.

### Scanner
- `nitrogen scan <chemin>` : inventaire d'un modèle (fichier ou dossier), format détecté par magic bytes, SHA-256 par fichier.
- Analyse pickle sans jamais charger le fichier : opcodes lus avec `pickletools.genops`, allowlist d'imports, memo reconstruit (`MEMOIZE`/`PUT`/`BINPUT`/`LONG_BINPUT` et leurs lectures). Couvre les protocoles 0 à 5, `GLOBAL`/`STACK_GLOBAL`/`INST`/`EXT`, les pickles concaténés, et les écarts connus entre `genops` et le vrai chargeur (entier en base 0, `GLOBAL`/`INST` sans saut de ligne final).
- Membres pickle des archives zip, zip PyTorch et tar hérité analysés sans extraction sur disque, décompression bornée.
- Validation safetensors : en-tête, offsets et tailles.
- Reconnaissance (sans analyse) des formats HDF5/Keras, GGUF, npy et ONNX.
- Détection multi-formats : chaque format qu'un chargeur accepterait est détecté et analysé.
- AI-BOM CycloneDX 1.6 avec `-o`, policy `allow_formats` avec `--policy`.
- Codes de sortie : `0` conforme, `1` risque, `2` erreur.

### Limites connues
- Keras, ONNX, GGUF, `trust_remote_code` : reconnus ou non, mais pas analysés.
- La signature (`cosign`), le webhook d'admission et le runtime arrivent dans les versions suivantes.
- La spécification exacte des échappements du digest `nitrogen-manifest-v1` sera figée pour la v0.2 (réimplémentation côté webhook).
