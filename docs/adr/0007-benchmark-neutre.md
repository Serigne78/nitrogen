# ADR 0007 — Benchmark neutre et honnête

- Statut : accepté
- Date : 2026-10-22
- Décideur : Mike

## Contexte
Le benchmark de la v0.1 (`scripts/benchmark.py`) tourne sur **notre** corpus, conçu autour de ce que Nitrogen sait détecter. Le 36/36 face à picklescan (26) et ModelScan (24) mesure la profondeur de notre analyse pickle, pas notre valeur générale : il est biaisé en notre faveur et ne dit rien de ce que **nous** ratons. Un chiffre défendable face à un expert doit mesurer deux axes séparés, sur un terrain que nous n'avons pas choisi.

## Décision
Ajouter un benchmark neutre, à deux axes :

1. **Profondeur (pickle)** : sur un corpus tiers, combien chaque outil détecte.
2. **Largeur (formats)** : quels formats chaque outil couvre réellement (pickle, zip PyTorch, tar, safetensors, Keras/HDF5, TensorFlow, ONNX, GGUF, remote code).

**Sources du corpus tiers** : vendorer dans `scanner/tests/vendor/` un sous-ensemble des suites de tests publiques de picklescan (MIT) et ModelScan (Apache-2.0). Échantillons déjà publics, inoffensifs, licences compatibles, reproductibles, sans réseau. Leur licence et leur provenance sont conservées dans `scanner/tests/vendor/NOTICE.md`. Ces fichiers ne sont **jamais chargés** (règle 1), seulement scannés.

**Faux positifs sur de vrais modèles** : un script manuel `scripts/false_positives.py` télécharge quelques petits modèles Hugging Face populaires et les scanne. Le réseau est autorisé car ce script est **hors de la suite pytest** (qui reste sans réseau). Les modèles ne sont pas versionnés ; seuls les chiffres vont dans le README.

## Conséquences
- Un nouveau script `scripts/benchmark_neutral.py` produit le tableau à deux axes, en traitant la vérité terrain du corpus tiers (ce que chaque outil *devrait* détecter).
- Le README remplace le tableau actuel (biaisé, signalé comme tel) par les deux tableaux neutres, en gardant la mention explicite des formats hors périmètre v0.1.
- Risque assumé : les échantillons tiers « malveillants » portent des payloads d'autres auteurs. Ils sont scannés statiquement, jamais exécutés ; un test vérifie qu'aucun ne contient autre chose que des commandes inoffensives connues, et les fichiers restent confinés à `tests/vendor/`.
- Attribution des licences tierces : `NOTICE.md` + mention dans le README.
- Ce benchmark peut montrer Nitrogen **en retard** sur la largeur (Keras/TF/ONNX). C'est le but : un chiffre honnête, pas flatteur.
