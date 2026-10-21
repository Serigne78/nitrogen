# nitrogen-scanner

Scanner statique de modèles d'IA : détecte les formats de sérialisation dangereux **sans jamais charger le modèle**, et produit un AI-BOM CycloneDX 1.6.

```
uv sync
uv run nitrogen scan ./model
uv run nitrogen scan ./model -o bom.cdx.json --policy policy.yaml
```

Codes de sortie : `0` conforme, `1` risque détecté, `2` erreur d'exécution.

Une session de démonstration : [`docs/demo.txt`](../docs/demo.txt).

## Ce qui est analysé

| Format | Analyse |
|---|---|
| pickle (brut, protocoles 0 à 5) | opcodes lus avec `pickletools.genops`, allowlist d'imports, memo reconstruit |
| zip PyTorch, zip, tar hérité | chaque membre pickle analysé, sans extraction sur disque |
| safetensors | en-tête, offsets et tailles vérifiés |
| HDF5/Keras, GGUF, npy, ONNX | reconnus, **non analysés** en v0.1 (une policy peut les refuser) |

Principe : **allowlist, pas denylist**. Tout import qui n'est pas explicitement autorisé est signalé. Un fichier peut avoir plusieurs formats, et chacun est analysé : l'attaquant ne choisit pas quelle analyse tourne.

## Comparaison

Sur le corpus de test du projet (`tests/corpus/`, 36 cas malveillants et 7 sains), via `scripts/benchmark.py` :

| Outil | Détectés (sur 36) | Faux positifs (sur 7) |
|---|---|---|
| Nitrogen | 36 | 0 |
| picklescan 1.0.5 | 26 | 0 |
| ModelScan 0.8.8 | 24 | 0 |

Les cas que picklescan et ModelScan manquent sont les contournements que le corpus exerce : noms passés par le memo (`PUT`/`GET`, `LONG_BINPUT`), entiers en base 0, `GLOBAL`/`INST` sans saut de ligne final, opcodes `EXT`, pickle caché dans un membre de zip autre que `data.pkl`, tar hérité, `torch.serialization.load`.

**Ce comparatif est biaisé** : le corpus a été conçu autour de ce que Nitrogen sait détecter. picklescan et ModelScan couvrent par ailleurs des formats hors du périmètre v0.1 de Nitrogen (TensorFlow/Keras, entre autres). Le tableau mesure la robustesse de l'analyse pickle, pas la couverture générale.
