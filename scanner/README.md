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

Deux mesures, pour rester honnête (ADR 0007). La profondeur se mesure sur un corpus **tiers** que nous n'avons pas conçu : les fixtures de test de picklescan (son propre terrain), vendorées sous `tests/vendor/`.

**Profondeur** — sur le corpus de picklescan (37 cas malveillants, 4 sains), via `scripts/benchmark_neutral.py` :

| Outil | Détectés | Faux positifs |
|---|---|---|
| Nitrogen | 36/37 | 1/4 |
| picklescan 1.0.5 | 34/37 | 0/4 |
| ModelScan 0.8.8 | 25/37 | 0/4 |

Le seul cas raté par les trois outils est un `.pt` tronqué de 8 octets. Notre unique « faux positif » est un `benign_password_protected.zip` : nous refusons de certifier un membre chiffré qu'on ne peut pas lire (`pickle.unanalyzable`, choix *fail-closed*), là où picklescan le laisse passer. C'est un parti pris de prudence, pas un bug.

**Largeur** — formats réellement analysés :

| Format | Nitrogen | picklescan | ModelScan |
|---|---|---|---|
| pickle / zip PyTorch / tar | oui | oui | oui |
| safetensors | oui | non | non |
| TensorFlow SavedModel | non | non | oui |
| Keras / HDF5 | non | non | oui |
| remote code (`.py`, `trust_remote_code`) | non | non | non |

En résumé : sur l'analyse pickle, Nitrogen est au niveau des meilleurs ; sur la largeur des formats, ModelScan couvre davantage (TensorFlow, Keras), hors du périmètre v0.1 de Nitrogen.

**Faux positifs sur de vrais modèles** : `scripts/false_positives.py` télécharge quelques petits modèles Hugging Face (dont un vrai `pytorch_model.bin`) et les scanne — 0 faux positif sur l'échantillon testé. Script manuel (réseau), hors de la suite de tests.

Un tableau sur notre propre corpus (où Nitrogen fait 36/36) reste disponible via `scripts/benchmark.py`, mais il est biaisé en notre faveur et ne sert qu'à suivre les non-régressions.
