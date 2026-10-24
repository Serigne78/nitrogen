# nitrogen-scanner

Scanner statique de modèles d'IA : détecte les formats de sérialisation dangereux **sans jamais charger le modèle**, et produit un AI-BOM CycloneDX 1.6.

```
uv sync
uv run nitrogen scan ./model
uv run nitrogen scan ./model -o bom.cdx.json --policy policy.yaml
```

Codes de sortie : `0` conforme, `1` risque détecté, `2` erreur d'exécution.

Trois niveaux de finding (ADR 0008) : `risk` (exécution possible ou structure trompeuse, bloquant), `warning` (angle mort sans chemin d'exécution prouvé, ex. membre d'archive chiffré — affiché, non bloquant par défaut), `info` (métadonnée). Une policy peut durcir (`fail_on: [warning]`) ou neutraliser un finding précis (`allow: [archive.encrypted]`).

Une session de démonstration : [`docs/demo.txt`](../docs/demo.txt).

## Ce qui est analysé

| Format | Analyse |
|---|---|
| pickle (brut, protocoles 0 à 5) | opcodes lus avec `pickletools.genops`, allowlist d'imports, memo reconstruit |
| zip PyTorch, zip, tar hérité | chaque membre pickle analysé, sans extraction sur disque ; les membres de stockage de tenseurs (`data/<clé>`) ne sont pas lus comme du pickle (ADR 0010) |
| safetensors | en-tête, offsets et tailles vérifiés |
| remote code (`trust_remote_code`) | `auto_map`/`custom_pipelines` des configs + fichiers `.py`, lus avec `ast.parse` (jamais importés), appels sensibles signalés (ADR 0011) |
| HDF5/Keras, GGUF, npy, ONNX | reconnus, **non analysés** en v0.1 (une policy peut les refuser) |

Principe : **allowlist, pas denylist**. Tout import qui n'est pas explicitement autorisé est signalé. Un fichier peut avoir plusieurs formats, et chacun est analysé : l'attaquant ne choisit pas quelle analyse tourne.

Identifiants de findings : `pickle.import`, `pickle.malformed`, `pickle.unanalyzable`, `archive.encrypted`, `policy.format-not-allowed`, et pour le remote code `remote_code.auto_map`, `remote_code.external`, `remote_code.python_file`, `remote_code.dangerous_call`, `remote_code.config_unreadable`.

## Comparaison

Deux mesures, pour rester honnête (ADR 0007). La profondeur se mesure sur un corpus **tiers** que nous n'avons pas conçu : les fixtures de test de picklescan (son propre terrain), vendorées sous `tests/vendor/`.

**Profondeur** — sur le corpus de picklescan (37 cas malveillants, 4 sains), via `scripts/benchmark_neutral.py` :

| Outil | Détectés | Faux positifs |
|---|---|---|
| Nitrogen | 34/37 | 0/4 |
| picklescan 1.0.5 | 34/37 | 0/4 |
| ModelScan 0.8.8 | 25/37 | 0/4 |

À égalité avec picklescan sur son propre corpus, 0 faux positif. Les cas que Nitrogen ne compte pas comme « détectés » sont deux membres de zip **chiffrés** (signalés en `warning`, pas en `risk` : aucun chargeur ne les ouvre sans mot de passe, donc pas de chemin d'exécution — code 0 par défaut, `fail_on: [warning]` pour bloquer) et un `.pt` tronqué de 8 octets que les trois outils manquent.

**Largeur** — formats réellement analysés :

| Format | Nitrogen | picklescan | ModelScan |
|---|---|---|---|
| pickle / zip PyTorch / tar | oui | oui | oui |
| safetensors | oui | non | non |
| TensorFlow SavedModel | non | non | oui |
| Keras / HDF5 | non | non | oui |
| remote code (`.py`, `trust_remote_code`) | non | non | non |

En résumé : sur l'analyse pickle, Nitrogen est au niveau des meilleurs ; sur la largeur des formats, ModelScan couvre davantage (TensorFlow, Keras), hors du périmètre v0.1 de Nitrogen.

**Faux positifs sur de vrais modèles** : `scripts/benchmark_false_positives.py` scanne une liste figée de modèles populaires du Hub, chacun épinglé à une révision de commit (`scripts/models.txt`). Sur les 18 modèles mesurés (bert, gpt2, t5, clip, mpnet…), **0 faux positif**. Ce banc a révélé que l'analyse de chaque membre d'archive prenait le **stockage brut des tenseurs** pour un pickle ou une archive : **61 % de ces modèles étaient faussement bloqués** avant les correctifs (ADR 0009 et 0010), qui n'analysent plus que ce qu'un chargeur dépickle réellement. Détail dans [`docs/benchmark-faux-positifs.md`](../docs/benchmark-faux-positifs.md). Script opt-in (réseau), hors de la suite de tests.

Un tableau sur notre propre corpus (où Nitrogen fait 36/36) reste disponible via `scripts/benchmark.py`, mais il est biaisé en notre faveur et ne sert qu'à suivre les non-régressions.
