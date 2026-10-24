# ADR 0012 — Points d'entrée de code par leur nom (hubconf, handler, modules.json, model_index.json)

- Statut : accepté
- Date : 2026-10-24
- Décideur : Mike
- Prolonge ADR 0011 (remote code). La policy d'autorisation par hash est renvoyée à une étape dédiée.

## Contexte
L'étape 10 (ADR 0011) détecte le code exécutable câblé par `auto_map` /
`custom_pipelines`. Mais plusieurs écosystèmes exécutent du code **par le nom du
fichier**, sans aucune déclaration dans une config :

- `hubconf.py` — exécuté par `torch.hub.load` ;
- `handler.py` — exécuté par les Inference Endpoints de Hugging Face (handler
  personnalisé) ;
- `modules.json` (sentence-transformers) — peut référencer des modules **custom**
  (hors `sentence_transformers.*`) ;
- `model_index.json` (diffusers) — `custom_pipeline` et composants pointant vers
  des modules custom.

Vérifié sur le code de l'étape 10 : les quatre ressortent en `warning` (donc
COMPLIANT, code 0). Un `hubconf.py` piégé et **obfusqué** (aucun appel dangereux
lisible par l'AST) passe inaperçu. Les deux `.json` ne sont même pas parsés
(`_is_config` ne les reconnaît pas). C'est un faux négatif réel.

## Décisions
1. **Points d'entrée `.py` par leur nom = `risk`, à n'importe quelle profondeur.**
   `hubconf.py` et `handler.py` sont des points d'entrée exécutés au chargement,
   indépendamment d'`auto_map` et de l'AST. Fail-closed sur la profondeur
   (décision Mike) : on ne sait pas quel dossier l'utilisateur passera à
   `torch.hub.load` / l'endpoint, donc tout fichier portant ce nom est `risk`.
   Finding `remote_code.entry_point`.
2. **`modules.json` et `model_index.json` sont analysés comme des configs qui
   câblent du code.** Une référence vers un module **custom** (racine hors des
   frameworks connus : `sentence_transformers`, `transformers`, `diffusers`,
   `torch`, `timm`, `huggingface_hub`) = `risk` ; `custom_pipeline` de diffusers
   = `risk`. Comme pour `auto_map`, le verdict ne dépend pas de la localisation
   exacte du `.py` (fail-closed) ; la localisation n'enrichit que le message et
   fait passer le `.py` correspondant en `risk`.
3. **UX : le message dit explicitement quoi faire.** Chaque finding remote code
   porte « exécuté au chargement — relisez ce fichier avant de charger », et la
   CLI ajoute, quand un tel `risk` existe, une note listant les fichiers de code
   à relire. Un modèle légitime à `trust_remote_code` sort bien en `risk`
   (voulu), mais l'utilisateur comprend pourquoi et quoi relire.
4. **Reporté à une étape dédiée (+ ADR) :** autoriser un dépôt précis après
   relecture, lié au **hash** des fichiers relus (l'autorisation tombe si le code
   change). C'est une vraie fonctionnalité de policy (nouveau schéma, résolution
   par fichier) ; la mélanger ici alourdirait le commit et le sujet.

## Conséquences
- **Code (`remote_code.py`)** :
  - `_ENTRY_POINTS = {"hubconf.py", "handler.py"}` → `remote_code.entry_point`
    (`risk`) sur tout `.py` portant ce nom, quelle que soit sa profondeur.
  - La boucle configs traite aussi `modules.json` (champs `type`) et
    `model_index.json` (`custom_pipeline` + composants `[module, classe]`), en
    ne signalant que les modules **hors frameworks** → `remote_code.auto_map`
    (`risk`).
  - Messages remote code reformulés (« relisez ce fichier avant de charger »).
- **Code (`cli.py`)** : quand un finding `remote_code.*` bloquant existe, une
  note liste les fichiers de code à relire.
- **Finding IDs** : ajout de `remote_code.entry_point` (`risk`).
- **Tests d'abord** : `hubconf.py` / `handler.py` (même obfusqués, même en
  sous-dossier) → `risk` ; `modules.json` / `model_index.json` avec module custom
  → `risk` ; un `modules.json` 100 % `sentence_transformers.*` → pas de faux
  positif ; la note CLI apparaît ; fuzz inchangé.
- **Non négociable inchangé** : toujours `ast.parse` / `json`, jamais
  d'exécution.

## Amendement (relecture) — racines sûres par écosystème
La décision #2 prévoyait une **liste unique** de frameworks « sûrs » appliquée à
`modules.json` comme à `model_index.json`. La relecture a trouvé que c'est un
**faux négatif critique** pour sentence-transformers : son chargeur ne traite
comme builtin que les `type` préfixés `sentence_transformers.` ; **tout autre**
`type` (y compris `torch.X`, `transformers.X`, `numpy.X`) est chargé **depuis le
dépôt** sous `trust_remote_code` (`get_class_from_dynamic_module`). Un
`modules.json` avec `type: "torch.Pooling"` + un `torch.py` piégé à la racine
sortait donc COMPLIANT alors que le code s'exécute.

Correction (les sévérités de #2 ne changent pas, seule la notion de « sûr »
devient **par écosystème**) :
- `modules.json` (sentence-transformers) : sûr **uniquement** si la racine est
  `sentence_transformers` ; tout le reste = `risk`.
- `model_index.json` (diffusers) : un composant `[library, Classe]` est importé
  depuis la **librairie installée** (`importlib.import_module`), donc les racines
  `transformers`, `diffusers`, `torch`, `timm`, `huggingface_hub` restent sûres.
- `auto_map` / `custom_pipelines` : inchangé, tout est custom.
- `numpy` (ajouté au code hors ADR) est **retiré** : non décidé et il élargissait
  l'angle mort.
- Régression : `modules.json` avec `type` à racine framework (`torch`,
  `transformers`, `numpy`, `timm`) + `.py` local → `risk` ; composant diffusers
  à racine framework → reste sûr.
