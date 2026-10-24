# ADR 0011 — Remote code (`trust_remote_code`) : sévérités et analyse AST

- Statut : accepté
- Date : 2026-10-23
- Décideur : Mike

## Contexte
Un modèle Hugging Face peut embarquer du **code Python exécutable** qui tourne au
chargement, sans aucun pickle piégé : des fichiers `.py` référencés par
`auto_map` (ou `custom_pipelines`) dans `config.json` / `tokenizer_config.json`
/ `preprocessor_config.json` / assimilés. Si l'utilisateur charge le modèle avec
`trust_remote_code=True`, ce code s'exécute. L'analyse pickle des étapes 1–9 ne
voit rien de ce vecteur. L'étape 10 l'inventorie.

Contraintes du projet : **on ne charge jamais** le code. On lit les `.py` avec
`ast.parse` uniquement (jamais `import`, `exec`, `compile(...)`+exec,
`importlib`). On lit les configs comme du **JSON** (données, pas de code). Le
hook anti-désérialisation et `test_no_deserialization.py` sont étendus à ces
interdits.

La roadmap fixe déjà : « présence d'un `.py` référencé par `auto_map` = `risk`
par défaut ». Restaient trois décisions de sévérité, tranchées par Mike.

## Options et décisions

**1. `.py` référencé par `auto_map` / `custom_pipelines` (code local présent).**
→ `risk`. Ce code s'exécute avec `trust_remote_code`. (Fixé par la roadmap.)

**2. `.py` présent mais NON référencé.**
- Options : `risk` (tout code livré est dangereux) / `warning` (présent mais non
  câblé) / `info` (simple métadonnée).
- **Décision : `warning`.** Le chargeur standard n'auto-exécute que le code
  référencé ; un `.py` non câblé n'a pas de chemin d'exécution automatique
  (critère ADR 0008), mais reste du code exécutable livré — angle mort signalé,
  code 0 par défaut. Évite les faux positifs sur les dépôts à `.py` utilitaires.

**3. Référence `auto_map` vers un AUTRE dépôt (`autre/repo--module.Classe`).**
- Options : `risk` / `warning`.
- **Décision : `risk`.** Le code s'exécute avec `trust_remote_code`, vient d'un
  dépôt tiers (vecteur d'attaque connu) et Nitrogen **ne peut pas l'analyser**
  (il n'est pas dans ce dépôt). Non analysable + externe + exécutable = pire cas.

**4. Appel dangereux détecté par l'AST dans un `.py` non référencé.**
- Options : élève `warning`→`risk` / enrichit seulement le détail.
- **Décision : élève `warning`→`risk`.** Concrètement, tout appel sensible
  trouvé par l'AST est un finding `remote_code.dangerous_call` de sévérité
  `risk` ; sa présence rend donc le verdict `risk`, que le `.py` soit référencé
  ou non. L'AST distingue un utilitaire bénin d'un script offensif.

**Appels sensibles classés par l'AST** (denylist, pas allowlist : le code est
trop varié pour une allowlist, et la roadmap les énumère) : import ou usage de
`os` (`system`, `popen`…), `subprocess`, `socket`, `requests`, `urllib`,
`eval`, `exec`, `compile`, `__import__`, `pickle`(`.loads`/`.load`), `base64`
(décodage, souvent suivi d'exécution). Chaque occurrence → `dangerous_call`
(`risk`), message avec le nom et la ligne.

## Conséquences
- **Code** : nouveau module `remote_code.py`, `analyze_remote_code(entries, root)`
  greffé dans le pipeline `_scan` après l'inventaire (comme `apply_policy`). Il
  lit les configs JSON (bornées en taille), extrait `auto_map` / `custom_pipelines`,
  résout les modules locaux (`module.Classe` → `module.py`), détecte les refs
  inter-dépôt (`--`), et analyse chaque `.py` par AST.
- **Finding IDs** (documentés README) :
  - `remote_code.auto_map` (`risk`) — une config câble du code local présent ;
  - `remote_code.external` (`risk`) — une config câble du code d'un autre dépôt ;
  - `remote_code.python_file` — un `.py` présent : `risk` si référencé, sinon
    `warning` ;
  - `remote_code.dangerous_call` (`risk`) — appel sensible trouvé par l'AST.
  - `remote_code.config_unreadable` — config non analysable (surdimensionnée → `risk`).
- **Jamais d'exécution** : `ast.parse` et `json` uniquement ; aucun `import` /
  `exec` / `compile`+exec / `importlib` sur un fichier scanné. Garde-fou :
  `test_no_deserialization.py` étendu.
- **Tests** : corpus (dépôt `auto_map`+`.py` → risk ; ref inter-dépôt → risk ;
  `.py` avec `os.system` → risk ; `.py` sain non référencé → warning ; modèle
  sans `.py` → safe) ; fuzz `hypothesis` sur l'AST (ne plante jamais sur une
  source aléatoire/malformée, une source illisible donne un finding, pas une
  exception).
- **Périmètre** : on n'analyse pas le code d'un autre dépôt (on le signale) ;
  l'analyse AST est volontairement une heuristique (denylist d'appels), pas une
  preuve d'innocuité — le `.py` référencé reste `risk` même si l'AST est propre.

## Amendement (relecture) — résolution fail-closed
La première implémentation était **fail-open** : le verdict reposait sur un
appariement de chaînes exact entre la référence `auto_map` et le chemin du `.py`.
La relecture a trouvé deux contournements critiques (modèle dans un sous-dossier
→ résolution relative à la racine de scan au lieu du dossier de la config ;
`config.json` rembourré au-delà du cap → tronqué puis parse échoué en silence),
plus des amplificateurs (évasions AST par import aliasé / from-import, casse du
nom de fichier, lecture non bornée). Corrections, sans changer les sévérités
décidées ci-dessus :
- **Le verdict ne dépend plus de l'appariement.** Une config qui **déclare** du
  code local (`auto_map`/`custom_pipelines` non `--`) est `risk` en soi, que le
  `.py` soit localisé ou non (`remote_code.auto_map` message « declared, file
  not located » quand l'appariement échoue). Localiser le fichier n'enrichit que
  le message et la sévérité du `remote_code.python_file`.
- **Résolution relative au dossier de la config** (comme transformers), et
  comparaison en casse repliée (`casefold`) pour les FS insensibles ; suffixe
  `.py`/`.PY` comparé en casse repliée.
- **Config surdimensionnée** (> cap) → `remote_code.config_unreadable` (`risk`) :
  valide pour transformers mais pas lisible par nous, on ne peut pas exclure un
  `auto_map`. Un JSON **malformé** n'est pas signalé : transformers échoue
  dessus aussi, donc aucun chemin d'exécution (ADR 0008).
- **Lecture bornée en streaming** (`open().read(cap)` via `entry.size`), jamais
  `read_bytes()[:cap]` : pas d'OOM sur un fichier de plusieurs Go.
- **AST alias-aware** : suit `import os as o`, `from os import system`,
  `import importlib as il`, `import builtins as b`. Restent hors de portée (angle
  mort documenté, non bloquant car l'AST n'est qu'un bonus) : `getattr(os,"sys"+
  "tem")`, construction de chaînes, indirection dynamique.
- **Tests** : le corpus mono-fichier ne modélise pas un dépôt multi-fichiers ;
  les cas remote code sont des tests unitaires/e2e sur dossiers temporaires,
  avec une régression par contournement trouvé (sous-dossier, config
  surdimensionnée, casse `.PY`, évasions AST).
