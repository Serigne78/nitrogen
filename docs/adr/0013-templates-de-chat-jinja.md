# ADR 0013 — Analyse des templates de chat Jinja

- Statut : accepté
- Date : 2026-10-24
- Décideur : Mike

## Contexte
Les modèles de chat embarquent un **template Jinja2** (dans `tokenizer_config.json`,
`chat_template.jinja`, `chat_template.json`) qui met en forme la conversation à
l'inférence. Ce template est **rendu**, donc exécuté. Jinja a un historique de
SSTI (injection de template) : un template piégé peut s'évader du bac à sable et
exécuter du code — c'est la **CVE-2024-34359** (`llama-cpp-python` rendait le
template chat avec un `Environment` non sandboxé). Ni l'analyse pickle ni le
remote code ne couvrent ce vecteur. L'étape 11 l'analyse.

Contrainte non négociable : **on ne rend jamais** le template. Il faut un AST
pour détecter les motifs d'évasion sans l'exécuter. Deux briques possibles : le
parser de jinja2, ou une détection maison (regex / mini-parser).

## Options
1. **jinja2 `Environment().parse()`** — ajouter jinja2 en dépendance runtime et
   utiliser son parser pour obtenir l'AST, sans jamais appeler `.render()` /
   `.compile()` / `Template(...)`.
   - Avantages : c'est le **parser de référence** (comme `pickletools` pour
     pickle) ; l'analyse est fidèle à ce que le chargeur verrait et **non
     contournable** par des espaces, de la concaténation ou de l'encodage ;
     jinja2 est pur-Python, ubiquitaire (transformers et llama.cpp s'en servent).
   - Inconvénients : une dépendance runtime de plus (le scanner n'avait que
     `pyyaml` et `typer`) ; il faut un garde-fou interdisant le rendu dans notre
     propre code.
2. **Détection maison (regex / mini-parser), sans dépendance.**
   - Avantages : zéro dépendance.
   - Inconvénients : fragile et **facilement contournable** (espaces,
     concaténation de chaînes, encodage) — exactement le piège qu'on évite pour
     pickle. Un scanner contournable ne vaut rien (leçon de l'étape 9). Réinvente
     un parser que jinja2 fournit déjà et maintient.

## Décision
Option 1, retenue par Mike le 2026-10-24 : jinja2 en dépendance **runtime**,
analyse via `Environment().parse()` uniquement. Jamais de rendu : le garde-fou
`test_no_deserialization` interdit `.render`, `.compile`, `Template` et
`from_string` dans le code du scanner (au même titre que les désérialiseurs).

## Conséquences
- **Dépendance** : `jinja2` passe dans `[project.dependencies]`.
- **Code** : nouveau module `chat_template.py`, `analyze_chat_templates(entries,
  root)` greffé dans le pipeline `_scan` après `analyze_remote_code`. Il extrait
  les templates (`tokenizer_config.json` → `chat_template` chaîne ou liste
  nommée ; `chat_template.jinja` ; `chat_template.json`), les parse avec jinja2
  (taille bornée, erreurs capturées en finding, jamais d'exception), et marche
  l'AST.
- **Finding IDs** :
  - `chat_template.sandbox_escape` (`risk`) — accès à un attribut dunder
    (`__class__`, `__globals__`, `__subclasses__`, `__builtins__`, `__mro__`,
    `__bases__`, `__base__`, `__init__`, `__dict__`, `__import__`), filtre
    `attr()`, ou gadget d'évasion connu.
  - `chat_template.hidden_instructions` (`warning`) — instructions système
    injectées, URL, longues chaînes base64.
  - `chat_template.obfuscated` (`warning`) — template anormalement long/obfusqué.
  - `chat_template.unparseable` (`warning`) — template illisible par jinja2 :
    angle mort, pas de chemin d'exécution prouvé.
- **Jamais d'exécution** : `Environment().parse()` tokenise et construit l'AST,
  sans rendre ni compiler. Garde-fou étendu.
- **Tests** : un template SSTI (dunder) → `risk` ; un template à instructions
  cachées → `warning` ; les templates ChatML réels (Llama/Qwen/Mistral) →
  **aucun faux positif** ; fuzz `hypothesis` (parse ne plante jamais, un template
  malformé donne un finding, pas une exception).

## Amendement (relecture) — fail-closed et non contournable
La première implémentation marchait une petite liste de motifs : c'était une
denylist de nœuds, contournable. Corrections (les findings et le principe « on
ne rend jamais » ne changent pas) :
- **Constant-folding** : la clé d'un `Getitem` est repliée via `as_const()`, donc
  `x['__cla' + 'ss__']` est détecté.
- **Toute chaîne constante dunder/gadget** (où qu'elle soit) = `risk` : couvre
  `{% set k='__class__' %}`, `map('attr','__class__')`, `selectattr('__globals__')`.
- **Tag `{% generation %}`** de transformers parsé via une extension no-op : plus
  de faux `unparseable`, et un escape caché dans un bloc `generation` est bien vu.
- **Fichier trop grand** (`tokenizer_config.json` rembourré au-delà du cap, relevé
  à 8 Mio) → finding `chat_template.oversized` (`risk`), jamais un skip silencieux.
- **Gadgets non-dunder** élargis (frame/coroutine/traceback : `cr_frame`,
  `tb_frame`…).
- Finding ajouté : `chat_template.oversized` (`risk`).
- Reste hors de portée (angle mort documenté, non bloquant) : une clé totalement
  dynamique non constante (`x[loopvar]`) n'est pas repliable — on ne la signale
  pas, pour ne pas générer de faux positif sur l'itération légitime d'un dict.
