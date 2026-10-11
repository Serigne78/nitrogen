# ADR 0003 — Détection de format multi-étiquettes

- Statut : accepté
- Date : 2026-10-11
- Décideur : Mike

## Contexte
Aujourd'hui, `detect_format` donne **une seule** étiquette par fichier : le premier test qui répond gagne (magic pickle, puis zip, puis sonde pickle, puis safetensors). L'analyse se branche sur cette étiquette, et un fichier `unknown` n'est pas analysé.

Or un même fichier peut être accepté par plusieurs chargeurs. Le code a révélé trois cas où l'étiquette unique fait sauter une analyse :

1. **Collision du champ de taille (relecture de l'étape 3, C2).** Un safetensors commence par sa taille d'en-tête sur 8 octets. Si cette taille vaut 640, 896, 1152 ou 1408 (plus des multiples de 65 536), le fichier commence par `\x80\x02`…`\x80\x05`, le magic d'un pickle. Il est alors classé `pickle` et **n'est pas validé comme safetensors**. Ça touche environ 1 fichier sain sur 2 000, et un attaquant le déclenche exprès en rembourrant son en-tête. Même chose pour une taille de 67 324 752 octets, qui fait commencer le fichier par `PK\x03\x04` (zip).
2. **Polyglotte safetensors/zip PyTorch.** Le fichier est un safetensors valide, et sa zone de données contient un zip avec un `data.pkl` malveillant. `torch.load` le lit comme un zip, `safetensors` comme un safetensors. Si on corrige le cas 1 en donnant la priorité à safetensors, le pickle de ce zip n'est plus jamais analysé.
3. **Polyglotte safetensors/pickle au-delà de 64 Kio (laissé ouvert à l'étape 2).** Un safetensors valide, dont la taille commence par l'opcode `V`, peut aussi être exécuté par `pickle.load`. Si l'import arrive après la fenêtre de 64 Kio, le fichier n'est étiqueté que `safetensors`.

Dans les trois cas, l'attaquant **choisit quel analyseur tourne**. Il faut trancher maintenant : l'analyse pickle (étape 5) et l'AI-BOM (étape 6) vont se brancher sur ce modèle.

## Options
1. **Garder une seule étiquette et ajuster l'ordre des tests.** Par exemple, valider strictement le safetensors avant de regarder le magic pickle.
   - Avantages : changement minimal, une seule colonne `FORMAT`, l'AI-BOM reste simple.
   - Inconvénients :
     - ne règle que le cas 1 : les cas 2 et 3 restent des contournements, puisqu'il faut toujours choisir une étiquette et que l'attaquant fabrique le fichier pour l'autre ;
     - chaque nouveau format (Keras, ONNX, GGUF) ajoute une énigme d'ordre ;
     - le défaut reste silencieux.
2. **Multi-étiquettes.** `detect_formats` renvoie l'**ensemble** des formats qu'un chargeur accepterait. Le prédicat de chaque format est calqué sur la *précondition de son chargeur*, pas sur un fichier typique :
   - zip : magic `PK` à l'offset 0, comme `torch.load` ;
   - pickle : magic PROTO, ou sonde d'opcodes ;
   - safetensors : taille plausible et `{` après d'éventuels espaces JSON.

   Tous les analyseurs concernés tournent, et leurs findings s'additionnent. `unknown` veut dire « aucun chargeur supporté ne l'accepte ».
   - Avantages :
     - règle les trois cas d'un coup ;
     - l'attaquant ne choisit plus l'analyseur ;
     - ajouter un format revient à ajouter un prédicat, sans ordre à régler ;
     - l'invariant devient testable : un fichier qu'un chargeur supporté accepte n'est jamais `unknown`.
   - Inconvénients :
     - le format devient un ensemble, ce qui touche la sortie humaine, l'AI-BOM (étape 6) et le code de l'étape 2 ;
     - l'étape 5 doit distinguer un pickle « interprétation secondaire » qui casse avant tout import (sans danger : un chargeur pickle échoue sans rien exécuter) d'un vrai pickle malformé.
3. **Lancer tous les analyseurs sur tous les fichiers**, sans se fier à la détection.
   - Avantages : aucun contournement possible par la détection.
   - Inconvénients :
     - coût : sonde pickle sur des Go de tenseurs, listage zip de chaque fichier ;
     - bruit : un `config.json` « analysé comme pickle » ;
     - il faut quand même une détection pour l'étiquette de l'AI-BOM.

## Recommandation
L'option 2. C'est la seule qui retire à l'attaquant le choix de l'analyseur sans payer le coût de l'option 3. Elle prolonge le principe « allowlist, pas denylist » : on ne parie pas sur une interprétation, on les vérifie toutes.

## Décision
Option 2, multi-étiquettes, retenue par Mike le 2026-10-11.

## Conséquences
- **Code :**
  - `formats.py` : `detect_formats(fh) -> frozenset[Format]`, sans ordre de priorité ;
  - `inventory.py` : `FileEntry.formats`, chaque analyseur concerné tourne sur le même descripteur ;
  - CLI : la colonne `FORMAT` affiche par exemple `pickle+safetensors`.
- **Étape 5 :**
  - un pickle qui casse **avant** d'avoir résolu un import ne donne pas de finding s'il n'est qu'une interprétation parmi d'autres ;
  - un fichier dont la seule étiquette est `pickle` et qui est malformé reste un finding.
- **Étape 6** : un composant de l'AI-BOM peut porter plusieurs formats.
- **Tests :**
  - toutes les tailles d'en-tête safetensors multiples de 8 jusqu'à 0x10000, dont 0x280, 0x380, 0x480 et 0x580 ;
  - la collision zip (`0x04034B50`, flux simulé) ;
  - un polyglotte safetensors/zip, étiqueté avec les deux formats ;
  - un test différentiel **au niveau du pipeline** : si la bibliothèque de référence accepte un fichier, `safetensors` fait partie de ses formats.
- **`CLAUDE.md`** : règle ajoutée aux règles non négociables, « **Un fichier peut avoir plusieurs formats** : chaque format qu'un chargeur supporté accepterait est détecté et analysé. Aucune priorité entre formats. »
