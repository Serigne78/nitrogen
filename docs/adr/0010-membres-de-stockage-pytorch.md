# ADR 0010 — Membres de stockage PyTorch : ne pas les analyser comme des pickles

- Statut : accepté
- Date : 2026-10-22
- Décideur : Mike
- Portée : précise et amende ADR 0004 (analyse de tout membre d'archive) ; englobe le cas que corrigeait ADR 0009 (tar imbriqué, conservé tel quel).

## Contexte
Le benchmark de faux positifs de l'étape 9, sur les modèles les plus
téléchargés du Hub, donne un résultat sans appel : **9 des 14 premiers modèles
réels scannés sortent RISK**, soit ~64 % de faux positifs. Exemples :
`google-bert/bert-base-uncased`, `openai-community/gpt2`, `google-t5/t5-small`,
`sentence-transformers/all-mpnet-base-v2`, `openai/clip-vit-base-patch32`.

Toutes les causes pointent une racine unique. ADR 0004 a décidé d'analyser
**tout membre d'archive détecté comme pickle**, pour couvrir des chargeurs
maison. Or un modèle PyTorch zip contient, à côté de `data.pkl`, des membres
`archive/data/<clé>` qui sont le **stockage brut des tenseurs** — des octets
quasi aléatoires. Les analyser comme des pickles ou des archives produit des
faux positifs en série :

| Modèle | Membre | Finding | Réalité |
|---|---|---|---|
| mpnet | `archive/data/1` | `pickle.unanalyzable` : member too large | un tenseur > 64 Mo |
| mpnet | `archive/data/27` | `pickle.malformed` : string4 de 1,3 Go | octets de tenseur lus comme pickle |
| bge-small | `pytorch_model/data/199` | `pickle.unanalyzable` : nested archive (tar) | un tenseur zéro-initialisé (ADR 0009) |
| clip, wav2vec2 | `archive/data/<n>` | `pickle.unanalyzable` | stockage de tenseur |

Le format **legacy non-zip** (`torch.save` avant 1.6, ex. gpt2 et t5-small,
magic `\x80\x02\x8a\n` = 0x1950a86a20f9469cfc6c) a le même défaut sous une autre
forme : le fichier est une suite de pickles de métadonnées **suivie des données
de stockage brutes**. Le scanner lit correctement les pickles, puis sonde les
octets de stockage comme un pickle concaténé et sort `pickle.malformed`.

Dans **tous** les cas observés, c'est un membre (ou une zone) de **stockage** qui
déclenche le finding — **jamais** `data.pkl`, dont l'allowlist suffit. Ce que fait
réellement `torch.load` le confirme : il ne dépickle que `data.pkl` (et
`*/data.pkl`) ; les `data/<clé>` sont chargés comme stockage via `persistent_id`
(type et taille déclarés), **jamais dépicklés**. Même en legacy, il lit un nombre
borné de pickles puis traite le reste comme du stockage. Le scanner analyse donc
du contenu qu'aucun chargeur standard n'exécute — contraire au critère du chemin
d'exécution (ADR 0008).

ADR 0009 (tar imbriqué) ne traitait qu'un symptôme (un tenseur zéro pris pour un
tar). La cause est générale : analyser les membres de stockage.

## Options

1. **Reconnaître le stockage ; n'analyser que ce qu'un chargeur dépickle.**
   Dans un zip PyTorch (présence d'un `*/data.pkl`), analyser `data.pkl` et les
   autres membres pickle hors stockage ; traiter les membres `<racine>/data/<n>`
   (clé numérique, la convention de stockage de torch) comme des **données
   brutes** : ni pickle, ni archive, et le plafond de taille ne les signale pas.
   Pour le legacy non-zip, scanner les pickles de tête et s'arrêter au STOP du
   dernier : les octets suivants sont du stockage, pas un pickle malformé.
   - Avantages : supprime toute la classe de faux positifs ; fidèle à ce que
     fait `torch.load` (ADR 0003, prédicat = précondition du chargeur) ; aligné
     sur le critère du chemin d'exécution (ADR 0008).
   - Inconvénients : un pickle caché dans un membre de stockage n'est pas
     analysé — mais aucun chargeur standard ne le dépickle (il le lit comme un
     stockage de type et taille déclarés). Assouplit ADR 0004 pour les membres
     de stockage.

2. **Tout analyser, mais seulement sur détection pickle à forte confiance.**
   Garder l'analyse de tous les membres, mais ne retenir un membre de stockage
   que s'il porte un **magic PROTO** (`\x80` + version 2–5) à l'offset 0 ;
   abandonner la sonde protocole 0/1 et les findings `too large` / nested
   archive sur ces membres. Même correctif STOP pour le legacy.
   - Avantages : conserve l'esprit « analyser tout membre » d'ADR 0004 ; un vrai
     pickle de protocole ≥ 2 glissé dans un membre reste détecté ; des octets de
     tenseur commencent rarement par `\x80\x02`–`\x80\x05`.
   - Inconvénients : un pickle de protocole 0/1 (sans magic) dans un membre de
     stockage passe ; `\x80\x02` peut encore apparaître par hasard au début d'un
     tenseur (faux positif résiduel, plus rare).

3. **Rétrograder les findings sur membres de stockage en `warning`/`info`.**
   Garder la détection, mais rendre non bloquants `too large` / `malformed` /
   nested archive quand ils viennent d'un membre de stockage.
   - Avantages : le code de sortie redevient 0.
   - Inconvénients : ne corrige pas la cause ; la sortie reste polluée d'un
     finding par gros tenseur ; traite le symptôme.

4. **Statu quo.** Ne rien changer : ~64 % de faux positifs sur les modèles les
   plus téléchargés. Le scanner est inutilisable sur de vrais modèles.

## Recommandation
**Option 1.** C'est la seule qui supprime la cause tout en restant fidèle au
chargeur : on analyse exactement ce que `torch.load` dépickle. Le risque
résiduel (un pickle planqué dans un membre de stockage) n'a pas de chemin
d'exécution par un chargeur standard, donc relève au plus d'un `warning` futur,
pas d'un `risk` par défaut. On peut, en complément conservateur, vérifier à bas
coût qu'un membre de stockage ne **commence pas** par un magic PROTO (part
d'Option 2) ; à décider dans la mise en œuvre.

Si l'option 1 est retenue, ADR 0004 est amendé (« tout membre pickle est
analysé » devient « tout membre pickle **hors stockage** ») et ADR 0009 devient
un cas particulier couvert par 0010 (il reste correct et sans effet de bord).

## Décision
Option 1, retenue par Mike le 2026-10-22, implémentée dans l'étape 9 : dans un
zip PyTorch, les membres de stockage `<racine>/data/<clé>` ne sont pas analysés
comme des pickles ni des archives ; en legacy non-zip, les octets qui suivent le
dernier pickle complet sont du stockage, pas un pickle malformé. ADR 0004 est
amendé en ce sens ; ADR 0009 est conservé tel quel (cas particulier, sans effet
de bord).

## Conséquences
Si l'option 1 est retenue :
- **Code (`archives.py`)** : identifier les membres de stockage d'un zip PyTorch
  (`<racine>/data/<entier>`, en présence d'un `data.pkl`) et ne pas les passer à
  l'analyse pickle / archive ni au plafond de taille.
- **Code (`pickle_scan.py`)** : en lecture d'un flux pickle concaténé, ne pas
  signaler `pickle.malformed` quand la rupture survient **après** un pickle
  complet (STOP atteint) et au **début** de ce qui n'est pas un pickle — c'est
  du stockage. Une rupture **à l'intérieur** d'un pickle (avant son STOP) reste
  `pickle.malformed`.
- **Tests d'abord** : un vrai `state_dict` PyTorch zip avec un gros membre de
  stockage et un membre de stockage aux octets « pickle-like » sort `safe` ; un
  `.bin` legacy (pickles + stockage) sort `safe` ; un `data.pkl` malveillant
  reste `risk` ; un pickle **réellement** tronqué reste `pickle.malformed`.
- **Corpus** : ajout d'un `.bin` zip avec membre de stockage volumineux et
  « pickle-like », et d'un `.bin` legacy sain ; maintien des cas malveillants.
- **Étape 9** : le rapport documente le taux de faux positifs avant/après 0010.
- **Règle non négociable ADR 0004** : « tout membre pickle est analysé » →
  « tout membre pickle hors stockage de tenseur est analysé ».
