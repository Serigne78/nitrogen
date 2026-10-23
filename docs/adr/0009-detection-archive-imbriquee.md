# ADR 0009 — Détection d'archive imbriquée : exiger une archive réelle

- Statut : accepté
- Date : 2026-10-22
- Décideur : Mike

## Contexte
L'étape 9 mesure les faux positifs du scanner sur une liste figée de vrais
modèles populaires du Hub. Dès les premiers scans, un modèle très téléchargé,
`BAAI/bge-small-en-v1.5`, ressort **RISK** avec un finding
`pickle.unanalyzable` : « pytorch_model/data/199: nested archive not scanned ».

Le membre `pytorch_model/data/199` n'est pas une archive : c'est le **stockage
brut d'un tenseur zéro-initialisé** (1536 octets à zéro). Le scanner le prend
pour un tar imbriqué parce que `_is_tar` reproduit le test d'entrée de
`torch.load` — `tarfile.open(fileobj=…, mode="r:")` — et que ce test **accepte
un bloc d'octets vide ou tout à zéro comme un tar vide valide**. Vérifié :

| Entrée | `_is_tar` actuel |
|---|---|
| 512 octets à zéro | **True** (faux positif) |
| 1536 octets à zéro (le tenseur `data/199`) | **True** (faux positif) |
| vrai tar à un membre | True (correct) |

Conséquence : **tout `.bin` PyTorch contenant un tenseur zéro-initialisé** (un
biais ou un buffer, cas très courant) a un membre `data/<clé>` pris pour une
archive imbriquée et classé `risk`. C'est un faux positif structurel, pas un cas
isolé. Un scanner de sécurité qui crie au loup sur les modèles les plus
téléchargés du Hub ne sera pas utilisé — c'est précisément ce que l'étape 9
doit empêcher.

Deux décisions antérieures encadrent le sujet :
- **ADR 0003** : le prédicat de chaque format est calqué sur la *précondition de
  son chargeur*, pas sur un fichier typique. Or la voie tar de `torch.load` ne
  se contente pas d'ouvrir le tar : elle lit des membres nommés (`pickle`,
  `storages`, `tensors`) et échoue si l'archive n'en a aucun. Le prédicat actuel
  est donc *moins* strict que la précondition réelle du chargeur.
- **ADR 0004** : « un membre qui est lui-même une archive est signalé », sans
  récursion en v0.1. Cette décision reste valable ; le présent ADR ne change pas
  *ce qu'on fait* d'une archive imbriquée, il corrige *ce qui compte* comme
  archive imbriquée.
- **ADR 0008** : critère du chemin d'exécution — un contenu qu'aucun chargeur
  standard n'exécute n'est pas un `risk`.

## Options

1. **Resserrer le prédicat : un membre n'est une archive imbriquée que s'il
   s'ouvre réellement comme telle** (tar : au moins un membre réel,
   `tar.next() is not None` ; zip : au moins une entrée au catalogue). Un bloc
   d'octets qui ressemble à un magic d'archive mais ne s'ouvre pas comme une
   archive réelle n'est plus signalé comme archive imbriquée ; il retombe sur la
   sonde pickle / `unknown` comme n'importe quel autre contenu.
   - Avantages : fidèle à la précondition réelle du chargeur (ADR 0003) ;
     supprime le faux positif mesuré ; préserve la détection des deux vrais tars
     du corpus (`legacy_tar_os_system.pt`, `safe_legacy_tar.pt`), qui ont des
     membres ; ne touche pas au périmètre d'ADR 0004 (une vraie archive
     imbriquée reste signalée `risk`) ; borné (on lit un seul en-tête de membre,
     pas tout le catalogue).
   - Inconvénients : un tar réel mais *vide* (zéro membre) ne serait plus
     signalé — mais `torch.load` ne lit rien d'un tar vide non plus, donc il n'y
     a pas de chemin d'exécution à couvrir.
2. **Reclasser « archive imbriquée » de `risk` à `warning`** (critère d'ADR
   0008) : aucun chargeur standard ne descend dans une archive imbriquée d'un
   modèle, donc pas de chemin d'exécution prouvé.
   - Avantages : cohérent avec le critère du chemin d'exécution ; le code de
     sortie redevient 0 sur ces modèles.
   - Inconvénients : **ne corrige pas la mauvaise détection** — chaque tenseur
     zéro-initialisé resterait signalé, en `warning`, donc du bruit sur
     quasiment tous les `.bin`. Traite le symptôme (la sévérité), pas la cause
     (la détection).
3. **Les deux** : resserrer la détection (option 1) *et* passer une vraie
   archive imbriquée de `risk` à `warning`.
   - Avantages : détection correcte *et* sévérité alignée sur le critère du
     chemin d'exécution.
   - Inconvénients : affaiblit la posture d'ADR 0004 sur un cas (vraie archive
     imbriquée) dont on n'a aucun exemple bénin réel ; une archive réellement
     imbriquée dans un modèle reste très anormale et mérite sans doute un `risk`
     tant qu'on n'a pas de preuve du contraire.
4. **Statu quo.** Ne rien changer.
   - Avantages : aucun.
   - Inconvénients : faux positif sur une large part des modèles `.bin` réels ;
     l'étape 9 documenterait un taux de faux positifs élevé sans correctif.

## Recommandation
**Option 1.** Elle corrige la cause à la racine, en rendant le prédicat de
détection *plus* fidèle au chargeur (dans la droite ligne d'ADR 0003), sans
rouvrir la décision d'ADR 0004 ni relâcher la sévérité. L'option 3 (archive
imbriquée réelle en `warning`) pourra être reconsidérée plus tard si le corpus
fait apparaître de vraies archives imbriquées bénignes ; aujourd'hui il n'y en a
aucune, donc garder `risk` reste le choix prudent et sans coût.

## Décision
Option 1, retenue par Mike le 2026-10-22 : un membre n'est une archive imbriquée
que s'il s'ouvre réellement comme telle (tar : au moins un membre réel ; zip : au
moins une entrée). Une archive imbriquée réelle reste `risk` (ADR 0004 inchangé).

## Conséquences
Si l'option 1 est retenue :
- **Code (`formats.py`)** : `_is_tar` ne renvoie `True` que si l'archive a au
  moins un membre réel (`tar.next() is not None`) ; la détection d'un membre zip
  imbriqué exige au moins une entrée au catalogue. Un bloc d'octets qui matche
  un magic mais ne s'ouvre pas comme archive réelle n'est plus étiqueté archive.
- **Détection de haut niveau** : un fichier de premier niveau tout à zéro, ou un
  tar vide, devient `unknown` au lieu de `tar` — correct, puisque `torch.load`
  échouerait dessus.
- **Tests d'abord** : un `.bin` sain avec un tenseur zéro-initialisé sort `safe`
  (régression du faux positif mesuré) ; un bloc de zéros n'est pas détecté
  `tar` ; les deux vrais tars du corpus restent détectés et jugés comme avant ;
  un vrai membre d'archive imbriquée reste signalé `risk`.
- **Corpus** : ajout d'un cas sain « pickle-zip avec tenseur zéro-initialisé »
  (le faux positif), et maintien d'un cas « membre réellement une archive
  imbriquée → risk ».
- **Étape 9** : le rapport `docs/benchmark-faux-positifs.md` documente le taux de
  faux positifs avant et après ce correctif.
- **Règles du projet** : aucune règle non négociable n'est modifiée ; le
  principe « détecter par magic bytes » reste, précisé par « un magic ne suffit
  pas : un membre n'est une archive que s'il s'ouvre comme telle ».
