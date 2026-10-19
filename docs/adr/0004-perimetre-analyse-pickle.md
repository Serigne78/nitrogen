# ADR 0004 — Périmètre de l'analyse pickle : EXT, membres de zip, tar

- Statut : accepté
- Date : 2026-10-19
- Décideur : Mike

## Contexte
L'étape 5 écrit l'analyseur pickle. La relecture du corpus (étape 4) a laissé trois questions ouvertes. Chacune décide si un fichier qui exécute du code au chargement peut sortir `COMPLIANT`.

1. **Opcodes EXT1/2/4.** Ils importent un nom désigné par un numéro du registre `copyreg` du processus qui charge le fichier. Ce registre est vide par défaut, mais une bibliothèque peut le remplir. Le scanner ne peut pas connaître l'état du registre au moment du chargement.
2. **Membres d'un zip.** `torch.load` ne lit que `<archive>/data.pkl`. Un zip peut pourtant contenir d'autres pickles, et d'autres chargeurs (TorchScript, chargeurs maison) peuvent les lire.
3. **Tar.** L'ancien format de `torch.save` (avant PyTorch 1.6) est un tar dont `torch.load` dépickle encore les membres. Il est détecté depuis l'étape 3, mais pas analysé : il sort `COMPLIANT (not analyzed yet)` avec le code 0.

## Options
**EXT1/2/4**
1. Finding `pickle.import` : un import qu'on ne peut pas résoudre vers un nom de l'allowlist est un import non autorisé.
2. Finding `pickle.malformed` : plus doux, mais faux sur le fond, puisque le flux est valide.
3. Ignorer : le registre est vide par défaut. C'est fail-open si le processus l'a rempli.

**Membres de zip**
1. Analyser chaque membre dont le contenu est un pickle, détecté par ses octets et non par son nom, dans un zip PyTorch comme dans un zip générique.
2. N'analyser que `*/data.pkl`, exactement ce que lit `torch.load`.

**Tar**
1. Analyser chaque membre pickle, comme pour le zip, en lisant sans extraire.
2. `RISK` d'office (format hérité non supporté).
3. Statu quo : détecté, non analysé.

## Décision
- **EXT1/2/4** : option 1, finding `pickle.import`. Conforme à la règle « allowlist, pas denylist ». `torch.save` n'utilise jamais EXT, donc aucun faux positif attendu.
- **Membres de zip** : option 1, tout membre détecté comme pickle est analysé. L'attaquant ne choisit plus le membre que le scanner ignore.
- **Tar** : option 1, mêmes règles que le zip, code partagé.

## Conséquences
- L'analyseur prend un **flux** en entrée, pas un fichier : il s'applique au fichier entier, à chaque membre de zip et à chaque membre de tar.
- **Conteneurs** : les membres sont lus depuis l'archive, sans jamais être écrits sur le disque (pas de zip slip possible). La décompression est **bornée** contre les zip bombs : au-delà du plafond, finding plutôt que lecture. Pas de récursion dans des archives imbriquées en v0.1 ; un membre qui est lui-même une archive est signalé.
- **Findings** : le nom du membre figure dans le message (ex. `archive/extra.pkl: os.system`).
- **Corpus** : ajout de cas EXT, d'un zip PyTorch avec un second pickle piégé, d'un zip générique contenant un pickle, et d'un tar hérité piégé et sain.
- **Périmètre v0.1** : le tar n'était pas listé dans le périmètre initial. Il entre dans le périmètre par cette décision, car `torch.load` le lit encore.
