# ADR 0014 — Source distante `hf://` et vérification d'intégrité

- Statut : accepté
- Date : 2026-11-02
- Décideur : Mike

## Contexte
Jusqu'ici le scanner travaille sur un **dossier local** et ne touche jamais le
réseau. L'étape 12 ajoute `nitrogen scan hf://org/model@révision` : télécharger
le modèle dans un cache (jamais chargé), résoudre la révision en SHA de commit,
puis le scanner normalement. Et une **vérification d'intégrité** : comparer ce
qu'on a téléchargé à ce que le Hub publie, pour détecter une altération.

Trois choix se posent : comment structurer la source (HF seul, ou extensible
vers S3/registries ?), l'ampleur de la vérification d'intégrité, et le fait
d'introduire du réseau et une dépendance runtime dans un outil jusqu'ici
hors-ligne.

## Décisions
1. **Abstraction de source pluggable.** `parse_target(str)` reconnaît un schéma
   `hf://` (et, par défaut, un chemin local). `hf://` est implémenté maintenant ;
   l'interface est conçue pour que `s3://` et les registries OCI s'ajoutent plus
   tard sans réécriture. Entre-temps, S3 reste couvert par « synchroniser puis
   `scan ./dossier` ».
2. **Réseau opt-in par le schéma.** Seul un `hf://` déclenche un accès réseau
   (avec timeouts) ; un chemin local ne touche **jamais** le réseau. On ne charge
   jamais le modèle (téléchargement d'octets uniquement, puis scan statique).
3. **`huggingface_hub` passe en dépendance runtime** (le scanner télécharge pour
   de vrai). Il n'était jusqu'ici qu'une dépendance de dev (benchmarks).
4. **Révision.** Épinglée (`@<40-hex>`) → utilisée telle quelle. Non épinglée
   (`hf://org/model` ou `@main`, un tag) → **résolue en SHA de commit**, inscrite
   dans l'AI-BOM, et signalée `source.unpinned_revision` (**warning**) : une
   attestation signée doit pointer un commit, pas une branche mouvante.
5. **Intégrité : SHA-256 des fichiers LFS.** Pour chaque fichier **LFS** (poids,
   pickles — là où vit le RCE), comparer le SHA-256 local à celui publié par le
   Hub (métadonnées LFS) → divergence = `integrity.mismatch` (**risk**). Les
   petits fichiers non-LFS ne sont pas vérifiés par ce biais (le Hub n'en publie
   qu'un SHA-1 de blob git, pas un SHA-256) : ils sont déjà analysés (remote
   code, templates) et leur SHA-256 est enregistré dans l'AI-BOM, qui reste la
   **référence d'intégrité qui fait foi, agnostique de la source**. Pour S3 plus
   tard, l'intégrité se fera contre un **manifeste de checksums maintenu par
   l'équipe**, pas contre un hash « officiel ».
6. **Findings de niveau source.** La révision non épinglée et une divergence
   d'intégrité sont des findings à part entière : comptés dans le verdict (même
   logique `is_blocking`) et inscrits dans l'AI-BOM (sur le composant modèle pour
   la source/révision, sur le composant fichier pour une divergence).

## Conséquences
- **Code** : nouveau module `source.py` (`parse_target`, `resolve`,
  `check_integrity`). La partie réseau (`resolve` d'un `hf://`) est isolée et
  injectable, pour que les tests restent hors réseau (règle 5). `cli.py` accepte
  une chaîne cible, route `hf://` vers la résolution puis scanne le cache ;
  `aibom.py` gagne `nitrogen:source` / `nitrogen:revision` et accepte les
  findings de niveau source.
- **Finding IDs** : `source.unpinned_revision` (`warning`), `integrity.mismatch`
  (`risk`).
- **Dépendance** : `huggingface-hub` dans `[project.dependencies]`.
- **Tests** : hors réseau — parse d'URI (épinglée / non / invalide), logique
  d'intégrité (concordant → rien, divergent → `risk`), warning non épinglée,
  chemin local inchangé et hors-ligne ; fuzz sur le parseur d'URI. Le
  téléchargement réel est un script/essai **opt-in**, jamais dans `pytest`.
- **Périmètre** : seul `hf://` est implémenté ; `s3://`/OCI sont préparés par
  l'interface mais hors périmètre v0.1.

## Amendement (relecture) — complétude de l'intégrité
La relecture a montré que comparer seulement le SHA-256 des fichiers LFS laisse
un `COMPLIANT` trompeur : un fichier **injecté** (présent localement mais non
listé par le Hub) ou **manquant** passait inaperçu, et rien ne disait quelle
part des fichiers avait réellement été vérifiée. Ajouts (décision Mike,
sévérités fail-closed-visibles) :
- `integrity.unlisted` (**warning**) : un fichier local que le Hub ne liste pas
  (injection probable) ; il est de toute façon analysé par ailleurs.
- `integrity.missing` (**warning**) : un fichier publié par le Hub mais absent
  localement (téléchargement incomplet ou altéré).
- `nitrogen:integrity.coverage = N/M` sur le composant modèle : N fichiers
  vérifiés par SHA-256 sur M publiés, pour rendre explicite la portée réelle.
- **Étiquetage honnête** : l'intégrité `hf://` compare un hash publié par le Hub
  aux octets livrés par le **même** Hub — elle détecte une **altération de
  transport/CDN**, pas la malveillance d'un uploadeur ni la confiance en
  l'auteur. L'ancre qui fait foi reste l'analyse statique + le digest de
  l'AI-BOM signé. Un `COMPLIANT` ne signifie jamais « modèle sûr ».
