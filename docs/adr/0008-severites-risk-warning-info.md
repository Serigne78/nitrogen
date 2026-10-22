# ADR 0008 — Sévérités risk / warning / info

- Statut : accepté
- Date : 2026-10-22
- Décideur : Mike

## Contexte
Jusqu'ici, tout finding rend le verdict RISK (code 1). Le benchmark neutre (ADR 0007) a montré la limite : un membre d'archive **chiffré** est un angle mort (aucun chargeur ne l'ouvre sans mot de passe), pas une preuve de danger ; le bloquer donne un faux positif sur un zip chiffré bénin, le laisser passer en silence perd l'information. Il manque un niveau intermédiaire.

## Décisions

**1. La sévérité est une propriété du `Finding`.**
`Finding(rule, message, severity)` avec trois niveaux : `risk`, `warning`, `info`. Défaut `risk` : un finding non qualifié bloque (fail-closed). La table centrale `rule → sévérité` de `aibom.py` disparaît ; la source unique est le finding émis. Critère (du « chemin d'exécution ») : un chargeur standard pourrait-il exécuter ce contenu ? Oui ou inconnu → `risk` ; non → `warning` ; métadonnée → `info`.

**2. Le code de sortie reste 0/1/2, les warnings ne bloquent pas par défaut.**
- Au moins un `risk` → `RISK`, code 1.
- Sinon au moins un `warning` → `COMPLIANT` (code 0), verdict « avec avertissements », warnings affichés et inscrits dans l'AI-BOM.
- `info` ne bloque jamais.
La policy peut durcir : `fail_on: [warning]` fait remonter les warnings au rang bloquant (code 1). `allow: [<rule>]` neutralise un finding précis (retiré du calcul du verdict, marqué autorisé dans l'AI-BOM). Jamais de défaut permissif sur un `risk` : on ne peut pas `fail_on` en dessous de risk, mais on peut `allow` un risk précis (décision explicite, tracée dans l'AI-BOM).

**3. Un membre d'archive chiffré est un `warning` (`archive.encrypted`).**
Aucun chargeur ne l'ouvre sans mot de passe : pas de chemin d'exécution. Les autres cas « non analysables » (archive imbriquée, membre trop gros, membre illisible pour une autre raison) restent `risk` : on ne peut pas exclure qu'un chargeur les exécute.

## Conséquences
- `Finding` gagne `severity` ; tous les analyseurs qualifient leurs findings (import/malformed pickle, layout/header safetensors → `risk` ; `archive.encrypted` → `warning` ; `findings.truncated` → `info`).
- `aibom.py` lit la sévérité du finding et la mappe vers CycloneDX (`critical`/`high`/`medium`/`info`). Le composant modèle porte `nitrogen:verdict` = `risk` / `compliant` / `compliant-with-warnings`.
- `policy.py` : clés `fail_on` et `allow`, en plus de `allow_formats`.
- `cli.py` : affichage et code de sortie à trois niveaux.
- Le README documente les trois niveaux et le code de sortie.

## Précisions (relecture)
- La décision « membre chiffré → warning » se lit sur la **structure** du zip (`ZipInfo.flag_bits` bit 0/6, méthode 99 AES), jamais sur le message d'erreur : un nom de membre est contrôlé par l'attaquant et ne doit pas décider d'une sévérité. Tout autre échec de lecture (CRC cassé, fonctionnalité non supportée) reste `risk`, quel que soit le nom. Le tar n'a pas de chiffrement : `archive.encrypted` n'y est jamais émis.
- Un finding `allow`-é n'est **pas supprimé** : il reste dans l'AI-BOM, marqué `analysis.state = resolved` (« allowed by policy »), et la policy appliquée (`allow`, `fail_on`, sha256) est inscrite sur le composant modèle. Un verdict « compliant » obtenu sous un `allow` reste donc distinguable d'un scan réellement propre dans l'attestation signée.
- `allow` s'applique par **règle**, globalement au scan (pas par fichier en v0.1) ; une granularité règle+fichier pourra être ajoutée plus tard.
- `fail_on` accepte `warning` et `info` (durcissement choisi), jamais `risk` (toujours bloquant).
