# ADR 0015 — Métadonnées dans l'AI-BOM et policy de licences

- Statut : accepté
- Date : 2026-11-11
- Décideur : Mike

## Contexte
L'AI-BOM n'a jusqu'ici que des hashes, formats et findings. L'étape 13 y ajoute
les **métadonnées** qui font d'un AI-BOM un vrai inventaire : architecture,
licence, lignage, datasets. Sources :
- `config.json` : `model_type`, `architectures`, dtype (`torch_dtype`), présence
  d'un `quantization_config` ;
- front matter YAML du `README.md` (model card) : `license`, `base_model`
  (lignage), `datasets`.

Ces données sont **contrôlées par l'auteur du modèle** : on les lit comme JSON /
YAML (jamais de code), on les borne en taille, et on les inscrit **échappées**
dans l'AI-BOM. Elles sont informatives (pas de finding), **sauf** la licence,
qu'une policy peut restreindre.

## Décisions
1. **Extraction, pas d'exécution.** `config.json` lu en JSON, `README.md` front
   matter lu en YAML (SafeLoader), tailles bornées, parse en erreur → métadonnée
   absente, jamais de plantage ni de finding. Aucune dépendance nouvelle.
2. **Place dans CycloneDX 1.6.** Bloc `modelCard.modelParameters`
   (`architectureFamily` = `model_type`, `modelArchitecture` = `architectures`,
   `datasets`), `component.licenses` (par **nom**, pour rester valide quel que
   soit le libellé — HF n'est pas toujours un id SPDX strict), et propriétés
   `nitrogen:dtype`, `nitrogen:quantization`, `nitrogen:base_model` (lignage).
3. **Policy `allow_licenses`** (liste blanche). Comparaison **insensible à la
   casse** ; la licence brute est conservée dans l'AI-BOM.
   - Licence déclarée **hors liste** → `policy.license-not-allowed` (**risk**),
     comme `policy.format-not-allowed` : si on restreint les licences, une
     licence non autorisée est une porte dure (décision Mike, défaut fail-closed).
   - **Aucune licence déclarée** alors que `allow_licenses` est défini →
     `policy.license-not-allowed` (**risk**, « aucune licence déclarée ») :
     fail-closed, on ne laisse pas passer l'inconnu sous liste blanche.
   - Sans `allow_licenses`, aucune restriction de licence.
4. Le finding licence est de **niveau modèle** : compté dans le verdict et
   rattaché au composant modèle dans l'AI-BOM (comme les findings de source,
   ADR 0014). Neutralisable/durcissable via `allow` / `fail_on`.

## Conséquences
- **Code** : nouveau module `metadata.py` (`extract_metadata`, `ModelMetadata`).
  `policy.py` gagne `allow_licenses` + un helper `license_finding(metadata,
  policy)`. `aibom.py` : `build_aibom` reçoit `metadata` et émet `modelCard` +
  `licenses` + propriétés. `cli.py` : extraction, finding licence threadé.
- **Finding ID** : `policy.license-not-allowed` (`risk`).
- **Tests** : l'AI-BOM d'un modèle réaliste contient architecture / licence /
  lignage et **valide contre le schéma CycloneDX 1.6** ; `allow_licenses`
  bloque une licence hors liste et une licence absente ; config/README malformés
  → métadonnée absente sans plantage ; fuzz du parseur de front matter.
- **Hors périmètre** : la détection de typosquatting sur `base_model` /
  l'organisation relève de l'étape 14 ; ici le lignage est seulement **inscrit**.

## Amendement (relecture)
- **Racine seulement** : les métadonnées ne sont lues que depuis `config.json` /
  `README.md` **à la racine** du scan (ce que lisent transformers / la model
  card HF). Un fichier de sous-dossier ne masque plus la racine, et ne peut donc
  pas fausser l'architecture/licence inscrites dans l'AI-BOM signé.
- **Jamais de plantage** : un `RecursionError` (YAML/JSON profondément imbriqué,
  bombe) est capturé comme une erreur de parse → métadonnée absente, jamais un
  exit-2, conformément à la décision 1.
- **Listes bornées** : au plus 32 éléments par liste (datasets, base_model…),
  en plus de la borne par chaîne, pour qu'une model card gonflée n'amplifie pas
  l'AI-BOM signé.
