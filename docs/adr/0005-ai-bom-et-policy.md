# ADR 0005 — Forme de l'AI-BOM et policy minimale

- Statut : accepté
- Date : 2026-10-20
- Décideur : Mike

## Contexte
L'étape 6 fait produire au scanner un AI-BOM CycloneDX 1.6 (`-o`) et applique une policy YAML (`--policy`). C'est ce document que `cosign` signera et que le webhook d'admission (v0.2) vérifiera : sa forme engage la suite de la chaîne. Quatre choix se posent.

## Options et décisions

**1. Production du JSON**
- Option A : la bibliothèque `cyclonedx-python-lib` en dépendance de production. Elle construit le document, au prix de 8 paquets installés chez chaque utilisateur.
- Option B : le JSON est construit par notre code avec la bibliothèque standard ; `cyclonedx-python-lib` n'est qu'une dépendance de développement, utilisée par les tests pour valider la sortie contre le schéma officiel 1.6 qu'elle embarque.
- **Décision : option B.** Un outil de sécurité de la supply chain doit limiter sa propre surface : chaque dépendance de production tourne dans la CI de l'utilisateur, à côté du modèle et du contexte de signature. La conformité reste garantie, par un test en CI plutôt que par la bibliothèque.

**2. Emplacement des findings**
- Option A : section `vulnerabilities`, chaque finding pointant (`affects`) vers le composant du fichier concerné.
- Option B : propriétés `nitrogen:*` sur chaque composant.
- **Décision : option A.** C'est l'emplacement standard des problèmes de sécurité en CycloneDX, lisible par les outils existants sans connaître Nitrogen.

**3. Périmètre de la policy v0.1** (révisée le 2026-10-20, voir Révision)
- Options considérées : interdire des formats (`deny_formats`), autoriser des formats (`allow_formats`), ignorer des règles, étendre l'allowlist.
- **Décision : autoriser des formats uniquement** (`allow_formats`). Un fichier dont un format n'est pas autorisé est un risque, même sans autre finding. Un fichier sans format reconnu porte le pseudo-format `unknown`, qu'il faut autoriser nommément. Exemple « safetensors uniquement » : `allow_formats: [safetensors, unknown]`. Sans clé `allow_formats`, aucune restriction de format. Ignorer des règles ou étendre l'allowlist affaiblirait la détection : écarté de la v0.1.

**4. Digest du modèle**
- Option A : un digest de manifeste, calculé sur la liste de chaque fichier avec son SHA-256.
- Option B : uniquement les hashes par fichier.
- **Décision : option A.** Il lie la preuve au modèle exact : modifier, ajouter ou retirer un seul fichier change le digest.
- **Spécification `nitrogen-manifest-v1`** (fixée le 2026-10-20, voir Révision) : SHA-256 des octets de `json.dumps(L, ensure_ascii=True, separators=(",", ":"))`, où `L` est la liste des paires `[chemin, sha256]` (chemin POSIX relatif, sha256 en hexadécimal minuscule), triée sur les octets UTF-8 du chemin (`surrogateescape` pour les noms non UTF-8). L'identifiant `nitrogen-manifest-v1` figure dans l'AI-BOM, pour qu'un vérificateur sache quel algorithme réimplémenter.

## Conséquences
- Aucune dépendance de production ajoutée ; `cyclonedx-python-lib` entre en dépendance de développement.
- L'AI-BOM est écrit pour un verdict conforme comme pour un risque (la preuve d'un refus compte autant), jamais en cas d'erreur d'exécution. Écriture atomique.
- La policy est lue avec un chargeur dérivé de `yaml.SafeLoader` qui refuse aussi les clés dupliquées ; une clé, un format ou un type inattendu est une erreur (code 2), jamais ignoré en silence. L'AI-BOM rappelle la policy appliquée (son SHA-256 et ses formats autorisés).
- Chaque fichier porte `nitrogen:analyzed` (`true` ou `false`) : un vérificateur peut refuser un modèle dont un fichier n'a pas été analysé.
- Le lien entre le digest de manifeste et le digest OCI vérifié par le webhook sera tranché en v0.2.

## Révision (2026-10-20)
La relecture de l'implémentation a montré deux défauts dans les décisions 3 et 4 telles qu'écrites au départ. Mike a tranché :
- **Décision 3** : `deny_formats` était une liste noire, contraire à la règle « allowlist, pas denylist » du projet. `deny_formats: [pickle]` laissait passer `pytorch-zip` et `tar`, qui contiennent du pickle ; un fichier de format non reconnu ne pouvait pas être interdit ; et tout format ajouté plus tard aurait été autorisé d'office. Remplacée par `allow_formats`.
- **Décision 4** : la liste de lignes « sha256  chemin » était ambiguë. Un nom de fichier contenant un retour à la ligne permettait de fabriquer deux ensembles de fichiers différents avec le même digest. L'encodage est désormais canonique et versionné (`nitrogen-manifest-v1`).
