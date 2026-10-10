# Nitrogen

Sécurité de la supply chain et du runtime des modèles d'IA déployés sur Kubernetes.

> Statut : en développement (v0.1 en cours, scanner uniquement). Ne pas utiliser en production.

Les équipes téléchargent des modèles (Hugging Face, S3, registries internes) et les font tourner sans savoir ce qu'ils contiennent. Un fichier pickle ou PyTorch peut exécuter du code au chargement. Nitrogen couvre trois moments :

1. **Scan (CI)** : `nitrogen scan` analyse le modèle sans l'exécuter et produit un AI-BOM CycloneDX.
2. **Admission** : l'AI-BOM signé (`cosign attest`) est vérifié par un webhook Kubernetes avant le déploiement.
3. **Runtime** : confinement réseau zero-egress et détection eBPF.

## Usage visé

```
# Utilisateur local
nitrogen scan ./model

# CI d'entreprise
nitrogen scan ./model -o bom.cdx.json --policy policy.yaml
cosign attest --predicate bom.cdx.json --type cyclonedx <modèle>
```

Codes de sortie : `0` conforme, `1` risque détecté, `2` erreur d'exécution.

## Structure

| Dossier | Contenu | Statut |
|---|---|---|
| [`scanner/`](scanner/) | CLI Python, analyse statique, AI-BOM | v0.1, en cours |
| [`webhook/`](webhook/) | Admission controller Go | v0.2 |
| [`runtime/`](runtime/) | Politiques réseau + eBPF | v0.3 |

## Développement

```
cd scanner
uv sync
uv run pre-commit install
uv run pytest
```
