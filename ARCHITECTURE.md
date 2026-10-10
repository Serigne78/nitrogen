# Architecture cible de Nitrogen (vision long terme)

> Document de vision, maintenu par l'architecte du projet.
> Il décrit où va Nitrogen, pas ce qu'on code maintenant. Le périmètre courant est celui de la roadmap v0.1.
> Ne pas implémenter ce qui est décrit ici sans demande explicite.

## Chaîne complète

```
1. Ingestion
   ├── Téléchargement du modèle (Hugging Face, S3, registry interne)
   ├── Révision source enregistrée (ex. SHA du commit Hugging Face, jamais "main")
   └── Digest SHA-256 du modèle entier (manifeste : hash de chaque fichier)

2. Scan statique (moteur Nitrogen)
   ├── Détection des formats par magic bytes
   ├── Analyse des opcodes pickle (allowlist, reconstruction via memo)
   ├── Validation safetensors, puis Keras, ONNX, GGUF
   ├── Remote code, provenance, templates de chat, licence, lignage
   └── AI-BOM CycloneDX lié au digest

3. Test dynamique (sandbox isolée, gVisor / Kata / Firecracker)
   ├── Déclenché seulement si le statique est incertain (import inconnu, format atypique)
   ├── Chargement éphémère du modèle, sans réseau
   ├── Observation des tentatives : DNS, connect, execve, accès fichiers (Falco / Tetragon)
   └── Signal d'appoint, jamais une preuve que le modèle est sûr
       (un payload peut détecter la sandbox, attendre, ou ne se déclencher qu'à l'inférence)

4. Signature et attestation
   ├── Attestation in-toto signée dans tous les cas, avec le verdict (pass / fail)
   ├── Payload : digest du modèle + AI-BOM + résultats des tests
   │            + version du scanner + version de la policy
   ├── Signature cosign / Sigstore, keyless de préférence (identité OIDC de la CI)
   └── Stockage à côté de l'artefact OCI et dans un journal de transparence (Rekor)

5. Admission Kubernetes (webhook Go)
   ├── Le pod référence le modèle par un digest immuable
   │   (artefact OCI, modelcar KServe, ImageVolume)
   ├── Refus des pods qui téléchargent un modèle au démarrage (ex. --model org/nom)
   ├── Vérification : attestation valide de Nitrogen pour ce digest ET verdict = pass
   │   ├── OUI ──► pod admis
   │   └── NON ──► pod refusé + alerte SIEM
   ├── failurePolicy: Fail (fermé), avec exceptions pour les namespaces système
   └── Si pas de référence OCI : init container qui revérifie le hash avant chargement

6. Runtime (production)
   ├── Default-deny réseau, allowlist par FQDN (Cilium)
   ├── Détection eBPF : shell lancé, connexion inattendue, lecture de secrets
   ├── Contrôle des privilèges sur les nœuds GPU (privileged, hostPath /dev/nvidia*, hostIPC)
   └── Hypothèse : le scan a pu rater quelque chose (assume breach)
```

## Découpage par versions

| Version | Contenu |
|---|---|
| v0.1 | Étapes 1 et 2 : scanner autonome (pickle, zip PyTorch, safetensors) + AI-BOM |
| v0.2 | Étapes 4 et 5 : signature et webhook d'admission |
| v0.3 | Étape 6 : runtime zero-egress et eBPF |
| v0.4+ | Étape 3 (sandbox dynamique) et nouvelles analyses (provenance, remote code, templates, formats) |

## Principes directeurs
- Un seul produit, une seule CLI. Chaque brique reste utilisable seule.
- Le scanner fonctionne sans Kubernetes et sans signature (usage local).
- Ne jamais exécuter un modèle hors de la sandbox prévue pour ça.
- Réutiliser les standards existants (CycloneDX, in-toto, Sigstore, Pod Security Admission, Cilium) plutôt que réinventer.
- Hors périmètre : détection de backdoors dans les poids.

## Décisions
Les décisions structurantes sont consignées dans `docs/adr/`.
