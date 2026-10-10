# nitrogen-scanner

Scanner statique de modèles d'IA : détecte les formats de sérialisation dangereux sans jamais charger le modèle, et produit un AI-BOM CycloneDX.

```
uv sync
uv run nitrogen scan ./model
uv run nitrogen scan ./model -o bom.cdx.json --policy policy.yaml
```

Codes de sortie : `0` conforme, `1` risque détecté, `2` erreur d'exécution.
