# ADR 0006 — Reconnaître les formats de modèle non analysés

- Statut : accepté
- Date : 2026-10-20
- Décideur : Mike

## Contexte
La policy `allow_formats` (ADR 0005) range dans le pseudo-format `unknown` tout fichier sans format reconnu. `unknown` mélangeait deux choses :
- les fichiers d'accompagnement d'un modèle (`config.json`, tokenizer, README) ;
- des formats de modèle que la v0.1 n'analyse pas : HDF5/Keras, GGUF, NumPy `.npy`, ONNX.

Comme tout modèle réel contient un `config.json`, toute policy réaliste devait autoriser `unknown`, et laissait donc passer un Keras avec une couche Lambda. Et `nitrogen:analyzed=false` sur un `config.json` rendait inutilisable la règle « refuser un modèle dont un fichier n'est pas analysé ».

## Options
1. Garder `unknown` tel quel et documenter la limite.
2. Reconnaître ces formats par leurs magic bytes, sans les analyser, et réserver `unknown` aux fichiers sans format de modèle.

## Décision
Option 2. Formats reconnus, non analysés en v0.1 :
- `hdf5` : `\x89HDF\r\n\x1a\n` à l'offset 0 (Keras `.h5`) ;
- `gguf` : `GGUF` à l'offset 0, suivi d'une version `1`, `2` ou `3` (little-endian), pour ne pas prendre un README qui commence par « GGUF » pour un fichier GGUF. Une version future (0 ou ≥ 4) ne serait pas reconnue et retomberait en `unknown` ;
- `npy` : `\x93NUMPY` à l'offset 0 ;
- `onnx` : protobuf commençant par le champ `ir_version` (`\x08` puis une version de 1 à 15) et contenant la trace d'un exportateur ONNX ou PyTorch dans ses 4 premiers Kio. C'est une **heuristique**, avec deux limites assumées : l'ordre des champs protobuf n'est pas garanti par la spec (un ONNX dont `ir_version` n'est pas le premier champ n'est pas reconnu), et un exportateur sans trace `onnx`/`pytorch` (tf2onnx, skl2onnx…) non plus. Un tel fichier retombe en `unknown`. ONNX reste hors du périmètre d'analyse v0.1 ; cette reconnaissance est un filet, pas une garantie.

## Conséquences
- Le périmètre d'**analyse** de la v0.1 ne change pas. Ces formats sont seulement reconnus, et marqués `nitrogen:analyzed=false`.
- `nitrogen:analyzed` vaut `false` uniquement pour un fichier qui porte un format de modèle non analysé. Un fichier sans format de modèle (`unknown`) n'a rien à analyser : `true`.
- Une policy `allow_formats: [safetensors, unknown]` refuse désormais un Keras, un GGUF, un `.npy` ou un ONNX.
- Limites connues : un `.keras` (Keras 3) est une archive zip, analysée comme telle, dont les poids HDF5 et la configuration ne sont pas inspectés ; les membres `.npy` d'un `.npz` non plus.
