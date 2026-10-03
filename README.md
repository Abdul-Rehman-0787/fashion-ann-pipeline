# fashion-ann-pipeline
hotfix
End-to-end reproducible ML pipeline: a fully-connected ANN on Fashion-MNIST,
versioned with Git (code) and DVC + Google Drive (data, models, metrics).

## Run
```bash
pip install -r requirements.txt
dvc pull      # fetch data/models from Google Drive (needs access)
dvc repro     # run prepare -> preprocess -> train -> evaluate
```
