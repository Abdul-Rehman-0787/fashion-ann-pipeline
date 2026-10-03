# fashion-ann-pipeline

End-to-end reproducible ML pipeline: a fully-connected ANN on Fashion-MNIST,
versioned with Git (code) and DVC + Google Drive (data, models, metrics).
End-to-end ML pipeline (dev version): ANN on Fashion-MNIST/
## Run
```bash
pip install -r requirements.txt
dvc pull      # fetch data/models from Google Drive (needs access)
dvc repro     # run prepare -> preprocess -> train -> evaluate
```
