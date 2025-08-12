## ProjFusNet: Protein Precursor Prediction Pipeline

An end-to-end workflow from sequence to structure, from structure to features, and finally to fusion model training and evaluation.

- Protein tertiary structure prediction: `protein_structure_prediction.py`
- Structure feature extraction from predicted structures: `extract_structure_features.py`
- ESM-2 sequence embedding extraction: `extract_ESM_features.py`
- ProjFusNet training (Projection Fusion + LSTM): `lstm_fusion_features_model.py`
- Example dataset: `Data/split_dataset/merge_train_val.fasta`
### Key layout

```
Data/
  split_dataset/
    merge_train_val.fasta        # merged sequences with labels (example)
    train_set.fasta              # training set (>id|label|...)
    val_set.fasta                # validation set (>id|label|...)
  structure_local/               # predicted .pdb files will be saved here
  structure_features/            # structure features (.csv / .pkl)
  esm_features/                  # ESM features (.csv / .pkl)
Results/experiments/             # training outputs (metrics, models, plots)
```

## Environment and dependencies

Recommended: Python 3.10+ and a virtual environment.

```bash
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Extra deps used by training and feature extraction
pip install torch fair-esm biotite scikit-learn seaborn matplotlib networkx python-louvain

# System tools required by structure feature extraction — macOS
brew install dssp freesasa
# Verify
mkdssp --version
freesasa --version
```

Notes:
- GPU is strongly recommended for structure prediction and ESM-2 embedding extraction.
- Install `torch` according to your CUDA version following the official PyTorch guide.

## Data format

- FASTA header format: `>PEPTIDE_ID|LABEL|...` where `LABEL ∈ {0,1}`.
- `merge_train_val.fasta` follows the header convention above.
- Structure/ESM features are written with `peptide_id` as the index and aligned by ID during training.

