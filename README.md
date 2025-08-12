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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Extra deps used by training and feature extraction
pip install torch fair-esm biotite scikit-learn seaborn matplotlib networkx python-louvain

# System tools required by structure feature extraction
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

## Quick start: from scratch to training

1) Structure prediction (ESMFold)

```bash
python protein_structure_prediction.py \
  --fasta Data/split_dataset/merge_train_val.fasta \
  --output-dir Data/structure_local \
  --batch-size 1 \
  --chunk-size 128 \
  --memory-efficient \
  --log-file protein_prediction.log
```

Outputs: `.pdb` files under `Data/structure_local/`, plus metrics under `Data/results/`.

2) Structure feature extraction

Ensure `mkdssp` and `freesasa` are installed (see above). Then run:

```bash
python extract_structure_features.py
```

Outputs:
- `Data/structure_features/structure_features.csv`
- `Data/structure_features/all_structure_features.pkl`

3) ESM-2 feature extraction

Extract embeddings from a FASTA file. You can pass your merged train/val FASTA.

```bash
python extract_ESM_features.py \
  --fasta Data/split_dataset/merge_train_val.fasta \
  --output_dir Data/esm_features \
  --model_name esm2_t33_650M_UR50D \
  --batch_size 1 \
  --gpu 0
```

Outputs:
- `Data/esm_features/esm_features.pkl`
- `Data/esm_features/esm_features.csv`

Available models: `esm2_t36_3B_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t12_35M_UR50D`.

4) Train ProjFusNet (Projection Fusion + LSTM)

The training script:
- loads IDs/labels from `Data/split_dataset/train_set.fasta` and `Data/split_dataset/val_set.fasta`;
- aligns features by ID from `Data/structure_features/structure_features.csv` and `Data/esm_features/esm_features.csv`;
- projects different modalities to the same dimension and concatenates them; 
- performs 5-fold cross validation and writes all artifacts to `Results/experiments/exp_YYYYMMDD_HHMMSS/`.

```bash
python lstm_fusion_features_model.py
```

## Script cheat sheet

- `protein_structure_prediction.py`
  - Inputs: `--sequence` single seq, `--fasta` FASTA, or `--file` CSV with `sequence`.
  - Key args: `--batch-size`, `--chunk-size`, `--memory-efficient`, `--max-retries`.
  - Outputs: `Data/structure_local/*.pdb`, plus `Data/results/prediction_metrics.csv` and `prediction_stats.json`.

- `extract_structure_features.py`
  - Scans `Data/structure_local/*.pdb` and extracts rich 3D structure descriptors using Biopython + DSSP + FreeSASA.
  - Requires system executables `mkdssp` and `freesasa` in `PATH`.
  - Outputs: `Data/structure_features/structure_features.csv` and `all_structure_features.pkl`.

- `extract_ESM_features.py`
  - Uses `fair-esm` ESM-2; mean-pools token representations to obtain a per-sequence embedding.
  - Outputs: `Data/esm_features/esm_features.pkl` and `esm_features.csv` (`peptide_id` as index).

- `lstm_fusion_features_model.py`
  - Projects structure + ESM features to a common space and feeds the concatenation to an LSTM classifier.
  - Runs 5-fold CV and writes results/plots to `Results/experiments/`.

## License

This project is intended for academic research only. Please also follow the licenses of upstream tools/models (ESM-2, ESMFold, DSSP, FreeSASA, etc.).