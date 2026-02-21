# FaRM System Package

This folder is the deployable unit for the fairness-aware music recommender.

## Structure
- `recommendation/` – core code, configs, and entrypoints  
  - `MainSystem.py` – CLI entrypoint  
  - `configs/base_local.yml` – run from repo root  
  - `configs/base_local_from_rec.yml` – run from inside `System/recommendation/`  
  - `experiments/` – legacy runners (keep for reference)  
- `data/` – CSV datasets (Git LFS)  
- `models/` – Optuna study DBs (Git LFS)  
- `SavedData2/`, `TensorBoardLogs/` – generated artifacts (gitignored)

## Running
From repo root:
```bash
python System/recommendation/MainSystem.py \
  --config System/recommendation/configs/base_local.yml \
  --recs ContentBased --k 5 --no-optim --eval-only
```

From `System/recommendation/`:
```bash
python MainSystem.py --config configs/base_local_from_rec.yml \
  --recs ContentBased --k 5 --no-optim --eval-only
```

## Fairness / tiers
- 7 artist tiers; weights set in configs under `tier_weights`.
- Recommenders apply tier-aware item weights and per-tier exposure limits.
- Objective loss combines listener metrics and artist exposure diversity (see `objective_loss.py`).

## Data & models
- Large CSVs and Optuna DBs are tracked with Git LFS. Run `git lfs pull` after cloning.
- If LFS isn’t available, place matching files in `System/data` and `System/models` as documented in `System/data/README.md`.

## Tests
Light smoke suite:
```bash
python -m pytest System/recommendation/testing -q
```
