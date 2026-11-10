# TRoPE-TUL

PyTorch code for **TRoPE-TUL: Trajectory-User Linking with Place-Aware Attention over Full Trajectories**.

Each point of a trajectory is described by four inputs: its coordinates, its time, the grid cells it falls in at four sizes (10, 50, 200 and 1,000 m), and a vector of a text that describes the named places around it. A one-layer Transformer mixes the four inputs of a point, an attention encoder with a spatial rotary encoding reads all points of the trajectory, and attention pooling gives one vector per trajectory, which a linear layer assigns to a user. Training uses cross-entropy and a supervised contrastive loss. At prediction time, a count of where each user has been can be blended with the model's scores (`tools/fuse_lookup.py`).

The first version of this repository (the original submission) is tagged `v1-submission`.

## Setup

```bash
pip install -r requirements.txt
```

The reported runs used Python 3.13, PyTorch 2.8 with CUDA, and one NVIDIA L40S GPU. Training logs to Weights & Biases; set `WANDB_MODE=offline` (or `disabled`) to run without an account.

## Data

The processed data are at https://dx.doi.org/10.6084/m9.figshare.32165550. Put the files in `dataset/` so that the paths in `settings/*.json` resolve:

```
dataset/
  4square-TrajTRoPE.csv               gowalla-Texas-TrajTRoPE.csv        geolife-Beijing-TrajTRoPE.csv
  tky-scale-TrajTRoPE.csv             tky-scale-u1000-TrajTRoPE.csv      tky-scale-u500-TrajTRoPE.csv
  splits/<dataset>_seed42.json        frozen 80 / 10 / 10 split, shared by every model in the paper
  pois/4square--poi-embed.npy         POI text vectors, one row per point (Qwen3-Embedding-4B)
  pois/gowalla-Texas-poi-embed.npy
  pois/geolife-Beijing-poi-embed--v2-fp16.npy        (Qwen3-Embedding-0.6B, 16-bit)
  pois/gemma/tky-scale-osm-qwen3-4b-2560.npy         one row per unique POI text
  pois/gemma/tky-scale{,-u500,-u1000}-osm-poi-index.npy   point -> row of the table above
```

| Config | Dataset | Users |
|---|---|---|
| `settings/foursquare.json` | Foursquare (Tokyo check-ins) | 213 |
| `settings/gowalla.json` | Gowalla (Texas check-ins) | 116 |
| `settings/geolife.json` | GeoLife (Beijing GPS) | 89 |
| `settings/foursquare_tky.json` | Foursquare-TKY (full TSMC2014 Tokyo file) | 2,048 |
| `settings/foursquare_tky_1000.json` | Foursquare-TKY-1000 (nested subset) | 1,000 |
| `settings/foursquare_tky_500.json` | Foursquare-TKY-500 (nested subset) | 500 |

The scripts in `scripts/` rebuild the Foursquare-TKY data from the raw TSMC2014 file and an OpenStreetMap extract (`build_tky_scale.py`, `make_tky_subsets.py`, `build_osm_pois.py`, `attach_pois.py`), the GeoLife trips file (`build_geolife.py`), and the cheaper POI vectors of the ablation (`poi_tables.py`, `embed_gemma.py`).

## Training and evaluation

```bash
python main.py --config settings/foursquare.json --data_dir dataset --seed 42
```

The paper reports the mean over seeds 42, 43 and 44. Each run writes `logs/<run_name>/` (the settings, the training log, `test_metrics.csv` and `<run_name>_preds.npz` with the validation and test scores) and keeps the best weights in `saved_model/`. The checkpoint with the highest validation ACC@1 is evaluated once on the test set. Set `LOG_SAVE_DIR` and `MODEL_CACHE_DIR` to write elsewhere.

### Lookup blend

```bash
python tools/fuse_lookup.py logs/<run_name>/settings.json logs/<run_name>/<run_name>_preds.npz --data_dir dataset --final
```

The form of the lookup (vote or naive Bayes), the cell size and the weight are chosen on validation ACC@1; `--final` also prints the test metrics of the chosen blend.

### Ablations

Each ablation in the paper changes one entry of a final config:

| Change | Config entry |
|---|---|
| No POI vectors | `model.poi: false` (and `model.poi_drop: 0`) |
| No POI vectors or location cells | as above, and remove `model.cell_embed` |
| BiLSTM in place of the attention encoder | `model.seq_arch: "bilstm"` |
| No spatial rotation | `model.rope.space: false` |
| Cross-entropy only | `finetune.config.supcon_weight: 0` |
| Add a focal term | `finetune.config.focal_weight: 0.6` |
| User-balanced batches | `finetune.config.sampler: {"type": "m_per_user", "m": 2}` |

## License

MIT (see `LICENSE`).
