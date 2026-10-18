# Clustered Imputation (GUIDE)

Run the following commands from this directory, with the datasets in `data/` and dependencies installed.

```bash
# Gothenburg: correlation-based clustering
python exe_gote.py --seed 42 --cluster_name gote

# Gothenburg: VAE-based clustering
python exe_gote.py --seed 42 --cluster_name gote_vae_latent_mean

# AQI-36: correlation-based clustering
python exe_aqi36.py --seed 42 --cluster_name pm25

# AQI-36: VAE-based clustering
python exe_aqi36.py --seed 42 --cluster_name pm25_vae_latent_mean

# METR-LA: VAE-based clustering, block missingness
python exe_metrla.py --seed 42 --missing_pattern block

# PEMS-BAY: VAE-based clustering, block missingness
python exe_pemsbay.py --seed 42 --missing_pattern block
```

For point missingness on METR-LA or PEMS-BAY, replace `block` with `point`. These two scripts currently select clustering through `cluster_name` inside the script, not a command-line argument.

Use seeds `38`, `39`, `40`, `41`, and `42` for five runs. Each command trains and evaluates a model; results are saved under `save/`. Defaults: GPU `cuda:0` and 100 generated samples.

For the Gothenburg evaluations, fixed test masks are encoded by missing-value positions in `data/goteborg/gote_missing.csv`. With the same data and mask settings, these masks are reused across runs, not randomly regenerated for each training seed.
