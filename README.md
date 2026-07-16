# Run

Prerequisites:
- `data/experiments/{experiment}/embeddings/embeddings.parquet`
- split metadata under `data/shared/`

Run the full pipeline:

```bash
uv run src/run_all_results.py
```

Run one experiment only:

```bash
uv run src/run_all_results.py B3_ECAPA
```
