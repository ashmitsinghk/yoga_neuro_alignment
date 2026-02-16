# Docker Quick Reference

## Building and Running

```bash
# Build the image
docker-compose build

# Start container in background
docker-compose up -d

# Enter the container
docker-compose exec yoga-neuro-alignment /bin/bash

# Or use the convenience script
./docker-run.sh
```

## Inside the Container

```bash
# Run full pipeline
./scripts/run_full_pipeline.sh

# Run individual steps
python training/run_dapt.py
python training/run_alignment.py
python training/statistical_evaluation.py
python training/visualize_embeddings.py

# Reproduce results only (skip training)
./scripts/reproduce_results.sh
```

## Managing the Container

```bash
# Stop the container
docker-compose down

# View logs
docker-compose logs -f

# Restart the container
docker-compose restart

# Remove everything (including volumes)
docker-compose down -v
```

## GPU Support

### Requirements
- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU drivers

### Verify GPU access inside container
```bash
docker-compose exec yoga-neuro-alignment nvidia-smi
```

### CPU-only mode
1. Remove the `deploy` section from `docker-compose.yml`
2. Change `device: cuda` to `device: cpu` in:
   - `experiments/configs/dapt_config.yaml`
   - `experiments/configs/alignment_config.yaml`

## Volumes

The container mounts:
- `.` → `/app` (project directory)
- `huggingface-cache` → `/app/.cache/huggingface` (model cache, persisted)

All results are saved to your local filesystem via the volume mount.

## Troubleshooting

### Out of memory
Reduce batch size in config files:
```yaml
batch_size: 8  # or 4
```

### Permission issues
```bash
# Fix ownership (run on host)
sudo chown -R $USER:$USER .
```

### Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```
