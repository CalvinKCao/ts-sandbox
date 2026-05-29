# Submitting Jobs on Alliance Canada / Slurm

This repository now uses a fully modular YAML pipeline. You no longer need to modify gigantic monolithic shell scripts (like the old `slurm_binary_anchor_92d3.sh`) to change model parameters or toggle pipeline phases.

## The New Workflow

All execution on Killarney or other Slurm clusters revolves around two files:

1. **`submit_grid.sh`**: The orchestrator script. You run this *on the login node*. It parses your requests and submits jobs for you.
2. **YAML Configs (e.g. `configs/binary_anchor.yaml`)**: The actual experiment definitions.

### Basic Usage

To run a single configuration across multiple datasets (using the default setup in the config):

```bash
# Submits jobs for ETTh1 and exchange_rate on L40S nodes
./submit_grid.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
```

### Smoke Testing

For quick 30-minute validation jobs that use minimal CPU/RAM and 1 trial/epoch to verify that your pipeline runs without crashing:

```bash
# Runs configs/smoke_test.yaml
./submit_grid.sh --smoke
```

### Grids & Chaining (Dependencies)

You can submit grids of different configs over multiple datasets/seeds:

```bash
./submit_grid.sh --configs configs/binary_anchor.yaml,configs/binary_anchor_large.yaml --datasets ETTh1,weather --seeds 42,1337
```

If you need to chain execution (e.g., waiting for a dataset preprocessing job or a pre-training job to finish):

```bash
# Say your pretrain job was ID 1234567
./submit_grid.sh --configs configs/finetune_only.yaml --datasets ETTh1 --dependency afterok:1234567
```

## Where are my logs?

The `submit_grid.sh` script is designed to capture the exact job ID and immediately tell you where your log file will be on `$SCRATCH`.

By default, logs go to:
`/scratch/ccao87/results/logs/MM-DD-{JOB_ID}-{dataset}-{config}.log`

This completely isolates your logs and W&B runs neatly so you aren't digging through a pile of generic `slurm-12345.out` files!
