#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for iTransformer guidance model.

Hyperparameters (from paper):
- Learning rate: {1e-3, 5e-4, 1e-4}
- Number of blocks (e_layers): {2, 3, 4}
- Embedding dimension (d_model): {256, 512}

Fixed:
- Optimizer: Adam
- Loss: MSE
- Batch size: 32
- Epochs: 10
"""

import argparse
import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

import optuna
from optuna.trial import TrialState

def run_itransformer_trial(
    trial,
    root_path: str,
    data_path: str,
    data_type: str,
    target_col: str,
    checkpoint_dir: str,
    model_id: str,
    n_epochs: int = 10,
    early_stop_patience: int = 3,
):
    """Run a single iTransformer trial with suggested hyperparameters."""
    
    # Sample hyperparameters
    # Continuous log-uniform LR search from 1e-5 to 1e-3
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # Fixed e_layers=4 and d_model=512 to match pretrained DiffusionTSF's embedded guidance model
    e_layers = 4
    d_model = 512
    d_ff = d_model  # Match d_ff to d_model as is common practice
    
    trial_id = f"trial_{trial.number}"
    trial_ckpt_dir = os.path.join(checkpoint_dir, trial_id)
    os.makedirs(trial_ckpt_dir, exist_ok=True)
    
    # Build command
    # Always use 'fair' data type for gap-based splits matching DiffusionTSF
    cmd = [
        "python3", "run.py",
        "--is_training", "1",
        "--model_id", f"{model_id}_{trial_id}",
        "--root_path", root_path,
        "--data_path", data_path,
        "--data", "fair",  # Use fair splits (same as DiffusionTSF)
        "--model", "iTransformer",
        "--features", "S",
        "--target", target_col,
        "--freq", "h",
        "--checkpoints", trial_ckpt_dir,
        "--seq_len", "512",
        "--label_len", "48",
        "--pred_len", "96",
        "--e_layers", str(e_layers),
        "--d_model", str(d_model),
        "--d_ff", str(d_ff),
        "--factor", "1",
        "--enc_in", "1",
        "--dec_in", "1",
        "--c_out", "1",
        "--des", f"{model_id}_{trial_id}",
        "--loss", "MSE",
        "--lradj", "type1",
        "--gpu", "0",
        "--train_epochs", str(n_epochs),
        "--batch_size", "32",
        "--learning_rate", str(lr),
        "--patience", str(early_stop_patience),
    ]
    
    print(f"\n[Trial {trial.number}] lr={lr}, e_layers={e_layers}, d_model={d_model}")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        # Parse validation loss from output
        # iTransformer prints: "Vali Loss: X.XXXX Test Loss: X.XXXX"
        val_loss = None
        for line in result.stdout.split('\n'):
            if 'Vali Loss:' in line:
                try:
                    # Extract the validation loss value
                    parts = line.split('Vali Loss:')
                    if len(parts) > 1:
                        val_loss_str = parts[1].strip().split()[0]
                        val_loss = float(val_loss_str)
                except (ValueError, IndexError):
                    continue
        
        if val_loss is None:
            # Try to find best validation loss from metrics file
            metrics_file = os.path.join(trial_ckpt_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    val_loss = metrics.get('best_val_loss', None)
        
        if val_loss is None:
            print(f"[Trial {trial.number}] Could not parse validation loss, using large value")
            print(f"stdout: {result.stdout[-500:]}")
            val_loss = 1e10
        
        print(f"[Trial {trial.number}] Val Loss: {val_loss:.6f}")
        
        # Save trial params
        trial_params = {
            "learning_rate": lr,
            "e_layers": e_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "val_loss": val_loss,
        }
        with open(os.path.join(trial_ckpt_dir, "params.json"), 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        return val_loss
        
    except subprocess.TimeoutExpired:
        print(f"[Trial {trial.number}] Timeout!")
        return 1e10
    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        return 1e10


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for iTransformer")
    parser.add_argument("--root-path", required=True, help="Root path to data directory")
    parser.add_argument("--data-path", required=True, help="CSV filename")
    parser.add_argument("--data-type", required=True, help="Data type (ETTh1, ETTh2, custom, etc.)")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save checkpoints")
    parser.add_argument("--model-id", required=True, help="Model ID prefix")
    parser.add_argument("--n-trials", type=int, default=8, help="Number of Optuna trials")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--output-checkpoint", required=True, help="Final best checkpoint path")
    
    args = parser.parse_args()
    
    # Change to iTransformer directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    itrans_dir = os.path.join(script_dir, "models", "iTransformer")
    os.chdir(itrans_dir)
    
    print(f"\n{'='*60}")
    print(f"iTransformer Optuna Hyperparameter Tuning (FAIR MODE)")
    print(f"{'='*60}")
    print(f"Dataset: {args.root_path}/{args.data_path}")
    print(f"Target: {args.target}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs/trial: {args.n_epochs}")
    print(f"Data split: FAIR (gap-based, matches DiffusionTSF)")
    print(f"  - Train: 70% | [GAP: 608 indices] | Val: 10% | [GAP] | Test: 20%")
    print(f"Search space:")
    print(f"  - learning_rate: [1e-5, 1e-3] (log-uniform)")
    print(f"  - e_layers: 4 (fixed to match pretrained DiffusionTSF)")
    print(f"  - d_model: 512 (fixed to match pretrained DiffusionTSF)")
    print(f"{'='*60}\n")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"itransformer_{args.model_id}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3),
    )
    
    def objective(trial):
        return run_itransformer_trial(
            trial,
            root_path=args.root_path,
            data_path=args.data_path,
            data_type=args.data_type,
            target_col=args.target,
            checkpoint_dir=args.checkpoint_dir,
            model_id=args.model_id,
            n_epochs=args.n_epochs,
            early_stop_patience=args.patience,
        )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print results
    print(f"\n{'='*60}")
    print("Optuna Study Complete")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val loss: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")
    
    # Find and copy best checkpoint
    best_trial_dir = os.path.join(args.checkpoint_dir, f"trial_{study.best_trial.number}")
    best_ckpt = None
    
    # Search for checkpoint.pth in the trial directory
    for root, dirs, files in os.walk(best_trial_dir):
        for f in files:
            if f == "checkpoint.pth":
                best_ckpt = os.path.join(root, f)
                break
        if best_ckpt:
            break
    
    if best_ckpt and os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
        shutil.copy(best_ckpt, args.output_checkpoint)
        print(f"✅ Best checkpoint copied to: {args.output_checkpoint}")
        
        # Also save best params
        best_params = study.best_trial.params
        best_params['val_loss'] = study.best_trial.value
        params_file = os.path.join(os.path.dirname(args.output_checkpoint), "best_itrans_params.json")
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✅ Best params saved to: {params_file}")
    else:
        print(f"❌ Could not find checkpoint in {best_trial_dir}")
        sys.exit(1)
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

