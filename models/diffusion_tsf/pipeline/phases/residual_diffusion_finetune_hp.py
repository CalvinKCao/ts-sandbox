"""Phase 3: Residual Diffusion Finetune HP.

Loads Phase 2 model to precompute predictions, then trains a new diffusion model
from the Phase 1b pretrain checkpoint to predict the residual (ground truth - Phase 2 prediction).
"""

from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader
from optuna import create_study
from optuna.samplers import TPESampler

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)

class ResidualDatasetWrapper(Dataset):
    """Wraps a standard dataset and precomputes Phase 2 predictions."""
    def __init__(self, base_dataset, phase2_model, device, batch_size=256, smoke_test=False):
        self.base_dataset = base_dataset
        self.device = device
        
        logger.info(f"Precomputing Phase 2 forecasts for residual training (N={len(base_dataset)})...")
        loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        phase2_model.eval()
        phase2_preds = []
        
        with torch.no_grad():
            for past, future in loader:
                past_t = past.to(device)
                
                # Use deterministic anchor or DDIM
                # Since we just want the Phase 2 base prediction quickly, we use Anchor or a fast sampler
                # Actually, if Phase 2 was trained with anchor, we should evaluate with it.
                res = phase2_model.generate(
                    past_t,
                    sampler="anchor",
                )
                pred = res['prediction_global_norm'].cpu() # (B, V, F)
                phase2_preds.append(pred)
                if smoke_test:
                    break
                    
        self.phase2_preds = torch.cat(phase2_preds, dim=0)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        past, future = self.base_dataset[idx]
        phase2_pred = self.phase2_preds[idx]
        
        # future has shape (V, 104) where the first K elements are overlap.
        # phase2_pred has shape (V, 96) because model.generate returns only the forecast.
        # We need to yield a residual of shape (V, 104). The first K elements
        # don't matter because DiffusionTSF.forward drops them anyway.
        residual = future.clone()
        K = future.shape[1] - phase2_pred.shape[1]
        if K > 0:
            residual[:, K:] = future[:, K:] - phase2_pred
        else:
            residual = future - phase2_pred
            
        return past, residual

class ResidualDiffusionFinetuneHPPhase(PipelinePhase):
    name = "residual_diffusion_finetune_hp"

    def should_skip(self, state: PipelineState) -> bool:
        subset_id = state.subset_id or state.dataset
        best_pt = os.path.join(state.checkpoint_dir, subset_id, "residual_best.pt")
        meta = os.path.join(state.checkpoint_dir, subset_id, "residual_metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            logger.info(f"  [{self.name}] cached: {best_pt}")
            state.diffusion_residual_finetune_ckpt = best_pt
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        import optuna
        from models.diffusion_tsf.train_multivariate_pipeline import (
            _promote_best_trial_to_final,
            load_dataset,
            load_itransformer_from_checkpoint,
            select_diffusion_batch_size,
            generate_dataset_job,
            diffusion_probe_max_candidate,
            create_diffusion_model,
            load_diffusion_state_keep_attached_guidance,
            anchor_kwargs_from_params,
            save_checkpoint,
            EarlyStopping,
            amp_context,
            FINETUNE_HP_LR_MIN,
            FINETUNE_HP_LR_MAX,
            FINETUNE_BATCH_SIZES,
            diffusion_arch_config_dict,
        )
        from models.diffusion_tsf.guidance import iTransformerGuidance
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        _patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

        device = state.resolve_device()

        # Require predecessors
        ft_itrans_ckpt = state.itrans_finetune_ckpt
        phase1b_ckpt = state.diffusion_pretrain_ckpt
        phase2_ckpt = state.diffusion_finetune_ckpt
        
        if not ft_itrans_ckpt or not os.path.exists(ft_itrans_ckpt):
            raise RuntimeError(f"Phase 3 requires Phase 2A iTransformer: {ft_itrans_ckpt}")
        if not phase1b_ckpt or not os.path.exists(phase1b_ckpt):
            raise RuntimeError(f"Phase 3 requires Phase 1B pretrain: {phase1b_ckpt}")
        if not phase2_ckpt or not os.path.exists(phase2_ckpt):
            raise RuntimeError(f"Phase 3 requires Phase 2B model: {phase2_ckpt}")

        n_iv = len(variate_indices)

        # 1. Load Phase 2 model
        ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        ft_itrans_guidance = iTransformerGuidance(ft_itrans_model)
        
        phase2_model = create_diffusion_model(n_variates=n_iv, **anchor_kwargs_from_params(state.finetune_best_params or {})).to(device)
        phase2_model.set_guidance_model(ft_itrans_guidance)
        p2_ckpt = torch.load(phase2_ckpt, map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(phase2_model, p2_ckpt['model_state_dict'])

        # 2. Load dataset and wrap to precompute residuals
        train_ds, val_ds, _, norm_stats = load_dataset(
            state.dataset, variate_indices,
            stride=state.window_stride, test_stride=1,
        )
        
        if state.smoke_test:
            from torch.utils.data import Subset
            train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

        residual_train_ds = ResidualDatasetWrapper(train_ds, phase2_model, device, smoke_test=state.smoke_test)
        residual_val_ds = ResidualDatasetWrapper(val_ds, phase2_model, device, smoke_test=state.smoke_test)

        # Clean up Phase 2 model to save memory during Phase 3 HP tuning
        del phase2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ft_diff_bs = select_diffusion_batch_size(
            phase_name=f"Diff Residual HP ({subset_id})",
            dataset=residual_train_ds,
            device=device,
            itrans_guidance=ft_itrans_guidance,
            max_candidate=diffusion_probe_max_candidate(n_iv, state.smoke_test),
            smoke_test=state.smoke_test,
        )

        n_trials = self.get("n_trials", 5)
        max_epochs = self.get("max_epochs", 20)
        patience_val = self.get("patience", 15)
        if state.smoke_test:
            n_trials = 1
            max_epochs = 1
            patience_val = 1

        subset_dir = os.path.join(state.checkpoint_dir, subset_id)
        os.makedirs(subset_dir, exist_ok=True)
        
        def residual_objective(trial):
            lr = trial.suggest_float('learning_rate', FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True)
            batch_size = trial.suggest_categorical('batch_size', [2, 4] if state.smoke_test else FINETUNE_BATCH_SIZES)
            
            # Phase 3 model initialized from Phase 1b pretrain
            model = create_diffusion_model(n_variates=n_iv, **anchor_kwargs_from_params()).to(device)
            model.set_guidance_model(ft_itrans_guidance)
            model.config.is_residual_model = True  # Tell model to NOT shift future by mean!
            
            p1b_ckpt = torch.load(phase1b_ckpt, map_location=device, weights_only=False)
            load_diffusion_state_keep_attached_guidance(model, p1b_ckpt['model_state_dict'])

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            
            train_loader = DataLoader(residual_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(residual_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            
            early_stop = EarlyStopping(patience=patience_val)
            best_val_loss = float('inf')
            
            for epoch in range(1, max_epochs + 1):
                model.train()
                train_loss = 0.0
                for past, future in train_loader:
                    past, future = past.to(device), future.to(device)
                    optimizer.zero_grad()
                    with amp_context():
                        loss = model.get_loss(past, future)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= max(1, len(train_loader))
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for past, future in val_loader:
                        past, future = past.to(device), future.to(device)
                        with amp_context():
                            loss = model.get_loss(past, future)
                        val_loss += loss.item()
                val_loss /= max(1, len(val_loader))
                
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trial_best_path = os.path.join(subset_dir, f"_diff_residual_trial_{trial.number}_best.pt")
                    ckpt_config = diffusion_arch_config_dict()
                    ckpt_config.update({
                        'tuned_params': {'learning_rate': lr, 'batch_size': batch_size},
                        'trial_number': trial.number,
                    })
                    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, ckpt_config, trial_best_path)
                    
                if early_stop(val_loss):
                    break
                    
            return best_val_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(
            residual_objective,
            n_trials=n_trials,
            show_progress_bar=False,
            catch=(ValueError,),
        )
        if study.best_trial is None:
            logger.warning(f"All HP trials failed for {subset_id}")
            return state

        tuned_params = study.best_params
        tuned_params["batch_size"] = ft_diff_bs

        # Custom logic to promote the best residual trial
        subset_info = {"subset_id": subset_id, "variate_indices": variate_indices}
        
        best_trial_path = os.path.join(subset_dir, f"_diff_residual_trial_{study.best_trial.number}_best.pt")
        final_ckpt_path = os.path.join(subset_dir, "residual_best.pt")
        
        import shutil, json
        if os.path.exists(best_trial_path):
            shutil.copy2(best_trial_path, final_ckpt_path)
            # Cleanup other trials
            for fn in os.listdir(subset_dir):
                if fn.startswith("_diff_residual_trial_") and fn.endswith("_best.pt"):
                    try:
                        os.remove(os.path.join(subset_dir, fn))
                    except OSError:
                        pass
                        
            # Save metadata
            meta_path = os.path.join(subset_dir, "residual_metadata.json")
            meta_data = {
                "subset_info": subset_info,
                "dataset_name": state.dataset,
                "batch_size": ft_diff_bs,
                "val_loss": study.best_value,
                "learning_rate": study.best_params.get("learning_rate"),
                "smoke_test": state.smoke_test,
                "norm_stats": {k: v.tolist() for k, v in norm_stats.items()},
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
            state.diffusion_residual_finetune_ckpt = final_ckpt_path
            state.residual_finetune_best_params = tuned_params

            wandb_utils.log_summary({
                "hp/residual_ft_best_val_loss": study.best_value,
                "hp/residual_ft_best_trial": study.best_trial.number,
                "hp/residual_ft_best_lr": tuned_params.get("learning_rate"),
            })
            
        else:
            logger.error(f"Best trial path {best_trial_path} not found!")

        return state
