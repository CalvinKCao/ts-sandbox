from data_provider.dataset_mts import Dataset_MTS
from exp.exp_basic import Exp_Basic
from einops import rearrange

from utils.tools import EarlyStopping, adjust_learning_rate
from metrics.prob_metrics import probabilistic_metrics
from metrics.deterministic_metrics import deterministic_metrics
from metrics.multi_mode_metrics import multi_mode_metrics

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

class _Ema:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, avg in self.shadow.items():
            avg.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> dict:
        state = model.state_dict()
        backup = {key: state[key].detach().clone() for key in self.shadow}
        for key, avg in self.shadow.items():
            state[key].copy_(avg)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict) -> None:
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


from models.backbone_loss_model import BackboneLossModel
from models.backbones.decoder_only_transformer import DecoderOnlyTransformer
from models.loss_funcs.mmpd.mmpd_loss import MMPD_Loss

#custom backbones and loss functions
from models.backbones.encoder_decoder_transformer import EncoderDecoderTransformer
from models.backbones.mask_ae_transformer import MaskAETransformer
from models.loss_funcs.distribution.gaussian_loss import Gaussian_loss

from exp.normalization import get_statistics, normalize, denormalize

backbone_map ={
    'Decoder': DecoderOnlyTransformer,
    'EncoderDecoder': EncoderDecoderTransformer,
    'MaskAE': MaskAETransformer,
}

loss_map = {
    'MMPD': MMPD_Loss,
    'Gauss': Gaussian_loss,
}

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
    
    def _build_model(self):      
        backbone = backbone_map[self.args.backbone](self.args)
        loss_func = loss_map[self.args.loss_func](self.args)

        model = BackboneLossModel(backbone, loss_func).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim
    
    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        #permute the data to [batch_size, data_dim, seq_len]
        batch_x = rearrange(batch_x, 'b l d -> b d l')
        batch_y = rearrange(batch_y, 'b l d -> b d l')

        #normalize
        x_shift, x_scale = get_statistics(batch_x)
        normed_x = normalize(batch_x, x_shift, x_scale)
        normed_y = normalize(batch_y, x_shift, x_scale)

        batch_loss = self.model.compute_loss(normed_x, normed_y, point_weight=self.args.point_weight) # [batch_size, data_dim]
        
        #return the loss by std, align with previous works
        if self.args.weighted:
            weighted_loss = (x_scale**2) * batch_loss
        else:
            weighted_loss = batch_loss

        return weighted_loss.mean()

    def vali(self, vali_data, vali_loader):
        self.model.eval()

        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                loss = self._process_one_batch(vali_data, batch_x, batch_y)
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.output_root, 'checkpoints', '{}-{}'.format(self.args.backbone, self.args.loss_func), setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        if getattr(train_data, "scaler", None) is not None:
            scale_statistic = {'mean': train_data.scaler.mean_.tolist(), 'std': train_data.scaler.scale_.tolist()}
            with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
                pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        smoke_max_batches = int(os.environ.get("MMPD_SMOKE_MAX_TRAIN_BATCHES", "0") or "0")
        
        model_optim = self._select_optimizer()
        ema_decay = float(getattr(self.args, "ema_decay", 0.0) or 0.0)
        ema = _Ema(self.model, ema_decay) if ema_decay > 0.0 else None

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()

            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                if smoke_max_batches and i >= smoke_max_batches:
                    break
                iter_count += 1
                
                model_optim.zero_grad()
                loss = self._process_one_batch(train_data, batch_x, batch_y)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                if ema is not None:
                    ema.update(self.model)
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            ema_backup = ema.swap_in(self.model) if ema is not None else None
            vali_loss = self.vali(vali_data, vali_loader)
            if ema_backup is not None:
                ema.restore(self.model, ema_backup)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            if ema is not None:
                ema_backup = ema.swap_in(self.model)
                early_stopping(vali_loss, self.model, path)
                ema.restore(self.model, ema_backup)
            else:
                early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'model_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'model_checkpoint.pth')
        
        return self.model

    def test(self, setting, inverse = False, test_batch_num = 5):
        path = os.path.join(self.args.output_root, 'checkpoints', '{}-{}'.format(self.args.backbone, self.args.loss_func), setting)
        best_model_path = path+'/'+'model_checkpoint.pth'
        trained_model_dict = torch.load(best_model_path, map_location='cpu')
        
        model_state_dict = self.model.state_dict()
        for k, v in trained_model_dict.items():
            if 'gen_diffusion' not in k:
                model_state_dict[k] = v
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')

        instance_num = 0

        #probabilistic metrics
        CRPS_agg = 0
        timepoints_agg = 0
        topK = 5
        topK_mse_agg = torch.zeros(topK).to(self.device)
        topK_mae_agg = torch.zeros(topK).to(self.device)

        #deterministic metrics
        mse_agg = 0
        mae_agg = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_size, seq_len, data_dim = batch_x.shape

                instance_num += batch_size

                batch_x = rearrange(batch_x, 'b l d -> b d l')
                batch_y = rearrange(batch_y, 'b l d -> b d l')
                #normalize
                x_shift, x_scale = get_statistics(batch_x)
                normed_x = normalize(batch_x, x_shift, x_scale)
                
                deterministic_pred, multi_mode_pred, prob_samples = self.model.predict(normed_x, prob_pred=self.args.prob_pred, 
                        sample_num = self.args.sample_num, temperature = self.args.temperature, \
                        gmm=True, gmm_components=self.args.gmm_components, prior_pi_decay=self.args.prior_pi_decay, prior_precision_shape=self.args.prior_precision_shape, \
                        gmm_iterations=self.args.gmm_iterations)
                
                #probabilistic prediction
                if prob_samples is not None:
                    original_scale_pred_samples = denormalize(prob_samples, x_shift, x_scale) # [batch_size, data_dim, sample_num, seq_len]
                    crps = probabilistic_metrics(batch_y, original_scale_pred_samples) # [batch_size, data_dim, seq_len]
                    CRPS_agg += crps.sum().item()

                if multi_mode_pred is not None:
                    original_scale_multi_mode_pred = denormalize(multi_mode_pred['mode_center'], x_shift, x_scale) # [batch_size, data_dim, mode_num, seq_len]
                    multi_mode_preds = {'mode_prob': multi_mode_pred['mode_prob'], 'mode_center': original_scale_multi_mode_pred}
                    for k in range(topK):
                        topK_mse, topK_mae = multi_mode_metrics(batch_y, multi_mode_preds, K=k+1)
                        topK_mse_agg[k] += topK_mse.sum().item(); topK_mae_agg[k] += topK_mae.sum().item()

                if deterministic_pred is not None:
                    original_scale_deterministic_pred = denormalize(deterministic_pred, x_shift, x_scale)
                    mse, mae = deterministic_metrics(batch_y, original_scale_deterministic_pred)
                    mse_agg += mse.sum().item(); mae_agg += mae.sum().item()
                
                timepoints_agg += batch_y.shape[0] * batch_y.shape[1] * batch_y.shape[2]
                
                #only evaluate the some batches
                if i == test_batch_num: 
                    break

        CRPS = CRPS_agg / timepoints_agg
        topK_mse = topK_mse_agg / timepoints_agg
        topK_mae = topK_mae_agg / timepoints_agg
    
        MSE = mse_agg / timepoints_agg
        MAE = mae_agg / timepoints_agg

        # result save
        folder_path = os.path.join(self.args.output_root, 'results', '{}-{}'.format(self.args.backbone, self.args.loss_func), setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #visualize the results in a Table in command line
        print('test instance num:', instance_num)
        print('\n')

        print('CRPS:', CRPS)
        print('\n')

        for k in range(topK):
            print('top'+str(k+1)+'_mse:', topK_mse[k].item())
            print('top'+str(k+1)+'_mae:', topK_mae[k].item())
        print('\n')

        print('MSE:', MSE)
        print('MAE:', MAE)

        save_file = os.path.join(folder_path, 'metrics.txt')
        with open(save_file, 'w') as f:
            f.write(setting+'\n')
            f.write('test instance num: '+str(instance_num)+'\n')
            f.write('\n')

            f.write('CRPS: '+str(CRPS)+'\n')
            f.write('\n')

            for k in range(topK):
                f.write('top'+str(k+1)+' MSE: '+str(topK_mse[k].item())+'\n')
                f.write('top'+str(k+1)+' MAE: '+str(topK_mae[k].item())+'\n')
            f.write('\n')

            f.write('MSE: '+str(MSE)+'\n')
            f.write('MAE: '+str(MAE)+'\n')

        return
