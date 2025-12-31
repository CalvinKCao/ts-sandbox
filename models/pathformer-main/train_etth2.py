"""
Train Pathformer on ETTh2 Dataset with Optional DILATE Loss
"""

import torch
import numpy as np
import random
import argparse
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_main import Exp_Main
from dilate_loss_wrapper import CombinedLoss

fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class Exp_Main_With_Dilate(Exp_Main):
    """Extended Exp_Main with DILATE loss support"""
    
    def __init__(self, args):
        super().__init__(args)
        
    def _select_criterion(self):
        """Select loss function based on args"""
        if hasattr(self.args, 'loss_type') and self.args.loss_type == 'dilate':
            print(f"Using DILATE Loss (alpha={self.args.dilate_alpha}, gamma={self.args.dilate_gamma})")
            criterion = CombinedLoss(
                loss_type='dilate',
                alpha=self.args.dilate_alpha,
                gamma=self.args.dilate_gamma,
                device=self.device
            )
        elif hasattr(self.args, 'loss_type') and self.args.loss_type == 'fft_dilate':
            freq_thresh = getattr(self.args, 'freq_threshold', 80.0)
            print(f"Using FFT-DILATE Loss (alpha={self.args.dilate_alpha}, gamma={self.args.dilate_gamma}, "
                  f"freq_threshold={freq_thresh}%)")
            criterion = CombinedLoss(
                loss_type='fft_dilate',
                alpha=self.args.dilate_alpha,
                gamma=self.args.dilate_gamma,
                freq_threshold=freq_thresh,
                device=self.device
            )
        elif hasattr(self.args, 'loss_type') and self.args.loss_type == 'mse':
            print("Using MSE Loss")
            criterion = CombinedLoss(loss_type='mse', device=self.device)
        else:
            print("Using MAE Loss (default)")
            criterion = CombinedLoss(loss_type='mae', device=self.device)
        return criterion
    
    def _compute_loss(self, outputs, batch_y, criterion):
        """Compute loss and handle DILATE/FFT-DILATE loss special cases"""
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        
        if hasattr(self.args, 'loss_type'):
            if self.args.loss_type == 'dilate':
                loss, loss_shape, loss_temporal = criterion(outputs, batch_y)
                return loss, loss_shape, loss_temporal
            elif self.args.loss_type == 'fft_dilate':
                loss, low_freq_loss, high_freq_loss, freq_info = criterion(outputs, batch_y)
                # Return in similar format for compatibility
                return loss, low_freq_loss, high_freq_loss
        
        # Default case
        loss = criterion(outputs, batch_y)
        return loss, None, None
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        
        from utils.tools import EarlyStopping, adjust_learning_rate
        from torch.optim import lr_scheduler
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_shape_loss = []
            train_temporal_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'PathFormer':
                            outputs, balance_loss = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)
                        
                        loss, loss_shape, loss_temporal = self._compute_loss(outputs, batch_y, criterion)
                        
                        if self.args.model == "PathFormer":
                            loss = loss + balance_loss
                else:
                    if self.args.model == 'PathFormer':
                        outputs, balance_loss = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                    
                    loss, loss_shape, loss_temporal = self._compute_loss(outputs, batch_y, criterion)
                    
                    if self.args.model == "PathFormer":
                        loss = loss + balance_loss

                train_loss.append(loss.item())
                if loss_shape is not None:
                    train_shape_loss.append(loss_shape.item())
                    train_temporal_loss.append(loss_temporal.item())

                if (i + 1) % 100 == 0:
                    if loss_shape is not None:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | "
                              f"loss: {loss.item():.7f} (shape: {loss_shape.item():.7f}, "
                              f"temporal: {loss_temporal.item():.7f})")
                    else:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            
            # Use simple criterion for validation (MAE)
            simple_criterion = torch.nn.L1Loss()
            vali_loss = self.vali(vali_data, vali_loader, simple_criterion)
            test_loss = self.vali(test_data, test_loader, simple_criterion)

            if train_shape_loss:
                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                      f"Train Loss: {train_loss:.7f} (shape: {np.average(train_shape_loss):.7f}, "
                      f"temporal: {np.average(train_temporal_loss):.7f}) | "
                      f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            else:
                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                      f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pathformer for ETTh2 Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer', help='model name')
    parser.add_argument('--model_id', type=str, default='ETTh2')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../datasets/ETT-small/', 
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate individually')

    # model parameters
    parser.add_argument('--d_model', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=7, help='number of features in ETTh2')
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, 
                        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')
    parser.add_argument('--batch_norm', type=int, default=0)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', 
                        default=False)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    # loss function
    parser.add_argument('--loss_type', type=str, default='mae', 
                        help='loss function type: mae, mse, dilate, or fft_dilate')
    parser.add_argument('--dilate_alpha', type=float, default=0.5,
                        help='DILATE loss alpha (weight for shape vs temporal)')
    parser.add_argument('--dilate_gamma', type=float, default=0.01,
                        help='DILATE loss gamma (smoothing parameter)')
    parser.add_argument('--freq_threshold', type=float, default=80.0,
                        help='FFT-DILATE frequency threshold percentile (0-100). Higher = fewer high-freq components')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main_With_Dilate

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            loss_suffix = f"_{args.loss_type}"
            if args.loss_type == 'dilate':
                loss_suffix += f"_a{args.dilate_alpha}_g{args.dilate_gamma}"
            
            setting = '{}_{}_ft{}_sl{}_pl{}_{}{}'.format(
                args.model_id,
                args.model,
                args.features,
                args.seq_len,
                args.pred_len,
                ii,
                loss_suffix
            )

            exp = Exp(args)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.do_predict:
                print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        loss_suffix = f"_{args.loss_type}"
        if args.loss_type == 'dilate':
            loss_suffix += f"_a{args.dilate_alpha}_g{args.dilate_gamma}"
        
        setting = '{}_{}_ft{}_sl{}_pl{}_{}{}'.format(
            args.model_id,
            args.model,
            args.features,
            args.seq_len,
            args.pred_len,
            ii,
            loss_suffix
        )

        exp = Exp(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
