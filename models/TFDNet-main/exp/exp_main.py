from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear,TFDNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from torch.optim import lr_scheduler
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import sys
import time

# Add DILATE loss path to system path
dilate_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'losses', 'DILATE-master')
if dilate_path not in sys.path:
    sys.path.insert(0, dilate_path)

# Import DILATE loss modules
try:
    from loss import soft_dtw
    from loss import path_soft_dtw
except ImportError:
    pass # Will handle in DilateLoss class if needed or assume it works if path is correct

class DilateLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.01, device='cuda'):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
    def forward(self, outputs, targets):
        batch_size, N_output, n_features = outputs.shape
        loss_shape = 0
        loss_temporal = 0
        for feature_idx in range(n_features):
            outputs_feat = outputs[:, :, feature_idx:feature_idx+1]
            targets_feat = targets[:, :, feature_idx:feature_idx+1]
            D = torch.zeros((batch_size, N_output, N_output)).to(self.device)
            for k in range(batch_size):
                Dk = soft_dtw.pairwise_distances(
                    targets_feat[k, :, :].view(-1, 1),
                    outputs_feat[k, :, :].view(-1, 1)
                )
                D[k:k+1, :, :] = Dk
            softdtw_batch = soft_dtw.SoftDTWBatch.apply
            loss_shape += softdtw_batch(D, self.gamma)
            path_dtw = path_soft_dtw.PathDTWBatch.apply
            path = path_dtw(D, self.gamma)
            Omega = soft_dtw.pairwise_distances(
                torch.arange(1, N_output + 1).view(N_output, 1).float()
            ).to(self.device)
            loss_temporal += torch.sum(path * Omega) / (N_output * N_output)
        loss_shape = loss_shape / (batch_size * n_features)
        loss_temporal = loss_temporal / (batch_size * n_features)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class MixtureLoss(nn.Module):
    def __int__(self):
        super(MixtureLoss, self).__init__()

    def forward(self,pred,true):
        criterion =torch.mean((F.tanh(torch.abs(pred - true))) *torch.abs(pred-true) +(1 - F.tanh(torch.abs(pred - true))) * torch.square(pred-true))
        return criterion

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'TFDNet':TFDNet,
        }
        if self.args.use_dilate and self.args.model == 'TFDNet':
            self.args.output_seasonal = True
            
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)


        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='mse':
            criterion = nn.MSELoss()
        elif self.args.loss=='Mixtureloss':
            criterion = MixtureLoss()
        return criterion




    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                        outputs = self.model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)


                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_dilate:
            dilate_criterion = DilateLoss(self.args.dilate_alpha, self.args.dilate_gamma, self.device)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.CosineAnnealingLR( model_optim, T_max = self.args.train_epochs)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                            pred_seasonal = None
                            if isinstance(outputs, tuple):
                                outputs, pred_seasonal = outputs
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_sliced = batch_y[:, -self.args.pred_len:, f_dim:]
                        # batch_y_trend=torch.mean(batch_y[:, -self.args.pred_len:, :].reshape(batch_y.shape[0],4,int(self.args.pred_len/4),-1),dim=2).squeeze(dim=2).to(self.device)
                        loss = criterion(outputs, batch_y_sliced)
                        
                        if self.args.use_dilate and pred_seasonal is not None:
                            if isinstance(self.model, nn.DataParallel):
                                decomp = self.model.module.decompsition
                            else:
                                decomp = self.model.decompsition
                            seasonal_y_full, _ = decomp(batch_y)
                            seasonal_y = seasonal_y_full[:, -self.args.pred_len:, f_dim:]
                            pred_seasonal = pred_seasonal[:, :, f_dim:]
                            loss_dilate = dilate_criterion(pred_seasonal, seasonal_y)
                            loss = loss + self.args.dilate_weight * loss_dilate

                        train_loss.append(loss.item())
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                        outputs = self.model(batch_x)
                        pred_seasonal = None
                        if isinstance(outputs, tuple):
                            outputs, pred_seasonal = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_sliced = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y_sliced)
                    
                    if self.args.use_dilate and pred_seasonal is not None:
                        if isinstance(self.model, nn.DataParallel):
                            decomp = self.model.module.decompsition
                        else:
                            decomp = self.model.decompsition
                        seasonal_y_full, _ = decomp(batch_y)
                        seasonal_y = seasonal_y_full[:, -self.args.pred_len:, f_dim:]
                        pred_seasonal = pred_seasonal[:, :, f_dim:]
                        loss_dilate = dilate_criterion(pred_seasonal, seasonal_y)
                        loss = loss + self.args.dilate_weight * loss_dilate
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                scheduler.step()
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def save(self,setting,test=0):
        test_data, test_loader = self._get_data(flag = 'test')
        y_save = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            y_save.append(batch_y)
        y_save = np.concatenate(y_save, axis = 0)
        data_name = self.args.model_id + 'pred.npy'
        folder_path = './test_data/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + data_name, y_save)
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        preds_seasonal = []
        trues_seasonal = []
        trues_trend = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Model forward pass
                pred_seasonal = None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs, pred_seasonal = outputs
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                            if isinstance(outputs, tuple):
                                outputs, pred_seasonal = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Decompose ground truth
                if isinstance(self.model, nn.DataParallel):
                    decomp = self.model.module.decompsition
                else:
                    decomp = self.model.decompsition
                
                # Move batch_y to device for decomposition
                batch_y_device = batch_y.to(self.device)
                seasonal_y_full, trend_y_full = decomp(batch_y_device)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                # Slicing
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Collect components
                if pred_seasonal is not None:
                    pred_seasonal = pred_seasonal[:, -self.args.pred_len:, f_dim:]
                    preds_seasonal.append(pred_seasonal.detach().cpu().numpy())
                
                seasonal_y = seasonal_y_full[:, -self.args.pred_len:, f_dim:]
                trend_y = trend_y_full[:, -self.args.pred_len:, f_dim:]
                trues_seasonal.append(seasonal_y.detach().cpu().numpy())
                trues_trend.append(trend_y.detach().cpu().numpy())

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        inputx = np.concatenate(inputx, axis = 0)
        
        if len(preds_seasonal) > 0:
            preds_seasonal = np.concatenate(preds_seasonal, axis=0)
            np.save(folder_path + 'pred_seasonal.npy', preds_seasonal)
            
        trues_seasonal = np.concatenate(trues_seasonal, axis=0)
        trues_trend = np.concatenate(trues_trend, axis=0)
        
        np.save(folder_path + 'true_seasonal.npy', trues_seasonal)
        np.save(folder_path + 'true_trend.npy', trues_trend)
        np.save(folder_path + 'x.npy', inputx)

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,trend= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
