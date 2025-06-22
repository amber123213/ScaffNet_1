import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        # 为ScaffNet特殊处理，loss是字符串时创建真正的损失函数
        if isinstance(loss, str) and loss == 'scaffnet_loss':
            self.loss_func = torch.nn.L1Loss()  # 使用L1Loss作为预测损失
        else:
            self.loss_func = loss
            
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # 添加结构引导损失的权重
        self.lambda_guidance = args.lambda_guidance
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                
                # --- 【最终修正版代码】 ---
                if hasattr(self.args, 'model_name') and self.args.model_name == 'ScaffNet':
                    output = self.model(data) # 在eval模式下，ScaffNet只返回一个值
                else:
                    output = self.model(data, target, teacher_forcing_ratio=0)
                
                if isinstance(output, tuple):
                    output = output[0]  # 只获取预测输出，忽略hidden_history和A_scaffold
                    
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss_func(output.to(self.args.device), label.to(self.args.device))
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            
            # --- 【最终修正版代码】 ---
            if isinstance(self.loss, str) and self.loss == 'scaffnet_loss':
                # ScaffNet返回三个值
                output, hidden_history, A_scaffold = self.model(data)
                output_rescaled = self.scaler.inverse_transform(output)
                label_rescaled = self.scaler.inverse_transform(label)
                
                # 确保维度匹配
                # output shape: [B, T*C, N, 1], label shape: [B, 1, N, 1]
                # 我们需要将output reshape为[B, 1, N, 1]以匹配label
                B, T_C, N, D = output_rescaled.shape
                output_rescaled = output_rescaled.view(B, self.args.horizon, self.args.output_dim, N, D)
                # 只使用第一个预测步骤进行损失计算
                output_rescaled = output_rescaled[:, 0:1, :, :, :]  # [B, 1, 1, N, 1]
                output_rescaled = output_rescaled.squeeze(2)  # [B, 1, N, 1]
                
                loss_pred = self.loss_func(output_rescaled.to(self.args.device), label_rescaled.to(self.args.device))
                loss_guide = self.calculate_structural_guidance_loss(hidden_history, A_scaffold)
                
                # 动态尺度平衡
                loss_pred_data = loss_pred.detach()
                loss_guide_data = loss_guide.detach()
                dynamic_lambda = loss_pred_data / (loss_guide_data + 1e-9) * self.lambda_guidance
                
                loss = loss_pred + dynamic_lambda * loss_guide
                
            else: # 原始AGCRN的逻辑
                output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss_func(output.to(self.args.device), label.to(self.args.device))
            
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                if isinstance(self.loss, str) and self.loss == 'scaffnet_loss':
                    self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}, Pred Loss: {:.6f}, Guide Loss: {:.6f}'.format(
                        epoch, batch_idx, self.train_per_epoch, loss.item(), loss_pred.item(), loss_guide.item()))
                else:
                    self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                        epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                
                # --- 【最终修正版代码】 ---
                if hasattr(args, 'model_name') and args.model_name == 'ScaffNet':
                    output = model(data) # 在eval模式下，ScaffNet只返回一个值
                else:
                    output = model(data, target, teacher_forcing_ratio=0)
                
                if isinstance(output, tuple):
                    output = output[0]  # 只获取预测输出，忽略其他返回值
                    
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

    def calculate_structural_guidance_loss(self, hidden_states_history, A_scaffold):
        if hidden_states_history is None or A_scaffold is None:
            return torch.tensor(0.0, device=self.args.device)
        degree = torch.sum(A_scaffold, dim=1)
        d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        d_matrix_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = torch.eye(A_scaffold.size(0), device=self.args.device) - torch.mm(d_matrix_inv_sqrt, A_scaffold).mm(d_matrix_inv_sqrt)
        batch_size, seq_len, num_nodes, hidden_dim = hidden_states_history.shape
        H = hidden_states_history.view(-1, num_nodes, hidden_dim)
        H_H_T = torch.matmul(H, H.transpose(1, 2))
        loss = torch.einsum('ij,bij->b', normalized_laplacian, H_H_T).sum()
        return loss / (batch_size * seq_len)