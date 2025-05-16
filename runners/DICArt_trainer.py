import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from tqdm import tqdm
import wandb
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
from networks.gf_algorithms.discrete_number import *
from networks.d3pmnet import D3PM,D3PM_Flow,D3PM_Guassion_Flow

class D3PMTest:

    def __init__(self, cfg, trans_stats=None,pretrained_path=None):
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats  
        self.model = D3PM(
            cfg, 
            num_bins=360,  
            T=1000,  
            device=self.device
        ).to(self.device)
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        

    def test_step(self, batch_sample):
        self.model.eval()
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        true_trans = batch_sample['zero_mean_gt_pose'][:, -3:]
        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
    
        with torch.no_grad():
            pred = self.model.sample(pts_feat)
            pred_rot_bins = pred[:, :3]
            pred_trans_bins = pred[:, 3:6]  
            pred_euler_angles = euler_angles_from_bins(pred_rot_bins , 360)
            pred_euler_matrix = pytorch3d.transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
            pred_spherical_angles = bins_to_angles(pred_rot_bins, 360, 360, 360)
            pred_matrix = spherical_angles_to_matrix(pred_spherical_angles)
            pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, 360)
            diff_angle = rot_diff_degree(rot_matrix,pred_euler_matrix)
            diff_trans = torch.norm(pred_trans - true_trans, p=2, dim=-1)
            return diff_angle, diff_trans


class D3PMTrainer:
    def __init__(self, cfg, trans_stats=None,pretrained_path=None):
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats  
        self.model = D3PM(
            cfg, 
            num_bins=360,  
            T=1000,  
            device=self.device
        ).to(self.device)
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        # Optimizer & Scheduler
        self.optimizer = torch.optim.RAdam(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[

                torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    factor=1.0,  
                    total_iters=int(cfg.n_epochs * 0.4) 
                ),

                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(cfg.n_epochs * 0.6),  
                    eta_min=1e-6 
                )
            ],
            milestones=[int(cfg.n_epochs * 0.4)]  # 中间切换点
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=cfg.n_epochs,
        #     eta_min=2e-6
        # )

    def train_step(self, batch_sample):
        self.model.train()
        
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:]

        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)
        euler_angles = pytorch3d.transforms.matrix_to_euler_angles(rot_matrix, "ZYX")
        spherical_angles = matrix_to_spherical_angles(rot_matrix)
        discretized_angles = get_bin_index(spherical_angles, 360, 360, 360)
        discretized_euler = discretize_euler_angles(euler_angles,360)
        trans_bins = translation_to_bins(trans_part, self.trans_stats, 360)

        gt_pose = torch.cat([discretized_euler, trans_bins], dim=1)
        y = gt_pose.to(self.device)
        
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        loss,loss_description = self.model.loss(y, pts_feat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(),loss_description

    def eval_step(self, batch_sample):
        self.model.eval()
        
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        true_trans = batch_sample['zero_mean_gt_pose'][:, -3:]

        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        
        try:
            with torch.no_grad():
               
                pred = self.model.sample(pts_feat, steps=1000)
           
                pred_rot_bins = pred[:, :3]
                pred_trans_bins = pred[:, 3:6]  
                pred_euler_angles = euler_angles_from_bins(pred_rot_bins , 360)
                pred_euler_matrix = pytorch3d.transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
                pred_spherical_angles = bins_to_angles(pred_rot_bins, 360, 360, 360)
                pred_matrix = spherical_angles_to_matrix(pred_spherical_angles)
                pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, 360)
                diff_angle = rot_diff_degree(rot_matrix,pred_euler_matrix).mean().item()
                diff_trans = torch.norm(pred_trans - true_trans, p=2, dim=-1).mean().item()
                return diff_angle, diff_trans
            
        except Exception as e:
            print(f"Error in eval_step: {e}")
            return float('inf'), float('inf')  


def train_data(cfg, train_loader, val_loader, test_loader,trans_stats):

    trainer = D3PMTrainer(cfg, trans_stats,pretrained_path=None)
    wandb.watch(trainer.model, log="all", log_freq=100)
    save_dir = "D3PM_6D"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(cfg.n_epochs):

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        train_losses = []
        
        for batch in pbar:
            batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
            loss,loss_description = trainer.train_step(batch)
            train_losses.append(loss)
            
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",  
                "Details": loss_description   
            })
            
        avg_train_loss = np.mean(train_losses)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
        
        trainer.model.eval()
        eval_freq = cfg.eval_freq
        if (epoch + 1) % eval_freq == 0:
            val_diff_angles = []
            val_diff_trans = []
            
            print(f"Running full validation on epoch {epoch}...")
            for val_batch in tqdm(val_loader, desc=f"Epoch {epoch} Full Validation"):
                val_batch = process_batch(val_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                try:
                    diff_angle, diff_trans = trainer.eval_step(val_batch)
                    if diff_angle != float('inf') and diff_trans != float('inf'):
                        val_diff_angles.append(diff_angle)
                        val_diff_trans.append(diff_trans)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
            
            val_mean_angle_diff = sum(val_diff_angles) / len(val_diff_angles) if val_diff_angles else float('inf')
            val_mean_trans_diff = sum(val_diff_trans) / len(val_diff_trans) if val_diff_trans else float('inf')
            
            print(f"Epoch {epoch} Validation Mean Angle Difference: {val_mean_angle_diff:.4f} degrees")
            print(f"Epoch {epoch} Validation Mean Translation Difference: {val_mean_trans_diff:.4f} meters")

            #Test dataloader
            test_diff_angles = []
            test_diff_trans = []
            
            for test_batch in tqdm(test_loader, desc=f"Epoch {epoch} Full Validation"):
                test_batch = process_batch(test_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
        
                try:
                    diff_angle, diff_trans = trainer.eval_step(test_batch)
                    if diff_angle != float('inf') and diff_trans != float('inf'):
                        test_diff_angles.append(diff_angle)
                        test_diff_trans.append(diff_trans)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
            
            test_mean_angle_diff = sum(test_diff_angles) / len(test_diff_angles) if test_diff_angles else float('inf')
            test_mean_trans_diff = sum(test_diff_trans) / len(test_diff_trans) if test_diff_trans else float('inf')
            
            print(f"Epoch {epoch} testidation Mean Angle Difference: {test_mean_angle_diff:.4f} degrees")
            print(f"Epoch {epoch} testidation Mean Translation Difference: {test_mean_trans_diff:.4f} meters")

            wandb.log({
                "epoch": epoch, 
                "val_mean_angle_diff": val_mean_angle_diff,
                "val_mean_trans_diff": val_mean_trans_diff,
                "test_mean_angle_diff": test_mean_angle_diff,
                "test_mean_trans_diff": test_mean_trans_diff
            })
            
            save_path = os.path.join(
                save_dir,
                f"epoch_model_epoch_{epoch}_angle_diff_{val_mean_angle_diff:.4f}_trans_diff_{val_mean_trans_diff:.4f}.pt"
            )
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")
        else:
            wandb.log({
                "epoch": epoch
            })
            
        trainer.scheduler.step()

        if (epoch + 1) % eval_freq == 0:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Angle Diff {val_mean_angle_diff:.4f} | Val Trans Diff {val_mean_trans_diff:.4f}")
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | test Angle Diff {test_mean_angle_diff:.4f} | test Trans Diff {test_mean_trans_diff:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Validation skipped")

            
def Test_data(cfg, test_loader, trans_stats):

    pretrained_path = cfg.pretrained_model_path_test
    Test_Model = D3PMTest(cfg, trans_stats,pretrained_path=pretrained_path)
    Test_Model.model.eval()

    pbar = tqdm(test_loader)
    total_diff_angle = 0.0
    total_diff_trans = 0.0
    batch_count = 0
    for batch in pbar:
        test_batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
        with torch.no_grad():
            diff_angle, diff_trans = Test_Model.test_step(test_batch)  
            diff_avg_angle = diff_angle.mean().item()
            diff_avg_trans = diff_trans.mean().item()
            total_diff_angle += diff_avg_angle
            total_diff_trans += diff_avg_trans
            batch_count += 1
            count_less_than_5 = (diff_angle < 5).sum().item()
            count_less_than_10 = (diff_angle < 10).sum().item()
            count_less_than_20 = (diff_angle < 20).sum().item()
            total_count = diff_angle.size(0)
            ratio_less_than_5 = count_less_than_5 / total_count
            ratio_less_than_10 = count_less_than_10 / total_count
            ratio_less_than_20 = count_less_than_20 / total_count
            diff_trans_cm = diff_trans * 100
            count_less_than_2cm = (diff_trans_cm < 2).sum().item()
            count_less_than_5cm = (diff_trans_cm < 5).sum().item()
            count_less_than_10cm = (diff_trans_cm < 10).sum().item()
            ratio_less_than_2cm = count_less_than_2cm / total_count
            ratio_less_than_5cm = count_less_than_5cm / total_count
            ratio_less_than_10cm = count_less_than_10cm / total_count
            print(f"5 ratio: {ratio_less_than_5:.2f} | 10 ratio: {ratio_less_than_10:.2f} | 20 ratio: {ratio_less_than_20:.2f}")
            print(f"2cm: {ratio_less_than_2cm:.2f} | 5cm: {ratio_less_than_5cm:.2f} | 10cm: {ratio_less_than_10cm:.2f}")
            print(f"Val Angle Diff {diff_avg_angle:.4f} | Val Trans Diff {diff_avg_trans:.4f}")

    avg_diff_angle = total_diff_angle / batch_count
    avg_diff_trans = total_diff_trans / batch_count
    print(f"Avg Angle Diff across all batches: {avg_diff_angle:.4f}")
    print(f"Avg Trans Diff across all batches: {avg_diff_trans:.4f}")


def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    wandb.init(project="test", config={
        "batch_size": 96,
        "n_epochs": 200,
        "learning_rate": 3e-4,
        "encoder": "pointnet2",
        "num_bins": 360,

    })
    
    cfg = get_config()
    data_loaders = get_data_loaders_from_cfg(cfg, ['train', 'val', 'test'])
    train_loader = data_loaders['train_loader'] 
    val_loader = data_loaders['val_loader']   
    test_loader = data_loaders['test_loader'] 
    print('train_set: ', len(train_loader))
    print('val_set: ', len(val_loader))
    print('test_set: ', len(test_loader))
    trans_stats = [-0.3785014748573303, 0.39416784048080444, -0.4042277932167053, 0.39954620599746704, -0.30842161178588867, 0.7598943710327148]
    #trans_stats = get_dataset_translation_min_max(train_loader, cfg)

    if not (cfg.eval or cfg.pred):
        train_data(cfg, train_loader, val_loader,test_loader,trans_stats)
    else:
        Test_data(cfg,test_loader=test_loader,trans_stats=trans_stats)
            
if __name__ == "__main__":
    main()