import argparse
from ipdb import set_trace

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--synset_names', nargs='+', default=['dishwasher', 'drawer', 'eyeglasses', 'laptop', 'scissors'])
    parser.add_argument('--num_parts', type=int, default=2)
    parser.add_argument('--joint_num', type=int, default=1)
    parser.add_argument('--PTS_AUG_PARAMS', default = {
    'aug_bb_pro': 0.0,  
    'aug_rt_pro': 0.3,  
    'aug_pc_pro': 0.3,    
    'aug_pc_r': 0.01     
})
    parser.add_argument('--cate_id', type=int, default=1)
    
    """ dataset """

    parser.add_argument('--selected_classes', nargs='+')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--o2c_pose', default=True, action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batch_size', type=int, default=192)  # 192
    parser.add_argument('--mini_bs', type=int, default=1)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0) # 32
    
    """ model """
    parser.add_argument('--posenet_mode',  type=str, default='score')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sampler_mode', nargs='+')
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2') 

    
    
    """ training """
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')   
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--saved_model_name', type=str, default=None)
    
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--T0', type=float, default=1.0)
    parser.add_argument('--pretrained_model_path_test', type=str, default='')
   
    
    cfg = parser.parse_args()
    
    
    return cfg


