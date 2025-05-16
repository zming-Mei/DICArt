import os
import cv2
import math
import random
import sys
import copy
sys.path.insert(0,os.getcwd())
import mmengine
import numpy as np
import _pickle as cPickle
sys.path.append('/home/zming/diffpose/6D/code/D3PM-Pose')
from configs.config import get_config
# from config.config import *
# from datasets.data_augmentation import defor_2D, get_rotation
# FLAGS = flags.FLAGS
cfg = get_config()
import open3d as o3d
import torch
import pytorch3d
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.sgpa_utils import load_depth, get_bbox
# from tools.eval_utils import load_depth, get_bbox
from utils.datasets_utils import *
from pycocotools import mask as maskUtils
from datasets.dataset_Generator import SapienDataset_KPAGen
import os.path as osp

import json

class PoseDataset(data.Dataset):
    CLASSES = ('background', 'laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors')


    def __init__(self, source=None, mode='train', data_dir=None, cate_id=None):
        '''

        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.cate_id=cate_id
        
        self.num_parts = cfg.num_parts
        self.joint_num = cfg.joint_num

        assert mode in ['train', 'test','val']
        # img_list_path=['train.txt','test.txt','val.txt']
        # img_list_path=['train.txt','part_train.txt','part_train.txt']

        # if mode == 'train':
        #     del img_list_path[1:]

        # elif mode == 'test':
        #     del img_list_path[0]
        #     del img_list_path[1]
        
        # else :
        #     del img_list_path[0:-1]
        img_list_path = []
        if mode == 'train' or mode == 'val':
            img_list_path.append('train.txt')
        else:
            img_list_path.append('test.txt')
        #json_list = sorted(os.listdir(os.path.join(self.data_dir,self.CLASSES[self.cate_id],self.mode,'annotations')))
        #self.length=len(json_list)

        
        #  训练的样本
        self.img_list=[]
        
        for path in img_list_path:
            self.img_list += [line.rstrip('\n')
                         for line in open(os.path.join(data_dir,self.CLASSES[self.cate_id],path))]
            

        self.length = len(self.img_list)

        # self.intrinsics = np.array([[914, 0, 0], [0, 914, 0], [320, 320, 1]],#相机内参
        #                                   dtype=np.float)  # [fx, fy, cx, cy]
           
     
        #self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)
        self.camera_intrinsics = o3d.io.read_pinhole_camera_intrinsic(osp.join(self.data_dir,'camera_intrinsic.json'))

        print('{}: {} jsons found.'.format(mode,self.length))
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #  load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong

        # index=str(index).zfill(6)
        thres_r = 0.5

        try:
            img_path = os.path.join(self.data_dir, self.CLASSES[self.cate_id],'train',
                                    'annotations','{}'.format(self.img_list[index]))
         
            annotation=mmengine.load(img_path)  # mmengine.load是一个包装好的文件读取接口
     
        except:
            print('json is not in the train.txt')
        


        rgb_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],'train',annotation['color_path'])
        rgb=o3d.io.read_image(rgb_path)

        depth_path = osp.join(self.data_dir,self.CLASSES[self.cate_id],'train',annotation['depth_path'])
        depth = o3d.io.read_image(depth_path)



        instances=annotation['instances']
        # instances_info = instances[0]
        assert len(instances) == 1, 'Only support one instance per image'

        #加载urdf与关键信息
        urdf_id = annotation['instances'][0]['urdf_id']    # 不同的urdf代表了不同的电脑
        joint_para_path = osp.join(self.data_dir,self.CLASSES[self.cate_id],'urdf',str(urdf_id),'joint_infos.json')  # 加载轴信息（轴点、轴向量）
        joint_annotation = mmengine.load(joint_para_path)

        joint_state = [0. for _ in range(self.num_parts-1)]  # 轴角度
        joint_para = [[[0 for _ in range(3)] for _ in range(2)] for _ in range(self.joint_num)]  
        """
        if self.cate_id == 2:
            for idx in range(self.num_parts-1):
                link_category_id = annotation['instances'][0]['links'][idx+1]['link_category_id']-1
                joint_state[link_category_id] = annotation['instances'][0]['links'][idx+1]['state']
        else:
            for part_id in range(self.num_parts-1):
                joint_state[part_id] = annotation['instances'][0]['links'][part_id+1]['state']
        """
        for idx in range(self.num_parts-1):
            link_category_id = annotation['instances'][0]['links'][idx+1]['link_category_id']-1
            joint_state[link_category_id] = annotation['instances'][0]['links'][idx+1]['state'] # 取idx+1因为basepart没有state(角度)
        joint_state = np.array(joint_state) # 加载每一个轴的state

        joint_xyz = []
        co_xyz = []
        joint_rpy = []
        joint_id = 0
        for key in sorted(joint_annotation.keys()):   # key是数字，按顺序一个一个保存轴信息
            joint_para[joint_id][0] = joint_annotation[key]['xyz']   # 轴点
            joint_para[joint_id][1] = joint_annotation[key]['rpy']   # 轴向量
            co_xyz.append(joint_annotation[key]['xyz'])
            joint_xyz.append(joint_annotation[key]['xyz'])
            joint_rpy.append(joint_annotation[key]['rpy'])
            joint_id += 1 
        joint_id = 0

        co_xyz = np.array(co_xyz)
        joint_para_xyz = np.array(joint_xyz) # 加载每一个轴的xyz和rpy
        joint_para_xyz = joint_para_xyz.reshape(-1) # 转换为一维的便于处理
        joint_para_rpy = np.array(joint_rpy) # 加载每一个轴的xyz和rpy
        joint_para_rpy = joint_para_rpy.reshape(-1) # 转换为一维的便于处理
        
        joint_para = np.array(joint_para) # 加载每一个轴的xyz和rpy
        joint_para = joint_para.reshape(-1) # 转换为一维的便于处理
            
        var = 10
        if self.cate_id == 4:
            joint_state = joint_state*4
        else:
            joint_state = joint_state / var
        
   
        transformation=annotation['instances'][0]['links'][0]['transformation']  # basepart
        transformation=np.array(transformation)
        
        #切片操作提取变幻矩阵
        rotation = transformation[:3,:3] # 取出左上角 3×3 旋转矩阵
        translation = transformation[:3, 3]  # 取出右侧 3×1 平移向量
        link_info=annotation['instances'][0]['links']

        #点云
        pc_all_points = []
        pc_temps = []
        label_all = []
        pc_all = o3d.geometry.PointCloud()
        for part_id in range(self.num_parts):
            part_seg = link_info[part_id]['segmentation']
            rle = None
            try:
                """
                    maskUtils.frPyObjects 这个函数将其中的多边形（polygon）
                    或未压缩的RLE（uncompressed RLE）转换为压缩的RLE格式，以便于后续的处理和分析。
                """
                rle = maskUtils.frPyObjects(part_seg, 640,640)
            except:
                print(type(index))
                print('index = ',index)
            # if rle: 
            #     mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
            # else:
            #     pc_temps.append(None)
            #     # pc_temps.append([[-1.0,-1.0,-1.0]])
            #     continue
            mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
            #生成RGB-D图像
            color = o3d.geometry.Image(rgb * np.repeat(mask[..., np.newaxis], 3, 2))
            deep = o3d.geometry.Image(depth * mask)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, deep, 1000.0, 20.0, convert_rgb_to_intensity=True)
            #生成点云
            pc_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsics)
            pc_temps.append(pc_temp)
            points_temp = np.asarray(pc_temp.points)
            pc_all_points.append(points_temp)


        pc = np.concatenate(pc_all_points)  # 所有点构成的点云

        #点云标准化
        T_inv = np.linalg.inv(transformation) # R的逆
        parts_gts = [None] * cfg.num_parts  # 每一个part的点云
        for i in range(self.num_parts):
            canonical = pc_temps[i].transform(T_inv)   # 将原始点云转换到标准空间下
            parts_gts[i] = np.asarray(canonical.points)  # [N, 3] 

        pc_canonical = np.concatenate(parts_gts)

        # vis_points = np.array(pc_canonical)
        # np.savetxt('vis_points_canonical.txt',vis_points)

        offset_heatmap = [None] * (cfg.num_parts + cfg.joint_num - 1)
        offset_heatmap_filtered = [None] * (cfg.num_parts + cfg.joint_num - 1)
        mean_offset_heatmap = [None] * cfg.joint_num 
        offset_unitvec = [None] * cfg.joint_num 
        parts_gt = [None] * (cfg.num_parts + cfg.joint_num - 1)
        mean_parts_gt = [None] * cfg.joint_num 

        #热图计算
        for i in range(cfg.num_parts):
            if i == 0:
                for j in range(cfg.joint_num):
                    offset_heatmap[j] = self.get_heatmap(joint_xyz[j], parts_gts[i], thres_r=thres_r)
                    offset_heatmap_arr = np.array(offset_heatmap[j])
                    indices = np.where(offset_heatmap_arr > 0)[0]
                    offset_heatmap_filtered[j] = offset_heatmap[j][offset_heatmap[j] > 0] 
                    parts_gt[j] = parts_gts[i][indices]  # 取出了heatmap大于0的部分点云，实际上即点的距离小于一定阈值

            else:
                offset_heatmap[i+cfg.joint_num-1] = self.get_heatmap(joint_xyz[i-1], parts_gts[i], thres_r=thres_r)
                offset_heatmap_arr = np.array(offset_heatmap[i+cfg.joint_num-1])
                indices = np.where(offset_heatmap_arr > 0)[0]

                offset_heatmap_filtered[i+cfg.joint_num-1] = offset_heatmap[i+cfg.joint_num-1][offset_heatmap[i+cfg.joint_num-1] > 0] 
               
                parts_gt[i+cfg.joint_num-1] = parts_gts[i][indices]

                mean_offset_heatmap[i-1] = np.concatenate([offset_heatmap_filtered[i+cfg.joint_num-1],offset_heatmap_filtered[i-1]])
                mean_parts_gt[i-1] = np.concatenate([parts_gt[i+cfg.joint_num-1],parts_gt[i-1]])  # 两个部分连接

                mean_offset_heatmap_arr = np.array(mean_offset_heatmap[i-1])
               
                mean_offset_heatmap_arr_value = np.mean(mean_offset_heatmap_arr)
                mean_offset_heatmap[i-1] = mean_offset_heatmap_arr_value
                # mean_offset_heatmap[i-1] = np.mean(mean_offset_heatmap[i-1])

                mean_parts_gt_arr = np.array(mean_parts_gt[i-1])
                mean_parts_gt_arr_value = np.mean(mean_parts_gt_arr, axis=0)
                mean_parts_gt[i-1] = mean_parts_gt_arr_value
                # mean_parts_gt[i-1] = np.mean(mean_parts_gt[i-1], axis=0)

                offset_unitvec[i-1] = joint_xyz[i-1] / mean_offset_heatmap_arr_value - mean_parts_gt_arr_value
                # offset_unitvec[i-1] = joint_xyz[i-1] / mean_offset_heatmap[i-1] - mean_parts_gt[i-1]



        offset_heatmap_gt = np.array(mean_offset_heatmap).reshape(-1)  # [1,]   所有距离在阈值内的点的heatmap值的平均
        offset_unitvec_gt = np.array(offset_unitvec).reshape(-1)       # [3,]   平均的，轴点指向
        mean_parts_gt = np.array(mean_parts_gt).reshape(-1)            # [3,]   轴两边部分的点云的中心点


        """
        pcd_test = o3d.io.read_point_cloud(osp.join('/home/jhn/pose_state_segmentation_size/pc_full','laptop_{}_all.xyz'.format(str(urdf_id))))
        aabb_test = pcd_test.get_axis_aligned_bounding_box() 
        bbox_dims_test = aabb_test.get_extent()
        """
        ###scale###
        """
        scale = 0.1
        T = transformation
        T_inv = np.linalg.inv(T)
        pc_transform = o3d.geometry.PointCloud()
        pc_transform = pc_all.transform(T_inv)
        pc_transform.points = o3d.utility.Vector3dVector(np.asarray(pc_transform.points) * scale)
        pc_scale = pc_transform.transform(T)
        """
        # pc = np.asarray(pc.points)

        ###sample###
        samplenum = 1024
        len_pc = len(pc)
        if len_pc >= samplenum:
            replace_rnd = False
        else:
            replace_rnd = True

        indices = np.random.choice(len_pc, size=samplenum, replace=replace_rnd)
        pc_sample = pc[indices]
        #label_sample = label[indices]


        fsnet_scale, mean_shape_part,mean_shape = self.get_fs_net_scale_part(c=self.CLASSES[self.cate_id],urdf_id = urdf_id)
        #bbox_dims_res = (bbox_dims - mean_shape)*0.3#整体物体的size

        sym_info = self.get_sym_info_part(c=self.CLASSES[self.cate_id])

        data_dict = {}

        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()
    
        data_dict['pcl_in'] = torch.as_tensor(pc_sample,dtype=torch.float32).contiguous()
        #data_dict['gt_RTs'] = torch.as_tensor(transformation, dtype=torch.float32).contiguous()
        data_dict['transformation'] = torch.as_tensor(transformation,dtype=torch.float32).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous() 
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()#size all part
        data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        data_dict['mean_shape_part'] = torch.as_tensor(mean_shape_part, dtype=torch.float32).contiguous()
        # data_dict['model_point'] = torch.as_tensor(part_model, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info, dtype=torch.float32).contiguous()
        data_dict['gt_state'] = torch.as_tensor(joint_state, dtype=torch.float32).contiguous()
        #data_dict['gt_label'] = torch.as_tensor(label_sample,dtype=torch.int64).contiguous()
        #data_dict['gt_size'] = torch.as_tensor(bbox_dims_res, dtype=torch.float32).contiguous()
        data_dict['index'] = index
        data_dict['gt_joint_xyz'] = torch.as_tensor(joint_para_xyz, dtype=torch.float32).contiguous()
        data_dict['gt_joint_rpy'] = torch.as_tensor(joint_para_rpy, dtype=torch.float32).contiguous()
        data_dict['offset_heatmap_gt'] = torch.as_tensor(offset_heatmap_gt, dtype=torch.float32).contiguous()
        data_dict['offset_unitvec_gt'] = torch.as_tensor(offset_unitvec_gt, dtype=torch.float32).contiguous()
        data_dict['mean_parts_gt'] = torch.as_tensor(mean_parts_gt, dtype=torch.float32).contiguous()
        # data_dict['child_transformation'] = torch.as_tensor(child_transformation, dtype=torch.float32).contiguous()
        data_dict['path'] = img_path
        data_dict['cate_id'] = torch.as_tensor(self.cate_id, dtype=torch.int8).contiguous()

        return data_dict
    
    
    #数据增强
    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=25, ay=25, az=25, a=5):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    
    
    #计算尺度信息（scale）和形状信息（shape）
    def get_fs_net_scale_part(self, c, urdf_id):
        # model pc x 3 xyz轴上的长度，在自身坐标系下
    

        if c == 'laptop':
            size_list = []
            
            mean_shape = np.array([1.32,0.89,1.17])
            mean_shape_part = np.array([[1.301,0.074,0.92],
                                        [1.301,0.903,0.062]])
            
            for i in range(self.num_parts):
                part_model_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],
                                 'urdf',str(urdf_id),'part_point_sample_rest','{}.xyz'.format(i))
                model=np.asarray(o3d.io.read_point_cloud(part_model_path).points)
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                size_list.append(np.array([lx,ly,lz]))
            size =np.array(size_list)    
            residual_size = size - mean_shape_part

        if c == 'drawer':
            size_list = []
            mean_shape = np.array([1.04,1.47,1.03])
            mean_shape_part = np.array([[1.040,1.469,0.946],
                                        [1.012,0.498,0.873],
                                        [1.012,0.432,0.873],
                                        [1.012,0.303,0.872]])
            for i in range(self.num_parts):
                part_model_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],
                                 'urdf',str(urdf_id),'part_point_sample_rest','{}.xyz'.format(i))
                model=np.asarray(o3d.io.read_point_cloud(part_model_path).points)
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                size_list.append(np.array([lx,ly,lz]))
            size =np.array(size_list)    
            residual_size = size - mean_shape_part

        if c == 'dishwasher':
            size_list = []
            mean_shape = np.array([1.02,1.41,1.09])
            mean_shape_part = np.array([[1.015,1.406,0.987],
                                        [1.018,0.140,1.246]])
            for i in range(self.num_parts):
                part_model_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],
                                 'urdf',str(urdf_id),'part_point_sample_rest','{}.xyz'.format(i))
                model=np.asarray(o3d.io.read_point_cloud(part_model_path).points)
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                size_list.append(np.array([lx,ly,lz]))
            size =np.array(size_list)    
            residual_size = size - mean_shape_part         

        if c == 'scissors':
            size_list = []
            mean_shape = np.array([1.61,0.08,0.86])
            mean_shape_part = np.array([[1.576,0.073,0.604],
                                        [1.493,0.065,0.700]])
            for i in range(self.num_parts):
                part_model_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],
                                 'urdf',str(urdf_id),'part_point_sample_rest','{}.xyz'.format(i))
                model=np.asarray(o3d.io.read_point_cloud(part_model_path).points)
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                size_list.append(np.array([lx,ly,lz]))
            size =np.array(size_list)    
            residual_size = size - mean_shape_part            
        if c == 'eyeglasses':
            size_list = []
            mean_shape = np.array([1.24,0.39,1.02])
            mean_shape_part = np.array([[1.190,0.376,0.157],
                                        [0.082,0.232,1.008],
                                        [0.082,0.227,1.008]])
            for i in range(self.num_parts):
                part_model_path=osp.join(self.data_dir,self.CLASSES[self.cate_id],
                                 'urdf',str(urdf_id),'part_point_sample_rest','{}.xyz'.format(i))
                model=np.asarray(o3d.io.read_point_cloud(part_model_path).points)
                lx = max(model[:, 0]) - min(model[:, 0])
                ly = max(model[:, 1]) - min(model[:, 1])
                lz = max(model[:, 2]) - min(model[:, 2])
                size_list.append(np.array([lx,ly,lz]))
            size =np.array(size_list)    
            residual_size = size - mean_shape_part            

        # scale residual
        return residual_size.reshape(-1)*0.3, mean_shape_part.reshape(-1)*0.3 ,mean_shape*0.3
    
    #rgbd to point cloud
    def rgbd2pc(self,rgb, depth, rgb_path,depth_path,camera_intrinsic, vis=False, save_pcd=False):
                # color_raw = o3d.io.read_image(rgb_path)
                # depth_raw = o3d.io.read_image(depth_path)
                rgb = o3d.geometry.Image(rgb)
                depth = o3d.geometry.Image(depth)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                    depth,
                                                                    1000.0,
                                                                    20.0,
                                                                    convert_rgb_to_intensity=True)

                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_intrinsic)
                                                                
                # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                if vis:
                    o3d.visualization.draw_geometries([pcd])
                if save_pcd:
                    basename = os.path.basename(rgb_path)
                    pcd_save_name = basename.split('.')[0] + '.pcd'
                    o3d.io.write_point_cloud(pcd_save_name, pcd)

                return pcd  
    
    #热力图
    def get_heatmap(self,joint_xyz, point, thres_r):
        """
        joint: [x, y, z] or [[x, y, z] + [rx, ry, rz]]
        point: N * 3
        """
        # if len(joint) == 2:
        #     P0 = np.array(joint[0])
        #     P  = np.array(point)
        #     l  = np.array(joint[1]).reshape(1, 3)
        #     P0P= P - P0
        #     # projection of P in joint minus P
        #     PP = np.dot(P0P, l.T) * l / np.linalg.norm(l)**2  - P0P
        # # PP的shape是[N,3]
        # return PP
        P2P = point - joint_xyz
        distances = np.linalg.norm(P2P, axis=1)
        heatmap = 1 - distances / thres_r
        return heatmap
    


    #返回对称信息
    def get_sym_info_part(self, c,):
        if c == 'laptop':
            sym = np.array([0, 0, 0, 1], dtype=np.int)
        if c =='eyeglasses':
            sym = np.array([0, 0, 0, 1], dtype=np.int)
        if c =='dishwasher':
            sym = np.array([0, 0, 0, 1], dtype=np.int)
        if c == 'drawer':
            sym = np.array([0, 0, 0, 1], dtype=np.int)
        if c =='scissors':
            sym = np.array([0, 0, 0, 0], dtype=np.int)

        return sym
    
    def RotateAnyAxis(self,v1, v2, step):
        ROT = np.identity(4)

        axis = v2 - v1
        axis = axis / math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

        step_cos = math.cos(step)
        step_sin = math.sin(step)

        ROT[0][0] = axis[0] * axis[0] + (axis[1] * axis[1] + axis[2] * axis[2]) * step_cos
        ROT[0][1] = axis[0] * axis[1] * (1 - step_cos) + axis[2] * step_sin
        ROT[0][2] = axis[0] * axis[2] * (1 - step_cos) - axis[1] * step_sin
        ROT[0][3] = 0

        ROT[1][0] = axis[1] * axis[0] * (1 - step_cos) - axis[2] * step_sin
        ROT[1][1] = axis[1] * axis[1] + (axis[0] * axis[0] + axis[2] * axis[2]) * step_cos
        ROT[1][2] = axis[1] * axis[2] * (1 - step_cos) + axis[0] * step_sin
        ROT[1][3] = 0

        ROT[2][0] = axis[2] * axis[0] * (1 - step_cos) + axis[1] * step_sin
        ROT[2][1] = axis[2] * axis[1] * (1 - step_cos) - axis[0] * step_sin
        ROT[2][2] = axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]) * step_cos
        ROT[2][3] = 0

        ROT[3][0] = (v1[0] * (axis[1] * axis[1] + axis[2] * axis[2]) - axis[0] * (v1[1] * axis[1] + v1[2] * axis[2])) * (1 - step_cos) + \
                    (v1[1] * axis[2] - v1[2] * axis[1]) * step_sin

        ROT[3][1] = (v1[1] * (axis[0] * axis[0] + axis[2] * axis[2]) - axis[1] * (v1[0] * axis[0] + v1[2] * axis[2])) * (1 - step_cos) + \
                    (v1[2] * axis[0] - v1[0] * axis[2]) * step_sin

        ROT[3][2] = (v1[2] * (axis[0] * axis[0] + axis[1] * axis[1]) - axis[2] * (v1[0] * axis[0] + v1[1] * axis[1])) * (1 - step_cos) + \
                    (v1[0] * axis[1] - v1[1] * axis[0]) * step_sin
        ROT[3][3] = 1

        return ROT.T
    
    def fetch_factors_nocs(self,cate_id):
        norm_factors = {}
        corner_pts = {}
        urdf_metas = json.load(open(cfg.data_path + '/{}'.format(self.CLASSES[cate_id]) +'/urdf/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
            corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts'])

        return norm_factors, corner_pts


#欧拉角转3D旋转矩阵
def get_rotation(x_, y_, z_):
        # print(math.cos(math.pi/2))
        x = float(x_ / 180) * math.pi
        y = float(y_ / 180) * math.pi
        z = float(z_ / 180) * math.pi
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(x), -math.sin(x)],
                        [0, math.sin(x), math.cos(x)]])
    
        R_y = np.array([[math.cos(y), 0, math.sin(y)],
                        [0, 1, 0],
                        [-math.sin(y), 0, math.cos(y)]])
    
        R_z = np.array([[math.cos(z), -math.sin(z), 0],
                        [math.sin(z), math.cos(z), 0],
                        [0, 0, 1]])
        return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)

#batch*
def process_batch(batch_sample,
                  device,
                  pose_mode='rot_matrix',
                  mini_batch_size=None,
                  PTS_AUG_PARAMS=None):
    #rot_matrix
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
        f"the rotation mode {pose_mode} is not supported!"
    if PTS_AUG_PARAMS==None:
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)
        gt_state = batch_sample['gt_state'].to(device)
        gt_joint_xyz = batch_sample['gt_joint_xyz'].to(device)
        gt_joint_rpy = batch_sample['gt_joint_rpy'].to(device)
        gt_heatmap = batch_sample['offset_heatmap_gt'].to(device)
        gt_unitvec = batch_sample['offset_unitvec_gt'].to(device)
        gt_parts = batch_sample['mean_parts_gt'].to(device)
    else:        
        PC_da, gt_R_da, gt_t_da, gt_s_da = data_augment(
            pts_aug_params=PTS_AUG_PARAMS,
            PC=batch_sample['pcl_in'].to(device), 
            gt_R=batch_sample['rotation'].to(device), 
            gt_t=batch_sample['translation'].to(device),
            gt_s=batch_sample['fsnet_scale'].to(device), 
            mean_shape=batch_sample['mean_shape'].to(device),
            sym=batch_sample['sym_info'].to(device),
            aug_bb=batch_sample['aug_bb'].to(device), 
            aug_rt_t=batch_sample['aug_rt_t'].to(device),
            aug_rt_r=batch_sample['aug_rt_R'].to(device),
        )
        gt_state = batch_sample['gt_state'].to(device)
        gt_joint_xyz = batch_sample['gt_joint_xyz'].to(device)
        gt_joint_rpy = batch_sample['gt_joint_rpy'].to(device)
        gt_heatmap = batch_sample['offset_heatmap_gt'].to(device)
        gt_unitvec = batch_sample['offset_unitvec_gt'].to(device)
        gt_parts = batch_sample['mean_parts_gt'].to(device)

    processed_sample = {}
    processed_sample['pts'] = PC_da                # [bs, 1024, 3]
    processed_sample['pts_color'] = PC_da          # [bs, 1024, 3]
    # processed_sample['id'] = batch_sample['cat_id'].to(device)      # [bs]
    # processed_sample['handle_visibility'] = batch_sample['handle_visibility'].to(device)     # [bs]
    # processed_sample['path'] = batch_sample['path']
    if pose_mode == 'quat_xyzw':  # 将旋转矩阵转换为四元组
        rot = pytorch3d.transforms.matrix_to_quaternion(gt_R_da)
    elif pose_mode == 'quat_wxyz':
        rot = pytorch3d.transforms.matrix_to_quaternion(gt_R_da)[:, [3, 0, 1, 2]]
    elif pose_mode == 'euler_xyz':
        rot = pytorch3d.transforms.matrix_to_euler_angles(gt_R_da, 'ZYX')
    elif pose_mode == 'euler_xyz_sx_cx':
        rot = pytorch3d.transforms.matrix_to_euler_angles(gt_R_da, 'ZYX')
        rot_sin_theta = torch.sin(rot)
        rot_cos_theta = torch.cos(rot)
        rot = torch.cat((rot_sin_theta, rot_cos_theta), dim=-1)
    #rot_matrix
    elif pose_mode == 'rot_matrix':
        rot = pytorch3d.transforms.matrix_to_rotation_6d(gt_R_da.permute(0, 2, 1)).reshape(gt_R_da.shape[0], -1)
    else:
        raise NotImplementedError
    #gt_pose = translation+rotation
    location = gt_t_da # [bs, 3]
    processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)   # [bs, 4/6/3 + 3]
    
    """ zero center """
    num_pts = processed_sample['pts'].shape[1]  # 用完整点云进行训练，因为在推理的时候我们无法知道哪个是base_part，这里相当于网络自己学习到了哪些点属于base_part
    zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)   # 点云的中心点 [bs, 1024, 3]  ->  [bs, 3]
    processed_sample['gt_state'] = gt_state    # joint旋转角
    # processed_sample['gt_joint_para'] = gt_joint_para

    processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
    processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
    processed_sample['zero_mean_gt_pose'] = copy.deepcopy(processed_sample['gt_pose'])
    processed_sample['zero_mean_gt_pose'][:, -3:] -= zero_mean   # 位姿中的平移量移到中心

    processed_sample['pts_center'] = zero_mean
    processed_sample['gt_joint_xyz'] = gt_joint_xyz    # 轴点
    processed_sample['gt_joint_rpy'] = gt_joint_rpy    # 轴向量
    processed_sample['gt_heatmap'] = gt_heatmap
    processed_sample['gt_unitvec'] = gt_unitvec   # 用于预测轴点 
    processed_sample['gt_parts'] = gt_parts       # 每个part的点云
    processed_sample['cate_id'] = batch_sample['cate_id']
    # processed_sample['child_transformation'] = gt_child_transformation
   

    if 'color' in batch_sample.keys():
        pass
        # processed_sample['color'] = batch_sample['color'].to(device)       # [bs]

    if not mini_batch_size == None:
        for key in processed_sample.keys():
            processed_sample[key] = processed_sample[key][:mini_batch_size]
    
    if not 'color' in processed_sample.keys():
        pass
        # processed_sample['color'] = None
    # print(processed_sample['zero_mean_pts'].device)
    return processed_sample 

#数据增强
def data_augment(pts_aug_params, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         check_points=False):
    """
    PC torch.Size([32, 1028, 3])
    gt_R torch.Size([32, 3, 3])
    gt_t torch.Size([32, 3])
    gt_s torch.Size([32, 3])
    mean_shape torch.Size([32, 3])
    sym torch.Size([32, 4])
    aug_bb torch.Size([32, 3])
    aug_rt_t torch.Size([32, 3])
    aug_rt_r torch.Size([32, 3, 3])
    model_point torch.Size([32, 1024, 3])
    nocs_scale torch.Size([32])
    obj_ids torch.Size([32])
    """

    def aug_bb_with_flag(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, flag):
        # PC_new, gt_s_new = defor_3D_bb_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, sym, aug_bb)
        # gt_s_new = gt_s_new - mean_shape
        # PC = torch.where(flag.unsqueeze(-1), PC_new, PC)
        # gt_s = torch.where(flag, gt_s_new, gt_s)
        # model_point_new = torch.where(flag.unsqueeze(-1), model_point_new, model_point)
        return PC, gt_s

    def aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag):
        PC_new, gt_R_new, gt_t_new = defor_3D_rt_in_batch(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
        PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
        gt_R_new = torch.where(flag.unsqueeze(-1), gt_R_new, gt_R)
        gt_t_new = torch.where(flag, gt_t_new, gt_t)
        return PC_new, gt_R_new, gt_t_new

    def aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag):
        pc_new, s_new, ey_up, ey_down = defor_3D_bc_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, model_point,
                                                                nocs_scale)
        pc_new = torch.where(flag.unsqueeze(-1), pc_new, PC)
        s_new = torch.where(flag, s_new - mean_shape, gt_s)
        return pc_new, s_new, ey_up, ey_down

    def aug_pc_with_flag(PC, gt_t, flag, aug_pc_r):
        PC_new, defor = defor_3D_pc(PC, gt_t, aug_pc_r, return_defor=True)
        PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
        return PC_new, defor
    

    # augmentation
    bs = PC.shape[0]

    prob_bb = torch.rand((bs, 1), device=PC.device)
    flag = prob_bb < pts_aug_params['aug_bb_pro']
    PC, gt_s = aug_bb_with_flag(PC, gt_R, gt_t, gt_s,  mean_shape, sym, aug_bb, flag)

    prob_rt = torch.rand((bs, 1), device=PC.device)
    flag = prob_rt < pts_aug_params['aug_rt_pro']
    PC, gt_R, gt_t = aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag)

    prob_pc = torch.rand((bs, 1), device=PC.device)
    flag = prob_pc < pts_aug_params['aug_pc_pro']
    PC, _ = aug_pc_with_flag(PC, gt_t, flag, pts_aug_params['aug_pc_r'])

    return PC, gt_R, gt_t, gt_s

    #掩码进行随机腐蚀和膨胀操作
def defor_2D(roi_mask, rand_r=2, rand_pro=0.3):
    '''

    :param roi_mask: 256 x 256
    :param rand_r: randomly expand or shrink the mask iter rand_r
    :return:
    '''
    roi_mask = roi_mask.copy().squeeze()
    if np.random.rand() > rand_pro:
        return roi_mask
    mask = roi_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_erode = cv2.erode(mask, kernel, rand_r)  # rand_r
    mask_dilate = cv2.dilate(mask, kernel, rand_r)
    change_list = roi_mask[mask_erode != mask_dilate]
    l_list = change_list.size
    if l_list < 1.0:
        return roi_mask
    choose = np.random.choice(l_list, l_list // 2, replace=False)
    change_list = np.ones_like(change_list)
    change_list[choose] = 0.0
    roi_mask[mask_erode != mask_dilate] = change_list
    roi_mask[roi_mask > 0.0] = 1.0
    return roi_mask


# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s


def defor_3D_bb_in_batch(pc,  R, t, s, sym=None, aug_bb=None):
    pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
    sym_aug_bb = (aug_bb + aug_bb[:, [2, 1, 0]]) / 2.0
    sym_flag = (sym[:, 0] == 1).unsqueeze(-1)
    new_aug_bb = torch.where(sym_flag, sym_aug_bb, aug_bb)
    pc_reproj = pc_reproj * new_aug_bb.unsqueeze(-2)
    
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)
    s_new = s * new_aug_bb
    return pc_new, s_new


def defor_3D_bc(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1,  device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    model_point_resize = (model_point[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = max(model_point[:, 0]) - min(model_point[:, 0])
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * nocs_scale
    ly_t = ly * nocs_scale
    lz_t = lz * nocs_scale
    return pc_new, torch.tensor([lx_t, ly_t, lz_t], device=pc.device)


def defor_3D_bc_in_batch(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    bs = pc.size(0)
    ey_up = torch.rand((bs,1), device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand((bs, 1),  device=pc.device) * (1.2 - 0.8) + 0.8
    pc_reproj = torch.matmul(R.transpose(-1,-2), (pc-t.unsqueeze(-2)).transpose(-1,-2)).transpose(-1,-2)

    s_y = s[..., 1].unsqueeze(-1)
    per_point_resize = (pc_reproj[..., 1] + s_y / 2.0) / s_y * (ey_up - ey_down) + ey_down
    pc_reproj[..., 0] = pc_reproj[..., 0] * per_point_resize
    pc_reproj[..., 2] = pc_reproj[..., 2] * per_point_resize
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2,-1)) + t.unsqueeze(-1)).transpose(-2,-1)


    new_model_point = model_point*1.0
    model_point_resize = (new_model_point[..., 1] + s_y / 2) / s_y * (ey_up - ey_down) + ey_down
    new_model_point[..., 0] = new_model_point[..., 0] * model_point_resize
    new_model_point[..., 2] = new_model_point[..., 2] * model_point_resize

    s_new = (torch.max(new_model_point, dim=1)[0] - torch.min(new_model_point, dim=1)[0])*nocs_scale.unsqueeze(-1)
    return pc_new, s_new, ey_up, ey_down

# def defor_3D_pc(pc, r=0.05):
#     points_defor = torch.randn(pc.shape).to(pc.device)
#     pc = pc + points_defor * r * pc
#     return pc

def defor_3D_pc(pc, gt_t, r=0.2, points_defor=None, return_defor=False):

    if points_defor is None:
        points_defor = torch.rand(pc.shape).to(pc.device)*r
    new_pc = pc + points_defor*(pc-gt_t.unsqueeze(1))
    if return_defor:
        return new_pc, points_defor
    return new_pc


# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t


def defor_3D_rt_in_batch(pc, R, t, aug_rt_t, aug_rt_r):
    pc_new = pc + aug_rt_t.unsqueeze(-2)
    t_new = t + aug_rt_t
    pc_new = torch.matmul(aug_rt_r, pc_new.transpose(-2,-1)).transpose(-2,-1)

    R_new = torch.matmul(aug_rt_r, R)
    t_new = torch.matmul(aug_rt_r, t_new.unsqueeze(-1)).squeeze(-1)
    return pc_new, R_new, t_new

def get_data_loaders(
    batch_size=8,
    seed=0,
    percentage_data=1,
    source="ArtImage",
    mode='train',
    data_path=None,
    cate_id=1,
    num_workers=0,
):
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 加载训练数据集
    dataset = PoseDataset(
        source=source,
        mode=mode,  # 测试集从训练数据中抽取
        data_dir=data_path,
        cate_id=cate_id
    )
    total_size = len(dataset)
    test_size = int((1-percentage_data) * total_size)
    indices = torch.randperm(total_size).tolist()
    test_indices = indices[:test_size]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)


    # 数据加载器的 shuffle 参数设置
    shuffle = False  # 不打乱数据

    train_size = len(dataset)
    train_indices = list(range(train_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=False,
        drop_last=False,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        drop_last=False,
        pin_memory=True,
    )
    if mode != 'test':
        return dataloader, val_dataloader
    
    return dataloader



# def get_data_loaders(
#     batch_size=8,
#     seed=0,
#     percentage_data=1,
#     source="ArtImage",
#     mode='train',
#     data_path=None,
#     cate_id=1,
#     num_workers=0,
# ):
#     # 设置随机种子以确保可重复性
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     # 初始化数据集
#     dataset = PoseDataset(
#         source=source,
#         mode=mode,
#         data_dir=data_path,
#         cate_id=cate_id
#     )

#     # 数据加载器的 shuffle 参数设置
#     shuffle = False  # 不打乱数据

#     # 根据百分比计算训练集和验证集的大小
#     total_size = len(dataset)
#     train_size = int(percentage_data * total_size)
#     val_size = total_size - train_size

#     # 使用 torch.utils.data.Subset 按顺序切分数据集
#     indices = list(range(total_size))  # 按顺序生成索引
#     train_indices = indices[:train_size]  # 前部分作为训练集
#     val_indices = indices[train_size:]   # 后部分作为验证集

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     val_dataset = torch.utils.data.Subset(dataset, val_indices)

#     # 创建数据加载器
#     dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         persistent_workers=False,
#         drop_last=False,
#         pin_memory=True,
#     )
#     val_dataloader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         persistent_workers=False,
#         drop_last=False,
#         pin_memory=True,
#     )

#     # 返回数据加载器
#     if mode != 'test':
#         return dataloader, val_dataloader
    
#     return dataloader

# def get_data_loaders(
#     batch_size = 8,
#     seed = 0,
#     percentage_data= 0.8,
#     source = "ArtImage",
#     mode = 'train',
#     data_path=None,
#     cate_id = 1,
#     num_workers=0,
# ):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     dataset = PoseDataset(
#         source=source,
#         mode=mode,
#         data_dir=data_path,
#         cate_id=cate_id
#     ) 

#     if mode == 'train':
#         shuffle = False
        
#     elif mode == 'val':
#         shuffle = True
        
#     else:
#         shuffle = False
        
            

#     # sample
#     size = int(percentage_data * len(dataset))
#     dataset, val_dataset = torch.utils.data.random_split(dataset, (size, len(dataset) - size))  # 划分测试集和训练集，这里只取了训练集

#     # train_dataloader = torch.utils.data.DataLoader(
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         persistent_workers=False,
#         drop_last=False,
#         pin_memory=True,
#     )
#     val_dataloader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         persistent_workers=False,
#         drop_last=False,
#         pin_memory=True,
#     )

#     if mode != 'test':
#         return dataloader,val_dataloader
#     return dataloader



def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test']):
    data_loaders = {}
    if 'train' in data_type or 'val' in data_type:
        train_loader, val_loader = get_data_loaders(
            batch_size = cfg.batch_size,
            seed = cfg.seed,
            percentage_data=0.8,  
            source = "ArtImage",
            mode = 'train',
            data_path=cfg.data_path,
            cate_id = cfg.cate_id,
            num_workers=cfg.num_workers,
        ) 
        data_loaders['train_loader'] = train_loader

        data_loaders['val_loader'] = val_loader
        
    if 'test' in data_type:
        test_loader = get_data_loaders(
            batch_size = cfg.batch_size,
            seed = cfg.seed,
            percentage_data=1.0,
            source = "ArtImage",
            mode = 'test',
            data_path=cfg.data_path,
            cate_id = cfg.cate_id,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_loader'] = test_loader
        
    return data_loaders

if __name__ == '__main__':
    # train_dataset = PoseDataset(source='ArtImage', mode='train',
    #                             data_dir='/home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage',cate_id=1)

    from tqdm import tqdm
    train_loader, val_loader = get_data_loaders(data_path='/home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage')
    
        # 处理训练集
    for batch_sample in tqdm(train_loader):
        batch_sample = process_batch(
            batch_sample=batch_sample,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    #print(f"rotation shape:{}")

    # 处理验证集
    for batch_sample in tqdm(val_loader):
        batch_sample = process_batch(
            batch_sample=batch_sample,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    
        

