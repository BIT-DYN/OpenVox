
import torch
import argparse
import glob
import os
from natsort import natsorted
from tsdf_cuda import integrate_tsdf
import warnings
warnings.filterwarnings('ignore')
import time
import argparse
import os
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
# yolo-world
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
import supervision as sv

from PIL import Image, ImageDraw, ImageFont

import distinctipy
from scipy.spatial.transform import Rotation as R
import cv2
from skimage import measure
import open3d as o3d
from utils import Log
import numpy as np
from tqdm import tqdm, trange
import torch
from PIL import Image
from utils import load_config
import pickle

# from inst_class import InstFrame
import spacy
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndi_label
from sklearn.cluster import DBSCAN

# tap
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack

# sbert
from sentence_transformers import SentenceTransformer, util

class OpenVox():
    def __init__(self, config, save_dir, vis_gui):
        super().__init__()
        self.device = torch.device("cuda")
        self.vis_gui = vis_gui
        self.save_dir = save_dir
        self.config = load_config(config)
        self.voxel_size = self.config["tsdf"]["voxel_size"]
        self.sdf_trunc = 5 * self.voxel_size
        self.const = 256*256
        self.instance_skip = self.config["instance"]["instance_skip"]
        self.pro_thre = self.config["instance"]["pro_thre"]
        self.ins_min_count = self.config["instance"]["ins_min_count"]
        self.vox_min_count = self.config["instance"]["vox_min_count"]
        self.ins_min_voxel = self.config["instance"]["ins_min_voxel"]
        # extend bds for avoid frequence increment
        self.extent_bds = 1.0
        self.vis_h = None
        
        self.tsdf_vol = None
        self.weight_vol = None
        self.color_vol = None
        self.load_models()
        
        instance_file = "data/instance_colors.pt"
        if os.path.exists(instance_file):
            self.instance_colors = torch.load(instance_file).cuda().to(torch.int64)
        else:
            instance_colors = distinctipy.get_colors(1000, pastel_factor=1.0)
            instance_colors[0] = [0.2, 0.2, 0.2] # for background
            self.instance_colors = torch.tensor(instance_colors, dtype=torch.float32).cuda()*255
            self.instance_colors = self.instance_colors.to(torch.int64)
            torch.save(self.instance_colors, instance_file)
        
        
    def load_models(self): 
        # yolo-world model
        config = "/code1/dyn/codes/OpenWorld/YOLO-World/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py"
        cfg = Config.fromfile(config)
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
        checkpoint = "/data/dyn/weights/yolo-world/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth"
        cfg.load_from = checkpoint
        self.yolo_world = init_detector(cfg, checkpoint=checkpoint, device="cuda")
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.yolo_world_test_pipeline = Compose(test_pipeline_cfg)
        with open("data/yolo_labels.txt") as f:
            lines = f.readlines()
        self.yolo_texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
        self.yolo_world.reparameterize(self.yolo_texts)
        self.yolo_score = 0.1
        self.yolo_max_dets = 100

        # tap model 
        model_type = "tap_vit_l"
        checkpoint = "/home/dyn/outdoor/tokenize-anything/weights/tap_vit_l_03f8ec.pkl"
        concept_weights = "/home/dyn/outdoor/tokenize-anything/weights/merged_2560.pkl"
        self.nlp = spacy.load("en_core_web_sm")
        self.tap_model = model_registry[model_type](checkpoint=checkpoint)
        self.tap_model.concept_projector.reset_weights(concept_weights)
        self.tap_model.text_decoder.reset_cache(max_batch_size=1000)
        # SBERT model
        self.sbert_model = SentenceTransformer('/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2')
        

    def get_view_frustum_bds(self, depth_im, cam_intr, cam_pose):
        # Get the height and width of the depth image
        im_h, im_w = depth_im.shape
        max_depth = torch.max(depth_im)
        # Calculate frustum points in image space
        frustum_pts_x = (torch.tensor([0, 0, 0, im_w, im_w], dtype=torch.float32, device=depth_im.device) - cam_intr[2]) * \
                        torch.tensor([0, max_depth, max_depth, max_depth, max_depth], dtype=torch.float32, device=depth_im.device) / cam_intr[0]
        frustum_pts_y = (torch.tensor([0, 0, im_h, 0, im_h], dtype=torch.float32, device=depth_im.device) - cam_intr[3]) * \
                        torch.tensor([0, max_depth, max_depth, max_depth, max_depth], dtype=torch.float32, device=depth_im.device) / cam_intr[1]
        frustum_pts_z = torch.tensor([0, max_depth, max_depth, max_depth, max_depth], dtype=torch.float32, device=depth_im.device)
        # Stack to get (5, 3) points
        view_frustum_pts = torch.stack([frustum_pts_x, frustum_pts_y, frustum_pts_z], dim=1)
        # Apply rigid transformation (camera pose)
        xyz_h = torch.cat([view_frustum_pts, torch.ones(view_frustum_pts.shape[0], 1, dtype=view_frustum_pts.dtype, device=view_frustum_pts.device)], dim=1)
        view_frustum_pts_t = torch.matmul(xyz_h, cam_pose.T)
        # Return transformed points (5, 3)
        vol_bnds = torch.zeros((3,2)).to(self.device)
        vol_bnds[:,0] = torch.amin(view_frustum_pts_t[:, :3].T, axis=1)
        vol_bnds[:,1] = torch.amax(view_frustum_pts_t[:, :3].T, axis=1)
        vol_bnds = torch.round(vol_bnds / self.voxel_size) * self.voxel_size
        return vol_bnds
        
    
    def initialize_tsdfs(self):
        # Adjust volume bounds
        self.vol_bnds[:, 0] -= self.extent_bds
        self.vol_bnds[:, 1] += self.extent_bds
        self.vol_dim = torch.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size).clone().long()
        self.vol_bnds[:, 1] = self.vol_bnds[:, 0] + (self.vol_dim * self.voxel_size)
        self.vol_origin = self.vol_bnds[:, 0].clone()
        self.num_voxels = torch.prod(self.vol_dim).item()
        self.tsdf_vol = torch.ones(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        self.weight_vol = torch.zeros(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        self.color_vol = torch.zeros(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        self.get_vol_coord()
        # instance id and log for each voxel
        self.instance_id_vol = torch.zeros(tuple(self.vol_dim) + (3,), dtype=torch.int64, device=self.device)
        self.instance_pro_vol = torch.zeros(tuple(self.vol_dim) + (4,), dtype=torch.int64, device=self.device)
        self.instance_count_vol = torch.zeros(tuple(self.vol_dim), dtype=torch.int64, device=self.device)
        self.instance_feature = torch.zeros((1,384), device=self.device)
        self.instance_fea_count = torch.ones((1, 1), dtype=torch.int64, device=self.device)
        self.instance_fea_weight = torch.ones((1, 1), dtype=torch.float, device=self.device)
        self.last_num = 0
        Log("Initialize TSDF Voxel volume: {} x {} x {}".format(*self.vol_dim), tag="TSDF-Fusion")
        
        
    def incremental_tsdfs(self, new_bds):
        # merge bounds
        old_vol_bnds = self.vol_bnds.clone()
        self.vol_bnds[:, 0] = torch.min(self.vol_bnds[:, 0], new_bds[:, 0])
        self.vol_bnds[:, 1] = torch.max(self.vol_bnds[:, 1], new_bds[:, 1])
        if torch.equal(self.vol_bnds, old_vol_bnds):
            return
        min_smaller = new_bds[:, 0] < old_vol_bnds[:, 0]
        max_larger = new_bds[:, 1] > old_vol_bnds[:, 1]
        self.vol_bnds[min_smaller, 0] -= self.extent_bds
        self.vol_bnds[max_larger, 1] += self.extent_bds
        # new dims
        old_vol_dim = self.vol_dim
        self.vol_dim = torch.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size).clone().long()
        Log("Add TSDF Voxel Volume to: {} x {} x {}".format(*self.vol_dim), tag="TSDF-Fusion")
        self.vol_origin = self.vol_bnds[:, 0].clone()
        self.num_voxels = torch.prod(self.vol_dim).item()
        # Create new volume data with all values initialized to default (e.g., 1 for TSDF, 0 for weights and color)
        # Initialize the new TSDF, weight, and color volumes
        new_tsdf_vol = torch.ones(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        new_weight_vol = torch.zeros(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        new_color_vol = torch.zeros(tuple(self.vol_dim)).to(torch.float32).to(self.device)
        new_instance_id_vol = torch.zeros(tuple(self.vol_dim) + (3,), dtype=torch.int64, device=self.device)
        new_instance_pro_vol = torch.zeros(tuple(self.vol_dim) + (4,), dtype=torch.int64, device=self.device)
        new_instance_count_vol = torch.zeros(tuple(self.vol_dim), dtype=torch.int64, device=self.device)
        
        # Get the indices in the new grid that correspond to the old grid
        old_x = torch.arange(old_vol_dim[0], dtype=torch.long).to(self.device)
        old_y = torch.arange(old_vol_dim[1], dtype=torch.long).to(self.device)
        old_z = torch.arange(old_vol_dim[2], dtype=torch.long).to(self.device)
        old_grid = torch.meshgrid(old_x, old_y, old_z, indexing='ij')
        old_x, old_y, old_z = [g.flatten() for g in old_grid]  # Flatten the grids to 1D arrays
        new_x = old_x + (torch.clamp(old_vol_bnds[0, 0] - self.vol_bnds[0, 0], min=0) / self.voxel_size).long()
        new_y = old_y + (torch.clamp(old_vol_bnds[1, 0] - self.vol_bnds[1, 0], min=0) / self.voxel_size).long()
        new_z = old_z + (torch.clamp(old_vol_bnds[2, 0] - self.vol_bnds[2, 0], min=0)/ self.voxel_size).long()
        
        new_x = torch.clamp(new_x, min=0, max=self.vol_dim[0] - 1)
        new_y = torch.clamp(new_y, min=0, max=self.vol_dim[1] - 1)
        new_z = torch.clamp(new_z, min=0, max=self.vol_dim[2] - 1)
        
        # Copy the original data to the new volume for valid indices (masking invalid ones)
        # print(new_tsdf_vol.shape)
        # print(new_weight_vol.shape)
        # print(new_color_vol.shape)
        new_tsdf_vol[new_x, new_y, new_z] = self.tsdf_vol[old_x, old_y, old_z]
        new_weight_vol[new_x, new_y, new_z] = self.weight_vol[old_x, old_y, old_z]
        new_color_vol[new_x, new_y, new_z] = self.color_vol[old_x, old_y, old_z]
        new_instance_id_vol[new_x, new_y, new_z] = self.instance_id_vol[old_x, old_y, old_z]
        new_instance_pro_vol[new_x, new_y, new_z] = self.instance_pro_vol[old_x, old_y, old_z]
        new_instance_count_vol[new_x, new_y, new_z] = self.instance_count_vol[old_x, old_y, old_z]
        # Assign the new volumes to the instance variables
        self.tsdf_vol = new_tsdf_vol
        self.weight_vol = new_weight_vol
        self.color_vol = new_color_vol
        self.instance_id_vol = new_instance_id_vol
        self.instance_pro_vol = new_instance_pro_vol
        self.instance_count_vol = new_instance_count_vol
        # Finally, recompute the volume coordinates (if necessary)
        self.get_vol_coord()
        # # debug visuliazation the points before and after incremental
        # raw_pcd = o3d.geometry.PointCloud()
        # raw_points = self.world_coords[abs(self.tsdf_vol)<0.3]
        # raw_colors = np.tile(np.array([1,0.,0.]), (len(raw_points), 1))
        # raw_pcd.points = o3d.utility.Vector3dVector(raw_points.cpu().numpy())
        # raw_pcd.colors = o3d.utility.Vector3dVector(raw_colors)
        # new_pcd = o3d.geometry.PointCloud()
        # new_points = self.world_coords[abs(self.tsdf_vol)<0.3]
        # new_colors = np.tile(np.array([0,0.,1.]), (len(new_points), 1))
        # new_pcd.points = o3d.utility.Vector3dVector(new_points.cpu().numpy())
        # new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
        # o3d.visualization.draw_geometries([raw_pcd])
        # o3d.visualization.draw_geometries([new_pcd])
        # o3d.visualization.draw_geometries([raw_pcd, new_pcd])
        
        
    def merge_instance(self, masks, caption_fts, depth_im, cam_intr, cam_pose):
        '''obtain the 3D voxels for each mask in current image'''
        # get 3D points based on depth_im
        H, W = depth_im.shape
        depth_image_flat = depth_im.T.flatten()
        depth_mask = depth_image_flat>0.0
        f_x, f_y, c_x, c_y = cam_intr
        # Create meshgrid for pixel coordinates (x, y) from 0 to W-1 and 0 to H-1
        x_coords, y_coords = torch.meshgrid(torch.arange(W), torch.arange(H))  # x and y are the coordinate grids of W and H
        x_coords = x_coords.flatten().cuda()
        y_coords = y_coords.flatten().cuda()
        # Compute 3D points in the camera coordinate system
        Z = depth_image_flat  # Depth values
        X = (x_coords - c_x) * Z / f_x  # X coordinates
        Y = (y_coords - c_y) * Z / f_y  # Y coordinates
        # Stack the 3D points to form a (N, 3) tensor
        point_cloud_camera = torch.stack((X, Y, Z), dim=1)
        # Add a column of ones to the point cloud to make it homogeneous (N, 4)
        ones_column = torch.ones(point_cloud_camera.shape[0], 1, device=point_cloud_camera.device)
        point_cloud_homogeneous = torch.cat([point_cloud_camera, ones_column], dim=1)  # Shape: (N, 4)
        point_cloud_world_homogeneous = torch.matmul(point_cloud_homogeneous, cam_pose.T)  # Shape: (N, 4)
        point_cloud_world = point_cloud_world_homogeneous[:, :3]
        
        # get voxel_coords for input points
        voxel_coords = (point_cloud_world - self.vol_origin) / self.voxel_size
        voxel_coords = torch.floor(voxel_coords).long() 
        # avoid voxel_coords extend self.vol_dim
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], min=0, max=self.vol_dim[0] - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], min=0, max=self.vol_dim[1] - 1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], min=0, max=self.vol_dim[2] - 1)
        
        # current scene state
        count_mask = self.instance_count_vol>self.vox_min_count
        max_pro_indices = torch.argmax(self.instance_pro_vol[...,:3], dim=-1)
        max_pro_instance_id = torch.gather(self.instance_id_vol, dim=-1, index=max_pro_indices.unsqueeze(-1)).squeeze(-1)[count_mask].to(torch.long)
        
        # for each instance, we calculate the probablity and update the associated voxels
        for mask_id in range(masks.shape[0]):
            # print("*"*100)
            # associated voxels
            instance_voxel_coords = voxel_coords[masks[mask_id].T.flatten() & depth_mask]
            instance_voxel_coords = torch.unique(instance_voxel_coords, dim=0)
            # denosie
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(instance_voxel_coords.cpu().numpy())
            start_time = time.time()
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            instance_voxel_coords = instance_voxel_coords[ind]
            
            # # debug
            # inlier_cloud = pcd.select_by_index(ind)
            # outlier_cloud = pcd.select_by_index(ind, invert=True)  # 噪声点
            # outlier_cloud.paint_uniform_color([1, 0, 0])
            # inlier_cloud.paint_uniform_color([0, 0, 1]) 
            # print("remove_statistical_outlier time:", time.time() - start_time)
            # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            
            occupied_voxels_num = len(instance_voxel_coords)
            # if occupied_voxels_num < 10:
            #     continue
            # get the possible instance id and correspond  likehold probablity
            instance_vol_id = self.instance_id_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2]]
            exist_instance_ids = torch.unique(instance_vol_id[instance_vol_id != 0])
            # no informations, new instance for init
            if len(exist_instance_ids) == 0:
                self.instance_id_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2], 0] = len(self.instance_feature)
                self.instance_pro_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2], 0] += 1
                self.instance_feature = torch.cat((self.instance_feature, caption_fts[mask_id].unsqueeze(0)), dim = 0)
                self.instance_fea_count = torch.cat((self.instance_fea_count, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                self.instance_fea_weight = torch.cat((self.instance_fea_weight, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                self.instance_count_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2]] += 1
            else:
                # look for each possible instance id for current mask
                geometry_sim = []
                for id in exist_instance_ids:
                    indices = torch.nonzero(instance_vol_id == id)
                    # joint probabilites for id (for simply, use add)
                    this_id_pro_count = self.instance_pro_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], \
                                                    instance_voxel_coords[:,2]][[indices[:, 0], indices[:, 1]]]
                    this_id_all_count = self.instance_count_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], \
                                                    instance_voxel_coords[:,2]][indices[:, 0]]
                    this_id_pro = this_id_pro_count.float() / this_id_all_count.float()
                    # this_id_pro = torch.zeros_like(this_id_pro_count, dtype=torch.float32)
                    # have_count_mask = this_id_all_count != 0
                    # this_id_pro[have_count_mask] = this_id_pro_count.float()[have_count_mask] / this_id_all_count.float()[have_count_mask]
                    geometry_probability = this_id_pro.sum() / occupied_voxels_num
                    geometry_sim.append(geometry_probability)
                    
                geometry_sim = torch.stack(geometry_sim)
                # feature similarity
                this_instance_fea = caption_fts[mask_id]
                exist_instances_fea = self.instance_feature[exist_instance_ids]
                feature_sim = F.cosine_similarity(exist_instances_fea, this_instance_fea.unsqueeze(0), dim=1)
                # overall similartity
                overall_sim = geometry_sim * 0.8 + feature_sim * 0.2
                associated_id = None
                max_pro, max_index = torch.max(overall_sim, dim=0)
                if max_pro>self.pro_thre:
                    associated_id = exist_instance_ids[max_index]
                
                
                if associated_id is not None:
                    # visual ratio
                    associated_instance_volume = (max_pro_instance_id==associated_id).sum().float()
                    vis_ratio = torch.clip(occupied_voxels_num/associated_instance_volume, min=0., max=1.)*max_pro
                    
                    # update the feature
                    self.instance_feature[associated_id] = (self.instance_feature[associated_id] * self.instance_fea_weight[associated_id] + \
                                                                                this_instance_fea * vis_ratio ) / (self.instance_fea_weight[associated_id] + vis_ratio)
                    # self.instance_feature[associated_id] = (self.instance_feature[associated_id] * self.instance_fea_count[associated_id] + \
                    #                                                             this_instance_fea * vis_ratio ) / (self.instance_fea_count[associated_id] + 1)
                    
                    self.instance_fea_count[associated_id] += 1
                    self.instance_fea_weight[associated_id] += vis_ratio
                    # print(f"For new instance {mask_id} and for id {associated_id}, the max probability is {max_pro}")
                    # increase in existing probability for the voxles already has a instance id equal to the associated_id
                    associated_vol_indices = torch.nonzero(instance_vol_id == associated_id)
                    self.instance_pro_vol[instance_voxel_coords[:,0][associated_vol_indices[:, 0]], instance_voxel_coords[:,1][associated_vol_indices[:, 0]],\
                                                    instance_voxel_coords[:,2][associated_vol_indices[:, 0]], associated_vol_indices[:, 1]] += 1
                    # a new probable instance_id is added for the voxles with an empty space
                    if len(associated_vol_indices) < len(instance_vol_id):
                        mask = torch.all(instance_vol_id != associated_id, dim=1)
                        instance_ids = self.instance_id_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2]][mask]
                        is_zero = (instance_ids == 0).to(torch.int64)
                        the_first_zero_indices = torch.argmax(is_zero, dim=1)
                        have_full = is_zero.sum(dim=1) == 0
                        the_first_zero_indices[have_full] = self.instance_id_vol.size(-1)  # if the instance id list is full, assigned to others
                        # add the instance id to instance_id_vol
                        no_full_mask = mask.clone()
                        true_indices = no_full_mask.nonzero() 
                        mask_M_true_indices = have_full.nonzero()
                        true_indices_in_mask_M = true_indices[mask_M_true_indices[:, 0]]
                        no_full_mask[true_indices_in_mask_M[:, 0]] = False
                        self.instance_id_vol[instance_voxel_coords[:,0][no_full_mask], instance_voxel_coords[:,1][no_full_mask], \
                                                    instance_voxel_coords[:,2][no_full_mask], the_first_zero_indices[~have_full]] = associated_id
                        # update the probablity
                        self.instance_pro_vol[instance_voxel_coords[:,0][mask], instance_voxel_coords[:,1][mask], instance_voxel_coords[:,2][mask] \
                                                    , the_first_zero_indices] += 1
                else: # this is a new instance
                    new_instance_id = len(self.instance_feature)
                    self.instance_feature = torch.cat((self.instance_feature, caption_fts[mask_id].unsqueeze(0)), dim = 0)
                    self.instance_fea_count = torch.cat((self.instance_fea_count, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                    self.instance_fea_weight = torch.cat((self.instance_fea_weight, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                    instance_ids = self.instance_id_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2]]
                    is_zero = (instance_ids == 0).to(torch.int64)
                    the_first_zero_indices = torch.argmax(is_zero, dim=1)
                    have_full = is_zero.sum(dim=1) == 0
                    the_first_zero_indices[is_zero.sum(dim=1) == 0] = 3  # if the instance id list is full, assigned to others
                    self.instance_id_vol[instance_voxel_coords[:,0][~have_full], instance_voxel_coords[:,1][~have_full], instance_voxel_coords[:,2][~have_full] \
                                                , the_first_zero_indices[~have_full]] = new_instance_id
                    self.instance_pro_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2], the_first_zero_indices] += 1
                self.instance_count_vol[instance_voxel_coords[:,0], instance_voxel_coords[:,1], instance_voxel_coords[:,2]] += 1
        
        if (len(self.instance_feature)-self.last_num)>10:
            self.last_num = len(self.instance_feature)
            Log(f"Now the number of instance is: {len(self.instance_feature)}", tag="Open-Instance" )

        
    def erode_mask(self, mask, kernel_size=5):
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
        eroded_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=kernel_size//2)
        eroded_mask = eroded_mask.squeeze(0).squeeze(0) >= kernel_size*kernel_size
        return eroded_mask
    
    
    def get_instance_color(self, depth_im, cam_intr, cam_pose):
        '''obtain the instance color image for current image'''
        # get 3D points based on depth_im
        H, W = depth_im.shape
        depth_image_flat = depth_im.T.flatten() 
        depth_mask = depth_image_flat>0.0
        f_x, f_y, c_x, c_y = cam_intr
        # Create meshgrid for pixel coordinates (x, y) from 0 to W-1 and 0 to H-1
        x_coords, y_coords = torch.meshgrid(torch.arange(W), torch.arange(H))  # x and y are the coordinate grids of W and H
        x_coords = x_coords.flatten().cuda()
        y_coords = y_coords.flatten().cuda()
        # Compute 3D points in the camera coordinate system
        Z = depth_image_flat  # Depth values
        X = (x_coords - c_x) * Z / f_x  # X coordinates
        Y = (y_coords - c_y) * Z / f_y  # Y coordinates
        # Stack the 3D points to form a (N, 3) tensor
        point_cloud_camera = torch.stack((X, Y, Z), dim=1)
        # Add a column of ones to the point cloud to make it homogeneous (N, 4)
        ones_column = torch.ones(point_cloud_camera.shape[0], 1, device=point_cloud_camera.device)
        point_cloud_homogeneous = torch.cat([point_cloud_camera, ones_column], dim=1)  # Shape: (N, 4)
        point_cloud_world_homogeneous = torch.matmul(point_cloud_homogeneous, cam_pose.T)  # Shape: (N, 4)
        point_cloud_world = point_cloud_world_homogeneous[:, :3]
        # get voxel_coords for input points
        voxel_coords = (point_cloud_world - self.vol_origin) / self.voxel_size
        voxel_coords = torch.floor(voxel_coords).long() 
        # avoid voxel_coords extend self.vol_dim
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], min=0, max=self.vol_dim[0] - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], min=0, max=self.vol_dim[1] - 1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], min=0, max=self.vol_dim[2] - 1)
        # get the max_id for the voxel_coords
        max_pro_indices = torch.argmax(self.instance_pro_vol[...,:3], dim=-1)
        max_pro_instance_id = torch.gather(self.instance_id_vol, dim=-1, index=max_pro_indices.unsqueeze(-1)).squeeze(-1).to(torch.long)
        this_image_instance_id = max_pro_instance_id[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        
        this_image_instance_count = self.instance_count_vol[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        ok_mask = this_image_instance_count<=self.vox_min_count
        this_image_instance_id[ok_mask] = 0
        
        unique_labels, _ = torch.unique(this_image_instance_id, return_counts=True)
        unique_labels_count = self.instance_fea_count[unique_labels][:,0]
        labels_to_remove_min_count = unique_labels[unique_labels_count < self.ins_min_count]
        mask = torch.isin(this_image_instance_id, labels_to_remove_min_count)
        this_image_instance_id[mask] = 0
        
        unique_labels, _ = torch.unique(this_image_instance_id, return_counts=True)
        
        colors = torch.index_select(self.instance_colors, 0, this_image_instance_id)
        instance_image = torch.zeros((H * W, 3), dtype=torch.int64).cuda()
        pixel_coords = y_coords * W + x_coords 
        instance_image[pixel_coords] = colors
        self.images_vis[3] = instance_image.view(H, W, 3).cpu().numpy()
    
    
    def set_gui(self):
        # OpenCV window name
        self.window_name = "OpenVox - Visualization"
        self.images_vis = [np.zeros((self.vis_h, self.vis_w, 3), dtype=np.uint8)] * 5  # Placeholder for 4 images
        # Initialize OpenCV window (First time)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Display 4 images and 4 text lines in the OpenCV window
        self.edge = 50
        self.img_display = np.zeros(((self.edge*4+self.vis_h*2), (self.edge*3+self.vis_w*2), 3), dtype=np.uint8)  # Create a blank canvas
        if self.vis_w > 600:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 50)
        else:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 42)
        
        
    def integrate(self, color_im, depth_im, cam_intr, cam_pose, tstamp, obs_weight=1.0):
        if self.vis_gui:
            if self.vis_h is None:
                self.vis_h, self.vis_w = depth_im.shape[0], depth_im.shape[1]
                self.set_gui()
            self.images_vis[0] = torch.clamp(color_im.clone(), 0, 255).cpu().numpy()[:, :, ::-1] 
            min_depth, max_depth = 0.1, 5.0
            depth = depth_im.clone().cpu().numpy()
            depth = np.clip(depth, min_depth, max_depth)
            depth_norm = ((depth - min_depth) / (max_depth - min_depth)) * 255
            depth_norm = depth_norm.astype(np.uint8)
            self.images_vis[1] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        vol_bnds_this = self.get_view_frustum_bds(depth_im.float(), cam_intr.float(), cam_pose.float())
        if self.tsdf_vol is None:
            self.vol_bnds = vol_bnds_this
            self.initialize_tsdfs()
        else:
            self.incremental_tsdfs(vol_bnds_this)
        # self.tsdf_vol_last = self.tsdf_vol
        self.weight_vol_last = self.weight_vol.clone()
        if len(cam_intr==4):
            cam_intr_3x3 = torch.tensor([cam_intr[0], 0.0, cam_intr[2], 0.0, cam_intr[1], cam_intr[3]], dtype=torch.float32).to(self.device)
        # use cuda 
        im_h, im_w = depth_im.shape
        color_im_long = color_im.float()
        color_im_long = torch.floor(color_im_long[..., 2]*256*256 + color_im_long[..., 1]*256 + color_im_long[..., 0])
        
        other_params = torch.tensor(np.asarray([self.voxel_size, im_h, im_w, self.sdf_trunc,  obs_weight, 10.0])).to(torch.float32).to(self.device)
        # use cuda 
        tsdf_vol, weight_vol,  color_vol = integrate_tsdf(self.tsdf_vol.reshape(-1), self.weight_vol.reshape(-1), \
                            self.color_vol.reshape(-1),  self.vol_dim.to(torch.float32).reshape(-1), \
                            self.vol_origin.to(torch.float32).reshape(-1), cam_intr_3x3.to(torch.float32).reshape(-1), cam_pose.to(torch.float32).reshape(-1), \
                            other_params.reshape(-1), color_im_long.to(torch.float32).reshape(-1), depth_im.to(torch.float32).reshape(-1),)
        # the observed  voxels
        self.tsdf_vol = tsdf_vol.reshape(self.vol_dim.tolist())
        self.weight_vol = weight_vol.reshape(self.vol_dim.tolist())
        self.color_vol = color_vol.reshape(self.vol_dim.tolist())
        
        if tstamp%self.instance_skip != 0:
            if self.vis_gui:
                self.get_instance_color(depth_im, cam_intr, cam_pose)
                self.update_gui(tstamp)
            return
        
        img = color_im.cpu().numpy()[:,:,::-1]
        
        # start_time = time.time()
        
        '''[1] yolo-world'''
        data_info = dict(img=img, img_id=0, texts=self.yolo_texts)
        data_info = self.yolo_world_test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0), data_samples=[data_info['data_samples']])
        with torch.no_grad():
            output = self.yolo_world.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # score thresholding: only keep the instances with scores higher than the threshold
        pred_instances = pred_instances[pred_instances.scores.float() > self.yolo_score]
        # max detections: if the number of instances is more than the maximum allowed, keep the top-yolo_max_dets instances
        if len(pred_instances.scores) > self.yolo_max_dets:
            indices = pred_instances.scores.float().topk(self.yolo_max_dets)[1]
            pred_instances = pred_instances[indices]
        # bboxes
        min_rects = pred_instances['bboxes']
        min_rects = torch.unique(min_rects, dim=0).cpu().numpy() # min_rects are final detections
        # no object
        if len(min_rects) == 0:
            return
        
        '''[2] tap'''
        img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, img.shape[:2]
        img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
        inputs = self.tap_model.get_inputs({"img": img_batch})
        inputs.update(self.tap_model.get_features(inputs))
        batch_points = np.zeros((len(min_rects), 2, 3), dtype=np.float32)
        batch_points[:, 0, 0] = min_rects[:, 0]  # min x
        batch_points[:, 0, 1] = min_rects[:, 1]  # min y
        batch_points[:, 0, 2] = 2
        batch_points[:, 1, 0] = min_rects[:, 2]  # max x
        batch_points[:, 1, 1] = min_rects[:, 3]  # max y
        batch_points[:, 1, 2] = 3 
        inputs["points"] = batch_points
        inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
        outputs = self.tap_model.get_outputs(inputs)
        iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
        iou_score[:, 1:] -= 1000.0  # Penalize the score of loose points.
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)
        
        iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
        masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = self.tap_model.upscale_masks(masks, original_size).gt(0).squeeze(1)
        # print("detect the object number is ", len(masks))
        
        # sorted by mask area
        mask_areas = torch.tensor([mask.sum().item() for mask in masks])
        sorted_indices = torch.argsort(mask_areas, descending=True)
        sorted_masks = masks[sorted_indices]
        mask_id = torch.zeros(sorted_masks[0].shape)
        # smalls cover bigs
        ok_area_mask = []
        final_masks = []
        for i, mask in enumerate(sorted_masks):
            mask_id[mask] = i+1
        for new_id in range(len(sorted_masks)):
            new_mask = mask_id == new_id+1
            new_mask = self.erode_mask(new_mask)
            if torch.sum(new_mask) < 100:
                continue
            final_masks.append(new_mask)
            ok_area_mask.append(new_id)
        ok_area_mask = torch.tensor(np.stack(ok_area_mask)).long()
        final_masks = torch.stack(final_masks).cuda()
        
        if self.vis_gui:
            mask_image = np.ones(img.shape)*255*0.2
            for i in range(len(final_masks)):
                mask = final_masks[i].cpu().numpy()  
                color = np.random.random(3)*255
                mask_colored = np.stack([mask * color[0], mask * color[1], mask * color[2]], axis=-1)  
                mask_image = np.maximum(mask_image, mask_colored)  
            self.images_vis[2] = mask_image
        
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = self.tap_model.generate_text(sem_tokens)
        captions = captions[sorted_indices][ok_area_mask]
        new_captions = []
        for sentence in captions:
            doc = self.nlp(str(sentence))
            subject = ""
            for npp in doc.noun_chunks:
                if sentence.startswith(str(npp)):
                    subject = str(npp)
                    break
            if not subject:
                subject = sentence
            new_captions.append(subject)
        
        # print(len(captions))
        '''[3] sbert'''
        caption_fts = self.sbert_model.encode(new_captions, convert_to_tensor=True, device="cuda").detach()
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        
        # end_time = time.time()
        # print(end_time-start_time)
        
        '''[4] get 3D voxels'''
        self.merge_instance(final_masks, caption_fts, depth_im, cam_intr, cam_pose)
        
        if self.vis_gui:
            self.get_instance_color(depth_im, cam_intr, cam_pose)
            self.update_gui(tstamp)
    
    
    def update_gui(self, tstamp):
        # reset the text area
        self.img_display = np.zeros(((self.edge*4+self.vis_h*2), (self.edge*3+self.vis_w*2), 3), dtype=np.uint8) 
        # Arrange 4 images in a 2x2 grid
        self.img_display[self.edge*2:(self.edge*2+self.vis_h), self.edge:(self.edge+self.vis_w)] = self.images_vis[0]  # Top-left
        self.img_display[self.edge*2:(self.edge*2+self.vis_h), (2*self.edge+self.vis_w):(2*self.edge+self.vis_w*2)] = self.images_vis[2]  # Top-middle
        self.img_display[(self.edge*3+self.vis_h):(self.edge*3+self.vis_h*2), self.edge:(self.edge+self.vis_w)] = self.images_vis[1]  # Bottom-left
        self.img_display[(self.edge*3+self.vis_h):(self.edge*3+self.vis_h*2), (2*self.edge+self.vis_w):(2*self.edge+self.vis_w*2)] = self.images_vis[3]  # Bottom-middle
       
        # formatted_text = f"    Gaussians: {gaussian_count:6d}     KFs: {kf_len:3d}     Frame rate: {hz:2.2f} hz     Instance: {ins_num:3d}"
        formatted_text = f"Frame: {tstamp:4d}   Instance: {len(self.instance_feature):3d}"
        
        img_pil = Image.fromarray(self.img_display)
        draw = ImageDraw.Draw(img_pil)
        draw.text((300, 20), formatted_text, font=self.font, fill=(255, 255, 255))
        self.img_display = np.array(img_pil)
        
        img_display = cv2.resize(self.img_display, (int(self.img_display.shape[1] * 0.5), int(self.img_display.shape[0] * 0.5)))
        image_save_dir = f'{self.save_dir}/online_vis/'
        os.makedirs(image_save_dir, exist_ok=True)
        if tstamp % self.config["instance"]["instance_skip"] == 0:
            cv2.imwrite(f'{image_save_dir}/{tstamp}.jpg', img_display)
        cv2.imshow(self.window_name, img_display)
        cv2.resizeWindow(self.window_name, img_display.shape[1], img_display.shape[0])
        cv2.waitKey(1)  # Refresh window
        
        
    def vis_ply_final(self, vis=False):        
        # select the voxels that has been counted
        count_mask = self.instance_count_vol>self.vox_min_count
        points = self.world_coords[count_mask]
        pc_colors = self.color_vol[count_mask]
        max_pro_indices = torch.argmax(self.instance_pro_vol[...,:3], dim=-1)
        max_pro_instance_id = torch.gather(self.instance_id_vol, dim=-1, index=max_pro_indices.unsqueeze(-1)).squeeze(-1)[count_mask].to(torch.long)
        confidence  =  torch.max(self.instance_pro_vol[...,:3], dim=-1).values[count_mask]
        norm_confidence = confidence.float()/self.instance_count_vol[count_mask].float()
        # print(norm_confidence)
        cmap = plt.cm.rainbow
        confidence_colors = cmap(norm_confidence.cpu().numpy())[:, :3] # RGB channels only
    
        # remove the small label
        unique_labels, counts = torch.unique(max_pro_instance_id, return_counts=True)
        labels_to_remove_min_voxel = unique_labels[counts < self.ins_min_voxel]
        unique_labels_count = self.instance_fea_count[unique_labels][:,0]
        labels_to_remove_min_count = unique_labels[unique_labels_count < self.ins_min_count]
        labels_to_remove = torch.unique(torch.cat((labels_to_remove_min_voxel, labels_to_remove_min_count)))
        mask = torch.isin(max_pro_instance_id, labels_to_remove)
        
        points = points[~mask]
        pc_colors = pc_colors[~mask]
        confidence_colors = confidence_colors[~mask.cpu().numpy()]
        Log(f"Before move the instance is {len(unique_labels)}, after is {len(unique_labels)-len(labels_to_remove)}", tag="Open-Instance")
        max_pro_instance_id = max_pro_instance_id[~mask]
        colors = torch.index_select(self.instance_colors, 0, max_pro_instance_id)/255.0
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        if vis:
            o3d.visualization.draw_geometries([pc])
        o3d.io.write_point_cloud(f"{self.save_dir}/instance.ply", pc)
        print(len(pc.points))
        print(len(confidence_colors))
        pc.colors = o3d.utility.Vector3dVector(confidence_colors)
        o3d.io.write_point_cloud(f"{self.save_dir}/instance_confidence.ply", pc)
        # add real rgb color
        pc_colors = self.conver_color(pc_colors)
        # save instance pc
        instance_pc = np.hstack((points.cpu().numpy(), max_pro_instance_id.unsqueeze(dim=-1).cpu().numpy(), pc_colors))
        np.save(f"{self.save_dir}/instance_ids.npy", instance_pc)
        
        
    def conver_color(self, torch_color):
        if isinstance(torch_color, torch.Tensor):
            torch_color = torch_color.cpu().numpy()
        colors_b = np.floor(torch_color / self.const)
        colors_g = np.floor((torch_color - colors_b*self.const) / 256)
        colors_r = torch_color - colors_b*self.const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return colors
        
              
    def get_vol_coord(self, device="cuda"):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.vol_dim[0], device=device),
            torch.arange(self.vol_dim[1], device=device),
            torch.arange(self.vol_dim[2], device=device),
            indexing='ij'
        )
        coords = torch.stack([xv, yv, zv], dim=-1)
        self.world_coords = self.vol_origin.clone().to(device) + coords *  self.voxel_size
    
    
    def extract_triangle_mesh(self):
        """Extract a triangle mesh from the voxel volume using marching cubes.
        """
        tsdf_vol = self.tsdf_vol.cpu().numpy().reshape(self.vol_dim.tolist())
        color_vol = self.color_vol.cpu().numpy().reshape(self.vol_dim.tolist())
        vol_origin = self.vol_origin.cpu().numpy()
        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self.voxel_size + vol_origin
        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors = self.conver_color(rgb_vals)
        return verts, faces, norms, colors
    

    def meshwrite(self, filename, verts, faces, norms, colors):
        """Save a 3D mesh to a polygon .ply file.
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)  # 标准化颜色到[0, 1]
        o3d.io.write_triangle_mesh(filename, mesh)
    
    
    def save_eval(self, all_inputs):
        image_save_dir = f'{self.save_dir}/offline_vis/'
        os.makedirs(image_save_dir, exist_ok=True)
        for i in trange(len(all_inputs)):
            if i%self.instance_skip != 0:
                continue
            frame_data = all_inputs[i]
            depth_im = frame_data["depth"].clone().cuda()
            cam_intr = frame_data["intrinsics"].clone().cuda()
            cam_pose = frame_data["pose_44"].clone().cuda()
            self.get_instance_color(depth_im, cam_intr, cam_pose)
            bgr_image = self.images_vis[3]
            rgb_image = bgr_image[..., ::-1]
            cv2.imwrite(f'{image_save_dir}/{int(i)}.jpg', rgb_image)

        
    def finalize(self, all_inputs):
        verts, faces, norms, colors = self.extract_triangle_mesh()
        self.meshwrite(f'{self.save_dir}/tsdf_mesh.ply', verts, faces, norms, colors)
        self.vis_ply_final()
        # save instance caption feature,   N,384
        torch.save(self.instance_feature, f'{self.save_dir}/instance_feature.pt')
        self.save_eval(all_inputs)