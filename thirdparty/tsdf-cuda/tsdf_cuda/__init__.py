from typing import NamedTuple
import torch.nn as nn
import torch
from integrate_kernel_cuda import integrateTSDFVolume

class CalFusedTSDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tsdf_vol, weight_vol, color_vol,  vol_dim, vol_origin, cam_intr, cam_pose, other_params, color_im, depth_im):
        new_tsdf_vol, new_weight_vol, new_color_vol = integrateTSDFVolume(tsdf_vol, weight_vol, color_vol,  vol_dim, vol_origin, cam_intr, cam_pose, other_params, color_im, depth_im)
        return new_tsdf_vol, new_weight_vol, new_color_vol 
    

def integrate_tsdf(tsdf_vol, weight_vol, color_vol,  vol_dim, vol_origin, cam_intr, cam_pose, other_params, color_im, depth_im):
    new_tsdf_vol, new_weight_vol, new_color_vol  = CalFusedTSDF.apply(tsdf_vol, weight_vol, color_vol,  vol_dim, vol_origin, cam_intr, cam_pose, other_params, color_im, depth_im)
    return  new_tsdf_vol, new_weight_vol, new_color_vol 
    
