#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
integrateTSDFVolume(
    torch::Tensor &tsdf_vol,    // [D, H, W]
    torch::Tensor &weight_vol,  // [D, H, W]
    torch::Tensor &color_vol,   // [D, H, W]
    torch::Tensor &vol_dim,     // [3]
    torch::Tensor &vol_origin,  // [3]
    torch::Tensor &cam_intr,    // [3, 3]
    torch::Tensor &cam_pose,    // [4, 4]
    torch::Tensor &other_params,// [6]
    torch::Tensor &color_im,    // [H, W]
    torch::Tensor &depth_im     // [H, W]
);
