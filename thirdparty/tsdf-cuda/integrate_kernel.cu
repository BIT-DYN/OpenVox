#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

__global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
        // Get voxel index
        int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int vol_dim_x = (int) vol_dim[0];
        int vol_dim_y = (int) vol_dim[1];
        int vol_dim_z = (int) vol_dim[2];
        if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
            return;
        // Get voxel grid coordinates (note: be careful when casting)
        float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
        float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
        float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
        // Voxel grid coordinates to world coordinates
        float voxel_size = other_params[0];
        float pt_x = vol_origin[0]+voxel_x*voxel_size;
        float pt_y = vol_origin[1]+voxel_y*voxel_size;
        float pt_z = vol_origin[2]+voxel_z*voxel_size;
        float tmp_pt_x = pt_x-cam_pose[0*4+3];
        float tmp_pt_y = pt_y-cam_pose[1*4+3];
        float tmp_pt_z = pt_z-cam_pose[2*4+3];
        float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
        float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
        float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
        // Camera coordinates to image pixels
        int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
        int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
        // Skip if outside view frustum
        int im_h = (int) other_params[1];
        int im_w = (int) other_params[2];
        if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
            return;
        // Skip invalid depth
        float depth_value = depth_im[pixel_y*im_w+pixel_x];
        if (depth_value < 0.07)
            return;
        // Integrate TSDF
        float trunc_margin = other_params[3];
        float depth_diff = depth_value-cam_pt_z;
        if (depth_diff < -trunc_margin)
            return;
        float dist = fmin(1.0f,depth_diff/trunc_margin);
        float w_old = weight_vol[voxel_idx];
        float obs_weight = other_params[4];
        float w_new = w_old + obs_weight;
        weight_vol[voxel_idx] = w_new;
        tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
        // Integrate color
        float old_color = color_vol[voxel_idx];
        float old_b = floorf(old_color/(256*256));
        float old_g = floorf((old_color-old_b*256*256)/256);
        float old_r = old_color-old_b*256*256-old_g*256;
        float new_color = color_im[pixel_y*im_w+pixel_x];
        float new_b = floorf(new_color/(256*256));
        float new_g = floorf((new_color-new_b*256*256)/256);
        float new_r = new_color-new_b*256*256-new_g*256;
        new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
        new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
        new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
        color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
    }





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
integrateTSDFVolume(
    torch::Tensor &tsdf_vol,    // [D, H, W]
    torch::Tensor &weight_vol,  // [D, H, W]
    torch::Tensor &color_vol,   // [D, H, W]
    torch::Tensor &vol_dim,     // [3]
    torch::Tensor &vol_origin,  // [3]
    torch::Tensor &cam_intr,    // [3, 3]
    torch::Tensor &cam_pose,    // [4, 4]
    torch::Tensor &other_params,// [5]
    torch::Tensor &color_im,    // [H, W]
    torch::Tensor &depth_im     // [H, W]
) {
    TORCH_CHECK(tsdf_vol.is_cuda(), "tsdf_vol must be a CUDA tensor");
    TORCH_CHECK(weight_vol.is_cuda(), "weight_vol must be a CUDA tensor");
    TORCH_CHECK(color_vol.is_cuda(), "color_vol must be a CUDA tensor");
    TORCH_CHECK(vol_dim.is_cuda(), "vol_dim must be a CUDA tensor");
    TORCH_CHECK(vol_origin.is_cuda(), "vol_origin must be a CUDA tensor");
    TORCH_CHECK(cam_intr.is_cuda(), "cam_intr must be a CUDA tensor");
    TORCH_CHECK(cam_pose.is_cuda(), "cam_pose must be a CUDA tensor");
    TORCH_CHECK(other_params.is_cuda(), "other_params must be a CUDA tensor");
    TORCH_CHECK(color_im.is_cuda(), "color_im must be a CUDA tensor");
    TORCH_CHECK(depth_im.is_cuda(), "depth_im must be a CUDA tensor");

    int threads_per_block = 512;
    int num_voxels = tsdf_vol.numel();
    int blocks = (num_voxels + threads_per_block - 1) / threads_per_block;

    // // 打印张量的维度
    // std::cout << "tsdf_vol size: " << tsdf_vol.sizes() << std::endl;
    // std::cout << "weight_vol size: " << weight_vol.sizes() << std::endl;
    // std::cout << "color_vol size: " << color_vol.sizes() << std::endl;
    // std::cout << "vol_dim size: " << vol_dim.sizes() << std::endl;
    // std::cout << "vol_origin size: " << vol_origin.sizes() << std::endl;
    // std::cout << "cam_intr size: " << cam_intr.sizes() << std::endl;
    // std::cout << "cam_pose size: " << cam_pose.sizes() << std::endl;
    // std::cout << "other_params size: " << other_params.sizes() << std::endl;
    // std::cout << "color_im size: " << color_im.sizes() << std::endl;
    // std::cout << "depth_im size: " << depth_im.sizes() << std::endl;

    // // 从GPU拷贝vol_dim到CPU
    // std::cout << "Before kernel call: vol_dim = ";
    // for (int i = 0; i < vol_dim.size(0); ++i) {
    //     std::cout << vol_dim[i].item<float>() << " ";
    // }
    // std::cout << std::endl;

    integrate<<<blocks, threads_per_block>>>(
        tsdf_vol.data_ptr<float>(),
        weight_vol.data_ptr<float>(),
        color_vol.data_ptr<float>(),
        vol_dim.data_ptr<float>(),
        vol_origin.data_ptr<float>(),
        cam_intr.data_ptr<float>(),
        cam_pose.data_ptr<float>(),
        other_params.data_ptr<float>(),
        color_im.data_ptr<float>(),
        depth_im.data_ptr<float>()
    );

    return {tsdf_vol, weight_vol, color_vol};
}
