import torch
import argparse
import os
from natsort import natsorted
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import os.path as osp

from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from tqdm import tqdm, trange
import torch
from openvox import OpenVox
            
            
def rgbd_stream(rgbddir, depthdir, posefile, calib, undistort=False, cropborder=False, start=0, length=100000, max_depth=12.0, dataset="replica"):
    """ image generator """
    
    all_inputs = []

    calib = np.loadtxt(calib, delimiter=" ")
    if calib.ndim == 2:
        K = calib[:3,:3]
        calib = np.array([calib[0,0], calib[1,1], calib[0,2], calib[1,2]])
        depth_scale = 1000.0
    else:
        K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])
        depth_scale = calib[4]

    rgb_image_list = natsorted(os.listdir(rgbddir))[start:start+length]
    depth_image_list = natsorted(os.listdir(depthdir))[start:start+length]
    
    poses = []
    poses_4x4 = []
    
    with open(posefile, "r") as f:
        lines = f.readlines()
    for i in range(start, min(len(lines), start+length)):
        line = np.array(list(map(float, lines[i].split())))
        # for N,16
        c2w = line.reshape(4, 4)
        poses_4x4.append(c2w)
        w2c = np.linalg.inv(c2w)
        quat = R.from_matrix(w2c[:3, :3]).as_quat()
        pose = np.hstack((w2c[:3, 3], quat))
        poses.append(pose)
    poses_4x4 = torch.as_tensor(np.array(poses_4x4))
    poses = torch.as_tensor(np.array(poses))
    
    for t, (rgbfile, depthfile) in zip(trange(len(rgb_image_list)), zip(rgb_image_list, depth_image_list)):
        image = cv2.imread(os.path.join(rgbddir, rgbfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(depthdir, depthfile), cv2.IMREAD_UNCHANGED)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        h0, w0, _ = image.shape
        if h0%10 != 0:
            w1, h1 = 640, 480
        elif h0 == 680:
            w1, h1 = 600, 340
        elif h0 == 720:
            w1, h1 = 480, 270
        image = cv2.resize(image, (w1, h1))
        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)
        h0, w0 = depth.shape
        depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_NEAREST)
        depth = depth / depth_scale
        pose = poses[t]
        is_last = (t == len(rgb_image_list) - 1)
        
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            depth = depth[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder
            
        image = torch.as_tensor(image).permute(2, 0, 1)
        depth = torch.as_tensor(depth)
        depth[depth>max_depth]=0.0
        
        # Append a dictionary of data for the current frame
        frame_data = {
            'index': t,
            'image': image,
            'depth': depth,
            'pose': pose,
            'intrinsics': intrinsics,
            'pose_44': poses_4x4[t],
            'is_last': is_last,
            'depth_scale': depth_scale
        }
        # yield frame_data
        all_inputs.append(frame_data)
    return all_inputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="replica", help="dataset name")
    parser.add_argument("--scene", type=str, default="room_0", help="scene name")
    parser.add_argument("--max_depth", type=float, default=8.0, help="the max depth used for depth image")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")
    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")
    parser.add_argument("--vis_gui", action="store_true", help="use opencv to visuliazation the whole process")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--output", default='None', help="path to save output")
    args = parser.parse_args()
    
    if args.dataset == "replica":
        dataset_dir = "/data/dyn/object/vmap"
        rgbdir = f"{dataset_dir}/{args.scene}/imap/00/rgb"
        depthdir = f"{dataset_dir}/{args.scene}/imap/00/depth"
        pose = f"{dataset_dir}/{args.scene}/imap/00/traj_w_c.txt"
        args.calib = f"calib/{args.dataset}.txt"
    elif args.dataset == "scannet":
        dataset_dir = "/data/dyn/ScanNet/scans"
        rgbdir = f"{dataset_dir}/{args.scene}/color"
        depthdir = f"{dataset_dir}/{args.scene}/depth"
        pose = f"{dataset_dir}/{args.scene}/traj_w_c.txt"
        # pose = f"{dataset_dir}/{args.scene}/traj_hislam2_aligned.txt"
        args.calib = f"{dataset_dir}/{args.scene}/intrinsic/intrinsic_color.txt"
        args.cropborder = 6
    elif args.dataset == "owndata":
        dataset_dir = "/data/dyn/myself"
        rgbdir = f"{dataset_dir}/{args.scene}/color"
        depthdir = f"{dataset_dir}/{args.scene}/depth"
        pose = f"{dataset_dir}/{args.scene}/traj_w_c.txt"
        # pose = f"{dataset_dir}/{args.scene}/traj_hislam2_aligned.txt"
        args.calib = f"{dataset_dir}/{args.scene}/intrinsic/intrinsic_color.txt"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}.")
        
    args.config = f"config/{args.dataset}_config.yaml"
    
    if args.output == "None":
        args.output = f"outputs/{args.scene}"
    os.makedirs(args.output, exist_ok=True)
    
    all_inputs = rgbd_stream(rgbdir, depthdir, pose, args.calib, args.undistort, args.cropborder, args.start, args.length, args.max_depth)
    
    progress_bar = tqdm(range(0, len(all_inputs)), desc="Intergrating")
    
    ov = OpenVox(args.config, args.output, args.vis_gui)
    
    for i in trange(len(all_inputs)):
        frame_data = all_inputs[i]
        color_im = frame_data["image"].permute(1,2,0).clone().cuda()
        depth_im = frame_data["depth"].clone().cuda()
        cam_intr = frame_data["intrinsics"].clone().cuda()
        cam_pose = frame_data["pose_44"].clone().cuda()
        ov.integrate(color_im, depth_im, cam_intr, cam_pose, i)
    progress_bar.close()
    # end
    ov.finalize(all_inputs)   
    print("Done")