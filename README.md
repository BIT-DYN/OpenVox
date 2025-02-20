<p align="center">
<h1 align="center"><strong> OpenVox: Real-time Instance-level Open-vocabulary Probabilistic Voxel Representation</strong></h1>
</p>



<p align="center">
  <a href="https://open-vox.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
</p>


 ## 🏠  Abstract
In recent years, vision-language models (VLMs) have advanced open-vocabulary mapping, enabling mobile robots to simultaneously achieve environmental reconstruction and high-level semantic understanding. While integrated object cognition helps mitigate semantic ambiguity in point-wise feature maps, efficiently obtaining rich semantic understanding and robust incremental reconstruction at the instance-level remains challenging. To address these challenges, we introduce OpenVox, a real-time incremental open-vocabulary probabilistic instance voxel representation. In the front-end, we design an efficient instance segmentation and comprehension pipeline that enhances language reasoning through encoding caption. In the back-end, we implement probabilistic instance voxels and formalize the cross-frame incremental fusion process into two subtasks: instance association and live map evolution, ensuring robustness to sensor and segmentation noise. Extensive evaluations across multiple datasets demonstrate that OpenVox achieves state-of-the-art performance in zero-shot instance segmentation, semantic segmentation, and open-vocabulary retrieval. Furthermore, real-world robotics experiments validate OpenVox's capability for stable, real-time operation.

<img src="https://github.com/BIT-DYN/OpenVox/blob/master/assets/poster.jpg">

## 🛠  Install

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda env create -f environment.yml
conda activate openvox
```

###  Install YOLO-World Model
Follow the [instructions](https://github.com/AILab-CVC/YOLO-World#1-installation) to install the YOLO-World model and download the pretrained weights [YOLO-Worldv2-L (CLIP-Large)](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth).

###  Install TAP Model
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).


###  Install SBERT Model
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```


### Clone this repo

```bash
git clone https://github.com/BIT-DYN/OpenVox
cd OpenVox
```

## 📊 Prepare dataset
OpenVox has completed validation on Replica (as same with [vMap](https://github.com/kxhit/vMAP)) and Scannet. 
Please download the following datasets.

* [Replica Demo](https://huggingface.co/datasets/kxic/vMAP/resolve/main/demo_replica_room_0.zip) - Replica Room 0 only for faster experimentation.
* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip) - All Pre-generated Replica sequences.
* [ScanNet](https://github.com/ScanNet/ScanNet) - Official ScanNet sequences.



## 🏃 Run


### Main Code
Run the following command to identifie and comprehend object instances from color images.

```bash
# for replica
python main.py  --dataset replica --scene {scene} --vis_gui
# for scannet
python main.py  --dataset scannet --scene {scene} --vis_gui
```

You can see a visualization of the results in the ```outputs/{scene}``` folder.


###  visulization
You can interact with our visualization code, where the voxels are simplified to a point cloud.
```bash
cd vis/
python vis_interaction.py --scene {scene}
```


Then in the open3d visualizer window, you can use the following key callbacks to change the visualization.

Press ```R``` to color the voxels by RGB.

Press ```I``` to color the voxels by object instance ID.

Press ```F``` and type object text and num in the terminal, and the meshes will be colored by the similarity.
