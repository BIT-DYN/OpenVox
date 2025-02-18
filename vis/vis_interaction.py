import matplotlib
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')
import argparse


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="room_0", help="scene name")
    args = parser.parse_args()
    
    result_dir = f'../outputs/{args.scene}'
    instance_path = f"{result_dir}/instance_ids.npy"
    feature_path = f"{result_dir}/instance_feature.pt"
    
    random_colors = torch.load("../data/instance_colors.pt").cpu().numpy()/255.0
    
    # load our instance map
    
    instance_npy = np.load(instance_path)
    points = instance_npy[:, :3]
    ids = instance_npy[:, 3]
    colors = instance_npy[:, 4:]/255.0
    
    instance_feature = torch.load(feature_path)
    print(len(points))
    # separate different isntance
    instance_pcds = []
    instance_rgbs = []
    unique_ids = np.unique(ids)
    print(len(unique_ids))
    num = 0
    for unique_id in unique_ids:
        mask = ids == unique_id
        filtered_points = points[mask.flatten()]
        filtered_colors = colors[mask.flatten()]
        num += len(filtered_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        instance_rgbs.append(filtered_colors)
        instance_pcds.append(pcd)
    print(num)
        
    # useful_feature
    instance_feature = instance_feature[torch.tensor(unique_ids).to(torch.int64)]
        
    
    print("Initializing SBERT model...")
    sbert_model = SentenceTransformer("/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2")
    sbert_model = sbert_model.to("cuda")
    print("Done initializing SBERT model.")
    
    cmap = matplotlib.colormaps.get_cmap("rainbow")
    
    # open3d vis window
    vis_window = o3d.visualization.VisualizerWithKeyCallback()
    vis_window.create_window(window_name='Open3D', width=1280, height=720)
    
    for geometry in instance_pcds:
        vis_window.add_geometry(geometry)  
    

    def color_by_instance(vis_window):
        for i in range(len(instance_pcds)):
            color = random_colors[i]
            pcd = instance_pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
        for mesh in instance_pcds:
            vis_window.update_geometry(mesh)
        print("seted instance color")
        
            
            
    def color_by_rgb(vis_window):
        for i in range(len(instance_pcds)):
            colors = instance_rgbs[i]
            pcd = instance_pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        for mesh in instance_pcds:
            vis_window.update_geometry(mesh)
        print("seted rgb color")
        

    def sim_and_update(similarities, vis_window, top_num = 0):
        top_indices = None
        if top_num != 0:
            top_values, top_indices = similarities.topk(top_num)
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]  
        for i in range(len(instance_pcds)):
            pcd = instance_pcds[i]
            if top_indices is None:
                pcd.colors = o3d.utility.Vector3dVector(np.tile([similarity_colors[i, 0].item(), similarity_colors[i, 1].item(), similarity_colors[i, 2].item()], (len(pcd.points), 1)))
            elif top_indices is not None and i in top_indices:
                pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(pcd.points), 1)))
            else:
                colors = instance_rgbs[i]
                pcd.colors = o3d.utility.Vector3dVector(colors)
        for mesh in instance_pcds:
            vis_window.update_geometry(mesh)
        print("seted object color")      

    def color_by_sbert_sim(vis_window):
        text_query = input("Enter your query: ")
        text_queries = [text_query]
        top_num = int(input("Enter your top num (such as 0(all) or 1 or 2): "))
        with torch.no_grad():
            object_sbert_ft = sbert_model.encode(text_queries, convert_to_tensor=True)
            object_sbert_ft /= object_sbert_ft.norm(dim=-1, keepdim=True)
            object_sbert_ft = object_sbert_ft.squeeze()
        similarities_sbert = F.cosine_similarity(object_sbert_ft.unsqueeze(0), instance_feature, dim=-1)
        sim_and_update(similarities_sbert, vis_window, top_num = top_num)
        
    
        
    vis_window.register_key_callback(ord("I"), color_by_instance)
    vis_window.register_key_callback(ord("R"), color_by_rgb)
    vis_window.register_key_callback(ord("F"), color_by_sbert_sim)
        
    vis_window.run()   

if __name__ == "__main__":
    main()