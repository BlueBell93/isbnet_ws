import open3d as o3d
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

path_data =  Path.home() / Path(r"datasets")
path_s3did_area5 = path_data / Path(r"Stanford3dDataset_v1.2_Aligned_Version") / Path(r"Area_5")
path_instseg_pred = path_data / Path(r"out_own_pretrained_models__all_epochs_best_model") / Path(r"pred_instance")
#path_instseg_pred = path_data / Path(r"out_own_pretrained_models_only_two_epochs") / Path(r"pred_instance") # old results
path_instseg_pred_mask = path_instseg_pred / Path(r"predicted_masks")

SEMANTIC_IDX2NAME = {
    0: "unannotated",
    1: "ceiling",
    2: "floor",
    3: "wall",
    4: "beam",
    5: "column",
    6: "window",
    7: "door",
    8: "chair",
    9: "table",
    10: "bookcase",
    11: "sofa",
    12: "board",
    13: "clutter",
}

def visualize_pc():
    area5_folder_files = [item for item in  path_s3did_area5.iterdir() if item.is_dir()]
    for path_room in area5_folder_files:
        pcd_filename = list(path_room.glob("*.txt"))[0]
        print(f"pcd_filename: {pcd_filename}")
        pcd_data = np.loadtxt(pcd_filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pcd_data[:, 3:6]/255.0)
        o3d.visualization.draw_geometries([pcd])

def visualize_mask(): # erstmal nur für eine PCD
    area5_folder_files = [item for item in  path_s3did_area5.iterdir() if item.is_dir()]
    pcd_path = area5_folder_files[0]
    pcd_filename = list(pcd_path.glob("*.txt"))[0]
    pcd_data = np.loadtxt(pcd_filename) 
    extension = "Area_5_" + pcd_filename.name
    with open(path_instseg_pred / extension, "r") as mask_paths:
        # Liste über alle Zeilen
        mask_list = mask_paths.readlines()
    #print(mask_list)
    path_mask, label, _ = mask_list[20].split(" ")
    print(path_mask)
    print(label)
    # lade path_mask als numpy array
    pred_mask = np.loadtxt(path_instseg_pred / path_mask)
    #print(pred_mask)
    # jede Instanz einzeln visualisieren (z.B. alle Punkte grau, außer die Instanz) + Label ausgeben (herausfinden, was das Label bedeutet)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
    color_grey = np.array([213, 221, 227])/255.0
    color_blue = np.array([17, 131, 212])/255.0
    #print(f"num points: {pcd_data.shape}")
    colors = np.full((pcd_data.shape[0], 3), color_grey)
    mask = pred_mask[:] == 1
    #print(f"mask: {mask}")
    #print(f"type of pred_mask: {type(pred_mask[0])}")
    colors[mask] = color_blue
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def create_colors_for_instance_seg_vis(num_labels):
    """
    Erstellt eine Farbtabelle für die Visualisierung der Instanzsegmentierungsergebnisse.

    Args:
        num_labels (int): Anzahl der verschiedenen Instanzlabels.

    Returns:
        numpy.ndarray: Array mit RGB-Werten für jede Instanz.
    """
    cmap = plt.get_cmap("tab20", num_labels) # "tab20", "viridis"
    colors = cmap(np.arange(num_labels))
    #colors = (colors[:, :3] * 255).astype(np.uint8)
    colors = (colors[:, :3] * 255).astype(np.uint8)
    return colors

colors_table = np.array([
    [102, 220, 225],
    [ 95, 179,  61],
    [234, 203,  92],
    [  3,  98, 243],
    [ 14, 149, 245],
    [ 46, 106, 244],
    [ 99, 187,  71],
    [212, 153, 199],
    [188, 174,  65],
    [153,  20,  44],
    [203, 152, 102],
    [214, 240,  39],
    [121,  24,  34],
    [114, 210,  65],
    [239,  39, 214],
    [244, 151,  25],
    [ 74, 145, 222],
    [ 14, 202,  85],
    [145, 117,  87],
    [184, 189, 221],
    [116, 237, 109],
    [ 85,  99, 172],
    [226, 153, 103],
    [235, 146,  36],
    [151,  62,  68],
    [181, 130, 160],
    [160, 166, 149],
    [  6,  69,   5],
    [ 52, 253, 112],
    [ 14,   1,   3],
    [ 76, 248,  87],
    [233, 212, 184],
    [235, 245,  26],
    [213, 157, 253],
    [ 68, 240,  37],
    [219,  91,  54],
    [129,   9,  51],
    [  0, 191,  20],
    [140,  46, 187],
    [147,   1, 254]
])

#visualize_pc()
# TODO: Visualisierung von PCD- Numpy array: (x y z r g b sem_label inst_label)
def visualize_inst_seg(file_name):
    path_pcd_files = Path(__file__).parent.parent / "workspace" / "results" / "pointcloud_data"
    with open(path_pcd_files / f"{file_name}", "r") as pcd_file:
        pcd = np.loadtxt(pcd_file)
    # separates Laden der Daten
    xyz = pcd[:, :3]
    rgb = pcd[:, 3:6]
    sem_labels = pcd[:, 6].astype(dtype=np.int16)
    inst_labels = pcd[:, 7].astype(dtype=np.int32)
    # Darstellung (erstmal nur für ein einziges Element)
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(xyz)
    color_grey = np.array([213, 221, 227])/255.0
    color_blue = np.array([17, 131, 212])/255.0
    # Wie viele Masken gibt es insgesamt?
    #print(f"inst_labels: {inst_labels.shape}")
    masks = np.unique(inst_labels)
    print(f"masks: {masks}")
    mask_numbers = len(masks)
    #color_table = create_colors_for_instance_seg_vis(mask_numbers)
    #print(f"color_table: {color_table}")
    #print(f"type(color_table): {type(color_table)}")
    #print(f"color_table.shape: {color_table.shape}")
    colors = np.full(xyz.shape, np.array([0, 0, 0]), dtype=np.float64)
    print(xyz.shape)
    print(colors.shape)
    for mask_idx, mask_label in enumerate(masks):
        # if mask_label == -100:
        #     continue
        #print(f"(idx, label): {mask_idx, mask_label}")
        curr_mask = inst_labels[:] == mask_label
        #print(f"len(mask): {mask.sum()}")
        #print(f"color with mask_idx {mask_idx}: {np.array(colors_table[mask_idx]) / 255.0}")
        #print(f"type(np.array(colors_table[mask_idx]) / 255.0): {type(np.array(colors_table[mask_idx]) / 255.0)}")
        #new_color = np.array(colors_table[mask_idx]) / 255.0
        new_color = colors_table[mask_idx]/255.0
        #print(f"new_color: {new_color}")
        #print(f"type new_color: {type(new_color[0])}")
        #print(f"type red: {type(np.array([1.0, 0.0, 0.0])[0])}")
        colors[curr_mask] = new_color #np.array([1, 0, 0]) # color_table[mask_idx] / 255.0
        #colors[curr_mask] = np.array([1.0, 0.0, 0.0])
        # pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd_vis])
    #print(f"colors: {colors}")
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_vis])
    # fuer jede eine andere Farbe...
    # extra Label fuer -100 (-> Rauschen)
    # Visualisierung, indem ich die Farbe jedem gebe
    # print(f"masks: {masks}")
    # print(f"len masks: {len(masks)}")
    # mask_idx = 39
    # mask = inst_labels[:] == mask_idx
    # colors = np.full((pcd.shape[0], 3), color_grey)
    # colors[mask] = color_blue
    # pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd_vis])

#     masks: [-100    0    1    2    3    4    5    6    7    8    9   10   11   12
#    13   14   15   16   17   18   19   20   21   22   23   26   28   29
#    30   32   33   40   42   43   55   57   61]
    # mask = 0
    # colors[mask] = color_blue
    # colors = np.full((pcd.shape[0], 3), color_grey)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])


visualize_inst_seg("Area_5_WC_1.txt")


# Visualisierung der PCD mit ihren Instanzen -> ganze PC visualisieren + bestimmte Punkte farblich hervorheben (oder so)
# Problem: ein Punkt kann mehreren Instanzen zugeordnet sein -> Wie oft passiert das???

# Laden der PC
# Laden der Instanzmasken
# Übersetzung der Labels (Was für eine Benchmark wurde da benutzt? Wie ist das Mapping zum S3DIS-Datensatz)
#visualize_mask()

# BENCHMARK_SEMANTIC_IDXS = [i for i in range(15)] 
# scan_ids = ["id_1", "id_2", "id_3", "id_4", "id_5"]
# benchmark_sem_ids = [BENCHMARK_SEMANTIC_IDXS] * len(scan_ids)
# print(benchmark_sem_ids)
# benchmark_sem_id = benchmark_sem_ids[0]

# label_id = 5  # 1-> 18
# label_id = label_id + 1  # 2-> 19 , 0,1: background
# label_id = benchmark_sem_id[label_id]
# print(label_id)