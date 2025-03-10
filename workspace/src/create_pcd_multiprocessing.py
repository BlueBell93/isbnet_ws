import numpy as np
from pathlib import Path
import torch
import argparse
from multiprocessing import Pool

CLASS_LABELS_S3DIS = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
)

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

def get_predicted_labels(scene_name, mask_valid, dir):
    instance_file = Path(dir) / 'pred_instance' / f"{scene_name}.txt"
    with open(instance_file, "r") as file: 
        masks = file.readlines()
    masks = [mask.rstrip().split() for mask in masks] # pro Eintrag: (Dateinamen, Sem_label, Score)
    #inst_label_pred_rgb = np.zeros((mask_valid.sum(), 3)) # das brauche ich nicht -> inst-labels
    instance_number = len(masks)
    instance_pointnumber = np.zeros(instance_number)
    instance_label = -100 * np.ones(mask_valid.sum()).astype(np.int)
    semantic_label = -100 * np.ones(mask_valid.sum()).astype(np.int)

    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1] # aufsteigend sortierte Liste
    i = 0
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        mask_path = Path(dir) / f"pred_instance/{masks[i][0]}"
        if float(masks[i][2]) < 0.1:
            i += 1
            continue
        mask = np.loadtxt(mask_path).astype(np.int)
        mask = mask[mask_valid]
        #cls = SEMANTIC_IDX2NAME[int(masks[i][1]) - 1] # das ist der semantic label
        cls_id = int(masks[i][1]) # label-id
        instance_pointnumber[i] = mask.sum() # Anzahl der Punkte
        instance_label[mask == 1] = i # mueeste das nicht schon die Liste sein, die ich brauche????
        # jetzt brauche ich noch die Liste mit den semantic Labels -> dann bin ich fast schon fertig
        semantic_label[mask == 1] = cls_id
    print(f"number of instances with score under 0.1: {i}")
    return semantic_label, instance_label
    # return: predicted labels: fuer jeden Punkt angeben, welches label es hat (nach der Vorhersage)
    # predicted inst labels: fuer jeden Punkt angeben, zu welcher Instanz es gehört (beginnend mit Idx 0 bis N)

def create_save_pcd(scene_name, args, current_area, vis_tasks):
    print(f"Creating pointcloud for scene {scene_name}.")
    args.scene_name = scene_name
    xyz, rgb, semantic_label, instance_label = torch.load(
        f"{args.data_root}/{args.split}/{args.scene_name}_inst_nostuff.pth"
    )
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    semantic_label = semantic_label.astype(np.int)
    instance_label = instance_label.astype(np.int)

    # NOTE split 4 to match with the order of model's prediction
    inds = np.arange(xyz.shape[0])

    xyz_list = []
    rgb_list = []
    semantic_label_list = []
    instance_label_list = []
    for i in range(4):
        piece = inds[i::4]
        semantic_label_list.append(semantic_label[piece])
        instance_label_list.append(instance_label[piece])
        xyz_list.append(xyz[piece])
        rgb_list.append(rgb[piece])
    
    xyz = np.concatenate(xyz_list, 0)
    rgb = np.concatenate(rgb_list, 0)
    semantic_label = np.concatenate(semantic_label_list, 0)
    instance_label = np.concatenate(instance_label_list, 0)

    xyz = xyz - np.min(xyz, axis=0)
    rgb = (rgb + 1) * 255.0

    mask_valid = semantic_label != -100
    xyz = xyz[mask_valid]
    rgb = rgb[mask_valid]
    semantic_label = semantic_label[mask_valid]
    instance_label = instance_label[mask_valid]

    if "inst_pred" in vis_tasks:
        pred_sem_label, pred_inst_label = get_predicted_labels(args.scene_name, mask_valid, args.prediction_path)
        # zusammensetzen der pointcloud mit xyz, rgb, semantic label, instance label zu einer npy-file
        pred_sem_label = pred_sem_label.reshape(len(pred_sem_label), 1)
        pred_inst_label = pred_inst_label.reshape(len(pred_inst_label), 1)
        pcd = np.concatenate((xyz, rgb, pred_sem_label, pred_inst_label), axis=1)
        # save results
        dir_name = Path("/root/workspace/results") / current_area
        dir_name.mkdir(parents=True, exist_ok=True)
        pcd_save_path = dir_name / f"{scene_name}.txt"
        np.savetxt(pcd_save_path, pcd)
        print(f"Saved scene {args.scene_name} to {pcd_save_path}.")


def main():
    parser = argparse.ArgumentParser("S3DIS-Vis")

    parser.add_argument("--data_root", type=str, default="dataset/s3dis")
    parser.add_argument("--scene_name", type=str, default="area_1_latest")
    parser.add_argument("--split", type=str, default="preprocess")
    parser.add_argument(
        "--prediction_path", help="path to the prediction results", default="/root/workspace/results/isbnet_scannetv2_val"
    )
    parser.add_argument("--point_size", type=float, default=15.0)
    parser.add_argument(
        "--task",
        help="all/input/sem_gt/inst_gt/superpoint/inst_pred",
        default="all",
    )
    args = parser.parse_args()

    if args.task == "all":
        vis_tasks = ["input", "sem_gt", "inst_gt", "superpoint", "inst_pred"]
    else:
        vis_tasks = [args.task]

    # gehe über alle Dateien 
    dataset_path = Path(args.prediction_path) / "pred_instance"

    scene_names = [content.stem for content in dataset_path.iterdir() if content.is_file()]
    
    current_area = args.scene_name

    params_creating_pcd = [(scene_name, args, current_area, vis_tasks) for scene_name in scene_names]
    with Pool(12) as pool:
        pool.starmap(create_save_pcd, params_creating_pcd)
    print(f"Finished processing area {current_area}!")


    # num_scenes = len(scene_names)
    # num_processed_scenes = 0
    # while num_processed_scenes < num_scenes:
    #     params_creating_pcd = []
    #     while len(params_creating_pcd) < 8 and num_scenes - num_processed_scenes > 0:
    #         param = (scene_names[num_processed_scenes], args, current_area, vis_tasks)
    #         params_creating_pcd.append(param)
    #         num_processed_scenes += 1
    #     with Pool() as pool:
    #         pool.starmap(create_save_pcd, params_creating_pcd)
    # print(f"Finished processing area {current_area}!")

    


if __name__ == "__main__":
    main()