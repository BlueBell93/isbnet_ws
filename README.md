Das Ziel dieses Repositories ist ein Docker-Setup für ISBNet.

# Installation Guide
Klonen des ISBNet Repos
```
mkdir workspace
cd workspace
git clone https://github.com/VinAIResearch/ISBNet.git
```

Bauen des Docker-Images
```
docker build -t isbnet:v2 .
```

Ausführen des Docker-Containers
```
./run_isbnet_container.sh
```
Falls sich der Rechner aufgrund begrenzter Ressourcen aufhängt, setze MAX_JOBS Umgebungsvariable
```
export MAX_JOBS=1
```
Installation von pointnet2
``` 
cd ~/workspace/ISBNet/isbnet/pointnet2
python3 setup.py bdist_wheel
cd ./dist
pip3 install <.whl>
```

Setup für ISBNet
```
cd ~/workspace/ISBNet
python3 setup.py build_ext develop
```


# Setup beim Starten der Dockerfile
Teile der Installation gehen verloren und müssen bei 
jedem Start des Docker-Containers ausgeführt werden

```
cd ~/workspace
./isbnet_setup.sh
```

# Date Preprocessing für S3DIS
1. Download des S3DIS Datensatzes über dieses [Google Formular](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1).
Download des **Stanford3dDataset_v1.2_Aligned_Version.zip** und unzippen
im Verzeichnis von **~/workspace/ISBNet/dataset/s3dis**

Korrektur einiger Fehler im Datensatz via (siehe hierzu [ISBNet Issue 60](https://github.com/VinAIResearch/ISBNet/issues/60))
- Line 180389 of Stanford3dDataset_v1.2_Aligned_Version\Area_5\hallway_6\Annotations\ceiling_1.txt
- Line 741101 of Stanford3dDataset_v1.2_Aligned_Version\Area_2\auditorium_1\auditorium_1.txt
- Line 926337 of Stanford3dDataset_v1.2_Aligned_Version\Area_3\hallway_2\hallway_2.txt

Herunterladen der superpoints nach [Data Preprocessing](https://github.com/VinAIResearch/ISBNet/blob/master/dataset/README.md).
Unzippen und den Ordner **learned_superpoint_graph_segmentations** ablegen im Verzeichnis 
**~/workspace/ISBNet/dataset/s3dis**. 

Anpassen der Pfade im Skript **~/workspace/ISBNet/dataset/s3dis/prepare_s3dis.py**
```
parser.add_argument(
    "--data_dir", type=str, default="./Stanford3dDataset_v1.2_Aligned_Version/", help="Path to the original data"
)
```
und in prepare_s3dis-Methode:
``` 
save_dir = "preprocess"
``` 
und 
``` 
scene_pth = os.path.join(save_dir, f"{area}_{name}_inst_nostuff.pth")
```  

Anpassen der Pfade im Skript **~/workspace/ISBNet/dataset/s3dis/prepare_superpoints.py**
```
files = sorted(glob.glob("learned_superpoint_graph_segmentations/*.npy"))
``` 
und 
```torch.save((spp), f"superpoints/{area}_{room}.pth")

```

Vorbereitung der Daten 
```
cd ~/workspace/dataset/s3dis
mkdir preprocess
mkdir superpoints
bash prepare_data.sh
```

# Inferenz
Vortrainiertes Modell ablegen in 
```
cd ~/workspace/ISBNet/dataset/s3dis
```
erstellen eines out-Directories für die Ergebnisse der Inferenz
``` 
mkdir ~/workspace/ISBNet/dataset/s3dis/out
``` 

Inferenz mittels vortrainiertes Modell für den S3DIS-Datensatz:

```
cd ~/workspace/ISBNet
python3 tools/test.py configs/s3dis/isbnet_s3dis_area5.yaml dataset/s3dis/head_s3dis_area5.pth --out dataset/s3dis/out
```


# TODOs
In Dockerfile schreiben, denn 8.9 ist zu aktuell und wird nicht unterstützt, ansonsten muss jedes mal 
aufs Neue die Environment Variable gesetzt werden...
```
export TORCH_CUDA_ARCH_LIST="8.6"
```
