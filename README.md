Das Ziel dieses Repositories ist ein Docker-Setup für ISBNet.

# Installation Guide
Bauen des Docker-Images
```
docker build -t isbnet:v2 .
```

Falls sich der Rechner aufgrund begrenzter Ressourcen aufhängt, setze MAX_JOBS Umgebungsvariable

# Date Preprocessing für S3DIS
Erstellen eine Verzeichnisses namens **workspace/dataset/s3dis/** im **isbnet_ws**-Verzeichnis. 
Die folgenden Dateien werden in dem gerade angelegten Verzeichnis **workspace/dataset/s3dis/** abgelegt. 
1. Download des S3DIS Datensatzes über dieses [Google Formular](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1).
Download des **Stanford3dDataset_v1.2_Aligned_Version.zip** und unzippen
im gerade angelegten Verzeichnis **isbnet_ws/workspace/dataset/s3dis/**.

Korrektur einiger Fehler im Datensatz via (siehe hierzu [ISBNet Issue 60](https://github.com/VinAIResearch/ISBNet/issues/60))
- Line 180389 of Stanford3dDataset_v1.2_Aligned_Version\Area_5\hallway_6\Annotations\ceiling_1.txt
- Line 741101 of Stanford3dDataset_v1.2_Aligned_Version\Area_2\auditorium_1\auditorium_1.txt
- Line 926337 of Stanford3dDataset_v1.2_Aligned_Version\Area_3\hallway_2\hallway_2.txt

Außerdem folgenden Dateinamen korrigieren im Verzeichnis 
**isbnet_ws/workspace/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version/Area_6/copyRoom_1**
von **copy_Room_1.txt** zu **copyRoom_1.txt**.

Herunterladen der superpoints, zu finden in [Data Preprocessing](https://github.com/VinAIResearch/ISBNet/blob/master/dataset/README.md#s3dis-dataset).
Unzippen und den Unterordner **learned_superpoint_graph_segmentations** ablegen im Verzeichnis 
**isbnet_ws/workspace/dataset/s3dis**.


Die Ordnerstruktur für den S3DIS-Datensatz soll wie hier angegeben sein: 
[Data Preprocessing](https://github.com/VinAIResearch/ISBNet/blob/master/dataset/README.md#s3dis-dataset).

Erstellen folgender Verzeichnisse im **workspace/dataset/s3dis/** Ordner:
```
mkdir preprocess 
mkdir superpoints
mkdir out
```

Lade das [vortrainierte Modell](https://github.com/VinAIResearch/ISBNet/tree/master?tab=readme-ov-file#s3dis) und lege es 
ebenfalls in dem **workspace/dataset/s3dis/** Ordner ab. 


2. Ausführen des Docker-Containers
Falls **run_isbnet_container.sh** nicht ausführbar ist:
```
chmod +x run_isbnet_container.sh
```
Starten des Containers
```
./run_isbnet_container.sh
cd /workspace/ISBNet/dataset/s3dis
```

Anpassen der Pfade im Skript **/workspace/ISBNet/dataset/s3dis/prepare_s3dis.py** (Zeile 15-17)
zu
```
parser.add_argument(
    "--data_dir", type=str, default="./Stanford3dDataset_v1.2_Aligned_Version/", help="Path to the original data"
)
```

und in preprocess_s3dis-Methode (Z. 138):
``` 
save_dir = "./preprocess"
``` 
und (Z. 139)
``` 
scene_pth = os.path.join(save_dir, f"{area}_{name}_inst_nostuff.pth")
```  

Anpassen der Pfade im Skript **/workspace/ISBNet/dataset/s3dis/prepare_superpoints.py** (Z. 7)
```
files = sorted(glob.glob("learned_superpoint_graph_segmentations/*.npy"))
``` 
und (Z. 15) 
```
torch.save((spp), f"superpoints/{area}_{room}.pth")
```

Vorbereitung der Daten 
```
cd /workspace/ISBNet/dataset/s3dis
bash prepare_data.sh
```

# Inferenz
Inferenz mittels vortrainiertes Modell für den S3DIS-Datensatz (Test-Area: Area_5):

```
cd /workspace/ISBNet
python3 tools/test.py configs/s3dis/isbnet_s3dis_area5.yaml dataset/s3dis/head_s3dis_area5.pth --out dataset/s3dis/out
```

Es gibt weitere config-files für die anderen Test-Areas.
