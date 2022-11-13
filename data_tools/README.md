# Data tools
## Prequisites
- Data source structure
```
 <dataset_dir_name>
├──  raw_label.csv
├──  splited_label.csv
├──  train
│  ├──  1
│  ├──  2
│  ├──  3
    ...
```
## Tools
- Split to train/val
    ```
    python data_tools/split_dataset.py <path/to/target/data/source>
    ```
- COCO format to MMCLS format
    ```
    ln -s <path/to/coco/dataset> data
    python data_tools/coco_to_mmcls.py data/<dataset_dir_name>
    ```
    Example
    ```
    ln -sf /home/kiennt54/data/milo/liveness/train_images_1fps data
    python data_tools/coco_to_mmcls.py data/train_images_1fps
    ```
- ZALO AI to MMCLS
    ```
    ln -s <path/to/zalo_ai/dataset> data
    python data_tools/setup_dataset.py data/<dataset_dir_name>
    ```
    Example 
    ```
    ln -sf /home/kiennt54/data/milo/liveness/train_images_2fps data
    python data_tools/setup_dataset.py data/train_images_2fps
    ```
    Requirement: zalo dataset have structure as in prequisites