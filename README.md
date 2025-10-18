# e2vla

Project under refactoring

Checkpoints, evaluation results and videos have been uploaded to google drive:
* Checkpoints: [link](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing)
* LIBERO scores: [link](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing)
* LIBERO videos: [link](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing)


# Dependency


# Pretrain
## 1. Dataset Preparation
* Droid
  
  We use the processed data from [cadence/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1) as it has camera extrinsic attached. Download it to anywhere you like, and make a symbolic link to it as `./data_raw/droid_1.0.1`. Then run:
  ```bash
  conda activate lerobot
  python data_prepare/process_droid.py \
    --input_root ./data_raw/droid_1.0.1 \
    --alter_vid_root VIDEO_DOWNLOAD_PATH \
    --output_root ./data_converted/droid \
    --skip_saved
  ```
  **Note:**
  * This requires [lerobot](https://github.com/huggingface/lerobot) installed. We use version 0.1.0. You may need to create a new conda environment (e.g. named `lerobot`) and install the package via:
    ```bash
    pip install "lerobot==0.1.0"
    ```
  * The initial downloads of video files may be incomplete (test at 2025/04). We need to download the full video files and place them at `VIDEO_DOWNLOAD_PATH`. **TODO:** upload scripts to fix this.
  
* Maniskill
  
  First download the [data](https://www.tensorflow.org/datasets/catalog/maniskill_dataset_converted_externally_to_rlds) to anywhere you like, e.g.:
  ```bash
  mkdir -p ANYWHERE/maniskill
  gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 ANYWHERE/maniskill
  ln -s ANYWHERE/maniskill ./data_raw/maniskill
  ```
  Then run:
  ```bash
  conda activate tensorflow
  python data_prepare/process_maniskill.py \
    --input_root ./data_raw/maniskill/0.1.0 \
    --output_root ./data_converted/maniskill/0.1.0 \
    --visualize
  ```
  Note:
  * This requires [tensorflow](https://www.tensorflow.org/install) installed. You may need to create a new conda environment (e.g. named `tensorflow`) to install it and run the above command to generate data.
  
* Metaworld
  
  This doesn't require downloading extra data. However, you may still need to create a new conda environment (e.g. named `metaworld-v3`) and then install the [metaworld](https://github.com/Farama-Foundation/Metaworld) package via:
  ```bash
  pip install "metaworld==2.0.0"
  ```
  Then run:
  ```bash
  conda activate metaworld-v3
  python data_prepare/process_metaworld.py \
    --output_root ./data_converted/metaworld \
    --visualize \
    --skip_saved
  ```
  Note:
  * Although we install "metaworld==2.0.0", it is actually version 3.

If you have downloaded and processed all the data, the file structure would be like this: 
```
data_raw
├── droid_1.0.1
│   ├── README.md
│   ├── data
│   │   ├── chunk-000
│   │   ├── chunk-001
│   │   └── ...
│   ├── meta
│   │   ├── episodes.jsonl
│   │   ├── episodes_stats.jsonl
│   │   ├── info.json
│   │   └── tasks.jsonl
│   └── videos
│       ├── chunk-000
│       ├── chunk-001
│       └── ...
├── libero
│   ├── datasets
│   ├── libero_10
│   │   ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
│   │   ├── KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5
│   │   └── ...
│   ├── libero_90
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5
│   │   └── ...
│   ├── libero_goal
│   │   ├── open_the_middle_drawer_of_the_cabinet_demo.hdf5
│   │   ├── open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5
│   │   └── ...
│   ├── libero_object
│   │   ├── pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
│   │   ├── pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5
│   │   └── ...
│   └── ...
└── maniskill
    └── 0.1.0
        ├── dataset_info.json
        ├── features.json
        ├── maniskill_dataset_converted_externally_to_rlds-train.tfrecord-00000-of-01024
        ├── maniskill_dataset_converted_externally_to_rlds-train.tfrecord-00001-of-01024
        └── ...

data_converted
├── drawer
│   ├── 0000.h5
│   ├── 0001.h5
│   └── ...
├── droid
│   ├── data
│   │   ├── chunk-000
│   │   ├── chunk-001
│   │   └── ...
│   └── videos
│       ├── chunk-000
│       ├── chunk-001
│       └── ...
├── libero
│   ├── libero_10_no_noops
│   │   ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
│   │   ├── KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
│   │   └── ...
│   ├── libero_90_no_noops (not used in fine-tuning)
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it
│   │   └── ...
│   ├── libero_goal_no_noops
│   │   ├── open_the_middle_drawer_of_the_cabinet
│   │   ├── open_the_top_drawer_and_put_the_bowl_inside
│   │   └── ...
│   ├── libero_object_no_noops
│   │   ├── pick_up_the_alphabet_soup_and_place_it_in_the_basket
│   │   ├── pick_up_the_bbq_sauce_and_place_it_in_the_basket
│   │   └── ...
│   └── libero_spatial_no_noops
│       ├── pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
│       ├── pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
│       └── ...
├── maniskill
│   └── 0.1.0
│       ├── 00000.h5
│       ├── 00001.h5
│       └── ...
├── metaworld
│   ├── assembly-v3
│   │   ├── 0000.h5
│   │   ├── 0001.h5
│   │   └── ...
│   ├── basketball-v3
│   │   ├── 0000.h5
│   │   ├── 0001.h5
│   │   └── ...
│   └── ...
├── oven
│   ├── 0000.h5
│   ├── 0001.h5
│   └── ...
└── pick-place-can
    ├── 0000.h5
    ├── 0001.h5
    └── ...
```

## (1.5) Data Visualization
Visualize the processed data is recommended before training. Run:
```bash
python datavis.py {DATASET_NAME}
```
to visualize the specified dataset. Run `python datavis.py -l` to list all the available datasets.

## 2. Start Pre-training
You can use `python train.py -h` to see the help message. To pretrain on the above three datasets, run:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain -s EXPERIMENT_NAME
```
To pretrain on all the datasets mentioned in paper, run:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain_extra -s EXPERIMENT_NAME
```
This will save the log to `./logs/E2VLA/EXPERIMENT_NAME` and save the checkpoints to `./checkpoints/E2VLA/EXPERIMENT_NAME`.

We have uploaded two checkpoints [here](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing).

# Fine-tune and Evaluation on LIBERO
## 1. Dataset Preparation
First download the [LIBERO dataset](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) to anywhere and then make a symbolink to `./data_raw/libero`. Then run 
```bash
conda activate libero
python data_prepare/process_libero.py \
  --libero_task_suite libero_spatial \
  --libero_raw_data_dir ./data_raw/libero \
  --libero_target_dir ./data_converted/libero \
  --skip_saved \
  --visualize
```
Change the libero_spatial to [libero_object, libero_goal, libero_10] for finetuning and evaluation on other task-suites.

## 2. Fine-tuning
For example, if we wnat to fine-tune on libero-10 from pretrained models:
```bash
CUDA_VISIBLE_DEVICES=x python train.py \
  --config finetune_libero_10 \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s FINETUNE_EXPERIMENT_NAME
```
This will load the config and the pre-trained weights. The fine-tuned weights are saved to `./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/`. We save the weights every 10k iterations by default.

Finetuned checkpoints can be found [here](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing).

## 3. Evaluation
* First we need to launch the pyro4 naming server (something like roscore). Open a separate terminal and run:
  ```bash
  pyro4-ns
  ```
  By default the naming server runs on `localhost:9090`.

* Launch planning service of your fine-tuned model:
  ```bash
  CUDA_VISIBLE_DEVICES=x python -m infer_utils.remote_service \
    --ckpt ./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/ckpt_xxxxxxx.pt \
    --uri CUSTOM_URI_NAME
  ```

* Start evaluation in simulation:
  ```bash
  python -m examples.libero.eval \
    --task_suite libero_10 \
    --uri CUSTOM_URI_NAME \
    --save --video
  ```
  The results are saved to `./eval_results/TASK_SUITE/URI/`, and the videos are saved to `./eval_videos/TASK_SUITE/URI/`

Evaluation results and videos using our fine-tuned checkpoints can be found [here](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing) and [here](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing).

# Fine-tune on Own Data

