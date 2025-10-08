# e2vla

Codes under refactoring


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
TODO: add an image

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

**TODO:** upload self collected datasets.

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

# Fine-tune on Own Data

