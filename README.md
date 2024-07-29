# Hierarchical Decomposed Dual-domain Deep Learning for Sparse-View CT Reconstruction 
Implementation of <a href="https://pubmed.ncbi.nlm.nih.gov/38457843/">Hierarchical Decomposed Dual-domain Deep Learning for Sparse-View CT Reconstruction </a> in Pytorch.

## Requirements
___PLEASE INSTALL !!! DOCKER !!! TO CREATE CONTAINER.___

https://docs.docker.com/engine/install/ubuntu/

## Getting Started

1. Clone this repository
   ```
   git clone git@github.com:hanyoseob/HDD-DL-for-SVCT.git
   ```

2. Create, Start, and Run Docker Container
   ```
   docker run -it -v [HOST_DIR]:/workspace/[CONTAINER_DIR] --gpus all --name [DOCKER_NAME] pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel /bin/bash
   ```

   ex) 
   ```
   docker run -it -v ./HDD-DL-for-SVCT.:/workspace/HDD-DL-for-SVCT. --gpus all --name mpi pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel /bin/bash
   ```

3. Install basic libraries
   ```
   apt-get update
   ```

   ```
   apt-get install git vim wget unzip -y
   ```

4. Move to the repository
   ```
   cd HDD-DL-for-SVCT.
   ```

5. Install the requirements
   ```
   pip install -r requirements.txt
   ```

6. Download pre-trained models
   ```
   wget https://www.dropbox.com/scl/fi/ryvda57j7dgu8yleqi51i/checkpoints.zip?rlkey=s6mczko9ajsm5va2x40e4u51t
   ```
   
   ```
   mv checkpoints.zip?rlkey=s6mczko9ajsm5va2x40e4u51t checkpoints.zip 
   ```
   
   ```
   unzip checkpoints.zip
   ```

   ```
   rm checkpoints.zip __MACOSX
   ```

7. Reproduce Figure 6 (i, ii, v, vi) for Transverse view
   ```
    python demo_fig6.py
   ```

## Prepare the datasets
Before training the image- and projection-domain network, 

___YOU MUST PREPARE THE DATASET.___


## Train and Test the model
### Train mode
1. II-Net trained at decomposition level 1 (*decomposition level 1 is equal to nstage=0)
   ```python
   python main_for_img2img.py \
         --scope ct_img2img \
         --mode train \
         --num_epoch 100 \
         --batch_size 4 \
         --dir_data [PATH_OF_TRAIN_DATA_DIRECTORY] \
         --nstage 0 \
         --loss_type_img img \
         --lr_type_img residual \
   ```

2. II-Net trained at decomposition level [N] (*decomposition level [N] is equal to nstage=[N-1])
     ```python
     python main_for_img2img.py \
           --scope ct_img2img \
           --mode train \
           --num_epoch 100 \
           --batch_size 4 \
           --dir_data [PATH_OF_TRAIN_DATA_DIRECTORY] \
           --nstage [N] \
           --loss_type_img img \
           --lr_type_img residual \
     ```

3. PI-Net trained at decomposition level 1 (*decomposition level 1 is equal to nstage=0)
     ```python
     python main_for_prj2img.py \
           --scope ct_prj2img \
           --mode train \
           --num_epoch 100 \
           --batch_size 4 \
           --dir_data [PATH_OF_TRAIN_DATA_DIRECTORY] \
           --nstage 0 \
           --loss_type_prj img \
           --lr_type_prj consistency \
     ```

4. PI-Net trained at decomposition level [N] (*decomposition level [N] is equal to nstage=[N-1])
    ```python
    python main_for_prj2img.py \
           --scope ct_prj2img \
           --mode train \
           --num_epoch 100 \
           --batch_size 4 \
           --dir_data [PATH_OF_TRAIN_DATA_DIRECTORY] \
           --nstage [N] \
           --loss_type_prj img \
           --lr_type_prj consistency \
    ```

### Test mode
1. II-Net trained at decomposition level [N] (*decomposition level [N] is equal to nstage=[N-1])
 ```python
 python main_for_img2img.py \
       --scope ct_img2img \
       --mode test \
       --batch_size 1 \
       --dir_data [PATH_OF_TEST_DATA_DIRECTORY] \
       --nstage [N] \
       --loss_type_img img \
       --lr_type_img residual \
 ```

2. PI-Net trained at decomposition level [N] (*decomposition level [N] is equal to nstage=[N-1])
    ```python
    python main_for_prj2img.py \
           --scope ct_prj2img \
           --mode test \
           --batch_size 1 \
           --dir_data [PATH_OF_TEST_DATA_DIRECTORY] \
           --nstage [N] \
           --loss_type_prj img \
           --lr_type_prj consistency \
    ```
      
