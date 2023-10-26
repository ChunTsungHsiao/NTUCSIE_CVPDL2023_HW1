# CVPDL_hw1
To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone https://github.com/Jacob-codingnoob/NTUCSIE_CVPDL2023_HW1.git
# Device

NVIDIA GeForce RTX 1080 ti.
    
# Run train code - DETR

### Create environment
use python=3.7.3,pytorch=1.9.0,cuda=11.1


    cd DINO
    conda create --name DINO python=3.7.3
    conda activate DINO
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    
### Train from pre-train weights
	python -m torch.distributed.launch --nproc_per_node=3 main.py --pretrain_model_path ./ckpts/checkpoint0011_4scale.pth --finetune_ignore label_enc.weight class_embed --output_dir logs/DINO/R50-MS4 -c ./config/DINO/DINO_4scale.py --coco_path  ./hw1_dataset --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0

### Train from your own checkpoints
	python -m torch.distributed.launch --nproc_per_node=3 main.py --pretrain_model_path ./logs/DINO/R50-epoch24/checkpoint_best_regular.pth --finetune_ignore label_enc.weight class_embed --output_dir logs/DINO/R50-MS4 -c ./config/DINO/DINO_4scale.py --coco_path  ./hw1_dataset --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0

### Test and generate output.json 
    python main.py --test -c config/DINO/DINO_4scale.py --output_dir logs/DINO/R50-MS4-%j --test_path ./hw1_dataset/vlid --test_checkopint ./logs/DINO/R50-epoch24/checkpoint_best_regular.pth #valid
    python main.py --test -c config/DINO/DINO_4scale.py --output_dir logs/DINO/R50-MS4-%j --test_path ./hw1_dataset/test --test_checkopint ./logs/DINO/R50-epoch24/checkpoint_best_regular.pth #test
    python evaluate.py ./output.json ./hw1_dataset/annotations/val.json
    
    
