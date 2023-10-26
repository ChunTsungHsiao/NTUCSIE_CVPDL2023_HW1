coco_path=$1
python main.py \
	--pretrain_model_path /home/yehcl/Desktop/r11945037cvpdl_hw/jupyter/r11945037_cvpdl_hw1/DINO/ckpts/checkpoint0011_4scale.pth\
	--finetune_ignore label_enc.weight class_embed\
	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale.py --coco_path  /home/yehcl/Desktop/r11945037cvpdl_hw/jupyter/r11945037_cvpdl_hw1/DINO/hw1_dataset\
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

	python -m torch.distributed.launch --nproc_per_node=3 main.py --pretrain_model_path ./ckpts/checkpoint0011_4scale.pth --finetune_ignore label_enc.weight class_embed --output_dir logs/DINO/R50-MS4 -c ./config/DINO/DINO_4scale.py --coco_path  ./hw1_dataset --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0