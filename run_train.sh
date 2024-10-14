set -e
WORK_DIR=./work_dir
tag=$(date +%Y%m%d%H%M%S)
seed=$RANDOM
cd $WORK_DIR
echo $PWD
conda activate ?
pip install --upgrade pip setuptools
pip install timm==0.3.2
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
OUTPUT_DIR=$WORK_DIR/logs/$tag
mkdir -p $OUTPUT_DIR

LOG_PATH=$OUTPUT_DIR/log_output.txt

export LOCAL_RANK=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29501 main_pretrain.py --distributed   --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}  \
   --seed $seed | tee 2>&1 $LOG_PATH
