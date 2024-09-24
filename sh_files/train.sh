CUDA_VISIBLE_DEVICES=0 python  \
    -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    start.py \
    --dataset Geometry3K \
    --use_pretrain \
    --batch_size 128 \
    --dropout_rate 0.2 \
    --pretrain_path /mnt/pfs/jinfeng_team/MMGroup/lzz/software/3399.pth \
    --project_dropout 0.2 \
    --project_type cnn \
    --max_epoch 500 \
    --workers 32 \
    --dump_path ./log/0602/ \
    --d_ff_hidden 2048 
