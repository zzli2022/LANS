CUDA_VISIBLE_DEVICES=0,1 /mnt/pfs/jinfeng_team/MMGroup/dsp/run/miniconda3/envs/themgpt/bin/python  \
    -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    start.py \
    --dataset Geometry3K \
    --use_pretrain \
    --batch_size 128 \
    --dropout_rate 0.2 \
    --pretrain_path /mnt/pfs/jinfeng_team/MMGroup/lzz/software/3399.pth \
    --project_dropout 0.2 \
    --project_type cnn \
    --workers 32 \
    --dump_path ./log/0602/ \
    --d_ff_hidden 2048 