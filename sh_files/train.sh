CUDA_VISIBLE_DEVICES=0 /mnt/pfs/jinfeng_team/MMGroup/dsp/run/miniconda3/envs/themgpt/bin/python  \
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

# CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/pfs/jinfeng_team/RLHF/Yaguang/tool/anaconda3/bin/python3  \
#     -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=$((RANDOM + 10000)) \
#     start.py \
#     --dataset Geometry3K \
#     --use_pretrain \
#     --batch_size 256 \
#     --dropout_rate 0.3 \
#     --pretrain_path /mnt/pfs/jinfeng_team/RLHF/lzz/software/3399.pth \
#     --project_dropout 0.3 \
#     --project_type cnn \
#     --workers 32 \
#     --dump_path ./log/0206/group_ffn_hidden_size \
#     --d_ff_hidden 2048 \
#     --group_head_num 4 
    # longer epoch same iter number