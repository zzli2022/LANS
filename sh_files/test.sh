CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_pretrain \
--pretrain_path /mnt/pfs/jinfeng_team/MMGroup/lzz/software/3399.pth \
--evaluate_only \
--eval_method completion \
--project_dropout 0.2 \
--project_type cnn \
--resume_model log/0602/2024-06-02-15-19-26/480.pth \
--dump_path ./test_merge/0602

# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port=$((RANDOM + 10000)) \
# start.py \
# --dataset Geometry3K \
# --use_pretrain \
# --pretrain_path /mnt/pfs/jinfeng_team/RLHF/lzz/software/3399.pth \
# --evaluate_only \
# --eval_method completion \
# --project_dropout 0.2 \
# --project_type cnn \
# --resume_model log/0205/2024-02-05-12-47-34/best_model.pth \
# --dump_path ./test_merge/0206/dropout02_bs_256/

# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port=$((RANDOM + 10000)) \
# start.py \
# --dataset Geometry3K \
# --use_pretrain \
# --pretrain_path /mnt/pfs/jinfeng_team/RLHF/lzz/software/3399.pth \
# --evaluate_only \
# --eval_method completion \
# --project_dropout 0.2 \
# --project_type cnn \
# --resume_model log/0205/2024-02-05-14-22-04/best_model.pth \
# --dump_path ./test_merge/0206/dropout04_bs_256/


# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port=$((RANDOM + 10000)) \
# start.py \
# --dataset Geometry3K \
# --use_pretrain \
# --pretrain_path /mnt/pfs/jinfeng_team/RLHF/lzz/software/3399.pth \
# --evaluate_only \
# --eval_method completion \
# --project_dropout 0.2 \
# --project_type cnn \
# --resume_model log/0205/2024-02-05-07-34-11/480.pth \
# --dump_path ./test_merge/0206/dropout03_bs_256_480epoch/

# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port=$((RANDOM + 10000)) \
# start.py \
# --dataset Geometry3K \
# --use_pretrain \
# --pretrain_path /mnt/pfs/jinfeng_team/RLHF/lzz/software/3399.pth \
# --evaluate_only \
# --eval_method completion \
# --project_dropout 0.2 \
# --project_type cnn \
# --resume_model log/0205/2024-02-05-07-34-11/400.pth \
# --dump_path ./test_merge/0206/dropout03_bs_256_400epoch/

# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port=$((RANDOM + 10000)) \
# start.py \
# --dataset Geometry3K \
# --use_pretrain \
# --pretrain_path /mnt/pfs/jinfeng_team/MMGroup/lzz/code/LANS/log/2023-10-30-22-55-58/440.pth \
# --evaluate_only \
# --eval_method completion \
# --project_dropout 0.2 \
# --project_type cnn \
# --resume_model log/0205/2024-02-05-07-34-11/360.pth \
# --dump_path ./test_merge/0206/