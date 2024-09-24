CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_pretrain \
--pretrain_path ./pretraining_model/3399.pth \
--evaluate_only \
--eval_method completion \
--project_dropout 0.2 \
--project_type cnn \
--resume_model log/0602/2024-06-02-15-19-26/480.pth \
--dump_path ./test_merge/0602