import argparse
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
                   
criterion_list = ["CrossEntropy", "FocalLoss", "MaskedCrossEntropy"]
optimizer_list = ["SGD", "ADAM"]
scheduler_list = ["multistep",'cosine','warmup']
visual_backbone_list = ['ResNet10', 'mobilenet_v2']
encoder_list = ['lstm', 'gru', 'transformer']
decoder_list = ["rnn_decoder", "tree_decoder"]
eval_method_list = ["completion", "choice", "top3"]
dataset_list = ['Geometry3K', 'PGPS9K'] 

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch PGPS Training')
    # visual backbone
    ##############################################################################
    parser.add_argument('--visual_backbone', default="ResNet10", type=str, choices=visual_backbone_list)
    parser.add_argument('--diagram_size',  default=256, type=int)
    parser.add_argument('--img_patch_size',  default=32, type=int)
    # encoder model
    ##############################################################################
    parser.add_argument('--encoder_type', default="gru", type=str, choices=encoder_list)
    parser.add_argument('--encoder_layers', default=2, type=int)
    parser.add_argument('--encoder_embedding_size', default=256, type=int)
    parser.add_argument('--encoder_hidden_size', default=512, type=int)
    parser.add_argument('--max_input_len', default=400, type=int)
    # decoder model
    ##############################################################################
    parser.add_argument('--decoder_type', default="rnn_decoder", type=str, choices=decoder_list)
    parser.add_argument('--decoder_layers', default=2, type=int)
    parser.add_argument('--decoder_embedding_size', default=512, type=int)
    parser.add_argument('--decoder_hidden_size', default=512, type=int)
    parser.add_argument('--max_output_len', default=40, type=int)
    # general model
    ##############################################################################
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--beam_size', default=10, type=int)
    # optimizer
    ##############################################################################
    parser.add_argument('--optimizer_type', default="ADAMW", type=str, choices=optimizer_list)
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate without LM')
    parser.add_argument('--lr_LM', default=1e-4, type=float, help='initial learning rate of LM')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_epoch', default=540, type=int)
    parser.add_argument('--scheduler_type', default="warmup", type=str, choices=scheduler_list)
    parser.add_argument('--scheduler_step', default=[160, 280, 360, 440, 500], type=list)
    parser.add_argument('--scheduler_factor', default=0.5, type=float, help='learning rate decay factor')
    parser.add_argument('--cosine_decay_end', default=0.0, type=float, help='cosine decay end')
    parser.add_argument('--warm_epoch', default=40, type=int)
    # criterion
    ###############################################################################
    parser.add_argument('--criterion', default="MaskedCrossEntropy", choices=criterion_list, type=str)
    parser.add_argument('--eval_method', default="top3", choices=eval_method_list, type=str)
    # dataset      
    ################################################################################
    parser.add_argument('--dataset', default="PGPS9K", type=str, choices=dataset_list)
    parser.add_argument('--dataset_dir', default='/mnt/pfs/jinfeng_team/MMGroup/lzz/data/PGPS9K_all')
    parser.add_argument('--pretrain_vis_path', default='')
    parser.add_argument('--vocab_src_path', default='./vocab/vocab_src.txt')
    parser.add_argument('--vocab_tgt_path', default='./vocab/vocab_tgt.txt')
    parser.add_argument('--pretrain_emb_path', default='')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--random_prob', default=0.5, type=float)
    parser.add_argument('--without_stru', action='store_true', help='structure clauses are used or not')
    parser.add_argument('--trim_min_count', default=5, type=int, help='minimum number of word')
    parser.add_argument('--use_pretrain', action='store_true', help='use pretrain')
    parser.add_argument('--pretrain_path', default='./pretraining_model/LM_MODEL.pth')
    # print information
    ###################################################################################
    parser.add_argument('--dump_path', default="./log/", type=str, help='save log path')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval_epoch', default=40, type=int)
    # general config
    ###################################################################################
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--resume_model', default="", type=str, help='use pre-trained model')
    # DistributedDataParallel
    ###################################################################################
    parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--init_method', default="env://", type=str, help='distributed init method')
    parser.add_argument('--debug', action='store_true', help = "if debug than set local rank = 0")
    parser.add_argument('--seed', default=202302, type=int,help='seed for initializing training. ')
    ###################################################################################
    
    parser.add_argument('--project_dropout', default=0.2, type=float)
    parser.add_argument('--project_type', default='linear', type=str)
    parser.add_argument('--img_loc_match', action='store_true', help='structure clauses are used or not')
    #############################
    # --add_parameter for mm_group_attention
    parser.add_argument('--d_ff_hidden', default=2048, type=int,help='FFN dim For MMGroupAttention.')
    parser.add_argument('--group_attention_layers', default=1, type=int,help='Layernums For MMGroupAttention.')
    parser.add_argument('--group_head_num', default=8, type=int,help='Layernums For MMGroupAttention.')  
    ##############################
    return parser.parse_args()
