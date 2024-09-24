import torch
import torch.nn as nn
from model.backbone import get_visual_backbone
from model.fusion import get_fusion
from model.encoder import get_encoder, TransformerEncoder
from model.decoder import get_decoder
from utils.utils import *
import numpy as np
from einops.layers.torch import Rearrange


class TransformerPretrain(nn.Module):

    def __init__(self, cfg, src_lang, channels=3):
        super(TransformerPretrain, self).__init__()
        self.cfg = cfg
        
        # TODO （两种projection方式的比较）
        if self.cfg.project_type == 'linear':
            # img patch with linear projection 
            image_height = image_width = cfg.diagram_size 
            patch_height = patch_width = cfg.img_patch_size
            assert  image_height % patch_height ==0 and image_width % patch_width == 0
            self.num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, cfg.encoder_embedding_size)
            )
        else:
            # img patch with CNN projection
            image_height = image_width = cfg.diagram_size 
            patch_height = patch_width = cfg.img_patch_size
            assert  image_height % patch_height ==0 and image_width % patch_width == 0
            self.num_patches = (image_height // patch_height) * (image_width // patch_width)
            self.visual_extractor = get_visual_backbone(cfg)
            self.visual_emb_unify = nn.Linear(self.visual_extractor.final_feat_dim[0], cfg.encoder_embedding_size)
        
        # TODO (重要，不加则过拟合掉点，系数要调参）
        self.dropout = nn.Dropout(self.cfg.project_dropout)
        
        self.transformer_en = TransformerEncoder(self.num_patches, cfg.encoder_embedding_size)
        
        self.text_embedding_src = self.get_text_embedding_src(
            vocab_size = src_lang.n_words,
            embedding_dim = cfg.encoder_embedding_size,
            padding_idx = 0,
            pretrain_emb_path = cfg.pretrain_emb_path
        )
        self.class_tag_embedding = nn.Embedding(
            len(src_lang.class_tag), 
            cfg.encoder_embedding_size, 
            padding_idx=0
        )
        self.sect_tag_embedding = nn.Embedding(
            len(src_lang.sect_tag), 
            cfg.encoder_embedding_size, 
            padding_idx=0
        )
    
    def forward(self, diagram_src, text_dict, var_dict):
        '''
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'}
        '''
        
        # img feature
        # TODO (对应两种projection，目前的预训练模型采用的 linear projection)
        if self.cfg.project_type == 'linear':
            img_emb = self.to_patch_embedding(diagram_src)
        else:
            img_emb = self.visual_extractor(diagram_src) # B x dim x pn x pn
            img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1], -1).transpose(1,2)  # B x (pnxpn) dim
            img_emb = self.visual_emb_unify(img_emb)
        
        img_emb = self.dropout(img_emb)
        # text feature
        token_emb = self.text_embedding_src(text_dict['token'])
        class_tag_emb = self.class_tag_embedding(text_dict['class_tag'])
        sect_tag_emb = self.sect_tag_embedding(text_dict['sect_tag'])
        text_emb_src = token_emb.sum(dim=1) + sect_tag_emb + class_tag_emb
        transformer_outputs = self.transformer_en(img_emb, text_emb_src, self.num_patches, text_dict['len'])
        
        len_comb = text_dict['len'] + self.num_patches
        var_pos_comb = var_dict['pos'] + self.num_patches
        return transformer_outputs, len_comb, var_pos_comb
        
        # len_comb = text_dict['len'] 
        # var_pos_comb = var_dict['pos'] 
        # return transformer_outputs[:,self.num_patches:], len_comb, var_pos_comb

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict_model = pretrain_dict['state_dict'] \
                                if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict_model.items():
            if k in model_dict:
                if k.startswith("module"):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def get_text_embedding_src(self, vocab_size, embedding_dim, padding_idx, pretrain_emb_path):

        embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_emb_path!='':
            emb_content = []
            with open(pretrain_emb_path, 'r') as f:
                for line in f:
                    emb_content.append(line.split()[1:])
                vector = np.asarray(emb_content, "float32") 
            embedding_src.weight.data[-len(emb_content):]. \
                                    copy_(torch.from_numpy(vector))
        return embedding_src

class Network(nn.Module):
    
    def __init__(self, cfg, src_lang, tgt_lang):
        super(Network, self).__init__()
        self.cfg = cfg
        # define the encoder and decoder
        self.encoder = get_encoder(cfg)
        self.fusioner =  get_fusion(cfg)
        self.decoder = get_decoder(cfg, tgt_lang)
        # load pretrain model
        if cfg.use_pretrain:
            self.pretrain_module = TransformerPretrain(cfg, src_lang)
            if cfg.pretrain_path!='':
                self.pretrain_module.load_model(cfg.pretrain_path)
        else:
            self.text_embedding_src = self.get_text_embedding_src(
                vocab_size = src_lang.n_words,
                embedding_dim = cfg.encoder_embedding_size,
                padding_idx = 0,
                pretrain_emb_path = cfg.pretrain_emb_path
            )
            self.class_tag_embedding = nn.Embedding(
                len(src_lang.class_tag), 
                cfg.encoder_embedding_size, 
                padding_idx=0
            )
            self.sect_tag_embedding = nn.Embedding(
                len(src_lang.sect_tag), 
                cfg.encoder_embedding_size, 
                padding_idx=0
            )

        self.src_lang = src_lang

    def forward(self, diagram_src, text_dict, var_dict, exp_dict, is_train=False):
        '''
            diagram_src: B x C x W x H
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'} /
                        {'token', 'sect_tag', 'class_tag', 'subseq_len', 'item_len', 'item_quant'}
            var_dict = {'pos', 'len', 'var_value', 'arg_value'}
            exp_dict = {'exp', 'len', 'answer'}
        '''

        if self.cfg.use_pretrain:
            all_emb_src, len_comb, var_pos_comb = self.pretrain_module(diagram_src, text_dict, var_dict)
        else:
            # text feature
            token_emb = self.text_embedding_src(text_dict['token'])
            class_tag_emb = self.class_tag_embedding(text_dict['class_tag'])
            sect_tag_emb = self.sect_tag_embedding(text_dict['sect_tag'])
            # all feature
            all_emb_src = token_emb.sum(dim=1) + sect_tag_emb + class_tag_emb
        
        # encoder
        encoder_outputs, encode_hidden = self.encoder(all_emb_src, len_comb)
        # some_parameter
        fusioned_encoder_outputs = self.fusioner(encoder_outputs, text_dict['token'])
        problem_output = encode_hidden[-1:,:,:].repeat(self.cfg.decoder_layers, 1, 1)
        # decoder 
        outputs = self.decoder(fusioned_encoder_outputs, problem_output, \
                                len_comb, \
                                var_pos_comb, var_dict['len'], \
                                exp_dict['exp'], \
                                is_train)
        return outputs

    def freeze_module(self, module):
        self.cfg.logger.info("Freezing module of "+" .......")
        for p in module.parameters():
            p.requires_grad = False

    def load_model(self, model_path):
        # import pdb; pdb.set_trace()
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict_model = pretrain_dict['state_dict'] \
                            if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict_model.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        return pretrain_dict
    
    def get_text_embedding_src(self, vocab_size, embedding_dim, padding_idx, pretrain_emb_path):

        embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_emb_path!='':
            emb_content = []
            with open(pretrain_emb_path, 'r') as f:
                for line in f:
                    emb_content.append(line.split()[1:])
                vector = np.asarray(emb_content, "float32") 
            embedding_src.weight.data[-len(emb_content):]. \
                                    copy_(torch.from_numpy(vector))
        return embedding_src


def get_model(args, src_lang, tgt_lang):
    model = Network(args, src_lang, tgt_lang)
    args.logger.info(str(model))
    return model
