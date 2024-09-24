import torch
import torch.nn as nn
from utils.utils import sequence_mask
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
            x: [B, max_len, d_model]
            pe: [1, max_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LearnedPositionEncoding(nn.Module):

    def __init__(self, d_model, max_len = 100):
        super(LearnedPositionEncoding, self).__init__()
        self.embeddings = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        """
            x: [B, max_len, d_model]
        """
        position_ids = self.position_ids[:x.size(-2)]
        
        return x + self.embeddings(position_ids)[None, :, :]

class TransformerEncoder(nn.Module):

    def __init__(self, num_patches, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.2):
        super(TransformerEncoder,self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.position_text = PositionalEncoding(d_model=d_model)
        self.position_img = LearnedPositionEncoding(d_model=d_model)
        self.img_pos = nn.Embedding(num_patches, d_model)

        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        """
            Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_emb, text_emb_src, num_patches, len_src):
        
        # mask
        src_key_padding_mask = ~sequence_mask(num_patches+len_src)
        # position encoding
        img_emb = self.position_img(img_emb)
        # img_emb = self.position_text(img_emb)
        text_emb_src = self.position_text(text_emb_src) 
        emb_src = torch.cat([img_emb, text_emb_src], dim=1)
        # encoder   
        memory = self.encoder(emb_src.permute(1,0,2), src_key_padding_mask=src_key_padding_mask)

        return memory.permute(1,0,2)