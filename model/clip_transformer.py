import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer
from typing import Optional
from torch import nn, Tensor
import math
import torch.nn.functional as F
from modules.detr_transformer_and_embed import TransformerEncoder, build_position_encoding
    
    
class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        
        self.concatTransformerEncoder = TransformerEncoder()
        
        self.pool_frames = Transformer(config)
        self.text_projection = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        position_embedding, txt_position_embedding = build_position_encoding(config)
        self.position_embedding = position_embedding
        self.txt_position_embedding = txt_position_embedding

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        text_features = self.clip.get_text_features(**text_data)
        text_features_Sequential = self.clip.text_model(**text_data, return_dict = False)[0]
        
        text_features_Sequential = text_features_Sequential.reshape(-1, 512)
        text_features_Sequential = self.text_projection(text_features_Sequential)
        text_features_Sequential = text_features_Sequential.reshape(batch_size, -1, 512)
        
        video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        
        src_vid_mask = torch.ones(batch_size, self.config.num_frames).to(torch.device("cuda:"+self.config.gpu))
        src_txt_mask = text_data['attention_mask']
        
        pos_vid = self.position_embedding(video_features, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = torch.zeros_like(text_features_Sequential)  # (bsz, L_txt, d)
        
        video_features = video_features.unsqueeze(1)
        video_features = video_features.repeat(1, batch_size, 1, 1)
        text_features_Sequential = text_features_Sequential.unsqueeze(0)
        text_features_Sequential = text_features_Sequential.repeat(batch_size, 1, 1, 1)
        video_textSeq_concat = torch.cat((video_features, text_features_Sequential), dim=2)
        video_textSeq_concat = video_textSeq_concat.reshape(batch_size*batch_size, -1, 512)
        
        src_vid_mask = src_vid_mask.unsqueeze(1)
        src_vid_mask = src_vid_mask.repeat(1, batch_size, 1)
        src_txt_mask = src_txt_mask.unsqueeze(0)
        src_txt_mask = src_txt_mask.repeat(batch_size, 1, 1)
        mask = torch.cat((src_vid_mask, src_txt_mask), dim=2)
        mask = mask.reshape(batch_size*batch_size, -1).bool()
        
        pos_vid = pos_vid.unsqueeze(1)
        pos_vid = pos_vid.repeat(1, batch_size, 1, 1)
        pos_txt = pos_txt.unsqueeze(0)
        pos_txt = pos_txt.repeat(batch_size, 1, 1, 1)
        pos = torch.cat((pos_vid, pos_txt), dim=2)
        pos = pos.reshape(batch_size*batch_size, -1, 512)
        
        # make batch here, 32 still big
        memory = []
        b_size = 320
        n_batches = (video_textSeq_concat.shape[0] + b_size - 1) // b_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * b_size
            end_idx = min((batch_idx + 1) * b_size, video_textSeq_concat.shape[0])
            memory.append(self.concatTransformerEncoder(video_textSeq_concat[start_idx:end_idx], ~mask, pos))
        memory = torch.cat(memory)
        
        vid_mem = memory[:, :self.config.num_frames]  # (bsz, L_vid, d)
        
        vid_mem = vid_mem.reshape(batch_size, batch_size, 12, 512)
        
        video_features_pooled = self.pool_frames(text_features, vid_mem)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled
        
        return text_features, video_features_pooled



        # text_features_Sequential_projected = []
        # batch = 1000
        # num_batches = (text_features_Sequential.shape[0] + batch - 1) // batch
        # for batch_idx in range(num_batches):
        #     start_idx = batch_idx * batch
        #     end_idx = min((batch_idx + 1) * batch, text_features_Sequential.shape[0])
        #     text_features_Sequential_projected.append([start_idx:end_idx]))
        # text_features_Sequential_projected = torch.cat(text_features_Sequential_projected)
        # text_features_Sequential = text_features_Sequential_projected.reshape(num_sentences, num_words, 512)