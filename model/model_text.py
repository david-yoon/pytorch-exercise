#-*- coding: utf-8 -*-

"""
what    :  Single Encoder Model for text - bidirectional
data    : IEMOCAP
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

from model_attention_pytorch import luong_attention
from random import shuffle
import numpy as np

from params import *

class ModelText(nn.Module):
    
    def __init__(self,
                 dic_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 num_category,
                 dr,
                 use_glove,
                 glove_embedding = None,
                 embedding_finetune = True,
                 use_attention=0
                ):
        print('[DEBUG][object created] ', self.__class__.__name__)
        
        super().__init__()
        
        self.use_attention = use_attention
        
        if self.use_attention:
            print('[DEBUG] create attention memory, dim: ', hidden_dim)
            self.memory = torch.randn([1, hidden_dim], requires_grad=True)
        
        if use_glove == 1:
            embed_dim = 300
            
        print('[DEBUG] Embedding size:', embed_dim)
        
        # bulid model
        self._create_embedding(dic_size, embed_dim, use_glove, glove_embedding, embedding_finetune)
        self._create_gru_encoder(embed_dim, hidden_dim, num_layers, dr)
        self._create_output_layers(hidden_dim, num_category)


    def _create_embedding(self, dic_size, embed_dim, use_glove, glove_embedding, embedding_finetune):
        print('[DEBUG][launch-text] create embedding')
        print('[DEBUG] Glove finetuning:', embedding_finetune)
        
        if use_glove == 1:
            self.fn_embed = nn.Embedding.from_pretrained(embeddings=glove_embedding,
                                                         freeze=embedding_finetune,
                                                         padding_idx=0
                                                        )
        else:
            self.fn_embed = nn.Embedding(num_embeddings=dic_size,
                                         embedding_dim=embed_dim,
                                         padding_idx=0
                                        )
            
    
    def _create_gru_encoder(self, intput_dim, hidden_dim, num_layers, dr):
        print('[DEBUG][launch-text] create text_encoder (GRU):')
        
        self.fn_encoder = nn.GRU(input_size    = intput_dim,
                                   hidden_size = hidden_dim,
                                   num_layers  = num_layers,
                                   bias = True,
                                   batch_first = True,
                                   dropout = dr,
                                   bidirectional = False
                                  )


    def _create_output_layers(self, in_features_dim, out_features_dim):
        print('[DEBUG][launch-text] create output projection layer')

        # output * M + b
        self.fn_output = nn.Linear(in_features = in_features_dim,
                                   out_features = out_features_dim,
                                   bias=True
                                  )

    
    def forward(self, inputs, seq_mask):
        
        embed = self.fn_embed(inputs)
        seqN  = torch.sum(seq_mask, dim=-1)
        
        # outputs: (seq_len, batch, input_size)
        # h_n:     (num_layers * num_directions, batch, hidden_size)
        packed_outputs, h_n = self.fn_encoder(pack_padded_sequence(input=embed,
                                                            lengths=seqN,
                                                            batch_first=True,
                                                            enforce_sorted=False
                                                           )
                                      )
        
        outputs, output_lengths = pad_packed_sequence(packed_outputs,
                                                      batch_first=True,
                                                      padding_value=0.0,
                                                      total_length=embed.shape[1]
                                                     )
        
        if self.use_attention:
            self.query = self.memory.unsqueeze(-1)
            self.key   = outputs
            self.weighted_sum, self.norm_b_sim = luong_attention(self.key, self.query, seq_mask)
            final_output = self.weighted_sum
        
        else:
            # oonsider only the last layer
            final_output = self.fn_output(h_n[-1])
        
        return final_output
    