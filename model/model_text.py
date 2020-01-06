#-*- coding: utf-8 -*-

"""
what    :  Single Encoder Model for text - bidirectional
data    : IEMOCAP
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

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
                 use_glove,
                 dr
                ):
        print('[DEBUG][object created] ', self.__class__.__name__)
        
        super().__init__()
        
        if use_glove == 1:
            embed_dim = 300
            
        
        # bulid model
        self._create_embedding(dic_size, embed_dim, use_glove)
        self._create_gru_encoder(embed_dim, hidden_dim, num_layers, dr)
        self._create_output_layers(hidden_dim, num_category)


    def _create_embedding(self, dic_size, embed_dim, use_glove):
        print ('[DEBUG][launch-text] create embedding')
        
        if use_glove == 1:
            self.fn_embed = nn.Embedding.from_pretrained(embeddings=pretrained_embeds,
                                                         freeze=False
                                                        )
        else:
            self.fn_embed = nn.Embedding(num_embeddings=dic_size,
                                         embedding_dim=embed_dim,
                                         padding_idx=0
                                        )
    
    def _create_gru_encoder(self, intput_dim, hidden_dim, num_layers, dr):
        print ('[DEBUG][launch-text] create text_encoder (GRU):')
        
        self.fn_encoder = nn.GRU(input_size    = intput_dim,
                                   hidden_size = hidden_dim,
                                   num_layers  = num_layers,
                                   bias = True,
                                   batch_first = True,
                                   dropout = dr,
                                   bidirectional = False
                                  )


    def _create_output_layers(self, in_features_dim, out_features_dim):
        print ('[DEBUG][launch-text] create output projection layer')

        # output * M + b
        self.fn_output = nn.Linear(in_features = in_features_dim,
                                   out_features = out_features_dim,
                                   bias=True
                                  )

    
    def forward(self, inputs, seqN):
        
        embed = self.fn_embed(inputs)
        
        outputs, h_n = self.fn_encoder(pack_padded_sequence(input=embed,
                                                            lengths=seqN,
                                                            batch_first=True,
                                                            enforce_sorted=False
                                                           )
                                      )
        
        final_output = self.fn_output(h_n.squeeze(0))
        
        return final_output
    