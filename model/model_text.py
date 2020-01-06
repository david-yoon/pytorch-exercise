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
                 params,
                 device
                ):
        print('[DEBUG][object created] ', self.__class__.__name__)
        
        super().__init__()
        
        self.device       = device
        self.params       = params
        self.batch_size   = self.params.BATCH_SIZE
        self.dic_size     = self.params.DIC_SIZE
        self.encoder_size = self.params.ENCODER_SIZE
        self.num_layers   = self.params.NUM_LAYER
        self.use_glove    = self.params.USE_GLOVE
        self.hidden_dim   = self.params.HIDDEN_DIM
        
        if self.use_glove == 1:
            self.embed_dim = 300
        else:
            self.embed_dim = self.params.EMBED_DIM
        
        self.lr = self.params.LR
        self.dr = self.params.DR
        
        # bulid model
        self._create_embedding()
        self._create_gru_encoder()
        self._create_output_layers()


    def _create_embedding(self):
        print ('[DEBUG][launch-text] create embedding')
        
        if self.use_glove == 1:
            self.fn_embed = nn.Embedding.from_pretrained(embeddings=pretrained_embeds,
                                                         freeze=False
                                                        )
        else:
            self.fn_embed = nn.Embedding(num_embeddings=self.dic_size,
                                         embedding_dim=self.embed_dim,
                                         padding_idx=0
                                        )
    
    def _create_gru_encoder(self):
        print ('[DEBUG][launch-text] create text_encoder (GRU):')
        
        self.fn_encoder = nn.GRU(input_size  = self.embed_dim,
                                   hidden_size = self.hidden_dim,
                                   num_layers  = self.num_layers,
                                   bias = True,
                                   batch_first = True,
                                   dropout = self.dr,
                                   bidirectional = False
                                  )


    def _create_output_layers(self):
        print ('[DEBUG][launch-text] create output projection layer')

        # output * M + b
        self.fn_output = nn.Linear(in_features = self.hidden_dim,
                                   out_features = self.params.N_CATEGORY,
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
    
    
    
    