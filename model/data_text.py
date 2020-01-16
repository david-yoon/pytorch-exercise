#-*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from params import *


"""
what    : process data, generate batch
data    : IEMOCAP
"""
class DataText(Dataset):

    # initialize data
    def __init__(self, params, dataset_type):
        
        assert dataset_type in ['train', 'dev', 'test']
        
        self.params = params
        self.dataset_type = dataset_type
        
        # load data
        if self.dataset_type == 'train':
            self.dataset = self._load_data(self.params.DATA_TRAIN_TRANS,
                                                self.params.DATA_TRAIN_LABEL)
        
        elif self.dataset_type == 'dev':
            self.dataset  = self._load_data(self.params.DATA_DEV_TRANS,
                                               self.params.DATA_DEV_LABEL)
        
        elif self.dataset_type == 'test':
            self.dataset  = self._load_data(self.params.DATA_TEST_TRANS,
                                                self.params.DATA_TEST_LABEL)
       
        self.dic = {}
        self.dic_size = 0
        with open( self.params.DATA_PATH + params.DIC_NAME, 'rb' ) as f:
            tmp_data = f.readlines()
            tmp_data = [ x.strip() for x in tmp_data ]
            for tmp in tmp_data:
                self.dic[tmp] = len(self.dic)
                        
        self.dic_size = len( self.dic )

        
    def load_glove(self):
        if self.params.USE_GLOVE == 0:
            return None
        
        print('[INFO] load Glove pretrained model')
        return torch.tensor(np.load(open(self.params.DATA_PATH + self.params.GLOVE, 'rb')))
        
        
    def _load_data(self, text_trans, label):
     
        print ('[DEBUG] load data : ' + text_trans + ' ' + label)

        # load dataset
        tmp_text_trans         = np.load(self.params.DATA_PATH + text_trans)
        tmp_labels             = np.load(self.params.DATA_PATH + label)

        list_trans     = []
        list_text_seqN = []
        list_text_seqMask = []
        list_label     = []
        
        for text in tmp_text_trans:
            
            # trim tokens to max encoder size
            list_trans.append(text[:self.params.ENCODER_SIZE])
            
            # compute the length of sequence
            seqN = 0
            tmp_index = np.where( text == 0 )[0]
            if ( len(tmp_index) > 0 ) :                                # pad exists
                seqN =  np.min((tmp_index[0], self.params.ENCODER_SIZE))
            else :                                                    # no-pad
                seqN = self.params.ENCODER_SIZE

            seqMask = np.zeros(self.params.ENCODER_SIZE)
            seqMask[:seqN] = 1
            
            #list_text_seqN.append(seqN)
            list_text_seqMask.append(seqMask.tolist())
            
        '''
        for tmp_label in tmp_labels:
            
            tmp = torch.zeros(self.params.N_CATEGORY, dtype=torch.float16)
            tmp[tmp_label] = 1
            list_label.append(tmp)
        '''
        
        return TensorDataset(torch.as_tensor(list_trans),
                             torch.as_tensor(list_text_seqMask),
                             torch.from_numpy(tmp_labels)
                             )


    def __getitem__(self, index):
        return self.dataset[index]
            

    def __len__(self):
        return len(self.dataset)
    
    