#-*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
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
        
        # load data
        if dataset_type == 'train':
            self.trans, self.label = self.load_data(self.params.DATA_TRAIN_TRANS,
                                                    self.params.DATA_TRAIN_LABEL)
        
        elif dataset_type == 'dev':
            self.trans, self.label  = self.load_data(self.params.DATA_DEV_TRANS,
                                                     self.params.DATA_DEV_LABEL)
        
        elif dataset_type == 'test':
            self.trans, self.label  = self.load_data(self.params.DATA_TEST_TRANS,
                                                     self.params.DATA_TEST_LABEL)
       
        '''
        self.dic = {}
        self.dic_size = 0
        with open( self.params.data_path + params.dic_name, 'rb' ) as f:
            tmp_data = f.readlines()
            tmp_data = [ x.strip() for x in tmp_data ]
            for tmp in tmp_data:
                self.dic[tmp] = len(self.dic)
                        
        self.dic_size = len( self.dic )
        '''
        
    def load_data(self, text_trans, label):
     
        print ('load data : ' + text_trans + ' ' + label)
        output_set = []

        # text
        tmp_text_trans         = np.load(self.params.data_path + text_trans)
        tmp_label              = np.load(self.params.data_path + label)
        
        return torch.from_numpy(tmp_text_trans), torch.from_numpy(tmp_label)


    def __getitem__(self, index):
        return self.trans[index], self.label[index]
            

    def __len__(self):
        return len(self.trans)
    
    