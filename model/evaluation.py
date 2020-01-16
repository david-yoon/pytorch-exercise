#-*- coding: utf-8 -*-

"""
what    : evaluation
data    : IEMOCAP
"""

#from tensorflow.core.framework import summary_pb2
import torch
import torch.nn as nn
from random import shuffle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from params import *

#from torch.utils.tensorboard import SummaryWriter

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def evaluate(params, model, data_loader):
    
    model.eval()
    
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction='sum')

        list_loss = []
        list_pred = []
        list_label = []

        for b_trans, b_seqMask, b_label in data_loader:

            b_trans = b_trans.to(params.DEVICE)
            b_seqMask = b_seqMask.to(params.DEVICE)
            b_label = b_label.to(params.DEVICE)
            
            try:
                b_pred = model(b_trans, b_seqMask)
                b_loss = criterion(b_pred, b_label)

            except Exception as e:
                print ('excepetion occurs in evaluate step : ', e)

            # batch loss
            list_loss.append( b_loss )
            
            # batch prediction
            list_pred.append( torch.argmax(input=b_pred, dim=-1) )
            list_label.append( b_label )
            
        # flatten
        list_pred  = torch.cat(list_pred)
        list_label = torch.cat(list_label)

        
        # for log
        if params.TEST_LOG_FOR_ANALYSIS:
            with open( '../analysis/inference_log/mult_attn.txt', 'w' ) as f:
                f.write( ' '.join( [str(x) for x in list_pred] ) )

            with open( '../analysis/inference_log/mult_attn_label.txt', 'w' ) as f:
                f.write( ' '.join( [str(x) for x in list_label] ) )


        # macro : unweighted mean
        # weighted : ignore class unbalance
        accr_WA = precision_score(y_true=list_label.tolist(),
                               y_pred=list_pred.tolist(),
                               average=params.WA)

        accr_UA = precision_score(y_true=list_label.tolist(),
                               y_pred=list_pred.tolist(),
                               average=params.UA)

        sum_batch_ce = np.sum( list_loss )

        #writer.add_scalar('loss/val', sum_batch_ce, global_step)
        #writer.add_scalar('accuracy/val', accr_WA, global_step)

        return sum_batch_ce, accr_WA, accr_UA