#-*- coding: utf-8 -*-

"""
what    : train text
data    : IEMOCAP
"""
import os
import time
import argparse
import datetime

from model_text import *
from data_text import *
from params import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for training         
def train_step(params, model, b_trans, b_seqMask, b_label, optimizer, criterion):
    
    model.train()
    
    b_trans = b_trans.to(params.DEVICE)
    b_seqMask = b_seqMask.to(params.DEVICE)
    b_label = b_label.to(params.DEVICE)
    
    b_pred = model(b_trans, b_seqMask)
    b_loss = criterion(b_pred, b_label)
    
    optimizer.zero_grad()
    b_loss.backward()
    optimizer.step()
    
    return b_loss

    
def train_model(params, model, dataset_train, dataset_dev, dataset_test, valid_freq, is_save=0, graph_dir_name='default'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params.LR)
    criterion = nn.CrossEntropyLoss()
    
    early_stop_count = params.MAX_EARLY_STOP_COUNT
        
    # if exists check point, starts from the check point
    # TO DO
           
    # tensorboard
    writer = SummaryWriter('./graph/'+graph_dir_name)
        
    initial_time = time.time()
        
    min_ce = 1000000
    target_best    = 0
    target_dev_WA  = 0
    target_dev_UA  = 0
    target_test_WA = 0
    target_test_UA = 0
    
    
    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_size=params.BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4)
    
    data_loader_dev = DataLoader(dataset=dataset_dev,
                                   batch_size=params.BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=4)
    
    data_loader_test = DataLoader(dataset=dataset_test,
                                   batch_size=params.BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=4)
    
    index = 0
    while (index < params.MAX_TRAIN_STEPS and early_stop_count > 0):

        for b_trans, b_seqMask, b_label in data_loader_train:
        
            index = index + 1
            if index >= params.MAX_TRAIN_STEPS:
                break
            
            try:
                # run train 
                b_loss_train = train_step(params, model, b_trans, b_seqMask, b_label, optimizer, criterion)
                writer.add_scalar('loss/train', b_loss_train, index)
                
            except Exception as e:
                print ("excepetion occurs in train step: ", e)
                
                
            # run validation
            if (index + 1) % valid_freq == 0:

                dev_ce, dev_WA, dev_UA = evaluate(params=params,
                                                  model=model, 
                                                  data_loader=data_loader_dev
                                                 )
                writer.add_scalar('loss/dev', dev_ce, index)
                writer.add_scalar('accuracy/dev', dev_WA, index)


                if params.IS_TRAGET_OBJECTIVE_WA : target = dev_WA
                else                                   : target = dev_UA


                if ( target > target_best ):
                    target_best = target

                    # save best result
                    if is_save is 1:
                        print('TODO - impl save')

                    early_stop_count = params.MAX_EARLY_STOP_COUNT

                    test_ce, test_WA, test_UA = evaluate(params=params,
                                                         model=model, 
                                                         data_loader=data_loader_test
                                                        )                    

                    target_dev_WA  = dev_WA
                    target_dev_UA  = dev_UA
                    target_test_WA = test_WA
                    target_test_UA = test_UA

                else:
                    # early stopping
                    if early_stop_count == 0:
                        print ("early stopped")
                        break

                    early_stop_count = early_stop_count -1
                    

                print (str( int((time.time() - initial_time)/60) ) + " mins" + \
                    " step/seen/itr: " + str( index ) + "/ " + \
                                           str( index * params.BATCH_SIZE ) + "/" + \
                                           str( round( index * params.BATCH_SIZE / float(len(dataset_train)), 2)  ) + \
                    "\t(dev WA/UA): " + \
                    '{:.3f}'.format(dev_WA)  + '\t'  + \
                    '{:.3f}'.format(dev_UA)  + '\t' + \
                    "(test WA/UA): "  + \
                    '{:.3f}'.format(test_WA) + '\t'  + \
                    '{:.3f}'.format(test_UA) + '\t' + \
                    " loss: " + '{:.2f}'.format(dev_ce))

    writer.close()

    print ('Total steps : {}'.format(index) )

    print ('final result at best step \t' + \
                '{:.3f}'.format(target_dev_WA) + '\t' + \
                '{:.3f}'.format(target_dev_UA) + '\t' + \
                '{:.3f}'.format(target_test_WA) + '\t' + \
                '{:.3f}'.format(target_test_UA) + \
                '\n')

    # result logging to file
    with open('./TEST_run_result.txt', 'a') as f:
        f.write('\n' + \
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                params.DATA_PATH.split('/')[-2] + '\t' + \
                graph_dir_name + '\t' + \
                '{:.3f}'.format(target_dev_WA) + '\t' + \
                '{:.3f}'.format(target_dev_UA) + '\t' + \
                '{:.3f}'.format(target_test_WA) + '\t' + \
                '{:.3f}'.format(target_test_UA) )
        


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(params,
         is_save,
         graph_dir_name
         ):

    create_dir('save/')
    
    if is_save is 1:
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/'+graph_dir_name)
    
    # data class
    dataset_train = DataText(params, 'train')
    dataset_dev   = DataText(params, 'dev')
    dataset_test  = DataText(params, 'test')
    print('[INFO] #train\t:',len(dataset_train))
    print('[INFO] #dev\t:'  ,len(dataset_dev))
    print('[INFO] #test\t:' ,len(dataset_test))

    params.DIC_SIZE = dataset_train.dic_size
    
    model = ModelText(dic_size           = params.DIC_SIZE,
                      embed_dim          = params.EMBED_DIM,
                      hidden_dim         = params.HIDDEN_DIM,
                      num_layers         = params.NUM_LAYER,
                      num_category       = params.N_CATEGORY,
                      dr                 = params.DR,
                      use_glove          = params.USE_GLOVE,
                      glove_embedding    = dataset_train.load_glove(),
                      embedding_finetune = params.EMBEDDING_FINETUNE,
                      use_attention      = params.ATTENTION
                      )

    model.to(params.DEVICE)
    if params.ATTENTION:
        model.memory = model.memory.to(params.DEVICE)
    print("Initialization done!")

    valid_freq = int( len(dataset_train) * params.EPOCH_PER_VALID_FREQ / float(params.BATCH_SIZE)  ) + 1
    print ("[INFO] Valid Freq = " + str(valid_freq))

    train_model(params, model, dataset_train, dataset_dev, dataset_test, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':

    # Common
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max_train_steps', type=int, default=-1)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    
    # Text
    p.add_argument('--use_glove', type=int, default=0)
    p.add_argument('--embedding_finetune', type=int, default=1)
    p.add_argument('--encoder_size_text', type=int, default=750)
    p.add_argument('--num_layer_text', type=int, default=1)
    p.add_argument('--hidden_dim_text', type=int, default=50)
    p.add_argument('--dr_text', type=float, default=1.0)
    p.add_argument('--attn_text', type=int, default=0)
    args = p.parse_args()

    _params = Params()
    
    _params.DATA_PATH    = args.data_path
    _params.BATCH_SIZE   = args.batch_size
    _params.LR           = args.lr
    _params.USE_GLOVE    = args.use_glove
    _params.EMBEDDING_FINETUNE = args.embedding_finetune
    _params.ENCODER_SIZE = args.encoder_size_text
    _params.NUM_LAYER    = args.num_layer_text
    _params.HIDDEN_DIM   = args.hidden_dim_text
    _params.DR           = args.dr_text
    _params.ATTENTION    = args.attn_text
    _params.DEVICE       = device
    
    if (args.max_train_steps != -1): 
        _params.MAX_TRAIN_STEPS = args.max_train_steps
        print('[INFO] max_train_steps:\t', _params.MAX_TRAIN_STEPS)
    
    glove_finetune = ''
    if _params.EMBEDDING_FINETUNE == False:
        glove_finetune = 'F'
    
    
    graph_name = args.graph_prefix + \
                    '_D' + (args.data_path).split('/')[-2] + \
                    '_b' + str(args.batch_size) + \
                    '_esT' + str(args.encoder_size_text) + \
                    '_LT' + str(args.num_layer_text) + \
                    '_HT' + str(args.hidden_dim_text) + \
                    '_G' + str(args.use_glove) + glove_finetune + \
                    '_drT' + str(args.dr_text) + \
                    '_attnT' + str(args.attn_text)
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")

    print('[INFO] device:\t\t', _params.DEVICE)
    print('[INFO] data:\t\t', _params.DATA_PATH)
    print('[INFO] batch:\t\t', _params.BATCH_SIZE)
    print('[INFO] #-class:\t\t', _params.N_CATEGORY)
    print('[INFO] lr:\t\t', _params.LR)
    
    print('[INFO]-T encoder_size:\t', _params.ENCODER_SIZE)
    print('[INFO]-T num_layer:\t', _params.NUM_LAYER)
    print('[INFO]-T hidden_dim:\t', _params.HIDDEN_DIM)
    
    if _params.USE_GLOVE == 1:
        print('[INFO]-Glove Finetune:\t', _params.EMBEDDING_FINETUNE)
    
    print('[INFO]-T dr:\t\t', _params.DR)
    
    if _params.ATTENTION: 
        print('[INFO]-T attention:\t', _params.ATTENTION)

    main(params = _params,
         is_save = args.is_save,
         graph_dir_name = graph_name
        )