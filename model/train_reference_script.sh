###########################################
# Text
# encoder_size_audio = 750
# encoder_size_text  = 128
###########################################


CUDA_VISIBLE_DEVICES=0 python train_text.py --batch_size 128 --lr 0.001 --encoder_size_text 128 --num_layer_text 1 --hidden_dim_text 200 --dr_text 0.0 --is_save 0  --use_glove 1 --embedding_finetune 0 --graph_prefix 'NLP' --data_path '../data/target_seven_120/fold01/'