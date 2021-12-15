import os
import numpy as np

embedding_dim = 50
hidden_num = 128
num_layer = 2
dropout = 0.7
bidirectional = "False"
n_gram_num = 1
log_num = 10
batch_size = 16
learning_rate = 2*1e-3
epoch_num = 100
lstm_param_path = "path_to_the_model_param"
lstm_result_save_path = "path_to_the_lstm_module_result_file"
train_xgb_file = "path_to_featured_training_set"
val_xgb_file = "path_to_featured_validation_set"
test_xgb_file = "path_to_featured_test_set"
predict_xgb_classifer_file = "path_to_the_featured_sequnce_file_for_searching"