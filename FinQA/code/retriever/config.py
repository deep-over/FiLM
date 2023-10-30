import torch
import os

'''
CUDA_VISIBLE_DEVICES=1 nohup python downstream/FinQA/FinQA/code/retriever/RMain.py > 0102_rtrain.log 2>&1 &
nohup python downstream/FinQA/FinQA/code/retriever/Test.py > 0104_rtest_235.log 2>&1 &
tail -f 1117_train.log
ps -ef | grep
kill -9
'''
class parameters():

    prog_name = "retriever"

    # set up your own path here
    root_path = "/home/jaeyoung/FinQA/code/"
    dataset_path = "/home/jaeyoung/FinQA/"
    output_path = "output"
    cache_dir = "./cache"


    train_file = os.path.join(dataset_path, "dataset/train.json")
    valid_file = os.path.join(dataset_path, "dataset/dev.json")
    test_file = os.path.join(dataset_path, "dataset/test.json")

    op_list_file = os.path.join(root_path, prog_name, "operation_list.txt")
    const_list_file = os.path.join(root_path, prog_name, "constant_list.txt")

    # model choice: bert, roberta
    # pretrained_model = "bert"
    # model_size = "bert-base-uncased"

    mode = "train"
    # model_name
    pretrained_model = ""
    model_tokenizer = "roberta-base"
    model_size = "/home/jaeyoung/group_models/fin-roberta-retrain-random"
    # the name of your result folder.
    model_size_name = model_size.split("/")[-1]
    model_save_name = os.path.join(output_path, prog_name, f"{model_size_name}_{mode}") #"retriever-roberta-base-test"

    # pretrained_model = "roberta-retrained"
    # model_size = "/home/ailab/Desktop/JY/roberta-retrained/add-reuters"

    # pretrained_model = "sec-bert"
    # model_size = "nlpaueb/sec-bert-shape"

    # train, test, or private
    # private: for testing private test data
    GPU_NUM = 3 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # torch.device('cpu') # change allocation of current GPU
    # device = "cuda"
    resume_model_path = ""
    shape_token = ''

    ### to load the trained model in test time
    saved_model_path = "/home/jaeyoung/FinQA/output/output/retriever/fin-roberta-retrain-random_train_20230422000359/saved_model/model.pt"
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 5

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    max_program_length = 100
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 32 #16
    batch_size_test = 32 #16
    epoch = 100 #100
    learning_rate = 2e-5

    report = 300
    report_loss = 100
    
    
