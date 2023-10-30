import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class parameters():

    prog_name = "generator"

    # set up your own path here
    root_path = "/home/jaeyoung/submission_code/FinQA"
    output_path = os.path.join(root_path, "output")
    cache_dir = os.path.join(root_path, "cache")
    dataset_path = os.path.join(root_path, "dataset")
    model_save_name = ""

    model_tokenizer = "roberta-base"
    # roberta-retrained
    pretrained_model = "roberta"
    # model name
    model_size = "HYdsl/FiLM"

    # input the gpu number you want to use
    gpu_num = 0

    ### files from the retriever results
    # train_file = os.path.join(dataset_path, "retrieve_train_correct.json")
    train_file = os.path.join(dataset_path, f'roberta-base-correct', "retrieve_ctrain_train_t3_correct.json")
    valid_file = os.path.join(dataset_path, f'roberta-base-correct', "retrieve_ctrain_dev_t3_correct.json")
    test_file = os.path.join(dataset_path, f'roberta-base-correct', "retrieve_ctrain_test_t3_correct.json")

    # infer table-only text-only

    op_list_file = os.path.join(root_path, 'code', prog_name, "operation_list.txt")
    const_list_file = os.path.join(root_path, 'code', prog_name, "constant_list.txt")

    # # model choice: bert, roberta, albert

    # single sent or sliding window
    # single, slide, gold, none
    retrieve_mode = "single"

    # use seq program or nested program
    program_mode = "seq"

    # train, test, or private
    # private: for testing private test data


    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    # device = "cuda"
    mode = "train"
    shape_token = 'none'
    saved_model_path = os.path.join(output_path, "saved_model")
    build_summary = False

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512 # 2k for longformer, 512 for others
    max_program_length = 30
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 16
    batch_size_test = 16
    epoch = 300 #300
    learning_rate = 2e-5

    report = 300
    report_loss = 100

    max_step_ind = 11