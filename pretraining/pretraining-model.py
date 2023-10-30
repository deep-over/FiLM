import argparse
import os
from glob import glob
import random
import torch

from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

class RobertaRePretraining():
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path="roberta-base")
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base')

    def main(self, args):
        path, file_name = args.train_dir, '*.json'
        train_dataset=[]
        
        # shuffling data
        json_list = glob(os.path.join(path, 'train_dataset', '*', file_name), recursive=True)
        random.shuffle(json_list)
        for file in json_list:
            textdataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=file,
                block_size=512,
                cache_dir=os.path.join(path,'cached_dataset')
            )
            train_dataset.extend(textdataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        training_args = TrainingArguments(
                output_dir = args.output_dir,
                overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            save_strategy='epoch',
            fp16=True,
            fp16_opt_level='O1',
            dataloader_num_workers=4
        )

        training_args._n_gpu = args.gpus
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
            )

        trainer.train()
        trainer.save_model(os.path.join(args.output_dir))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(prog="train", description="Re pretrain Roberta or BART")
    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output_dir", type=str, required=False, default='save_model')
    g.add_argument("--epochs", type=int, default=1, help="the numnber of training epochs")
    g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
    g.add_argument("--gpus", type=int, default=1, help="the number of gpus")
    g.add_argument("--batch-size", type=int, default=16, help="training batch size")
    g.add_argument("--train-dir", default='document', help="the directory of training data")

    retrain = RobertaRePretraining()

    retrain.main(parser.parse_args())