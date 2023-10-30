import functools
import os
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from torch.nn.modules import padding
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
# using transformers early stopping
from transformers import EarlyStoppingCallback
import spacy
from itertools import product

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

metric = evaluate.load("seqeval")

spacy_tokenizer = spacy.load("en_core_web_sm")

model_dict = {
    "SALT-NLP/FLANG-BERT":["SALT-NLP/FLANG-BERT"],
    'roberta-base':[
    "models/FiLM_2B"
    'roberta-base',
    ],
    "ProsusAI/finbert":["models/finbert"],
    "bert-base-uncased":["bert-base-uncased"],
    "yiyanghkust/finbert-tone":["yiyanghkust/finbert-tone"],
    "nlpaueb/sec-bert-base":["nlpaueb/sec-bert-base"],
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def align_labels_with_tokens(labels, word_ids):
   new_labels = []
   current_word = None
   for word_id in word_ids :
      if word_id != current_word:
         # Start of a new word!
         current_word = word_id
         label = -100 if word_id is None else labels[word_id]
         new_labels.append(label)
      elif word_id is None :
         # Special token
         new_labels.append(-100)
      else:
         # Same word as previous token
         label = labels[word_id]
         # If the label is B-XXX, we change it to I-XXX
         if label % 2 == 1:
            label +=1 
         new_labels.append(label)
   return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512, padding="max_length"
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def load_to_model(model_tokenizer, raw_datasets):

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer, add_prefix_space=True, padding=True, truncation=True)
    
    return tokenizer, raw_datasets

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

raw_datasets = load_dataset("tner/fin")

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# Greedy Search
SEED = [777]
BS = [8,16]
LR = [1e-5, 1e-4, 1e-3, 2e-5, 3e-5, 5e-5, 1e-6]

outputs = []
model_ckpt_list = []
search_list = []

for model_key, model_list in model_dict.items():
    model_tokenizer = model_key
    tokenizer, raw_datasets = load_to_model(model_tokenizer, raw_datasets)
    for model_checkpoint in model_list:
        tokenize_and_align_labels_partial = functools.partial(tokenize_and_align_labels, tokenizer=tokenizer)
        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels_partial,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        for seed, bs, lr in product(SEED, BS, LR):
            set_seed(seed)
            batch_size = bs
            learning_rate = lr
            # showing progress bar
            tqdm.pandas()

            model_ckpt_list.append(model_checkpoint.split("/")[-1])
            # train Models
            model = AutoModelForTokenClassification.from_pretrained(
                    model_checkpoint,
                    id2label=id2label,
                    label2id=label2id
                    )

            # Early Stopping
            early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

            training_args = TrainingArguments(
                os.path.join("NER","results", f"model_checkpoint"),
                evaluation_strategy = "epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=100,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_strategy="epoch",
                save_total_limit=1,
            )

            trainer = Trainer(
                model,
                training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping]
            )

            trainer.train()

            # check the result score
            val_output = trainer.evaluate(tokenized_datasets["validation"], metric_key_prefix="test")
            output = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
            output['val_f1'] = val_output['test_f1']
            output['seed'] = seed
            output['batch_size'] = batch_size
            output['learning_rate'] = learning_rate
            outputs.append(output)
            torch.cuda.empty_cache()

            df = pd.DataFrame(outputs, index=model_ckpt_list)
            print(df)
            df.to_csv("grid_result.csv")