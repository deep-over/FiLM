# README

## **1. Pretraining**

For pretraining, we have prepared a preprocessed temporary text dataset which is placed in the **`pretraining/document`** folder. Before executing the pretraining, you have the option to modify the arguments in the **`submission_code/pretraining/pretraining-model.py`** file. All the set values are configured to those that were used during the actual training.

### Execution Command

```
python pretraining/pretraining-model.py
```

## **2. FiNER**

Before running FiNER, open **`FiNER/configurations/transformer.json`** and modify the **`train_parameters`**.

### Execution Command:

```python
python FiNER/run_experiment.py
```

If you want to specify only one GPU, you can run the command with **`CUDA_VISIBLE_DEVICES=0`**.

## **3. FinQA**

Before running FinQA, you need to set the **`root_path`** in **`"submission_code/FinQA/code/generator/config.py"`**.

### Training Command:

```
python FinQA/code/generator/Main.py
```

After the training is completed, find the saved best model and run **`FinQA/code/generator/Test.py`**.

### Test Command:

```
python FinQA/code/generator/Test.py
```

## **4. FPB**

Before starting the FPB training, set the **`root_path`** in the **`FPB/train_for_loop.py`** file.

### Training Command:

```
python FPB/train_for_loop.py
```

Before testing, make sure to check the .ckpt file and tokenizer to be used for testing.

### Test Command:

```
python FPB/test.py
```

## **5. Headline**

Before running Headline, set the **`root_path`** in the **`submission_code/Headline/model_dictionary.py`**. To execute, run python from the headline directory.

### Execution Command:

```
python train_for_loop.py
```

## **6. NER**

### Execution Command:

```
python NER/train_greedy.py
```

## 7. FOMC

### Execution Command:

```
python fomc-hawkish-dovish-main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py
```

The experiments were performed in the Anaconda virtual environment with the following configurations:

- GPU: RTX3090
- CUDA version: 11.7
- Python version: 3.8.13
- Anaconda version: 22.9.0

Here are the required libraries:

- torch 1.13.1+cu117
- transformers 4.26.0
- tensorflow 2.9.1
- six 1.16.0
- nltk
- protobuf 3.19.4
- scipy
- datasets
- spacy
- scikit-learn

Please make sure to replace the paths and filenames with your actual paths and filenames as necessary.