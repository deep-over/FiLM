import json

from sklearn.model_selection import train_test_split
import pandas as pd

def make_csv(json_file, train_save_path, valid_save_path) :
    keys = list(json_file.keys())
    sentence = [json_file[key]['sentence'] for key in keys]
    senti_score = [float(json_file[key]['info'][0]['sentiment_score']) for key in keys]
    target = [json_file[key]['info'][0]['target'] for key in keys]
    aspects = [json_file[key]['info'][0]['aspects'] for key in keys]
    
    df = pd.DataFrame({'sentence' : sentence, 'score' : senti_score, 'target' : target, 'aspects' : aspects})
    x_train, x_valid, y_train, y_valid = train_test_split(df[['sentence','target','aspects']], df['score'], test_size=0.3, shuffle=True, random_state=34)

    train = pd.DataFrame({'sentence' : x_train['sentence'], 'target' : x_train['target'], 'aspects' : x_train['aspects'], 'score' : y_train})
    valid = pd.DataFrame({'sentence' : x_valid['sentence'], 'target' : x_valid['target'], 'aspects' : x_valid['aspects'], 'score' : y_valid})
    train.to_csv(train_save_path)
    valid.to_csv(valid_save_path)