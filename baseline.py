import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn import svm
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')

from catboost import Pool, CatBoostClassifier

def catboost_train_and_make_pred(X_train, X_test, y_train, y_test, data_test):
    
    text_feats = [
        'q_clean',
        'ans_clean'
                  ]
    
    train_pool = Pool(
        X_train, 
        y_train, 
        text_features=text_feats, 
        feature_names=list(X_train)
    )
    
    valid_pool = Pool(
        X_test, 
        y_test,
        text_features=text_feats, 
        feature_names=list(X_test)
    )

    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'eval_metric': 'F1',
        # 'task_type': 'GPU', # GPU/CPU ?
        'early_stopping_rounds': 400,
        'use_best_model': True,
    }
    
    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool)
    
    y_pred = model.predict(data_test)
    return y_pred


def clean_text(text: string, edge: int)->string:
    text = text[:edge]
    words = word_tokenize(text.lower().strip().translate(table)) 
    
    return " ".join(words)

if __name__ == "__main__":
    
    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    
    cringe_symbs = ['\n','\t','\r','\0']
    cringe_symbs = dict({i:' ' for i in cringe_symbs})
    table = str.maketrans(cringe_symbs)

    data['original_text_len'] = data['ans_text'].str.len()
    data['ans_clean'] = data['ans_text'].apply(
        lambda x: pd.Series(clean_text(x, 1500))
        )
    data['q_clean'] = data['q_title'].apply(
        lambda x: pd.Series(clean_text(x, 130))
        )

    
    data_test['original_text_len'] = data_test['ans_text'].str.len()

    data_test['ans_clean'] = data_test['ans_text'].apply(
        lambda x: pd.Series(clean_text(x, 1500))
        )
    data_test['q_clean'] = data_test['q_title'].apply(
        lambda x: pd.Series(clean_text(x, 130))
        )
    

    # data['ans_clean_len'] = data['ans_clean'].str.len()
    # data['q_clean_len'] = data['q_clean'].str.len()
    
    data['label'] = data['label'].map({'ai_answer': 1, 'hu_answer': 0}) # 1 - AI
    
    y = data['label']
    
    drop_list = [
        'q_title',
        'q_id',
        'ans_text',
        'label',
        'line_id',
        # 'tabs_count',
        # 'ans_clean_len',
        #  'original_text_len', worse without
        # 'ans_clean',
        # 'q_clean',
        # 'q_clean_len'
        ]
        
        
    X = data.drop(drop_list, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # models_name = ['bert-base-nli-mean-tokens',
    #            'sentence-transformers/all-MiniLM-L6-v2',
    #            'paraphrase-multilingual-MiniLM-L12-v2'
    #            ]
    # model = SentenceTransformer(models_name[1])
    
    preparing_test = data_test.drop([
        'q_title',
        'q_id',
        'ans_text',
        'line_id',
    ])
    
    y_pred = catboost_train_and_make_pred(X_train, X_test, y_train, y_test, preparing_test)
    
    data_test["label"] = y_pred
    data_test["label"] = data_test["label"].map({1: 'ai_answer', 0: 'hu_answer'})
    data_test[["line_id", "label"]].to_csv("data/submission.csv", sep=",", index=False)
    
    
    
    
    
    # df_train = pd.read_csv("data/train.csv")
    # df_test = pd.read_csv("data/test.csv")

    # ans_train = df_train["ans_text"].values
    # y_train = df_train["label"].map({'ai_answer': 1, 'hu_answer': 0}).values
    # ans_test = df_test["ans_text"].values

    # feat_extractor = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # emb_train = feat_extractor.encode(ans_train.tolist())
    # emb_test = feat_extractor.encode(ans_test.tolist())

    # y_pred = train_and_make_predictions(emb_train, y_train, emb_test)

    # df_test["label"] = y_pred
    # df_test["label"] = df_test["label"].map({1: 'ai_answer', 0: 'hu_answer'})
    # df_test[["line_id", "label"]].to_csv("data/submission.csv", sep=",", index=False)
    
