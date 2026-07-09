import argparse
import os
import pathlib
import pickle
import re

from datasets import load_dataset
import pandas as pd
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


def load_data(dataset_name, **dataset_kwargs):
    return load_dataset(dataset_name, **dataset_kwargs)

def generate_training_df(data):
    df = pd.DataFrame(data).rename(columns={"sms": "text"})
    return df

def preprocess_text(X):
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    X['text'] = X['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())
    return X

# ==================================================================================================
# === MAIN =========================================================================================
# ==================================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sms_spam')
    parser.add_argument('--hf_token', type=str, default=os.getenv('HF_TOKEN', ''))

    args = parser.parse_args()
    artifact_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "model_artifacts")
    os.makedirs(artifact_path, exist_ok=True)

    if args.dataset.lower() == 'sms_spam':
        print("Loading SMS spam dataset...")
        data = load_data("ucirvine/sms_spam", token=args.hf_token, split="train")
        train_df = generate_training_df(data)

        print("Preprocessing data...")
        X = train_df.drop(columns=['label'])
        X = preprocess_text(X)
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X['text'])

        y = train_df['label']

        print("Training XGBoost model...")
        param_grid =  {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        grid_search.fit(X_vec, y)
        clf = xgb.XGBClassifier(
                    max_depth=grid_search.best_params_['max_depth'],
                    learning_rate=grid_search.best_params_['learning_rate'],
                    subsample=grid_search.best_params_['subsample'],
                    random_state=42
        )
        clf.fit(X_vec, y)

        print(f"Saving training artifacts to {artifact_path}...")
        pickle.dump(vectorizer, open(f'{artifact_path}/vectorizer.pkl', 'wb'))
        pickle.dump(clf, open(f'{artifact_path}/model.pkl', 'wb'))

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported")
