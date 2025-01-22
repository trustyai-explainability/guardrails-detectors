import argparse
import datasets
import dataset_configs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
from sentence_transformers import SentenceTransformer
import torch
from tqdm.autonotebook import tqdm
import umap

from detectors.embedding_classification.build.dataset_configs.base_dataset_config import BaseDatasetConfig


plt.style.use('https://raw.githubusercontent.com/RobGeada/stylelibs/main/material_rh.mplstyle')


# === DATA LOADING =================================================================================
def load_data(dataset_name, **dataset_kwargs):
    return datasets.load_dataset(dataset_name, **dataset_kwargs)


def generate_training_df(data, dataset_config: BaseDatasetConfig):
    df = pd.DataFrame()
    df['text'] = dataset_config.get_text(data)
    df['label'] = dataset_config.get_label(data)
    return df


# === EMBEDDING ====================================================================================
def get_torch_device():
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        device = "cuda"
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"
    print("Using {} backend for sentence transformer.".format(device))
    return torch.device(device)


def get_embedding_model():
    device = get_torch_device()
    return SentenceTransformer(os.path.join("model_artifacts","dunzhang","stella_en_1"), trust_remote_code=True).to(device)


def get_embeddings(train_df, batch_size, model):
    query_prompt_name = "s2p_query"

    nrows = len(train_df)
    embeddings = np.zeros([nrows, 1024])
    for idx in tqdm(range(0, nrows, batch_size)):
        text = train_df['text'].iloc[idx: idx+batch_size]
        embeddings[idx:idx+batch_size] = model.encode(text, prompt_name=query_prompt_name)
    return embeddings


def generate_embedding_df(train_df, reduced_embedding):
    embedding_df = pd.DataFrame(reduced_embedding)
    embedding_df.columns = [str(i) for i in range(reduced_embedding.shape[1])]
    embedding_df['Label'] = train_df['label']
    return embedding_df


# === CENTROIDS ====================================================================================
def get_centroids(embedding_df, reduced_embedding):
    return embedding_df.groupby("Label").agg({str(d): "mean" for d in range(reduced_embedding.shape[1])})


# ==================================================================================================
# === MAIN =========================================================================================
# ==================================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mmlu')

    args = parser.parse_args()
    artifact_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "model_artifacts")
    os.makedirs(artifact_path, exist_ok=True)

    if args.dataset.lower() == 'mmlu':
        # load data
        print("Loading MMLU dataset...")
        data = load_data("cais/mmlu", name='all')
        train_df = generate_training_df(data['test'], dataset_configs.MMLUDatasetConfig())

        # get embeddings
        embedding_artifact_path = os.path.join(artifact_path, args.dataset.lower()+"_embeddings.npy")

        if not os.path.exists(embedding_artifact_path):
            print("Loading embedding model...")
            embedding_model = get_embedding_model()

            print("Generating embeddings for MMLU")
            embeddings = get_embeddings(train_df, batch_size=4, model=embedding_model)
            np.save(embedding_artifact_path, embeddings)
        else:
            print("Loading pre-trained embeddings...")
            embeddings = np.load(embedding_artifact_path)

        # get dimensionality reduction
        print("Fitting dimensionality reduction...")
        reducer = umap.UMAP(n_components=3)
        reduced_embedding = reducer.fit_transform(embeddings)
        embedding_df = generate_embedding_df(train_df, reduced_embedding)

        # centroids
        print("Generating centroids...")
        centroids = get_centroids(embedding_df, reduced_embedding)

        # save artifacts
        print("Saving training artifacts to {}...".format(artifact_path))
        pickle.dump(reducer, open(os.path.join(artifact_path, "umap.pkl"), "wb"))
        centroids.to_pickle(os.path.join(artifact_path, "centroids.pkl"))
        print("Training completed successfully!")
    else:
        raise NotImplementedError("Dataset {} not yet supported".format(args.dataset))





