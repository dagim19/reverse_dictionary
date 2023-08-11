import os
import requests
import configparser
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

config = configparser.ConfigParser()
config.read('config.ini')



embeddings = np.load('dictionary_embeddings.npy')
dictionary = pd.read_csv('dictionary.csv')

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"
hf_token = config.get('HF_TOKEN', 'token')


api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()


def embeddings_to_np(embeddings):
    return np.array(embeddings)


def pick_top_k(res, k):
    indices = np.argpartition(res, -k)[-k:]
    return indices[np.argsort(res[indices])]

def reverse_dict(sentence):
    e = embeddings_to_np(query([sentence]))
    cosine_similarities_np = cosine_similarity(embeddings, e)
    best_matches = pick_top_k(cosine_similarities_np.squeeze(axis=1), 5)
    return best_matches


def print_row(row):
    print('=======================================')
    print(f"Word: {row['word']}")
    print(f"Definition: {row['definition']}")
    print('=======================================')


def print_rows(data_frame):
    print('These are the possible matches: ')
    for _, row in data_frame.iterrows():
        print_row(row)




if __name__ == "__main__":
    while(True):
        os.system("cls")
        sentence = input("Please enter your sentence: ")
        e = embeddings_to_np(query([sentence]))
        cosine_similarities_np = cosine_similarity(embeddings, e)
        best_matches = pick_top_k(cosine_similarities_np.squeeze(axis=1), 5)
        row = dictionary.iloc[best_matches]
        print_rows(row)
        ex = input("Press any key except 'e' to continue. ...")
        if ex == 'e':
            break


