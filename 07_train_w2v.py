import os
import sys

import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS

from dask.distributed import Client


if __name__ == '__main__':
    year = int(sys.argv[1])
    print(f"Current Directory:{os.getcwd()}")
    
    print("Starting Cluster...")
    client = Client()
    print(client.dashboard_link)
    
    print(f"Loading {year} Sentences...") 
    w2v_sentences = dd.read_parquet(f'data/ngram_sentences/sentences_{year}.parquet')
    w2v_sentences['sentence'] = w2v_sentences['sentence'].map(lambda x: [token for token in x if token not in STOPWORDS])
    w2v_sentences['length'] = w2v_sentences['sentence'].map(lambda x: len(x))
    w2v_sentences = w2v_sentences[w2v_sentences['length'] > 10]
    w2v_sentences = w2v_sentences.compute()
    sentence_list = w2v_sentences['sentence'].to_list()
    
    # Close Cluster
    print("Closing Cluster...")
    client.close()
    
    print(f"Building Vocab {year}...") 
    w2v_model = Word2Vec(workers=20, vector_size=500, window=11)
    w2v_model.build_vocab(sentence_list)

    print(f"Training {year}...") 
    w2v_model.train(
        sentence_list, 
        total_examples=w2v_model.corpus_count, 
        epochs=30
    )

    print(f"Saving Model...")
    w2v_model.save(f'models/adult_notes/w2v_{year}.model')

    
