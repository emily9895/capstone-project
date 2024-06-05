import os
import sys
import time

import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db

from gensim.models.phrases import Phrases, Phraser 


if __name__ == '__main__':
    year = int(sys.argv[1])
    print(f"Current Directory:{os.getcwd()}")
    
    print(f"Reading Sentences {year}...")
    w2v_sentences = dd.read_parquet(f'data/processed_sentences/sentences_{year}.parquet').compute()

    print(f"Training Bigrams {year}...")
    phrases = Phrases(w2v_sentences['sentence'])
    ngram = Phraser(phrases)

    print(f"Saving Bigrams {year}...")
    ngram.save(f"models/bigrams/bigrams_{year}.pkl")

