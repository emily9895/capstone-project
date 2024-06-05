import os
import sys
import time

import numpy as np
import pandas as pd

import re

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db

import pyarrow as pa

from gensim.models.phrases import Phrases, Phraser 
from gensim.models import Word2Vec
# from gensim.parsing.preprocessing import STOPWORDS

from dask.distributed import Client

if __name__ == '__main__':
    year = int(sys.argv[1])
    print(f"Current Directory:{os.getcwd()}")
    
    print("Starting Cluster...")
    client = Client()
    print(client.dashboard_link)
    
    print(f"Loading Ngrams Model {year}...")
    ngrams = Phrases.load(f'models/ngrams/ngrams_{year}.pkl')

    print(f"Applying Ngrams {year}...")
    w2v_sentences = dd.read_parquet(f'data/bigram_sentences/sentences_{year}.parquet')
    
    w2v_sentences['sentence'] = w2v_sentences['sentence'] \
        .map(lambda x: ngrams[x])
    
    print(f"Saving Ngram Sentences {year}...")
    w2v_sentences.to_parquet(f'data/ngram_sentences/sentences_{year}.parquet',
                             write_index = True, 
                             overwrite = True, 
                             schema = {'sentence': pa.list_(value_type = pa.string())})
    
    client.close()