import os
import sys
import time

import numpy as np
import pandas as pd

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('extraordinary')

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db

import pyarrow as pa

from dask.distributed import Client

if __name__ == '__main__':
    year = int(sys.argv[1])
    print(f"Step 02_{year}")
    
    print("Starting Cluster...")
    client = Client()
    print(client.dashboard_link)
    
    notes = dd.read_parquet(f'data/raw_note_text/notes_{year}.parquet').repartition(npartitions = 100)
    
    print(f"Processing Data {year}...")
    notes['sentence'] = notes['note_text'] \
        .dropna() \
        .map(nltk.sent_tokenize)

    notes = notes \
        .explode(column = 'sentence')\
        .dropna()
        

    notes['sentence'] = notes['sentence'] \
        .map(lambda x: re.split('\n', x))

    notes = notes \
        .explode(column = 'sentence') \
        .dropna()
    
    notes = notes[
        ~notes['sentence'].str.contains(r'(_|\b)egfr(_|\b)', regex=True, case=False)
    ]

    notes['sentence'] = notes['sentence'] \
        .str.replace('*****', ' zzredactedzz ', regex=False) \
        .str.replace('\d+\S\d+\S\d+', ' zzdatezz ', regex=True) \
        .str.replace('[0-9]+', ' zzdigitzz ', regex=True) \
        .str.replace('\n', ' ', regex=True) \
        .str.replace('[^\w\s]', ' ', regex=True) \
        .str.replace(' +', ' ', regex=True) \
        .map(lambda x: re.sub(r'\b(zzredactedzz+|zzdigitzz+)( \1\b)+', r'\1', x)) \
        .str.replace('[^A-Za-z0-9 ]+', '', regex=True) \
        .str.lower() \
        .map(
            lambda sent: 
            [
                lemmatizer.lemmatize(token) 
                for token in word_tokenize(sent) 
            ]
        )
    
    notes['complete'] = notes['sentence'].map(lambda x: len(x) > 10)
    notes = notes.loc[notes['complete'], ['patientdurablekey', 'encounterkey', 'sentence']]
            
    print(f"Saving Data {year}...")
    notes.to_parquet(f'data/processed_sentences/sentences_{year}.parquet',
                     write_index = True, overwrite = True, 
                     schema = {'sentence': pa.list_(value_type = pa.string())})
    
    client.close()