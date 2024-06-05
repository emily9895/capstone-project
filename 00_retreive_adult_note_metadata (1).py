import os
import sys
import time

import numpy as np
import pandas as pd

import datetime

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db

from dask_jobqueue import SGECluster
from dask.distributed import Client

def setup_note_metadata(year,
                        loc = '/wynton/protected/project/ic/data/parquet/DEID_CDW/note_metadata/',
                        ped_keywords = ['pediatric', 'child', 'bcho', 'benioff']):
    """Takes in the file path for the note_metadata parquet file and returns the dask dataframe.

    Filters out notes from pediatric departments.

    Args:
        loc: File path for the note_metadata parquet file.
        ped_keywords: A list of keywords associated with pediatric departments.

    Returns:
        A dask dataframe
    """
    inclusion_note_types = pd.read_csv('data/inclusion_note_types/inclusion_note_types.csv')
    note_metadata = dd.read_parquet(loc)
    
    note_metadata = note_metadata[note_metadata['note_type'].isin(inclusion_note_types['note_type'])]
    
    note_metadata['deid_service_date'] = dd.to_datetime(note_metadata['deid_service_date'], errors='coerce')
    note_metadata['year'] = note_metadata['deid_service_date'].apply(lambda x: x.year, meta=('deid_service_date', 'int64'))
    note_metadata = note_metadata[note_metadata['year'] == year]
    
    note_metadata['enc_dept_name'] = note_metadata['enc_dept_name'].fillna('')
    note_metadata['keep'] = note_metadata['enc_dept_name'].map(lambda x: any(keyword in x for keyword in ped_keywords))
    note_metadata = note_metadata.loc[note_metadata['keep'] == False, ['year', 'deid_note_key', 'encounterkey', 'patientdurablekey']]
    
    return note_metadata.compute()

def setup_patientdim(loc = '/wynton/protected/project/ic/data/parquet/DEID_CDW/patientdim/'):
    """Takes in the file path for the patientdim parquet file and returns the dask dataframe.

    Select specific columns in retain

    Args:
        loc: File path for the patientdim parquet file.

    Returns:
        A dask dataframe
    """
    patientdim = dd.read_parquet(loc)
    patientdim = patientdim.astype({'patientkey': str})
    patientdim = patientdim[
        ['patientkey', 'sexassignedatbirth', 'genderidentity', 
         'birthdate', 'deathdate'] 
    ]
    
    return patientdim

def setup_encounterfact(loc = '/wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact/'):
    """Takes in the file path for the encounterfact parquet file and returns the dask dataframe.

    Select specific columns in retain

    Args:
        loc: File path for the encounterfact parquet file.

    Returns:
        A dask dataframe
    """
    encounterfact = dd.read_parquet('/wynton/protected/project/ic/data/parquet/DEID_CDW/encounterfact/')
    encounterfact = encounterfact.astype({'patientkey': str, 'encounterkey': str})
    encounterfact = encounterfact[['encounterkey', 'patientkey']]
    
    return encounterfact

def merge_patient_metadata(patientdim, encounterfact, note_metadata):
    """Takes in the patientdim, encounterfact, note_metadata dask dataframes and returns a pandas dataframe of them merged.
    
    Select to notes for patients age >= 18.

    Args:
        patientdim: Dask dataframe of patientdim table.
        encounterfact: Dask dataframe of encounterfact table.
        note_metadata: Dask dataframe of note_metadata table.

    Returns:
        A dask dataframe
    """
    note_metadata_merged = patientdim.merge(
        encounterfact,
        left_on = 'patientkey', 
        right_on = 'patientkey',
        how = 'inner'
    )
    
    note_metadata_merged['birthdate'] = dd.to_datetime(note_metadata_merged['birthdate'], errors='coerce')
    note_metadata_merged['birthyear'] = note_metadata_merged['birthdate'].dt.year
    
    note_metadata_merged = note_metadata_merged.merge(
        note_metadata,
        left_on = 'encounterkey',
        right_on = 'encounterkey',
        how = 'inner'
    )
    
    note_metadata_merged['age'] = note_metadata_merged['year'] - note_metadata_merged['birthyear']
    note_metadata_merged = note_metadata_merged[
        note_metadata_merged['age'] >= 18
    ]
    
    return note_metadata_merged

if __name__ == '__main__':
    
    year = int(sys.argv[1])
    print(f"Step 00_{year}")
    print(f"Cores: {os.cpu_count()}")
    
    print("Starting Cluster...")
    client = Client()
    print(client.dashboard_link)
    
    # Get data
    print("Retrieving Data...")
    note_metadata = setup_note_metadata(year = year)
    patientdim = setup_patientdim()
    encounterfact = setup_encounterfact()
    note_metadata_merged = merge_patient_metadata(patientdim, encounterfact, note_metadata)
    note_metadata_merged = note_metadata_merged.compute()
    
    # Save results to parquet file
    note_metadata_merged.set_index('deid_note_key', inplace=True)
    
    print(f"Savings Metadata Results {year}...")
    note_metadata_merged[note_metadata_merged['year'] == year] \
        .to_parquet(f'data/raw_note_metadata/adult_notes_metadata_{year}.parquet', index = True)
    
    # Close Cluster
    print("Closing Cluster...")
    client.close()