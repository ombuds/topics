# -*- coding: utf-8 -*-
import click
import gc
import logging
import pickle
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
import random
import thinc
import os
# from dotenv import find_dotenv, load_dotenv

DATA_RAW_DIR = "data/raw/"
DATA_RAW_NAME = "yahoo_answers_csv.tar.gz"
DATA_INTERIM_DIR = "data/interim/"
DATA_PROCESSED_DIR = "data/processed/"
SEED_TRAINING_SET_SAMPLER = 101

def untar(input_dir=DATA_RAW_DIR,input_name=DATA_RAW_NAME):
    """ Opens the dataset from the tar file and returns the training, test, and classes list dataframes.
    """
    # Open dataset tar file
    with tarfile.open(input_dir+"/"+input_name, "r:*") as tar:
        df_train = pd.read_csv(tar.extractfile('yahoo_answers_csv/train.csv'),names=['classid','title','qtext','answer'], sep=",")
        df_test  = pd.read_csv(tar.extractfile('yahoo_answers_csv/test.csv'),names=['classid','title','qtext','answer'], sep=",")
        df_classes = pd.read_csv(tar.extractfile('yahoo_answers_csv/classes.txt'), names=['class'], sep=",")

    # Create the classid field from the index (classes are numbered from 1 to 10)
    df_classes['classid']=df_classes.index.values+1
    
    #Readily merge the class names with the class IDs to ease data exploration.
    df_train = df_train.merge(df_classes,on='classid')
    df_test  = df_test.merge(df_classes,on='classid')

    return df_train, df_test, df_classes

def straighten(df_raw):
    """Create a single list of documents from the raw dataframes."""
    # Concatenate the title, question text, and answer in a single column, with their associated category.
    df = pd.concat([ df_raw[['title' ,'classid']].rename(columns={'title' :'doc'}), \
                     df_raw[['qtext' ,'classid']].rename(columns={'qtext' :'doc'}), \
                     df_raw[['answer','classid']].rename(columns={'answer':'doc'}), ] ).dropna()
    # Remove any NaN entries.
    df = df.dropna()
    gc.collect()
    # Return X and Y values in separate arrays, for ingestion into the tokenizer
    return list(zip( df['doc'].values.tolist(), df['classid'].values.tolist() ))

def dump(train_processed, val_processed,classes,processed_path=DATA_PROCESSED_DIR):
    """Save the processed dataset to disk."""
    pickle.dump( train_processed, open( processed_path + "train_processed.p", "wb" ) )
    pickle.dump(   val_processed, open( processed_path +   "val_processed.p", "wb" ) )
    pickle.dump(         classes, open( processed_path +         "classes.p", "wb" ) )

def load_processed(processed_path=DATA_PROCESSED_DIR):
    """Load the processed data from disk"""
    train_processed = pickle.load( open( processed_path + "train_processed.p", "rb" ) )
    val_processed   = pickle.load( open( processed_path +   "val_processed.p", "rb" ) )
    classes         = pickle.load( open( processed_path +         "classes.p", "rb" ) )
    return train_processed, val_processed, classes

# TODO: Set default limit to 0, to keep full dataset.
def load_data(input_dir=DATA_RAW_DIR,limit=25000, split=0.8):
    """Load the Yahoo question-answers dataset."""
    # load the data
    train,test,classes=untar(input_dir)
    data_train = straighten(train)
    data_test  = straighten(test )
    # print statistics
    logging.info(f"Number of training samples: {len(data_train):10}")
    logging.info(f"Number of test     samples: {len(data_test ):10}")
    # Create a seedable random shuffler
    # TODO: Add flow control to create different models with random seed.
    random_shuffler = random.Random(SEED_TRAINING_SET_SAMPLER)
    # Combine texts with their categories to shuffle them together, then sumsample a random training set
    # TODO: replace this subsampling step with a balanced sampling from each class.
    random_shuffler.shuffle( data_train )
    logging.info(f"Completed data shuffling.")
    data_train = data_train[-limit:]
    # Split the text and labels for gold data formatting
    texts, labels = zip(*data_train)
    # Transform the 1D arrays X into lists and the Y array into categories, using the classes list [{"CLASS1":1,"CLASS2":2,...}]
    cats=[]
    for y in labels:
        df_cats = pd.Series(classes['classid'].values.astype(int)==int(y),index=classes['class'])
        dic_cats = df_cats.to_dict()
        cats+=[dic_cats]
    logging.info(f"Completed categories list.")
    split = int(len(data_train) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:]), classes

# TODO: Add flow control for input/output filepath
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_processed, dev_processed, classes = load_data()
    # train_processed = (train_texts, train_cats)
    # dev_processed = (dev_texts, dev_cats)

    # Processed data saving
    dump( train_processed, dev_processed, classes )

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
