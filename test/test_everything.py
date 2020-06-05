import hashlib
import os
import gc
from src.data.make_dataset import load_data, load_processed
from src.models.train_model import train
from src.models.predict_model import evaluate, load
import pandas as pd
from src.serving import app

DATA_RAW_PATH = "data/raw/yahoo_answers_csv.tar.gz"
DATA_RAW_MD5SUM = "f3f9899b997a42beb24157e62e3eea8d"

# TODO : update with latest dataset
def test_dataset_integrity():
    '''Compare dataset md5sum to avoid data corruption'''
    # TODO: replace with chunk-based hash to reduce memory requirements.
    def checksum(file):
        def file_as_bytes(file):
            with file:
                return file.read()
        # Return md5 hash from file fully loaded in memory
        return hashlib.md5( file_as_bytes( open(file, 'rb') ) ).hexdigest()
    # Assert the datafile exists
    assert os.path.exists(DATA_RAW_PATH)
    # Compare current datahash to reference hash to avoid data corruption
    assert DATA_RAW_MD5SUM == checksum(DATA_RAW_PATH)

def test_preprocessing():
    '''Test data preprocessing'''
    # Load raw dataset
    train_data, valid_data, classes = load_data("data/raw/")
    # Compare some strings
    idx=1543
    # TODO: Increase comparison robustness
    train_str = train_data[0][idx]
    train_cat = train_data[1][idx]
    valid_str = valid_data[0][idx]
    valid_cat = valid_data[1][idx]
    classlist = str(classes)
    # compare strings
    assert train_str == "where did Fredrick Mckinley Jonesdie?"
    assert train_cat['Education & Reference']
    assert valid_str == "Depends alot on fitness level."
    assert valid_cat['Health']
    assert classlist == '                    class  classid\n0       Society & Culture        1\n1   Science & Mathematics        2\n2                  Health        3\n3   Education & Reference        4\n4    Computers & Internet        5\n5                  Sports        6\n6      Business & Finance        7\n7   Entertainment & Music        8\n8  Family & Relationships        9\n9   Politics & Government       10'

def test_training():
    '''Launch a dummy training and ensure results repeatability'''
    # Load preprocessed dataset
    train_data, valid_data, classes = load_processed("data/processed/")
    # Launch model training
    train(train_data, valid_data, classes, output_dir="test/models/", n_iter = 1)
    # TODO: Compare files hash (depends on object stability)
    pass
    # TODO: Cleanup the model directory

def test_predict():
    '''Load the dummy model and compare prediction on the test set'''
    # Load model
    nlp=load(model_dir="test/models/")
    # Load preprocessed data
    train_data, valid_data, classes = load_processed("data/processed/")
    train_texts, train_cats = train_data
    dev_texts, dev_cats = valid_data
    # Cleanup unnecessary data
    del train_data
    del valid_data 
    del classes
    gc.collect()
    # Evaluate performance
    # TODO: Implement performance comparison
    train_scores = evaluate(nlp.tokenizer, nlp.get_pipe("textcat"), train_texts, train_cats)
    dev_scores   = evaluate(nlp.tokenizer, nlp.get_pipe("textcat"), dev_texts  , dev_cats  )
    # Compare results
    # TODO: Implement a more serious test.
    idx=432
    text=train_texts[idx]
    doc = nlp(text)
    # doc=nlp.tokenizer(text)
    # doc = nlp.get_pipe("textcat").pipe(doc)
    cat = pd.Series(doc.cats).idxmax()
    assert cat == "Family & Relationships"

def test_service():
    '''Test model service'''
    # TODO: Have the app load the testing model
    tester = app.app.test_client()
    query = "hockey"
    response = tester.get('/?q='+query+"&cat_only=yes", content_type='html/text')
    assert response.data == b"Sports"
