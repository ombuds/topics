from src.models.train_model import evaluate, metrics_log
from src.data.make_dataset import load_processed
import gc
import logging
import spacy

MODEL_DIR="models/"
TEST_TEXT = "This movie was great!"

def load(model_dir=MODEL_DIR):
    logging.info("Loading model from: "+model_dir)
    nlp = spacy.load(model_dir)
    return nlp

def predict(test_text=TEST_TEXT,nlp=None,model_dir=MODEL_DIR):
    '''Predict text class from trained nlp model.'''

    # Load a trained model if necessary
    if nlp is None:
        load(model_dir)

    # test the trained model
    doc = nlp(test_text)
    logging.info("Evaluating string.")
    return test_text, doc.cats



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logging.info("Loading data.")
    train_data, valid_data, classes = load_processed("data/processed/")

    train_texts, train_cats = train_data
    dev_texts, dev_cats = valid_data

    # Cleanup unnecessary data
    del train_data
    del valid_data 
    del classes
    gc.collect()

    # Load model
    nlp=load(model_dir="models/")

    # Evaluate performance
    train_scores = evaluate(nlp.tokenizer, nlp.get_pipe("textcat"), train_texts, train_cats)
    dev_scores   = evaluate(nlp.tokenizer, nlp.get_pipe("textcat"), dev_texts  , dev_cats  )

    # Print loss
    # TODO: fix printout in log in case of missing loss
    metrics_log({"textcat":"-9.999"}, scores, prefix="training  :")
    metrics_log({"textcat":"-9.999"}, scores, prefix="validation:")

