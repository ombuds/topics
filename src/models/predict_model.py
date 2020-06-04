import logging
import spacy

MODEL_DIR="models/"
TEST_TEXT = "This movie was great!"

def predict(test_text=TEST_TEXT,nlp=None,model_dir=MODEL_DIR):
    '''Predict text class from trained nlp model.'''

    # Load a trained model if necessary
    if nlp is None:
        print("Loading from", model_dir)
        nlp = spacy.load(model_dir)

    # test the trained model
    doc = nlp(test_text)
    logging.info("Evaluating string.")
    print(test_text, doc.cats)
