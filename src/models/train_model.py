from __future__ import unicode_literals, print_function
# import plac
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import random
import spacy
from spacy.util import minibatch, compounding
from src.data.make_dataset import load_data, load_processed

MODEL_DIR="models/"
# TODO: add flow control for minibatch random seed
SEED_MINIBATCH=101

def train(train_data=None, valid_data=None, classes=None, model="en_core_web_sm", output_dir=MODEL_DIR, n_iter=20, n_texts=2000, init_tok2vec=None):
    '''Launch model training'''

    # TODO: Improve flow control for missing input data
    if train_data is None or valid_data is None or classes is None:
        logging.info(f"No data provided: reloading saved processed data.")
        train_data, valid_data, classes = load_processed()
        
    # Split the dataset tuples
    (train_texts, train_cats) = train_data
    (dev_texts, dev_cats) = valid_data

    # prepare the output dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    # load the model from files
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        logging.info(f"Loaded model {model}")
    else:
        nlp = spacy.blank("en")  # create blank Language class
        logging.info("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    for _,row in classes.iterrows():
        textcat.add_label(row['class'])


    train_texts = train_texts[:n_texts]                              # TODO: Investigate this second split 
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    # This contains a dict of the categories inside a dict of one entry with key "cats"
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))   

    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.Random(SEED_MINIBATCH).shuffle(train_data)                                             # TODO: control this random loop as well
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:                                                  # TODO: add timer
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            # Show metrics
            metrics_log(losses,scores)

    # Save model to disk    
    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

def metrics_log(losses,scores,prefix=""):
    '''Show performance metrics in log.'''
    # TODO: add error protection for prefix type
    logging.info(prefix+
        "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
            losses["textcat"],
            scores["textcat_p"],
            scores["textcat_r"],
            scores["textcat_f"],
        )
    )

def evaluate(tokenizer, textcat, texts, cats):
    '''Evaluate model performance from multiclass precision/recall built from confusion matrix.'''
    docs = (tokenizer(text) for text in texts)
    # initialize the confusion matrix
    nclasses = len(cats[0])
    confusion_matrix = np.zeros([nclasses,nclasses])
    # populate the confusion matrix
    for i, doc in enumerate(textcat.pipe(docs)):
        # use the class with maximum score
        idx_gold = pd.Series(cats[i],name="gold").argmax()
        idx_pred = pd.Series(doc.cats,name="pred").argmax()
        confusion_matrix[idx_gold,idx_pred]+=1
    # Compute class-wise precision and recall
    # TODO: Implement error detection to avoid division by zero in case not all classes are represented.
    precisions = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
    recalls = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    # Average the precision and recall computed over the classes
    # (this gives more weight to minority classes compared to using accuracy)
    precision, recall = precisions.mean(), recalls.mean()
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    train()