topics
==============================

Document classification model and service built using the Yahoo! Question-Answer dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires an Anaconda installation.


### Installing

Get the project from git.
```
git clone https://github.com/ombuds/topics.git
```

Browse to the project's root directory from the Anaconda terminal, and create a new environment.
```
conda env create -f environment.yml
```


Activate the environment.

```
conda activate topics
```

Install the project for development:

```
pip install -e .
```

Download the language model:

```
python -m spacy download en_core_web_sm
```

Download the Yahoo! Question-Answer dataset from here:
```
https://drive.google.com/open?id=1BHICkntwHlD_KaaG2_0n6obV9fi_TqBv
```

Move the dataset tar file to the raw data directory. From the project's root directory, use the following path:
```
data/raw/yahoo_answers_csv.tar.gz
```

Run the dataset preprocessing step.
```
python -m src.data.make_dataset
```

Train the model.
```
python -m src.models.train_model
```

Evaluate the model.
```
python -m src.models.train_model
```

The test are run from the root directory using pytest.
```
pytest
```


### Serve the model.

Start the service.
```
python -m src.serving.app
```

The service runs locally on port 5000. You may now perform queries on the http entry point such as:
```
http://127.0.0.1:5000/?q=This is a sample query
```

## Model architecture

The first objective of this proof-of-concept solution is to present coding style and provide a conversation starting point for realizing similar projects. Therefore the focus was rather on efficiency and ease of integration. Therefore the project and model architecture are based on the python module `spacy` and the text classification tutorial available [here](https://spacy.io/usage/examples). The model use a simple CNN architecture over word vector sequences computed from the `en_core_web_sm` english model.

## Performance metrics

Out of any broader context for the application, it is safe to base performance on well-established metrics such as precision, recall and f-score to get insight on the global performance. However these metric usually refer to binomial classification.

Precision and recall can be computed for each class from the confusion matrix. True positive for each class are available on the diagonal, while the total number of sample in the gold standard are the sum of each line (for recall calculation), and the total number of samples predicted for each class are the sum of each column (for precision calculation). The overall precision and recall is calculated as the average precision and recall of all classes. This gives a higher weight to minority classes in the performance metric compared to using a global prediction accuracy.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `0.1-treid-data_extract.ipynb`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

