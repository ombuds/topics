from src.models.predict_model import load,predict
from flask import Flask, request, render_template
import pandas as pd
app = Flask("serving")

# load the default model
# TODO: Add option to load the model
nlp=load()

@app.route('/')
def service():
    '''Serve a html document returning a sorted table with the score for each category.'''
    # TODO: Add json serialized output
    try:
        # parse argument
        query = request.args.get('q')
        cat_only = request.args.get('cat_only')
        # predict category
        test_text, cats = predict(query,nlp=nlp)
        # store categories into dataframe
        cats_df=pd.Series(cats,name="Score").sort_values(ascending=False).to_frame()*100
        # flow control for testing
        if cat_only == "yes":
            cat = cats_df['Score'].idxmax()
            return cat
        # create the categories html table
        cats_str=cats_df.to_html(float_format='{:.0f}%'.format)
    except:
        # TODO: Catch exception by type to specialized the server error message.
        cats_str="Server error!"
    output = f'<!DOCTYPE html><body>{test_text}<P>{cats_str}:</body></html>'
    return output