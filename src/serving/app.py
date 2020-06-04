from src.models.predict_model import load,predict
from flask import Flask, request, render_template
import pandas as pd
app = Flask("serving")

# load model
nlp=load()

@app.route('/service')
def service():
    try:
        query = request.args.get('q')
        test_text, cats=predict(query,nlp=nlp)
        cats_df=pd.Series(cats,name="Probability").sort_values(ascending=False).to_frame()*100
        cats_str=cats_df.to_html(float_format='{:.0f}%'.format)
    except:
        cats_str="Server error!"
    output = f'<!DOCTYPE html><body>{test_text}<P>{cats_str}:</body></html>'
    return output