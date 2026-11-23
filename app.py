import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / 'credit_risk_model.pkl')
features = joblib.load(BASE / 'feature_names.pkl')

labels = [
    'Credit limit (NT$)', 'Sex (1=male, 2=female)',
    'Education (1=graduate school, 2=university, 3=high school, 4=other)',
    'Marital status (1=married, 2=single, 3=other)', 'Age (years)',
    'September payment status', 'August payment status', 'July payment status',
    'June payment status', 'May payment status', 'April payment status',
    'September bill (NT$)', 'August bill (NT$)', 'July bill (NT$)',
    'June bill (NT$)', 'May bill (NT$)', 'April bill (NT$)',
    'September payment amount (NT$)', 'August payment amount (NT$)',
    'July payment amount (NT$)', 'June payment amount (NT$)',
    'May payment amount (NT$)', 'April payment amount (NT$)'
]
assert len(features) == len(labels) == 23

def predict(*values):
    if any(v is None for v in values):
        raise gr.Error('Please fill in all 23 fields.')
    row = pd.DataFrame([list(values)], columns=features)
    result = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
    label = 'Higher predicted default risk' if result == 1 else 'Lower predicted default risk'
    return label, f'{prob:.1%}'

initial = [50000, 2, 2, 2, 25, 0, 0, 0, 0, 0, 0,
           10000, 10000, 10000, 10000, 10000, 10000,
           2000, 2000, 2000, 2000, 2000, 2000]
with gr.Blocks(title='Credit Card Default Prediction') as demo:
    gr.Markdown('# Credit Card Default Prediction')
    gr.Markdown('Student ML deployment demo using a tuned Random Forest model trained on the UCI Default of Credit Card Clients dataset. Enter all 23 features. Payment status values follow the dataset coding (e.g. -1 = paid duly, 1 = one month delay, 2 = two months delay). This is a class project, **not financial advice or a lending decision tool**.')
    inputs = []
    with gr.Accordion('Customer details', open=True):
        for i in range(5):
            inputs.append(gr.Number(label=f'{features[i]} — {labels[i]}', value=initial[i], precision=0))
    with gr.Accordion('Payment history (6 months)', open=True):
        for i in range(5, 11):
            inputs.append(gr.Number(label=f'{features[i]} — {labels[i]}', value=initial[i], precision=0))
    with gr.Accordion('Bill amounts (6 months)', open=False):
        for i in range(11, 17):
            inputs.append(gr.Number(label=f'{features[i]} — {labels[i]}', value=initial[i], precision=0))
    with gr.Accordion('Previous payments (6 months)', open=False):
        for i in range(17, 23):
            inputs.append(gr.Number(label=f'{features[i]} — {labels[i]}', value=initial[i], precision=0))
    submit = gr.Button('Predict default risk', variant='primary')
    result = gr.Textbox(label='Model prediction')
    probability = gr.Textbox(label='Estimated probability of default')
    submit.click(predict, inputs=inputs, outputs=[result, probability])

demo.launch()
