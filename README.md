---
title: Credit Card Default Prediction
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
---

# Credit Card Default Prediction

Module 10 deployment of a tuned Random Forest classifier trained on the UCI Default of Credit Card Clients dataset. This is an educational demo only and should not be used for actual lending decisions.

The model expects 23 numerical inputs in the original UCI feature order (X1 through X23).

To run locally: `pip install -r requirements.txt && python app.py`.
