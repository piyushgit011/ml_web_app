import os
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_scaled_inputs(inputs):
    data = get_clean_data()
    scaled_dict = {}
    for key, value in inputs.items():
        max_val = data[key].max()
        min_val = data[key].min()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict

def get_radar_chart(inputsc):
    categories = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
                  'concave points', 'symmetry', 'fractal_dimension']
    inputsc = get_scaled_inputs(inputsc)
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            inputsc['radius_mean'],
            inputsc['texture_mean'],
            inputsc['perimeter_mean'],
            inputsc['area_mean'],
            inputsc['smoothness_mean'],
            inputsc['compactness_mean'],
            inputsc['concavity_mean'],
            inputsc['concave points_mean'],
            inputsc['symmetry_mean'],
            inputsc['fractal_dimension_mean'],
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            inputsc['radius_worst'],
            inputsc['texture_worst'],
            inputsc['perimeter_worst'],
            inputsc['area_worst'],
            inputsc['smoothness_worst'],
            inputsc['compactness_worst'],
            inputsc['concavity_worst'],
            inputsc['concave points_worst'],
            inputsc['symmetry_worst'],
            inputsc['fractal_dimension_worst'],
        ],
        theta=categories,
        fill='toself',
        name='Worst Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            inputsc['radius_se'],
            inputsc['texture_se'],
            inputsc['perimeter_se'],
            inputsc['area_se'],
            inputsc['smoothness_se'],
            inputsc['compactness_se'],
            inputsc['concavity_se'],
            inputsc['concave points_se'],
            inputsc['symmetry_se'],
            inputsc['fractal_dimension_se'],
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )),
        showlegend=True
    )

    return fig


def get_clean_data():
    df = pd.read_csv('data.csv')
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def add_sidebar():
    st.sidebar.title('Cell Nuclei Details')
    st.sidebar.markdown('Please enter the details of the cell nuclei')
    df = get_clean_data()
    slider_labels = (
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst")
    inputs = {}
    for i in range(30):
        inputs[slider_labels[i]] = st.sidebar.slider(slider_labels[i], 0.0, float(df[slider_labels[i]].max()),
                                                     float(df[slider_labels[i]].mean()), 0.1)
    return inputs


def add_prediction(inputs):
    #load model
    # model_dt = pickle.load(open('model_dt.pkl', 'rb'))
    # model_dt = pickle.load(open('model_rf.pkl', 'rb'))
    # model_dt = pickle.load(open('model_xg.pkl', 'rb'))
    model_dt = pickle.load(open('model_lbm.pkl', 'rb'))
    # model_dt = pickle.load(open('model_log.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))


    data = pd.DataFrame.from_dict([inputs])
    # print(data)
    input_arraysc = scaler.transform(data)
    # print(input_arraysc)
    pred = model_dt.predict(input_arraysc)
    st.write("probability of being benign: ",float(model_dt.predict_proba(input_arraysc)[0][0]))
    st.write("probability of being malicious: ",model_dt.predict_proba(input_arraysc)[0][1])
    # st.write(input_arraysc)
    # st.write(pred)
    if pred[0] == 0:
        st.write('The cell is benign')
    else:
        st.write('The cell is malignant')

def main():
    st.set_page_config(page_title='cancer prediction', layout="wide", initial_sidebar_state="expanded")
    inputs = add_sidebar()
    with st.container():
        st.title('Cancer Prediction')
        st.write('This is a simple cancer prediction app built with Streamlit and Scikit-learn')
    # columns in streamlit
    col1, col2 = st.columns([4, 1])

    with col1:
        st.write('Column1')
        fig = get_radar_chart(inputs)
        st.plotly_chart(fig)
    with col2:
        st.write('Column2')
        add_prediction(inputs)

if __name__ == '__main__':
    main()
