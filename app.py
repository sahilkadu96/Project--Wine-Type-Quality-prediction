
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import StandardScaler

st.title('Wine Quality Prediction')

model = load_model('/content/wine_quality_NN.h5')
df = pd.read_csv('/content/wine_preprocessed.csv')

X = df.drop(['quality', 'is_red'], axis = 1).values 
sc = StandardScaler()
X_sc = sc.fit_transform(X)

fa = st.sidebar.number_input('Fixed acidity', min_value = 3.0, max_value = 16.0, step = 0.1)
va = st.sidebar.number_input('Volatile acidity', min_value = 0.0, max_value = 1.5, step = 0.1)
ca = st.sidebar.number_input('Critic acid', min_value = 0.0, max_value = 2.0, step = 0.1)
rs = st.sidebar.number_input('Residual sugar', min_value = 0.0, max_value = 66.0, step = 0.1)
c = st.sidebar.number_input('Chloride', min_value = 0.0, max_value = 1.0, step = 0.1)
fsd = st.sidebar.number_input('Free SO2', min_value = 1, max_value = 133, step = 1)
tsd = st.sidebar.number_input('Total SO2', min_value = 6, max_value = 344, step = 1)
d = st.sidebar.number_input('Density', min_value = 0.0, max_value = 1.5, step = 0.1)
ph = st.sidebar.number_input('pH', min_value = 2.0, max_value = 4.0, step = 0.1)
s = st.sidebar.number_input('Sulphates', min_value = 0.0, max_value = 2.0, step = 0.1)
a = st.sidebar.number_input('Alcohol', min_value = 8.0, max_value = 15.0, step = 0.1)

inputs = [[fa, va, ca, rs, c, fsd, tsd, d, ph, s, a]]
inputs_sc = sc.transform(inputs)

if st.button('Predict'):
  res = model.predict(inputs_sc)
  wine_quality = np.round(res[0][0][0])
  st.write(f'Wine quality is {wine_quality}')

  wine_type = np.round(res[1][0][0])
  if wine_type > 0:
    st.write("Wine type is red wine")
  else:
    st.write("Wine type is white wine")


