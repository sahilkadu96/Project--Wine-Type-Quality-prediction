
# Wine Quality Prediction
- By using Functional Model Neural Network - 2 Outputs. 
- First output is by using Regression & the second output is using Classification.




## Description

Datset fownloaded from: [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)


- There are two files **winequality-red.csv** and **winequality-white.csv**
 - We will combine both datasets & will preprocess the data.
 - After that we will build a neural network model that will output the wine quality (Regression) and wine type (Classification).
 - Refer **Project Wine Quality Type Prediction.ipynb** for model building.
 - There are **11 input variables** -  **fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide, total sulphur dioxide, density, pH, sulphates, alcohol**
 - There are **2 output variables** - **quality, is_red**
 - I have built a **Neural Network model having 2 Hidden Layers with 128 units each & 2 Output Layers for the output variables respecvtively.**
 - After training fror 40 epochs & predicting on the test we get the **Test Accuracy for Wine type as 99.5%** & **the Mean Absolute Error for Wine Quality is approx 0.4** which is pretty good.
 - After that I have deployed the app on Streamlit





## Streamlit interface
![image](https://github.com/sahilkadu96/Project--Wine-Type-Quality-prediction/assets/106151994/e1ff3b14-8193-4da6-b5c0-875965eaf50f)

## 🛠 Skills
Python, Machine Learning, Deep Learning, Tensorflow, Keras, Streamlit

