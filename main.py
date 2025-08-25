import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle

st.title("Flower Species Predictor")
petal_length = st.number_input('Please choose a petal length',placeholder='Enter a valid number b/w 1.0 and 6.9',min_value=1.0,max_value=6.9,value = None)
petal_width = st.number_input('Petal width',placeholder='Enter a valid number b/w 1.0 and 2.5',min_value=1.0,max_value=2.5,value = None)
sepal_length = st.number_input('Sepal length',placeholder='Enter a valid number b/w 4.3 and 7.9',min_value=4.3,max_value=7.9,value = None)
sepal_width = st.number_input('Sepal width',placeholder='Enter a valid number b/w 2.0 and 4.4',min_value=2.0,max_value=4.4,value = None)


df_user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns =['sepal_length','sepal_width','petal_length','petal_width'])
st.write(df_user_input)

model_path = path.join('model','iris_classifier.pkl')
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)

species = {0 : 'setosa',1 : 'versicolor',2 : 'virginica'}

if st.button("Predict Species"):
     if(petal_length==None or petal_width==None or sepal_length==None or sepal_width==None):
         st.text("Please fill all values") #will be executed when any of the value is not entered properly
     else:
         predicted_species = iris_predictor.predict(df_user_input)
         st.write('The species is',species[predicted_species[0]]) #predictions can be done here
