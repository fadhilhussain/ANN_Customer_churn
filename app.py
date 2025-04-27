import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# load the trained model 
model = tf.keras.models.load_model('model.h5')


# load the encoders and scaler 
with open('label_encoder_gender.pkl','rb') as file_object:
    gender_encoder = pickle.load(file_object)

with open('onehot_geography.pkl','rb') as file_object:
    geo_encoder = pickle.load(file_object)

with open('scaler.pkl','rb') as file_object:
    scaler = pickle.load(file_object)


## streamlit app
st.title('Customer churn prediction')

# user input 
geography = st.selectbox('Geagraphy', geo_encoder.categories_[0])
gender = st.selectbox('Gender',gender_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number Of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_memeber = st.selectbox('Is Activate Member',[0,1])


# prepare the input data
input_data = pd.DataFrame(
    {
        'CreditScore' : [credit_score],
        'Gender' : [gender_encoder.transform([gender])[0]],
        'Age' : [age],
        'Tenure' : [tenure],
        'Balance' : [balance],
        'NumOfProducts' : [num_of_product],
        'HasCrCard' : [has_cr_card],
        'IsActiveMember' : [is_active_memeber],
        'EstimatedSalary' : [estimated_salary]
    }
)

# onehot for geography 
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))


# combine input data with one hot df 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)


# scaling the data 
input_data_scaled = scaler.transform(input_data)

# predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability : {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn')



