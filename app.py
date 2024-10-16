import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

# Load your trained model and vectorizer
with open('FFM.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

def convert_to_sparse_tensor(csr_matrix):
    coo = csr_matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.SparseTensor(indices, coo.data, coo.shape)
    

# Streamlit app
st.title("Bangla Emotion Detection: বাংলা লেখা হতে আবেগ নির্ণয়")

input_text = st.text_area("Enter text here:")

if st.button("Predict"):
    if input_text:
        # Preprocess the input text
        input_vec = vectorizer.transform([input_text])
        input_sparse = convert_to_sparse_tensor(input_vec)
        input_sparse = tf.sparse.reorder(input_sparse)
        input_dense = tf.sparse.to_dense(input_sparse).numpy()

        # Make a prediction
        prediction = model.predict(input_dense)
        predicted_index = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_index])
        st.write(f"The predicted class is: {predicted_class[0]}")
        

    else:
        st.write("Please enter some text to predict.")

