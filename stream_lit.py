from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten 
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO
import numpy as np
st.write(st.__version__)
st.title("Web App to Predict Xray Result Image and detect Pneumonia")


def pnuemonia_router():
    model = define_model()
    model.load_weights('weights.h5')

    path = st.file_uploader('Enter Image URL to detect if there is Pneumonia.. ','')
      
    image = Image.open(BytesIO(path))
    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((64, 64))
    image = img_to_array(image)/255.0
    image = image.reshape(1, 64, 64, 1)
     # image = Image.open(io.BytesIO(path))

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      st.markdown("**The result of your analysis is**")
   #   graph = tf.compat.v1.get_default_graph()
    #  with graph.as_default():
        #prediction = model.predict_proba(image)
      label = model.predict(image)
    #  label = model.predict(decode_img(image))
      predicted_class = 'pneumonia' if label[0] > 0.5 else 'normal'
      if predicted_class == 'pneumonia':
        st.markdown(f"The Xray scan model has revealed that it has a case of {predicted_class} with {label} Probability)
      else:                                  
        st.markdown(f"The Xray scan model revealed that this is a {predicted_class})
    
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
  pnuemonia_router()
