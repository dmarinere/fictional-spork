from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import streamlit as st
from keras.preprocessing.image import img_to_array
import streamlit as st
import requests
from PIL import Image
import requests
from io import BytesIO
import numpy as np
st.write(st.__version__)
st.title("Web App to Predict Xray Result Image and detect Pneumonia")


def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  
  return tf.image.resize(image,[64,64])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

def pnuemonia_router():
    model = define_model()
    model.load_weights('weights.h5')

    path = st.text_input('Enter Image URL to Classify.. ','https://raw.githubusercontent.com/happilyeverafter95/pneumonia-detection/master/fixtures/pneumonia_2.jpeg')
    if path is not None:
      image = requests.get(path).content
      image = Image.open(io.BytesIO(path))

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      st.markdown("**The result of your analysis is**")
      graph = tf.compat.v1.get_default_graph()
      with graph.as_default():
        #prediction = model.predict_proba(image)
        label = model.predict(decode_img(image))
      predicted_class = 'pneumonia' if label[0] > 0.5 else 'normal'
      st.write(predicted_class) 
      st.write(label)
    
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