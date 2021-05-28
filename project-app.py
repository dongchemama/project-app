import streamlit as st 
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title("Image Classifier using Machine learning")
st.subheader("upload the image")

model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("choose an image..")
if uploaded_file is not None:
  img= Image.open(uploaded_file)
  st.image(img,caption="Image Uploaded")

  if st.button('PREDICT'):
    CATEGORIES = ['animal dog','animal elephant','animal giraffe','animal horse', 'animal monkey','insects','reptile snake']
    st.write('Result....')
    flat_data = []
    img = np.array(img)
    img_resized = resize(img,(100,100,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.write(f'PREDICTED OUTPUT : {y_out}')
    q=model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item} : {q[0][index]*100}')
