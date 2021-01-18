import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib

@st.cache
def load_model():
	return tf.keras.models.load_model('model.h5')

def set_target_labels(labels):
	return [x[1].decode('utf-8') for x in labels]

def preprocess_image(img):
	pimg = img.copy()
	pimg = np.sum(img/3, axis=3, keepdims=True)
	pimg = (pimg - 128) / 128
	return pimg


model = load_model()
labels = [( 0, b'Speed limit (20km/h)'),
( 1, b'Speed limit (30km/h)'),
( 2, b'Speed limit (50km/h)'),
( 3, b'Speed limit (60km/h)'),
( 4, b'Speed limit (70km/h)'),
( 5, b'Speed limit (80km/h)'),
( 6, b'End of speed limit (80km/h)'),
( 7, b'Speed limit (100km/h)'),
( 8, b'Speed limit (120km/h)'),
( 9, b'No passing'),
(10, b'No passing for vehicles over 3.5 metric tons'),
(11, b'Right-of-way at the next intersection'),
(12, b'Priority road'),
(13, b'Yield'),
(14, b'Stop'),
(15, b'No vehicles'),
(16, b'Vehicles over 3.5 metric tons prohibited'),
(17, b'No entry'),
(18, b'General caution'),
(19, b'Dangerous curve to the left'),
(20, b'Dangerous curve to the right'),
(21, b'Double curve'),
(22, b'Bumpy road'),
(23, b'Slippery road'),
(24, b'Road narrows on the right'),
(25, b'Road work'),
(26, b'Traffic signals'),
(27, b'Pedestrians'),
(28, b'Children crossing'),
(29, b'Bicycles crossing'),
(30, b'Beware of ice/snow'),
(31, b'Wild animals crossing'),
(32, b'End of all speed and passing limits'),
(33, b'Turn right ahead'),
(34, b'Turn left ahead'),
(35, b'Ahead only'),
(36, b'Go straight or right'),
(37, b'Go straight or left'),
(38, b'Keep right'),
(39, b'Keep left'),
(40, b'Roundabout mandatory'),
(41, b'End of no passing'),
(42, b'End of no passing by vehicles over 3.5 metric tons')]
labels = set_target_labels(labels)


# main page
st.title("Sign Traffic Classifier")
st.write("""
Source Notebook: [link](https://github.com/hendraronaldi/machine_learning_projects/blob/main/MLProject%20LeNet%20Traffic%20Sign%20Classifier.ipynb)
""")

uploaded_img = st.file_uploader("Upload Image")
if uploaded_img is not None:
	try:
		file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
		frame = cv2.imdecode(file_bytes, 1)

		if frame.shape[0] > 500:
			st.image(frame, channels="BGR", width=500)
		else:
			st.image(frame, channels="BGR")

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (32, 32))
		img = image.img_to_array(frame)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = preprocess_image(img)
		y_pred = np.argmax(model.predict(img))
		st.subheader('Prediction Sign')
		st.text(labels[y_pred])
	except:
		st.subheader("Please upload an image file")
