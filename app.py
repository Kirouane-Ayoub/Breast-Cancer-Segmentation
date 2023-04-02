import streamlit as st
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from model_definition import SegmentationModel
import matplotlib.pyplot as plt 
import tensorflow as tf
import cv2

tab0 , tab1 = st.tabs(["Home" , "Segmentation"])
with st.sidebar : 
    st.image('icon.png' , width=250)
    save = st.radio("You want to save your Result ? " , ("Yes" , "No"))
with tab0 : 
    st.header("About This project : ")
    st.image("des.png")
    st.write("""
In recent years, using Deep Learning methods to apply medical and biomedical image analysis has seen many advancements. 
In a clinical, using Deep Learning-based approaches for cancer image analysis is one of the key applications for cancer detection and treatment.
However, the scarcity and shortage of labeling images make the task of cancer detection and analysis difficult to reach high accuracy. 

In 2015, the Unet model was introduced and gained much attention from researchers in the field. 
The success of Unet model is the ability to produce high accuracy with very few input images.
Since the development of Unet, there are many variants and modifications of Unet related architecture. 

In this project, I've developed a breast cancer image analysis system using the U-Net architecture. The aim of the system is to assist doctors in detecting breast cancer in medical images accurately and efficiently. The U-Net architecture is a type of convolutional neural network that is widely used for image segmentation tasks.

The system was trained on a large dataset of breast cancer images to learn the features of cancerous and non-cancerous cells. It then uses this knowledge to segment and classify the cells in new medical images as cancerous or non-cancerous.

The project involved various steps, including data preprocessing, model training, and model evaluation. The performance of the system was evaluated using metrics such as accuracy, precision, and recall, and the results were compared to those of other state-of-the-art methods.

Overall, this project has the potential to significantly improve the accuracy and efficiency of breast cancer diagnosis and treatment, ultimately leading to better patient outcomes.
""")



with tab1 :
    model = SegmentationModel().model
    model.load_weights("cancer_weights.h5")
    fi_le = st.file_uploader("Uplead Your image : " , type=["png" , "jpeg" , "jpg"])
    if fi_le : 
        name = fi_le.name
        if st.button("Click To start") : 
            with st.spinner('Wait for it ...'):
                image = tf.keras.utils.load_img(
                    name , 
                    target_size=(256 , 256) , 
                    color_mode = "rgb" , 
                    interpolation="nearest" , 
                    keep_aspect_ratio=False
                )
                y_hat = model.predict(tf.expand_dims(image , axis=0))
                y_hat = np.squeeze(np.where(y_hat > 0.3, 1.0, 0.0))
                x = cv2.imread(name)
                fig, ax = plt.subplots(1,6, figsize=(30,30))
                for i in range(6):
                    ax[i].imshow(y_hat[:,:,i])
                if save == "Yes" : 
                    fig.savefig(f"{name.split('.')[0]}_result.png" , format="png")
                else : 
                    pass
                st.header("Segmentation Resuls : ")
                st.pyplot(fig)
                st.success('Done !! ', icon="✅")
