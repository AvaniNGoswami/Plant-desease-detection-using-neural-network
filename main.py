import streamlit as st
import tensorflow as tf
import numpy as np
import cohere


def model_prediction(test_image):
    model = tf.keras.models.load_model(
        "C:/Users/Avani N. Goswami/Desktop/jupyter notebook/plant disease detection/plant_model_keras3.h5"
    )
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  
    predictions = model.predict(input_arr) 
    return np.argmax(predictions)  



co = cohere.Client("25xymcEmgWCL7V2xDtP9j1stk3kXRGehhkUcY9i2")

documents = [
    "Apple___Black_rot : Prune and destroy infected fruit and branches. Use fungicides like captan or sulfur.",
    "Apple___Cedar_apple_rust: Remove nearby juniper trees. Apply fungicides like myclobutanil, mancozeb, sulfur, or copper early in the season.",
    "Cherry___Powdery_mildew: Prune for good air flow. Apply sulfur-based or oil-based fungicides preventively.",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot, Common_rust, Northern_Leaf_Blight: Use resistant hybrid seeds. Apply fungicides like chlorothalonil, mancozeb, or azoxystrobin during risk periods.",
    "Grape___Black_rot, Esca, Leaf_blight: Remove old fruit and infected leaves. Spray fungicides like copper or systemic products early in the season.",
    "Orange___Haunglongbing_(Citrus_greening): Control psyllid insects using insecticides. Remove infected trees. Experimental methods include heat treatment.",
    "Peach___Bacterial_spot: Use resistant varieties. Avoid overhead watering. Spray copper-based bactericides.",
    "Potato___Early_blight, Late_blight: Rotate crops and remove infected debris. Spray chlorothalonil or mancozeb preventively.",
    "Strawberry___Leaf_scorch: Use resistant varieties. Apply fungicides like captan.",
    "Tomato___Leaf_Mold: Improve air circulation and lower humidity. Apply protective fungicides.",
    "Tomato___Septoria_leaf_spot: Use mancozeb or chlorothalonil at early stages.",
    "Tomato___Spider_mites Two-spotted_spider_mite: Spray plants with water. Use predatory mites or miticides.",
    "Tomato___Target_Spot: Apply fungicides like chlorothalonil.",
    "Tomato___Tomato_mosaic_virus: Remove infected plants. Practice tool sanitation and avoid touching plants when wet.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus: Use resistant varieties. Control whitefly insects with insecticides.",
]

doc_response = co.embed(
    model="embed-english-v3.0", input_type="search_document", texts=documents
)
doc_embeddings = doc_response.embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))






tab1, tab2, tab3 = st.tabs(["Home", "About", "Disease Recognition"])


with tab1:
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "C:/Users/Avani N. Goswami/Desktop/jupyter notebook/plant disease detection/home_page.jpeg"
    st.image(image_path,width=300)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    """)

with tab2:
    st.header("About")
    st.markdown(
        """
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.  
        The original dataset can be found on GitHub.  
        It consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes.  
        The dataset is divided into 80/20 ratio of training and validation set.  
        A new directory containing 33 test images is created later for prediction purpose.

        #### Content
        1. Train (70,295 images)  
        2. Test (33 images)  
        3. Validation (17,572 images)  
        """
    )


with tab3:
    st.header("Disease Recognition")
    test_image = st.file_uploader("📷 Choose a Plant Leaf Image:")

    if test_image is not None:
        st.image(test_image, width=200)

    if st.button("🔍 Predict"):
        st.subheader(" Our Prediction")
        result_index = model_prediction(test_image)

        
        class_name = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]

        predicted_class = class_name[result_index]
        st.success(f"🌿 The model predicts this is: **{predicted_class}**")


        query = f"What is solution of {predicted_class}?"
        query_response = co.embed(
            model="embed-english-v3.0", input_type="search_query", texts=[query]
        )
        query_embedding = query_response.embeddings[0]

        scores = [cosine_similarity(query_embedding, doc) for doc in doc_embeddings]
        best_idx = np.argmax(scores)

        st.subheader("Suggested Remedy")
        st.info(documents[best_idx])

        st.markdown(
            """
        ---
        **🌍 General Tips for All Diseases**
        - Use disease-resistant plant varieties  
        - Clean up plant debris  
        - Avoid overhead watering when possible  
        - Apply appropriate fungicides or insecticides early, especially in humid or rainy weather  
        - Rotate crops yearly to reduce disease buildup in the soil  
        """
        )
