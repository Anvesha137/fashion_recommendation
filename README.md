Fashion Recommendation System 👗🤖

This is a deep learning-based fashion recommendation system that suggests visually similar clothing items from a dataset. The system uses ResNet50, a powerful CNN architecture, for extracting image features and recommends products based on their visual similarity.

🚀 Features
Upload an image and get visually similar fashion recommendations

Uses ResNet50 pretrained on ImageNet for feature extraction

Embedding-based similarity search using cosine similarity / FAISS

Web interface using Flask or Streamlit

Easily extendable with custom datasets

🧠 Technologies Used
Python

Google Colab

TensorFlow / Keras (with ResNet50)

scikit-learn

NumPy & Pandas

Streamlit

NearestNeighbors (sklearn)

Dataset used

https://www.kaggle.com/paramaggarwal/datasets

🧩 How It Works
Image Feature Extraction:
The input images are passed through the ResNet50 model (excluding top layers) to obtain a 2048-dimensional feature vector.

Similarity Matching:
These vectors are compared with the pre-computed embeddings of the dataset using cosine similarity.

Recommendation:
The top 5 most similar items are returned based on similarity scores.

Screenshot
![image](https://github.com/user-attachments/assets/1f59262f-c8a3-4665-acc1-57fe298ba112)
![image](https://github.com/user-attachments/assets/dd95da2c-7ab0-48b5-bcfc-cc2ec06f75df)

