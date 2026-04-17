#  AI Recommendation System

##  Project Overview
This is an end-to-end AI-powered recommendation system built using the MovieLens dataset.  
The system uses a hybrid approach combining collaborative filtering (SVD) and content-based filtering (TF-IDF) to generate personalized movie recommendations.

---

## Features
- Hybrid Recommendation Model (SVD + TF-IDF)
- Real-time recommendations using Flask API
- Cold start handling using popular items
- Filters already watched movies
- Frontend UI for user interaction
- Deployed on cloud (Render)

---

##  Architecture
Data → Preprocessing → Model Training → Flask API → UI → Recommendations

---

##  Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML/CSS
- MovieLens Dataset

---

## Project Structure

ai_recommendation_system/
- app.py (Flask API)
- model.py (Hybrid model: SVD + TF-IDF)
- preprocess.py (Data preprocessing)
- requirements.txt
- Procfile
- data/
- templates/

---

##  Setup

1. Install dependencies  
pip install -r requirements.txt

2. Preprocess data  
python preprocess.py

3. Train model  
python model.py

4. Run application  
python app.py

---

## Run in Browser

http://127.0.0.1:5000

---

##  Live Demo

https://ai-movie-recommendation-mpwm.onrender.com
---

##  Features

- Hybrid recommendation system (SVD + TF-IDF)
- Real-time recommendations using Flask
- Handles new users (cold start)
- Removes already watched items
- Simple UI for interaction
