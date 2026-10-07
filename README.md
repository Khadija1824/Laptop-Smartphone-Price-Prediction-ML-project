# 💻 Tech Gadget Valuation: Advanced Laptop Price Predictor

An end-to-end Machine Learning web application designed to predict real-time laptop market prices based on core hardware specifications. Built using Python, Scikit-Learn, and Streamlit, featuring a modern dark-mode aesthetic dashboard.

---

## 🚀 Project Overview
Navigating the secondhand or retail laptop market can be challenging without transparent valuation data. This project bridges that gap by leveraging a **Supervised Tabular Regression** pipeline powered by a **Random Forest Regressor**. Users can configure specific technical hardware properties (such as brand, RAM, storage, screen size, and weight) through an intuitive user interface to receive instant, accurate price estimations.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.10+
* **Data Processing & Manipulation:** Pandas
* **Machine Learning Model:** Scikit-Learn (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`, `OneHotEncoder`)
* **Web Interface:** Streamlit (Customized with modern CSS, glassmorphism elements, and smooth transition animations)
* **Model Serialization:** Pickle (`model.pkl`)

⚙️ How It Works (Pipeline Workflow)
Data Ingestion (laptop_data.csv): Loads raw tabular specifications tracking parameters like company brand, laptop type, screen inches, RAM size, storage type/capacity, weight, and target price.

Preprocessing (train_model.py):

Uses a ColumnTransformer with OneHotEncoder to encode categorical text features (Company, TypeName, Memory) into numerical columns.

Passes numerical properties (Inches, Ram, Weight) directly through the pipeline.

Model Training: Fits a robust RandomForestRegressor ensemble to capture non-linear relationships and interactions between hardware components.

Deployment (app.py): Deploys the pickled pipeline into an interactive web interface where real-time user selections are fed into the model for instant evaluation.

✨ Key Features & UI Design Highlights
Aesthetic Dark Theme: Styled with deep space gradient backgrounds and custom CSS styling.

Streamlined Layout: Features a dedicated header box, wide-format configuration blocks, and custom-styled interactive widgets.

Smooth Transitions & Feedback: Incorporates animated loading spinners and fade-in container scaling when calculating valuations.

Instant Predictions: Renders estimated market values clearly in a highlighted metric card format.

📈 Future Enhancements
Integrating a larger, live-scraped Kaggle dataset with thousands of modern models.

Adding interactive data visualization charts (e.g., Price vs. RAM distributions) inside the Streamlit app.

Implementing advanced hyperparameter tuning using GridSearchCV to optimize model accuracy.
