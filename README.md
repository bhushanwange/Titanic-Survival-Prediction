Titanic Survival Prediction using Logistic Regression

This project demonstrates an end-to-end Machine Learning pipeline in Python for predicting passenger survival on the Titanic using Logistic Regression.

📌 Features & Project Workflow

Data Loading & Cleaning: Loads local Titanic CSV data, cleans column names, and handles missing values for continuous attributes.

Categorical Encoding: Transcribes categorical variables (such as Sex) into numerical format via One-Hot Encoding (pd.get_dummies).

Data Splitting & Feature Scaling: Splits data into an 80% training set and 20% test set while applying standard scaling (StandardScaler).

Model Training & Evaluation: Trains a Logistic Regression model and computes key metrics (Accuracy, Precision, Recall, and F1-Score).

ROC-AUC Visualization: Generates and displays a Receiver Operating Characteristic (ROC) curve using matplotlib.

📁 Directory Structure

Ensure your project directory is formatted as follows:

Titanic_Project/
├── titanic.csv          # The Titanic dataset file
├── titanic_model.py     # Python script executing the pipeline
└── README.md            # Project documentation


⚙️ Prerequisites & Installation

Make sure you have Python installed on your system. Install the required Python packages using pip:

pip install pandas scikit-learn matplotlib


🚀 How to Run

Open your terminal or command prompt and navigate to your project directory:

cd path/to/Titanic_Project


Execute the main Python script:

python titanic_model.py