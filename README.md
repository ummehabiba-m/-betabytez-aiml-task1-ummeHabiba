# Heart Disease Prediction — ML Classifier with Web UI

A machine learning project that predicts the likelihood of heart disease in a 
patient based on clinical features, deployed behind a simple Streamlit web 
app so anyone — technical or not — can get a prediction without touching code.
---

## 📌 Project Overview

The goal of this task was to understand the full lifecycle of a machine 
learning project: from raw data to a usable, deployed application. 
Specifically:

1. Explore and understand a real-world medical dataset
2. Clean and preprocess the data for modeling
3. Train and compare multiple classification models
4. Evaluate them using standard ML metrics
5. Deploy the best-performing model behind a simple web interface

---

## 🗂️ Dataset

**UCI Heart Disease Dataset**
- 303 patient records
- 13 clinical features (age, sex, chest pain type, cholesterol, resting 
  blood pressure, max heart rate, etc.)
- 1 binary target column: presence (`1`) or absence (`0`) of heart disease
- No missing values
- Fairly balanced classes: 165 positive cases (54.5%), 138 negative (45.5%)

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Verified data shape, types, and confirmed zero missing values
- Analyzed class distribution to check for imbalance
- Built a correlation heatmap to identify which features relate most 
  strongly to heart disease (chest pain type and max heart rate showed the 
  strongest positive correlation; exercise-induced angina and ST depression 
  showed the strongest negative correlation)

### 2. Preprocessing
- One-hot encoded 8 categorical features (expanding from 13 to 22 columns)
- Split data 80/20 into train/test sets, stratified to preserve class balance
- Scaled numerical features using `StandardScaler`, fitted only on the 
  training set to prevent data leakage into the test set

### 3. Model Training & Evaluation
Two classification models were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **86.9%** | **85.7%** | **90.9%** | **0.882** |
| Random Forest (100 trees) | 73.8% | 74.3% | 78.8% | 0.765 |

**Logistic Regression was selected** as the final model. It outperformed 
Random Forest across every metric — likely because the dataset is 
relatively small and the underlying relationships between features and the 
target are fairly linear, making the simpler model less prone to 
overfitting. Its high recall (90.9%) is especially valuable in a medical 
context, since it means the model misses very few actual heart disease cases.

### 4. Deployment
The trained model, fitted scaler, and expected column structure were saved 
using `joblib`, then loaded into a Streamlit app where a user can input 
patient details through form fields and get an instant prediction with a 
confidence score.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python | Core language |
| Google Colab | Development environment |
| Pandas | Data loading & manipulation |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Preprocessing, model training, evaluation |
| Joblib | Model serialization |
| Streamlit | Web UI for live predictions |
| Git & GitHub | Version control with incremental commits |

---

## ⚠️ Challenges Faced

- **Broken dataset URL:** the original dataset source returned a 404 error; 
  resolved by sourcing a reliable mirror of the UCI Heart Disease dataset.
- **Logistic Regression convergence warning:** the model initially failed 
  to converge within the default iteration limit. Fixed by increasing 
  `max_iter` and ensuring all features (including one-hot encoded columns) 
  were consistently typed as floats before training.
- **Matching live input to training format:** a single user entry in the 
  Streamlit app doesn't automatically generate every one-hot encoded column 
  present in the full training set. This was solved by saving the exact 
  training column list (`model_columns.pkl`) and programmatically adding any 
  missing columns as `0` before prediction, ensuring the input always 
  matches the shape the model expects.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ummehabiba-m/betabytez-aiml-task1-ummeHabiba.git
cd betabytez-aiml-task1-ummeHabiba
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn streamlit joblib matplotlib seaborn
```

### 3. Run the notebook (optional — to see the full EDA/training process)
Open `notebooks/heart_disease_classifier.ipynb` in Jupyter or Google Colab.

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```
This opens a browser window where you can enter patient details and get a 
heart disease risk prediction.

---

## 📁 Project Structure

```
betabytez-aiml-task1-ummeHabiba/
├── notebooks/
│   └── heart_disease_classifier.ipynb
├── model/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
├── app/
│   └── app.py
└── README.md
```
---
**URL:** https://heart-disease-classifiers.streamlit.app/

## ✍️ Author

**Umme Habiba Malik**

**Note:** _The model's feature contributions reflect patterns learned from this 
specific 303-patient dataset, scaled relative to the training data's mean 
and spread — not universal medical thresholds. A value like cholesterol=180 
mg/dl, while medically considered healthy, may still show as increasing 
predicted risk if it happens to correlate with the positive class in this 
particular sample. This highlights a common limitation of small medical 
datasets: learned patterns may not always align with established clinical 
guidelines._
