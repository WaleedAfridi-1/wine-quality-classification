# wine-quality-classification
machine learning pipeline using XGBoost and SMOTE to classify wine quality as "good" or "not good" based on chemical properties. Achieves 91.9% accuracy with strong precision/recall. Built in Python using scikit-learn, XGBoost, and imbalanced-learn. Ideal for binary classification tasks.



# 🍷 Wine Quality Classification Project

---

## 📋 Project Overview  
This project implements a machine learning pipeline to classify wine quality as **"good" (quality ≥ 7)** or **"not good" (quality < 7)** based on physicochemical properties. The solution uses **XGBoost with SMOTE** for handling class imbalance and achieves **91.9% accuracy**.

---

## 🎯 Objective  
To build a classification model that predicts wine quality (binary classification) using various chemical features like acidity, sugar, sulfur dioxide, alcohol content, etc.

---

## 📊 Dataset  
The dataset contains **1,599 samples** with **12 features**:

- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  
- Quality (target variable, converted to binary)

---

## 🛠️ Technologies Used  
- Python 3  
- Pandas & NumPy (data manipulation)  
- Matplotlib & Seaborn (visualization)  
- Scikit-learn (machine learning utilities)  
- XGBoost (gradient boosting algorithm)  
- Imbalanced-learn (SMOTE for handling class imbalance)

---

## 📈 Key Steps  

### 1. Data Preprocessing  
- Converted quality score to binary classification (1 if ≥7, else 0)  
- Explored feature correlations using a heatmap visualization  

### 2. Model Building  
Created a pipeline with:
- `MinMaxScaler` for feature normalization  
- `SMOTE` for oversampling the minority class  
- `XGBClassifier` with optimized hyperparameters  

### 3. Model Evaluation  
- Split data into 80% training and 20% testing sets  
- Achieved **91.9% accuracy** on test data  

**Precision/Recall:**
- Class 0 (not good): 96% precision, 94% recall  
- Class 1 (good): 70% precision, 79% recall  

---

## 📋 Results  

| Metric              | Value |
|---------------------|--------|
| Overall Accuracy     | 91.9%  |
| Precision (Class 0)  | 96%    |
| Recall (Class 0)     | 94%    |
| Precision (Class 1)  | 70%    |
| Recall (Class 1)     | 79%    |

---

## 🔧 Installation & Usage  

1. **Clone the repository**  
2. **Install required packages:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

3. **Run the Jupyter notebook:**
```bash
jupyter notebook wine_quality_classification.ipynb
```

---

## 📁 Project Structure

- wine-quality-classification/
- ├── wine_quality_classification.ipynb  # Main Jupyter notebook
- ├── wine_dataset.csv                   # Dataset file
- ├── requirements.txt                   # Python dependencies
- └── README.md                          # Project documentation

---


## 🚀 Future Improvements

- Experiment with different classification algorithms  
- Perform more extensive hyperparameter tuning  
- Implement feature engineering to improve model performance  
- Develop a web application for real-time predictions  

---

## 👥 Author

**[Waleed Afridi]**  

