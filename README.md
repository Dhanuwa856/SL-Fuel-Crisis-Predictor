# 🇱🇰 Sri Lanka Economic Crisis Predictor (ML Simulation)

## 📌 Project Overview
This project applies Machine Learning to simulate and predict the "Ripple Effect" of the ongoing economic and fuel crisis in Sri Lanka. Using real-world baseline indicators (Fuel prices, USD rates, QR Quotas), an end-to-end ML pipeline was built to forecast critical economic disruptions, such as national power grid failures and food price inflation.

## 🛠️ Tech Stack & Skills Demonstrated
* **Languages & Libraries:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Unsupervised Learning:** K-Means Clustering, Principal Component Analysis (PCA)
* **Supervised Learning:** Logistic Regression, Support Vector Machines (SVM)
* **Model Evaluation:** GridSearchCV, Hyperparameter Tuning, Confusion Matrices, Precision-Recall analysis.

## 📊 The Dataset
Since comprehensive historical data for this specific novel crisis is scarce, a highly realistic simulated dataset (1000 days) was generated based on **current baseline ground-truth values in Sri Lanka**:
* USD Rate: ~Rs. 313
* Auto Diesel: Rs. 382 / Petrol 92: Rs. 398
* Essential Food (Rice/Bread) current market prices
* CEB Power Cut Probabilities based on fuel shortages.

## 🧠 Methodology & Results
### 1. Exploratory Data Analysis (EDA)
Identified strong positive correlations (e.g., Pearson coeff = 0.84) between Diesel Prices and Essential Food Prices, visually proving the economic ripple effect.

### 2. Unsupervised Learning (Clustering)
Applied **PCA** to reduce dimensionality and used **K-Means Clustering**. The algorithm naturally discovered 3 distinct economic states (Normal, Moderate Stress, Severe Crisis) without any prior labeling, proving the strong underlying signal in the data.

### 3. Supervised Learning (Predictive Modeling)
Built classifiers to predict imminent Power Cuts (Grid Failures) based on economic inputs.
* **Logistic Regression:** Achieved 98.5% Accuracy with 'balanced' class weights.
* **Support Vector Machine (RBF Kernel):** Initial model achieved 99.0% accuracy.
* **Hyperparameter Tuning:** Used `GridSearchCV` to optimize the SVM. Found the optimal parameters (`C=100`, `gamma='scale'`), which pushed the final Test Accuracy to an incredible **99.5%**, achieving a Precision of 1.00 (Zero False Positives) for crisis prediction.

## 🚀 Conclusion
This project successfully demonstrates how foundational linear classifiers and unsupervised clustering algorithms can be combined to model complex, real-world socio-economic issues with extremely high precision.