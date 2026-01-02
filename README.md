# House Price Prediction using Ridge & Lasso Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project applies **regularized linear regression techniques**—**Ridge** and **Lasso**—to predict house prices and control overfitting using **hyperparameter tuning**.

---

## Dataset
- **Name:** House Price Prediction Dataset
- **Type:** Supervised regression dataset
- **Target Variable:** House Price
- **Features:** Numerical and categorical features (preprocessed before modeling)

---

## Models Implemented
- **Ridge Regression**
- **Lasso Regression**

---

## Workflow
1. Define a range of **alpha values** for regularization  
2. Perform **hyperparameter tuning using GridSearchCV**  
   - RidgeRegressor as estimator  
   - LassoRegressor as estimator  
3. Fit the best model on training data  
4. Predict values for `X_test`  
5. Evaluate performance using **Mean Absolute Error (MAE)**  
6. Visualize predictions and residuals using **Matplotlib** and **Seaborn**

---

## Evaluation & Visualization
- **Metric Used:** Mean Absolute Error (MAE)  
- **Plots:**
  - Actual vs Predicted values
  - Residual distribution using `sns.displot(y_test - y_pred)`

These plots help in understanding model bias, variance, and prediction spread.

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## Project Structure
├── ridge_lasso_house_price_prediction.ipynb

├── README.md

---

## Key Takeaways
- Ridge regression reduces overfitting by shrinking coefficients
- Lasso regression performs feature selection by driving some coefficients to zero
- GridSearchCV helps identify optimal regularization strength
- MAE provides an interpretable evaluation metric for price prediction

---

## Author
**Nisarg Patel**  
Aspiring Data Scientist | Machine Learning Enthusiast

---

*If you found this project useful, feel free to star the repository!*
