# Linear Regression From Scratch (NumPy)

This project implements **Linear Regression from scratch** using **Python and NumPy**, without relying on machine learning libraries like `scikit-learn`.

The goal of this project is to deeply understand:
- How linear regression works mathematically
- How gradient descent optimizes model parameters
- How weights and bias are updated during training
  
## 🚀 Features
- Manual implementation of:
  - Hypothesis function
  - Mean Squared Error (MSE) loss
  - Gradient Descent optimization
- Separate handling of **weights** and **bias**
- Fully vectorized operations using NumPy
- Beginner-friendly and well-commented code

## 📌 Mathematical Background

### Hypothesis Function
y_pred = X · w + b

Where:
(X) = feature matrix  
(w) = weight vector  
(b) = bias  

### Loss Function (Mean Squared Error)
J(w, b) = (1/m) * Σ (y_pred - y)²

### Gradients

**Derivative w.r.t. weights**
∂J/∂w = (1/m) * Xᵀ · (y_pred - y)


**Derivative w.r.t. bias**
∂J/∂b = (1/m) * Σ (y_pred - y)

### Gradient Descent Update Rules
```python
w = w - learning_rate * dw
b = b - learning_rate * db
