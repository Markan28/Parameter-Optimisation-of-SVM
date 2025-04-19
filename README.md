# Parameter-Optimisation-of-SVM

SVMs have hyperparameters that control how the model behaves. Parameter optimization (or hyperparameter tuning) means finding the best combination of these hyperparameters to maximize the model’s performance (like accuracy, F1 score, etc.) on your data.

🔧 Key Hyperparameters in SVM
1. C (Regularization Parameter)
Controls trade-off between achieving a low error on training data and minimizing the margin (maximizing generalization).

Low C: Makes the margin wide but allows more misclassifications (underfitting).

High C: Makes the margin narrow, focusing on correctly classifying all training points (overfitting).

Think of C like a “penalty for misclassification”.

2. Kernel Type
SVM can map data into higher dimensions using kernel functions.

linear: For linearly separable data.

rbf (Radial Basis Function): Popular, non-linear, handles complex data well.

poly: Polynomial kernel (degree also needs tuning).

sigmoid: Similar to a neural network’s activation function.

3. Gamma (γ) for RBF/Poly/Sigmoid Kernels
Determines how far the influence of a single training example reaches.

Low gamma: Large similarity radius (more general model).

High gamma: Small radius (model gets very tight around data — overfitting risk).

4. Degree (only for Polynomial kernel)
Defines the degree of the polynomial when using a polynomial kernel.

⚙️ How is Optimization Done?
1. Grid Search (GridSearchCV)
Try all combinations of hyperparameter values in a grid.

Time-consuming but exhaustive.

python
Copy
Edit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
2. Randomized Search (RandomizedSearchCV)
Samples random combinations of hyperparameters.

Faster than grid search, especially with large parameter spaces.

3. Bayesian Optimization / Genetic Algorithms
Advanced optimization methods for smarter, faster tuning.

Libraries: Optuna, Hyperopt, BayesSearchCV

🧠 Evaluation Strategy
Use cross-validation (e.g., 5-fold CV) to evaluate performance on unseen data during tuning, reducing overfitting.

📝 Example Workflow
Preprocess the data.

Choose kernel (start with RBF or linear).

Set ranges for C and gamma.

Perform GridSearchCV with cross-validation.

Use best model on the test set.
