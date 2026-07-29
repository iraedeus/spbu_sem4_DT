# SPbU Sem 4 — Machine Learning & Data Technologies

This repository contains the coursework, algorithm implementations, experimental notebooks, and unit tests for the **Machine Learning** course (Semester 4) at Saint Petersburg State University (SPbU).

---

## 📁 Repository Structure

```text
.
├── ml_spbu/               # Main package directory containing homework assignments
│   ├── homework_1/        # Exploratory & Visual Data Analysis (EDA / VDA)
│   ├── homework_2/        # Custom ML Framework (KNN, KDTree, Metrics, Normalizers)
│   ├── homework_3/        # Linear Regression
│   ├── homework_4/        # Gradient Descent Optimization
│   ├── homework_5/        # Support Vector Machines (SVM)
│   ├── homework_6/        # Ensemble Methods (Random Forest, CatBoost)
│   ├── homework_7/        # Unsupervised Learning & Clustering
│   └── homework_8/        # Natural Language Processing & Text Processing
├── tests/                 # Unit tests (PyTest) for custom modules
├── pyproject.toml         # Project dependencies and tool configurations
└── README.md              # Project documentation
```

---

## 📑 Homework Overview

### 🔹 [Homework 1: Exploratory & Visual Data Analysis (EDA & VDA)](ml_spbu/homework_1)

Focuses on foundational data exploration and visualization techniques applied to the classic **Titanic** dataset.

- **EDA ([`titanic_eda.ipynb`](ml_spbu/homework_1/EDA/titanic_eda.ipynb))**: Initial data analysis, handling missing values (Age, Cabin, Embarked), feature distribution analysis, and statistical summary.
- **VDA ([`titanic_vda.ipynb`](ml_spbu/homework_1/VDA/titanic_vda.ipynb))**: Visual data analysis using Matplotlib and Seaborn. Examines feature interactions, survival rates across passenger classes, gender, age groups, and ticket pricing.

---

### 🔹 [Homework 2: Custom Machine Learning Framework from Scratch](ml_spbu/homework_2)

Implements core machine learning algorithms, data structures, preprocessing utilities, and metrics from scratch in pure Python and NumPy with full unit test coverage.

- **K-Nearest Neighbors ([`KNNClassifier`](ml_spbu/homework_2/KNNClassifier/knn_classifier.py))**: Custom k-NN classification supporting brute-force search as well as optimized KD-Tree queries.
- **Spatial Indexing ([`KDTree`](ml_spbu/homework_2/KDTree/kd_tree.py))**: Implementation of a k-dimensional tree along with `KDNode` and `KDMaxHeap` for efficient nearest-neighbor lookup in multidimensional spaces.
- **Distance Metrics ([`metrics/distance`](ml_spbu/homework_2/metrics/distance))**: Custom implementations of Euclidean, Manhattan, and Minkowski distance functions.
- **Classification Metrics ([`metrics/classification`](ml_spbu/homework_2/metrics/classification))**: Implementation of Accuracy score and F1-Score (macro/micro/weighted averaging).
- **Data Normalizers ([`normalizers`](ml_spbu/homework_2/normalizers))**: Custom feature scalers including `StandardScaler`, `MinMaxScaler`, and `RobustScaler` extending from `AbstractScaler`.
- **Data Splitting ([`train_test_split.py`](ml_spbu/homework_2/train_test_split.py))**: Custom dataset splitting utility into training and testing sets.
- **Experiments & Validation**: Evaluation of custom algorithms against standard baselines on Breast Cancer ([`cancer.ipynb`](ml_spbu/homework_2/notebooks/cancer.ipynb)) and Spam ([`spam.ipynb`](ml_spbu/homework_2/notebooks/spam.ipynb)) datasets.
- **Unit Tests ([`tests/homework_2`](tests/homework_2))**: Comprehensive PyTest suite verifying algorithm correctness, edge cases, and numerical properties.

---

### 🔹 [Homework 3: Linear Regression](ml_spbu/homework_3)

Covers theoretical foundations and practical applications of **Linear Regression**.

- **Notebook ([`linreg.ipynb`](ml_spbu/homework_3/linreg.ipynb))**: Analytical (normal equation) and iterative solutions for linear regression models.
- **Key Concepts**: Feature engineering, evaluation metrics (MSE, RMSE, MAE, $R^2$), multicollinearity analysis, and regularization methods ($L_1$ Lasso and $L_2$ Ridge regression).
- **Datasets**: `train.csv` and `test.csv` for model training and out-of-sample performance evaluation.

---

### 🔹 [Homework 4: Gradient Descent Optimization](ml_spbu/homework_4)

Explores numerical optimization algorithms for training machine learning models.

- **Notebook ([`gradient_descent.ipynb`](ml_spbu/homework_4/gradient_descent.ipynb))**: Implementation and comparative study of optimization techniques:
  - **Batch Gradient Descent (BGD)**
  - **Stochastic Gradient Descent (SGD)**
  - **Mini-Batch Gradient Descent**
- **Analysis**: Effect of learning rates, learning rate decay/schedules, convergence dynamics, loss landscape visualization, and momentum optimization.

---

### 🔹 [Homework 5: Support Vector Machines (SVM)](ml_spbu/homework_5)

Investigates linear and non-linear classification using **Support Vector Machines**.

- **Notebook ([`svm.ipynb`](ml_spbu/homework_5/svm.ipynb))**:
  - **Hard-margin & Soft-margin SVM**: Dual problem formulation and solving via Quadratic Programming using `cvxopt`.
  - **Kernel Trick**: Linear, Polynomial, and Radial Basis Function (RBF / Gaussian) kernels.
  - **Hyperparameter Analysis**: Impact of regularization parameter $C$ and kernel parameter $\gamma$ on margin width, support vector selection, and overfitting/underfitting boundaries.

---

### 🔹 [Homework 6: Ensemble Methods](ml_spbu/homework_6)

Focuses on tree-based algorithms and ensemble learning applied to a VK dataset (`vk.csv`).

- **Notebook ([`ensembles.ipynb`](ml_spbu/homework_6/ensembles.ipynb))**:
  - **Decision Trees**: Splitting criteria (Gini impurity, Entropy) and tree pruning.
  - **Bagging & Random Forests**: Bootstrap aggregating, out-of-bag (OOB) score evaluation, variance reduction.
  - **Gradient Boosting**: Comparison of state-of-the-art gradient boosting frameworks (CatBoost, XGBoost/LightGBM).
  - **Analysis**: Feature importance ranking and hyperparameter tuning.

---

### 🔹 [Homework 7: Unsupervised Learning & Clustering](ml_spbu/homework_7)

Covers unsupervised learning methods and cluster analysis techniques.

- **Notebook ([`clustering.ipynb`](ml_spbu/homework_7/clustering.ipynb))**:
  - **Algorithms**: K-Means, Agglomerative Hierarchical Clustering (with dendrograms), and DBSCAN.
  - **Cluster Evaluation**: Elbow method, Silhouette Coefficient, Davies-Bouldin Index.
  - **Image Compression Application**: Color quantization of images (`meme.jpg` -> `result.jpg`) using cluster centroids to represent dominant image colors.

---

### 🔹 [Homework 8: Natural Language Processing & Text Processing](ml_spbu/homework_8)

Introduces natural language processing pipelines and text classification.

- **Notebook ([`texts.ipynb`](ml_spbu/homework_8/texts.ipynb))**:
  - **Text Preprocessing**: Tokenization, stop-words removal, lowercasing, stemming, and lemmatization using NLTK.
  - **Feature Vectorization**: Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).
  - **Text Classification**: Spam filtering on `spam.txt` using Naive Bayes and Logistic Regression models evaluated with Precision, Recall, F1-Score, and ROC-AUC.

---

## 🛠️ Environment & Prerequisites

This project is managed using modern Python packaging tooling configured in [`pyproject.toml`](pyproject.toml).

### Key Dependencies

- **Python**: `>= 3.11`
- **Core ML & Data Science**: `numpy`, `pandas`, `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`
- **Optimization & Boosting**: `cvxopt`, `catboost`
- **NLP**: `nltk`
- **Environment & Testing**: `jupyterlab`, `pytest`, `hypothesis`, `ruff`, `mypy`

### Running Unit Tests

To execute the unit test suite for custom implementations in Homework 2:

```bash
pytest tests/
```
