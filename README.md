# K-Means Clustering & Cross-Validation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter)](https://jupyter.org/)

A machine learning project demonstrating the integration of **unsupervised learning** (K-Means Clustering) and **supervised learning** (Logistic Regression) evaluated through **k-fold cross-validation** on the Fisher's Iris dataset.

---

## Table of Contents

- [Overview](#overview)
- [Workflow Pipeline](#workflow-pipeline)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Step-by-Step Methodology](#step-by-step-methodology)
- [Results & Performance](#results--performance)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

In traditional machine learning workflows, clustering is typically used strictly for unsupervised pattern discovery. In this project, unsupervised clustering is leveraged both to discover natural groupings in the data and to act as a **feature engineering** technique:

1. **Clustering Analysis**: Identifying natural groupings in flower morphology using Sepal Length and Petal Length.
2. **Elbow Method**: Evaluating inertia across cluster counts ($K = 1$ to $6$) to empirically verify the optimal cluster number ($K = 3$).
3. **Cluster Feature Augmentation**: Encoding cluster assignments as an additional feature in the dataset.
4. **Supervised Classification & Cross-Validation**: Training a Logistic Regression model with 5-fold cross-validation to assess classification accuracy across species.

---

## Workflow Pipeline

```
+-------------------------------------------------------------+
|                     Iris Dataset                            |
| (150 samples: Sepal Length, Sepal Width,                    |
|               Petal Length, Petal Width, Target Species)   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   Feature Selection                         |
|         X = [Sepal Length, Petal Length]                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              Elbow Method & K-Means Optimization            |
|       Compute Inertia for K in [1, 6] -> Optimal K = 3      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Cluster Assignment                       |
|           df['Cluster'] = kmeans.fit_predict(X)             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              Supervised Model & Cross-Validation            |
|   Features: Original 4 Features + Cluster Assignment        |
|   Model: Logistic Regression (max_iter=1000)                |
|   Validation: 5-Fold Stratified Cross-Validation            |
+-------------------------------------------------------------+
```

---

## Key Features

- **Exploratory Data Analysis**: Loaded Fisher's 150-sample Iris dataset using `scikit-learn` and structured it with `pandas`.
- **Elbow Heuristic**: Evaluated within-cluster sum of squares (inertia) over $K \in \{1, 2, 3, 4, 5, 6\}$ to identify the inflection point.
- **K-Means Model**: Configured with `n_clusters=3`, `random_state=42`, and `n_init=10` for reproducible centroid initialization.
- **Feature Augmentation**: Integrated unsupervised cluster assignments back into the classification feature matrix.
- **Cross-Validation**: Performed 5-fold cross-validation (`cross_val_score`) to avoid data leakage and prevent overfitting.

---

## Project Structure

```text
K-Means-Clustering-Cross-Validation/
├── .gitignore          # Git ignore specifications
├── LICENSE             # Apache 2.0 open-source license
├── README.md           # Comprehensive project documentation
└── program.ipynb       # Jupyter notebook with end-to-end implementation
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- `pip` or `conda` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/K-Means-Clustering-Cross-Validation.git
cd K-Means-Clustering-Cross-Validation
```

### 2. Create a Virtual Environment

Using `venv`:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn matplotlib jupyter
```

### 4. Launch the Notebook

```bash
jupyter notebook program.ipynb
# or
jupyter lab
```

---

## Step-by-Step Methodology

### 1. Data Preparation
The dataset contains 150 instances with 4 morphological features:
- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`
- `species` (Target: 0 = *Iris setosa*, 1 = *Iris versicolor*, 2 = *Iris virginica*)

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
```

### 2. Finding Optimal $K$ with the Elbow Method
The algorithm runs K-Means on `["sepal length (cm)", "petal length (cm)"]` across cluster counts $1$ through $6$ and tracks model inertia:

```python
from sklearn.cluster import KMeans

X = df[["sepal length (cm)", "petal length (cm)"]]
inertia = []

for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)
```

The resulting elbow plot reveals a clear bend at $K = 3$, corresponding to the 3 known biological species.

### 3. Clustering & Feature Augmentation
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)
```

### 4. Supervised Training & Cross-Validation
The augmented dataset is evaluated using a Logistic Regression model with 5-fold cross-validation:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X_class = df.drop('species', axis=1)
y_class = df['species']

model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X_class, y_class, cv=5)
```

---

## Results & Performance

| Metric | Score / Value |
|---|---|
| **Optimal Clusters ($K$)** | 3 |
| **Cross-Validation Strategy** | 5-Fold Stratified CV |
| **Fold 1 Accuracy** | 93.33% |
| **Fold 2 Accuracy** | 96.67% |
| **Fold 3 Accuracy** | 83.33% |
| **Fold 4 Accuracy** | 96.67% |
| **Fold 5 Accuracy** | 96.67% |
| **Mean Cross-Validation Accuracy** | **93.33%** |

The high average accuracy across all 5 folds demonstrates that combining unsupervised cluster features with linear classification provides strong generalization across unseen subsets of the data.

---

## Technologies Used

- **[Python](https://www.python.org/)**: Core programming language.
- **[scikit-learn](https://scikit-learn.org/)**: Implementation of K-Means clustering, Logistic Regression, and k-fold cross-validation.
- **[pandas](https://pandas.pydata.org/)**: Tabular data manipulation and feature engineering.
- **[Matplotlib](https://matplotlib.org/)**: Visual generation of the Elbow plot.
- **[Jupyter Notebook](https://jupyter.org/)**: Interactive computing and experimentation environment.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete terms and details.
