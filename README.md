# AI Fundamentals

A collection of hands-on Artificial Intelligence and Machine Learning projects developed during the **AI Fundamentals** course.

The repository focuses on understanding how Artificial Intelligence algorithms work internally by implementing several models and intelligent agent architectures from scratch, evaluating their performance, and comparing them with established implementations from libraries such as **scikit-learn**.

## Highlights

- Implemented multiple Machine Learning algorithms from scratch
- Built reusable evaluation pipelines for classification and regression
- Compared custom implementations with scikit-learn baselines
- Performed data preprocessing, normalization, feature encoding, and model tuning
- Designed and tested multiple intelligent agent architectures in a custom GridWorld environment
- Evaluated models using metrics such as Accuracy, F1 Score, Precision, Recall, ROC-AUC, MSE, RMSE, and MAE
- Explored both supervised Machine Learning and classical Artificial Intelligence concepts

---

## Repository Overview

The repository is mainly divided into two areas:

1. **Machine Learning models and experiments**
2. **Intelligent Agents in GridWorld**

The goal of the project was not only to use existing Machine Learning libraries, but also to understand the mathematical and algorithmic foundations behind the models by implementing them directly.

---

# Machine Learning Project

The main Machine Learning notebook works on a dataset related to **obesity levels and lifestyle habits**.

The dataset contains:

- **2,111 samples**
- **17 features**
- Demographic information
- Eating habits
- Physical activity indicators
- Transportation habits
- Obesity-related labels

The project follows a complete experimental workflow:

```text
Data preprocessing
        ↓
Feature encoding
        ↓
Model implementation
        ↓
Training
        ↓
Hyperparameter tuning
        ↓
Evaluation
        ↓
Comparison with scikit-learn
```

---

## Data Preprocessing

Before training the models, the dataset is processed to make it suitable for different algorithms.

The preprocessing phase includes:

- Conversion of categorical variables into numerical values
- Creation of encoded representations for non-numerical attributes
- Dataset shuffling
- Train/test preparation
- Feature normalization
- Conversion to NumPy matrices when required
- Separate preprocessing strategies depending on the model

For some experiments, categorical attributes were also transformed using a dummy-variable representation.

---

# Implemented Machine Learning Models

## Linear Regression

A Linear Regression model was implemented from scratch using the closed-form solution based on the pseudo-inverse of the feature matrix.

The model estimates the coefficient vector using:

```text
w = X⁺ y
```

where `X⁺` represents the pseudo-inverse of the input matrix.

The custom implementation is then compared against:

```python
sklearn.linear_model.LinearRegression
```

This experiment focuses on:

- Regression fundamentals
- Matrix-based optimization
- Feature encoding
- Model comparison

---

## Decision Tree

A Decision Tree classifier was implemented from scratch.

The model recursively builds the tree by:

- Selecting a feature for each node
- Splitting the dataset into sub-groups
- Measuring the quality of each split using information gain
- Recursively generating child nodes
- Stopping when terminal conditions are reached

The implementation allows manual control over feature ranges and splitting strategies, making it possible to study how the tree structure affects classification performance.

The custom implementation is compared with:

```python
sklearn.tree.DecisionTreeClassifier
```

---

## Stochastic Gradient Descent

A regression model based on **Stochastic Gradient Descent** was implemented to optimize model parameters iteratively.

The implementation includes several techniques designed to improve training stability and efficiency:

- Mini-batch training
- Learning-rate reduction
- Early stopping
- Tracking of the best-performing weights

At each training step, the gradient of the error function is computed and the weights are updated in the opposite direction of the gradient.

The model is compared against:

```python
sklearn.linear_model.SGDRegressor
```

This experiment focuses on:

- Gradient-based optimization
- Iterative training
- Convergence behavior
- Early stopping
- Hyperparameter tuning

---

## K-Nearest Neighbors

A K-Nearest Neighbors classifier was implemented from scratch.

The model:

1. Computes the N-dimensional Euclidean distance between the input sample and all training samples
2. Sorts the samples by distance
3. Selects the `k` closest neighbors
4. Predicts the class using majority voting

Before inference, the features are normalized to prevent attributes with larger numerical ranges from dominating the distance calculation.

Different values of `k` were experimentally tested.

Example results:

| k | Accuracy |
|---:|---:|
| 1800 | 55.92% |
| 4 | 92.89% |
| 3 | 93.36% |
| 2 | **94.79%** |

The custom implementation achieved results comparable to:

```python
sklearn.neighbors.KNeighborsClassifier
```

---

## Logistic Regression

A binary Logistic Regression classifier was implemented using the sigmoid function and a Gradient Descent-based optimization process.

The sigmoid function transforms the linear model output into a probability:

```text
σ(z) = 1 / (1 + e^-z)
```

Several training configurations were tested by changing:

- Learning rate
- Number of epochs
- Batch size

Example experiments:

| Learning Rate | Epochs | Batch Size | Accuracy |
|---:|---:|---:|---:|
| 0.01 | 20,000 | 100 | 90.52% |
| 0.001 | 20,000 | 10 | 94.79% |
| 0.01 | 20,000 | 10 | **99.53%** |

The custom model is compared against:

```python
sklearn.linear_model.LogisticRegression
```

---

## Neural Network

A configurable feed-forward Neural Network was also implemented.

The model includes:

- Multiple hidden layers
- ReLU activation functions
- Forward propagation
- Backpropagation
- Gradient-based weight optimization

The number of layers and neurons can be configured before training.

Different architectures and training configurations were tested to study their impact on model performance.

The results are compared with:

```python
sklearn.neural_network.MLPRegressor
```

This experiment also provided an opportunity to analyze cases where an optimized library implementation significantly outperforms a custom implementation and to investigate the reasons behind the performance difference.

---

# Model Evaluation

Reusable evaluation functions were developed for both classification and regression tasks.

## Classification Metrics

The classification evaluation pipeline includes:

- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve
- ROC-AUC

These metrics make it possible to evaluate model behavior beyond simple accuracy.

---

## Regression Metrics

Regression models are evaluated using:

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error

The final section of the Machine Learning project compares the performance of the implemented models using visualizations.

---

# Intelligent Agents

The repository also contains exercises focused on **Intelligent Agent architectures**.

A custom **GridWorld** environment was implemented using Python and Pygame.

The environment includes:

- A starting position
- A target position
- Obstacles
- Movement constraints
- Observable and non-observable information
- A performance measure based on the number of steps required to reach the goal

The main objective is to analyze how different types of intelligent agents behave under different levels of observability and decision-making complexity.

---

## Simple Reflex Agent

The Simple Reflex Agent selects an action based only on the current observable state.

It does not maintain an internal representation of the environment.

This exercise explores the capabilities and limitations of purely reactive behavior.

---

## Model-Based Reflex Agent

The Model-Based Reflex Agent maintains an internal representation of the environment.

This allows the agent to reason about information that may not be directly observable in the current state.

The exercise focuses on:

- Partial observability
- Internal state
- Environment modelling
- Rational action selection

---

## Utility-Based Agent

The Utility-Based Agent evaluates possible actions according to a utility function.

Instead of only trying to reach the goal, the agent considers which actions provide the best outcome according to the defined performance measure.

The experiment explores:

- Utility functions
- Rational behavior
- Action evaluation
- Performance optimization

---

## Goal-Based Agent

The Goal-Based Agent chooses actions according to a target state.

The exercise investigates how goal-oriented reasoning differs from purely reactive or utility-based approaches.

This architecture is particularly useful when an agent must satisfy intermediate goals before reaching its final destination.

---

# Technologies

The repository mainly uses:

- **Python**
- **NumPy**
- **pandas**
- **scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Pygame**
- **Jupyter Notebook**
- **Google Colab**

---

# Key Concepts Explored

The project covers several core concepts in Artificial Intelligence and Machine Learning:

- Supervised Learning
- Regression
- Classification
- Gradient Descent
- Stochastic Gradient Descent
- Decision Trees
- K-Nearest Neighbors
- Logistic Regression
- Neural Networks
- Data preprocessing
- Feature encoding
- Feature normalization
- Hyperparameter tuning
- Model evaluation
- Intelligent Agents
- Rational Agents
- Partial Observability
- Utility-Based Decision Making
- Goal-Based Reasoning

---

# Repository Structure

```text
AIFundamentals/
│
├── Homework_Fondamenti_di_IA_AA_2023_24.ipynb
│
├── Copia_di_Homework_Fondamenti_di_IA_AA_2023_24.ipynb
│
├── EsercitazioniFondamentiAI/
│   └── Esercitazione1.ipynb
│
└── README.md
```

The Machine Learning notebooks contain the main supervised learning experiments and custom model implementations.

The `EsercitazioniFondamentiAI` directory contains exercises focused on classical Artificial Intelligence and intelligent agents.

---

# What This Project Demonstrates

This repository reflects my interest in understanding Artificial Intelligence beyond the use of high-level libraries.

The project gave me practical experience in:

- Translating mathematical concepts into working code
- Implementing Machine Learning algorithms from first principles
- Designing training procedures
- Performing hyperparameter tuning
- Evaluating models using multiple metrics
- Comparing custom implementations with established ML libraries
- Preprocessing and transforming real-world datasets
- Debugging Machine Learning models
- Understanding optimization techniques
- Designing intelligent agent architectures
- Experimentally analyzing model behavior

The project strengthened both my theoretical understanding of Artificial Intelligence and my ability to implement, test, and evaluate AI systems in practice.

---

# Author

**Federico Santavicca**

Master's student in Engineering in Computer Science at Sapienza University of Rome.

Main areas of interest:

- Artificial Intelligence
- Machine Learning
- Generative AI
- Agentic AI
- Computer Vision
- Software Engineering
