# AI Fundamentals

A collection of hands-on Artificial Intelligence and Machine Learning projects developed during the **AI Fundamentals** course.

The repository focuses on understanding how AI algorithms work internally by implementing several models and agent architectures from scratch, evaluating their performance, and comparing them with established implementations from libraries such as **scikit-learn**.

## Highlights

- Implemented multiple Machine Learning algorithms from scratch
- Built reusable evaluation pipelines for classification and regression
- Compared custom implementations with scikit-learn baselines
- Performed data preprocessing, normalization, feature encoding, and model tuning
- Designed intelligent agents with different reasoning strategies in a GridWorld environment
- Explored model performance using metrics such as Accuracy, F1 Score, Precision, Recall, ROC curves, RMSE, MSE, and MAE

---

## Machine Learning Project

The main notebook explores supervised learning techniques using a dataset containing information about eating habits, physical activity, and obesity levels.

The dataset contains **2,111 samples and 17 features**, including demographic information, lifestyle habits, physical activity, and transportation choices.

The project covers the full experimental workflow:

**Data preprocessing → Model implementation → Training → Hyperparameter tuning → Evaluation → Comparison**

### Implemented Models

#### Linear Regression

Implemented Linear Regression using the closed-form solution based on the pseudo-inverse:

```text
w = X⁺ y
