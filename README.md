# ShopperIntentLab
## Online Shoppers Purchasing Intention — Machine Learning Classification

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to predict whether an online shopping session results in a purchase.

The target variable is `Revenue`:

- `True`: The shopping session resulted in a purchase.
- `False`: The shopping session did not result in a purchase.

The project investigates how different classification algorithms behave on the same dataset, particularly when the target classes are imbalanced.

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

| Property | Description |
|---|---|
| Original observations | 12,330 |
| Original columns | 18 |
| Input features | 17 |
| Target variable | Revenue |
| Cleaned observations | 12,205 |
| Missing values | None |
| Problem type | Binary classification |

### Duplicate analysis

During data inspection, exact duplicate rows were identified.

There were 125 extra duplicate occurrences. These were removed before modelling, resulting in 12,205 observations.

No feature combinations with conflicting target labels were identified.

An important consideration is that identical shopping-session records can represent genuinely repeated behaviour. Therefore, duplicate removal was treated as a data-cleaning decision, not proof that every repeated session was erroneous.

## 3. Exploratory Data Analysis: What We Discovered

### 3.1 Class imbalance

The target distribution was:

| Target class | Observations | Percentage |
|---|---:|---:|
| No purchase (`False`) | 10,297 | 84.37% |
| Purchase (`True`) | 1,908 | 15.63% |
| Total | 12,205 | 100% |

The dataset is imbalanced because non-purchase sessions are much more frequent than purchase sessions.

This matters because a model can achieve relatively high accuracy simply by predicting the majority class.

The majority-class baseline on the test set is approximately 84.35% accuracy.

Consequently, accuracy alone is insufficient to evaluate the models. Recall, F1-score, MCC and ROC-AUC are also important.

### 3.2 Zero-heavy numerical features and mean versus median

One important discovery was that several numerical features contain a large proportion of zero values.

Examples include:

| Feature | Percentage of zero values |
|---|---:|
| SpecialDay | 89.77% |
| Informational_Duration | 80.29% |
| Informational | 78.44% |
| PageValues | 77.63% |

These zero values are meaningful observations, not automatically missing values.

For these features, the median is zero because more than half of the observations are zero.

However, the mean can still be positive because a smaller number of observations contain positive values. Some of those positive values can be considerably larger than the rest.

This creates a difference between the mean and median and indicates that the distribution is not symmetric.

Several numerical features also showed substantial positive skewness, with the maximum observed skewness being approximately 7.54.

**Our interpretation:** The data contains many sessions with no activity for particular features, alongside a smaller number of sessions with substantial activity. This long-tailed behaviour needs to be considered during preprocessing.

### 3.3 Why we did not blindly use IQR for outlier removal

The Interquartile Range method identifies potential outliers using:

IQR = Q3 - Q1

Lower fence = Q1 - 1.5 × IQR

Upper fence = Q3 + 1.5 × IQR

However, a problem appeared with zero-heavy features.

When both Q1 and Q3 are zero:

- IQR becomes zero.
- The upper fence becomes zero.
- Every positive observation becomes an IQR outlier.

This does not mean every positive observation is an error.

For example, a positive PageValues value or a positive informational duration can represent genuine user behaviour.

Removing or capping all such values would destroy useful information.

**Decision:** We did not use IQR as an automatic rule to remove or cap these observations. We retained meaningful zero and non-zero values and used a transformation approach for numerical features.

## 4. Data Preprocessing

A common preprocessing strategy was used for all five machine learning algorithms.

### 4.1 Numerical feature transformation: Yeo-Johnson

The numerical features showed skewed distributions, long tails and, in several cases, a high concentration of zeros.

We selected the Yeo-Johnson power transformation.

Reasons for this decision:

1. It can handle zero values.
2. Unlike the Box-Cox transformation, it can also handle negative values.
3. It can reduce skewness and make numerical distributions more symmetric.
4. It is useful when numerical variables have substantially different distribution shapes.
5. Standardisation was enabled as part of the transformation.

The implementation used:

`PowerTransformer(method="yeo-johnson", standardize=True)`

The objective was not to guarantee perfectly normal distributions. It was to make the numerical features more suitable for algorithms that are sensitive to feature scale and distribution.

### 4.2 Categorical feature encoding: One-Hot Encoding

Seven features were treated as categorical variables.

These included:

- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

Some categorical features contain integer values, such as browser or operating-system identifiers.

Although these values look numerical, they represent categories rather than meaningful quantities.

For example, Browser category 4 is not necessarily greater than Browser category 2.

Treating these identifiers as ordinary numerical values could introduce artificial ordering and distance relationships.

We therefore selected One-Hot Encoding.

Reasons:

1. It represents each category using a separate binary indicator.
2. It avoids imposing an artificial numerical order on category identifiers.
3. It allows machine learning algorithms to learn category-specific effects.
4. `handle_unknown="ignore"` allows the preprocessor to handle previously unseen categories during prediction.

The seven categorical features expanded into 64 encoded columns.

Together with the 10 numerical features, the preprocessed dataset contained 74 features.

### 4.3 Avoiding data leakage

Preprocessing was included inside the machine learning pipelines.

This means that during cross-validation, the numerical transformations and categorical encoders are fitted using only the training portion of each fold.

The validation fold does not influence the fitting of its own preprocessing transformations.

This helps prevent information leakage during model evaluation.

## 5. Train-Test Split and Cross-Validation

### 5.1 Train-test split

The cleaned dataset was divided using an 80:20 stratified split.

| Dataset | Observations |
|---|---:|
| Training set | 9,764 |
| Testing set | 2,441 |
| Total | 12,205 |

The random state was set to 42.

Stratification was used to preserve approximately the same target-class proportions in both sets.

### 5.2 Why we used Cross-Validation

A single train-validation split can give a result that depends heavily on which observations happen to be selected.

To obtain a more reliable estimate during model tuning, we used five-fold Stratified Cross-Validation.

The training data was divided into five folds.

For each iteration:

- Four folds were used for fitting.
- One fold was used for validation.
- The process was repeated until every fold had been used for validation.

Stratification helped preserve the minority and majority class proportions across folds.

The mean validation ROC-AUC across the five folds was used to compare hyperparameter configurations.

### 5.3 Why we used GridSearchCV and RandomizedSearchCV

The performance of a machine learning algorithm depends not only on the algorithm itself but also on its hyperparameters.

For example, Logistic Regression depends on regularisation strength, while KNN depends on the number of neighbours and distance weighting.

We used hyperparameter-search methods rather than selecting arbitrary values.

**GridSearchCV**

GridSearchCV systematically evaluates the combinations of hyperparameters specified in a search grid.

It was used for:

- Logistic Regression
- KNN
- Gaussian Naive Bayes

**RandomizedSearchCV**

For models with broader or more flexible search spaces, evaluating every possible combination can be computationally expensive.

RandomizedSearchCV samples a specified number of parameter combinations rather than exhaustively evaluating every combination.

It was used for:

- Decision Tree
- Support Vector Machine

Both approaches used five-fold Stratified Cross-Validation.

ROC-AUC was selected as the optimisation metric because the dataset is imbalanced and we wanted to assess how well models distinguish purchasing sessions from non-purchasing sessions across classification thresholds.

### 5.4 Why the test set was kept untouched

The held-out test set was not used to choose hyperparameters.

It was reserved for the final evaluation after model selection.

This provides a more independent estimate of how the selected models perform on unseen observations.

## 6. Machine Learning Models

Five classification algorithms were implemented and tuned:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model used the same train-test split and the same preprocessing strategy.

## 7. Evaluation Metrics

The models were evaluated using six metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| ROC-AUC | Ability to distinguish the two target classes across thresholds |
| Precision | Proportion of predicted purchases that were actual purchases |
| Recall | Proportion of actual purchases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Matthews Correlation Coefficient (MCC) | Balanced measure of classification quality using all four confusion-matrix outcomes |

Using multiple metrics is particularly important because of the class imbalance in this dataset.

## 8. Model Comparison

All values below were calculated on the same held-out test set.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8669 | 0.9226 | 0.5494 | 0.8298 | 0.6611 | 0.6017 |
| Decision Tree | 0.8263 | 0.9234 | 0.4704 | 0.8743 | 0.6117 | 0.5534 |
| KNN | 0.8939 | 0.9157 | 0.7016 | 0.5602 | 0.6230 | 0.5670 |
| Gaussian Naive Bayes | 0.8247 | 0.8737 | 0.4645 | 0.7880 | 0.5845 | 0.5097 |
| SVM | 0.8755 | 0.9235 | 0.5728 | 0.8037 | 0.6688 | 0.6077 |

## 9. Model Observations and Trade-offs

The test results demonstrate that different models prioritise different aspects of classification.

- Logistic Regression achieved good recall and a strong ROC-AUC.
- Decision Tree achieved the highest recall, identifying the largest proportion of actual purchasing sessions among the five models.
- KNN achieved the highest accuracy and precision, but its lower recall indicates that it missed a comparatively larger proportion of actual purchasing sessions.
- Gaussian Naive Bayes had the lowest overall performance among the five tested models.
- SVM achieved the highest F1-score and MCC, together with the highest ROC-AUC.

The SVM and Decision Tree ROC-AUC values were extremely close: 0.9235 and 0.9234 respectively. This difference should not be interpreted as a meaningful performance gap by itself.

The results show why model selection should depend on the objective rather than accuracy alone.

## 10. Overall Model Selection

Based on the held-out test results, SVM is the strongest overall choice when considering F1-score, MCC and ROC-AUC together.

However, the best model depends on the application requirement:

- SVM offers the strongest overall balance according to F1 and MCC.
- Decision Tree is preferable when identifying as many purchasing sessions as possible is the primary concern.
- KNN is preferable when higher precision and accuracy are prioritised.

Therefore, no single metric should be considered in isolation.

## 11. Streamlit Application

The project includes a Streamlit application for interacting with the saved models.

Application features:

- Upload a CSV test dataset.
- Select an individual trained model.
- Generate predictions using the selected model.
- Display Accuracy, AUC, Precision, Recall, F1 and MCC.
- Display a confusion matrix and classification report.
- Compare the results of all five trained models.
- Download prediction and model-comparison reports.

The application loads previously trained model artifacts. It does not retrain the models when a test CSV is uploaded.

## 12. Repository Structure

```text
ShopperIntentLab/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    └── svm.joblib
```

## 13. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- Jupyter Notebook

## 14. Project Links

**GitHub Repository:**  
https://github.com/kulshresth007/ShopperIntentLab

**Streamlit Community Cloud:**  
To be added after deployment.

## 15. Conclusion

This project involved more than training five classification algorithms. It required understanding the dataset, investigating class imbalance, examining zero-heavy and skewed numerical features, selecting appropriate preprocessing methods, avoiding data leakage, tuning hyperparameters and evaluating models using multiple metrics.

The comparison demonstrates that different algorithms make different precision-recall trade-offs, and that a reliable model-selection decision requires a broader evaluation than accuracy alone.
