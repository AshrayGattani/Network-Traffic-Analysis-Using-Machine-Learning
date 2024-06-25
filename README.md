# ML Classification - Network Traffic Analysis

This project is designed to analyze and classify real network traffic data to differentiate between malicious and benign traffic records. By comparing and fine-tuning several Machine Learning algorithms, it aims to achieve the highest accuracy with the lowest false positive and negative rates.

## Dataset: Aposemat IoT-23

The dataset utilized in this project is [CTU-IoT-Malware-Capture-34-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/), a part of the [Aposemat IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23). This labeled dataset includes malicious and benign IoT network traffic and was created through the Avast AIC laboratory with support from Avast Software.

## Data Classification Process

The project consists of four phases, each represented by a corresponding notebook within the [notebooks]() directory. Intermediate data files are stored in the [data]() directory, while trained models are kept in the [models]() directory.

### Phase 1: Initial Data Cleaning

> Notebook: [initial-data-cleaning.ipynb]()

This phase involves the initial exploration and cleaning of the dataset:

1. Load the raw dataset into a pandas DataFrame.
2. Review dataset summary and statistics.
3. Fix combined columns.
4. Remove irrelevant columns.
5. Correct unset values and validate data types.
6. Inspect the cleaned dataset.
7. Save the cleaned dataset to a CSV file.

### Phase 2: Data Processing

> Notebook: [data-preprocessing.ipynb]()

In this phase, the focus is on processing and transforming the data:

1. Load the dataset into a pandas DataFrame.
2. Review dataset summary and statistics.
3. Analyze the target attribute.
4. Encode the target attribute using [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
5. Handle outliers with the [Inter-quartile Range (IQR)](https://en.wikipedia.org/wiki/Interquartile_range).
6. Encode IP addresses.
7. Manage missing values:
   - Impute missing categorical features using [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
   - Impute missing numerical features using [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).
8. Scale numerical attributes with [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
9. Handle categorical features: manage rare values and apply [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
10. Verify the processed dataset and save it to a CSV file.

### Phase 3: Model Training

> Notebook: [model-training.ipynb])

This phase includes training and evaluating various classification models:

1. Naive Bayes: [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
2. Decision Tree: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
3. Logistic Regression: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
4. Random Forest: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. Support Vector Classifier: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
6. K-Nearest Neighbors: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
7. XGBoost: [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html#)

Evaluation Method:
- Cross-Validation: [Stratified K-Folds Cross-Validator](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- Number of folds: 5
- Shuffling: Enabled

The results of each model are analyzed and compared.

### Phase 4: Model Tuning

> Notebook: [model-tuning.ipynb]()

This phase focuses on fine-tuning the best-performing model:

- Model: Support Vector Classifier ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC))
- Tuning Method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

The performance of the model is analyzed both before and after tuning.

---

By following these steps, the project aims to effectively classify network traffic and detect malicious activities with high accuracy.
