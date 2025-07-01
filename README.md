# Active Learning for Classification

---
## 📂 Repository Structure 

│──📁 data/

│   │── binary_data.xlsx                 # Generated binary dataset

│   │── multiclass_data.xlsx             # Generated multiclass dataset

│──📁 src/

│   │── 📁 utils/

││       │── active_loop.py               # Main Active Learning loop

││       │── initialization.py            # Initialization methods

││       │── models.py                    # Classification models

││       │── preprocessing.py             # Data preprocessing

││       │── prueba.py                    # Code to see how the functions can be used

││       │── selection.py                 # Data selection strategies that are integrated into the active learning loop for initial selection within the loop in main.py

││       │── stopping_criteria.py         # Stopping criteria strategies

││       │── TSVM.py                      # TSVM model modified

│   │── main.py                      # Active Learning execution

│── baselines.py                     # Random model comparisons

│ requirements.txt                   # Required dependencies

README.md                            # This file

---
## TSVM Model

The file [`TSVM.py`] was taken and modified from the repository [LAMDA-SSL](https://github.com/YGZWQZD/LAMDA-SSL), developed by [@YGZWQZD](https://github.com/YGZWQZD).

Original repository: [https://github.com/YGZWQZD/LAMDA-SSL](https://github.com/YGZWQZD/LAMDA-SSL)  

> Some parts of the code have been modified to fit the requirements of this project.

---
## 📊 Implemented Methods  

### Initial Selection Methods 🎯
- **Minimum Distance**: Selects samples by maximizing the minimum distance between labeled and unlabeled samples.  
- **Mean Distance**: Selects samples by maximizing the mean distance between labeled and unlabeled samples.

### Active Learning Strategies 🔍
- **Coreset**: Selects the most representative samples by entropy + euclidean distance.  
- **Entropy**: Selects samples with the highest uncertainty, calculated by entropy.  
- **Uncertainty Distance**: Combines least uncertainty and euclidean distance.
- **Variance**: Selects samples with highest variance (intended for GP Classifiers).
- **Least Confidence**: Selects samples the model is the least confidence in its prediction.
- **Margin Sampling**: Selects samples with lowest difference between probabilities of being in the two most probable classes.
- **Support Vector**: Selects samples that are used as a support vector in the creation of the decision boundary and are NOT LABELED, if no unlabeled samples as SVs, then coreset method (Intended ONLY for TSVM and Multiclass TSVM). 

### Implemented Models 🧠  
- **Transductive SVM (TSVM) and Multiclass TSVM**, for binary and multiclass classification 
- **Standard SVM**
- **Gaussian Process Classifier**
- **Logistic Regression Classifier**
- **Random Forest**

### Stopping Criteria 🛑
- **Maximum Uncertainty**: Given a threshold, training of model stops when the most uncertain sample is below that threshold.
- **Overall Uncertainty**: Given a threshold, training of model stops when theaverage of unceratinty os samples is below that threshold.
- **Maximum Expected Error**: Given a threshold, training of model stops when the prediction error of the model is lower than that threshold.
