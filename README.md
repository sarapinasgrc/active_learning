# Active Learning for Classification

## 📂 Repository Structure 
📁 data/

│── arcene_df.xlsx           # Original MALDI-TOF (Arcene) dataset

│── binary_data.xlsx         # Generated binary dataset

│── multiclass_data.xlsx     # Generated multiclass dataset

📁 src/

│── 📁 utils/

│   │── active_loop.py       # Main Active Learning loop

│   │── initialization.py    # Initialization methods

│   │── models.py            # Classification models

│   │── preprocessing.py     # Data preprocessing

│   │── selection.py         # Data selection strategies

│   │── main.py              # Active Learning execution

│── baselines.py             # Random model comparisons

│ requirements.txt           # Required dependencies

README.md                    # This file

## 📊 Implemented Methods  

### Initial Selection Methods  
- **Minimum Distance**: Selects samples by maximizing the minimum distance between labeled and unlabeled samples.  
- **Mean Distance**: Selects samples by maximizing the mean distance between labeled and unlabeled samples.
- 
### Active Learning Strategies  
- **Coreset**: Selects the most representative samples by entropy + euclidean distance.  
- **Entropy**: Selects samples with the highest uncertainty, calculated by entropy.  
- **Uncertainty Distance**: Combines uncertainty and euclidean distance.  

### Implemented Models  
- **Transductive SVM (TSVM)**, for binary and multiclass classification 
- **Standard SVM**
- **Gaussian Process Classifier**
- **Logistic Regression Classifier**
- **Random Forest** 
