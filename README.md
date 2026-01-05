# RandonForestCustom
Implementation of a Random Forest classifier from scratch to understand ensemble methods and decision tree bagging logic.

This project is the result of a learning process: entirely recoding a Random Forest Classifier model from scratch.

The main objective was to deeply understand the mechanisms of ensemble learning, specifically Bagging (Bootstrap Aggregating) and Feature Selection, using only NumPy for the core logic. The code is then validated through a benchmark against the professional standard: the RandomForestClassifier from Scikit-learn.

To validate the implementation, I used the Breast Cancer Dataset from Scikit-learn. Featuring 30 real-valued dimensions that describe breast mass characteristics.

We compare the performances (Accuracy) after training both models on a classification dataset.

| Model | Precision | execution time |
| :--- | :--- | :--- | 
| **scikit-learn** | 0.9649 | 0.369s | 
| **RandomForestCustom** | 0.9649 | 0.434s |

git clone https://github.com/evrardlecureur/RandomForestCustom.git
cd RandomForestCustom
python test.py
