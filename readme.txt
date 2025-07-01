# 💸 Predicting Financial Crises Using Machine Learning (Scratch Implementation)

> 🔍 CS613 Final Project  
> 👨‍💻 Team: S. Kamien, R. Sairajath, C. Shashi Kumar, **Ram Kishore KV(Lead Developer)**  
> 📅 Fall 2023 | 🏫 Drexel University  

---

## 📌 Project Overview

This project explores predicting four types of financial crises (currency, inflation, banking, systemic) using **machine learning models implemented entirely from scratch**, without any external ML libraries like `scikit-learn`.

---

## 📄 Project Report Summary (CS613 Project Report)

A financial crisis is an important event in a country’s financial health and has a significant potential impact. Traditionally, crisis prediction relies on manual analysis of indicators like inflation, debt, or exchange rate. This project explores how machine learning models — Logistic Regression, Decision Tree, and Naive Bayes — can automate this process.

Our models were trained on data from 70 countries covering banking, currency, inflation, and systemic crises. Each classifier was trained independently per crisis type, and results were compared based on accuracy, precision, and recall. A simple **ensemble method** was implemented to maximize recall by combining the outputs of all models.

**Key Takeaways:**
- Logistic Regression was the most accurate overall
- Ensemble consistently had the highest recall
- Inflation crises were easiest to detect; currency crises were the hardest

---

## 🧠 Technical Summary from Python Code (`CS613project_FINAL.py`)

### Models Implemented (No scikit-learn used):
- **Logistic Regression**:
  - Gradient descent on log-loss
  - Sigmoid activation
  - Manual convergence check with tolerance

- **Decision Tree**:
  - Custom tree using entropy-based splits
  - Recursively built with leaf probability handling

- **Naive Bayes**:
  - Binary probabilistic decision rule
  - Class-conditional feature evaluation

- **Ensemble**:
  - Logical OR combination of all 3 base models
  - Focused on maximizing recall (minimizing false negatives)

---

### Data Workflow:
- Read `global_crisis_data.csv` from `Downloads/` folder
- Cleaned invalid/unknown values
- Converted multi-label crisis columns into binary
- Zero-mean normalization
- Train/test split: 2/3 training, 1/3 validation
- Results are averaged across all 70 countries

---

## 📈 Evaluation Metrics

Each model and crisis type is evaluated using:

- **Accuracy**: % of correct predictions  
- **Precision**: % of true positives out of predicted positives  
- **Recall**: % of true positives out of actual positives (critical for crises)

> Final output is a printed table of average accuracy, precision, and recall for each model-crisis combination.

---

## ▶️ How to Run

### 🔧 Requirements:
- Python 3.8+
- `numpy` (no external ML libraries)

### 📦 Steps:

# Step 1: Ensure dataset is placed in
Downloads/global_crisis_data.csv

# Step 2: Run the script
python CS613project_FINAL.py

🧪 Sample Output
Currency Crisis
Logistic Regression → Accuracy: 0.92, Precision: 0.86, Recall: 0.19

Inflation Crisis
Logistic Regression → Accuracy: 0.95, Precision: 0.88, Recall: 0.57
Ensemble → Recall: 0.59
📄 References
Hennig, T., Varghese, R., & Iossifov, P. (2023). Predicting Financial Crises: The Role of Asset Prices. IMF Working Paper

Reinhart, C., Rogoff, K., Trebesch, C. – Global Crises Dataset by Country
| Name                                          | Contribution                                      |
| --------------------------------------------- | ------------------------------------------------- |
| **Ram Kishore KV* | Lead Developer (Logistic Regression, Integration) |
| S. Kamien                                     | Decision Tree, QA                                 |
| R. Sairajath                                  | Report Drafting, Cleaning                         |
| C. Shashi Kumar                               | Naive Bayes, Validation                           |

"Predicting crises isn't about perfection — it's about warning before the fall."
⭐ Star this repo if you’re impressed by pure Python ML from scratch!



