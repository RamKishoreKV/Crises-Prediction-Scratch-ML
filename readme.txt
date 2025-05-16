Predicting Financial Crises Using Scratch ML Methods (CS613 Project)
This project aims to predict various forms of financial crises—such as currency, inflation, banking, and systemic crises—by analyzing global economic indicators using scratch implementations of core machine learning algorithms.

📊 Problem Statement
Financial crises can severely impact a country’s economic health. Traditional prediction methods rely on manual analysis of indicators like GDP, debt, and inflation. This project attempts to automate that process using machine learning models built from scratch.

Can we use machine learning to predict different types of financial crises using historical economic data?

🧠 Models Implemented
Implemented entirely from scratch:

Logistic Regression

Naive Bayes

Decision Tree

Ensemble Method (logical OR across models)

Each model was trained separately for four types of crises:

Currency Crisis

Inflation Crisis

Banking Crisis

Systemic Crisis

📂 Dataset
Source: Carmen Reinhart and collaborators

Data spans 70+ countries from 1800 to present

Crisis types are binary labels; features include macroeconomic indicators

Note: Due to noise and sparsity, preprocessing steps include null removal, categorical encoding, and feature normalization.

🛠️ How to Run
Place global_crisis_data.csv inside a folder named Downloads:

bash
Copy
Edit
/project-root/
└── Downloads/
    └── global_crisis_data.csv
Run the Python script:

nginx
Copy
Edit
python CS613project_FINAL.py
Output:

Average accuracy, precision, and recall for each model and crisis type (across all countries)

🧪 Results Snapshot
Crisis Type	Model	Accuracy	Precision	Recall
Currency Crisis	Logistic Regression	~0.92	~0.86	~0.19
Inflation Crisis	Logistic Regression	~0.95	~0.88	~0.57
Banking Crisis	Logistic Regression	~0.94	~0.77	~0.27
Systemic Crisis	Logistic Regression	~0.96	~0.84	~0.32
Ensemble		—	—	Best recall overall

Ensemble model helps catch more true positives, reducing false negatives (important for crisis prediction).

🔍 Key Insights
Logistic Regression performed best overall, especially in precision and accuracy.

Recall was relatively low across models, meaning they missed many actual crisis events.

The ensemble method improved recall at the expense of some precision.

🔮 Future Work
Implement SVMs, ANNs, or stronger ensemble techniques

Use larger, more diverse datasets

Explore real-time or online learning models

👥 Authors
Ram Kishore KV

Sam Kamien

Sairajath

Chaitanya Shashi Kumar

📄 References
Hennig, T., Varghese, R., & Iossifov, P. (2023). Predicting financial crises: The role of asset prices. IMF Working Paper.
https://www.imf.org/en/Publications/WP/Issues/2023/08/03/Predicting-Financial-Crises-The-Role-of-Asset-Prices-536491

Samitas, A. (2020). Machine learning as an early warning system.
https://www.sciencedirect.com/science/article/abs/pii/S1057521920301514
