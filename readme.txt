# Predicting Financial Crises Using Machine Learning

This project explores the use of machine learning models to predict various types of financial crises, an important aspect of a country's financial health with significant societal impact. Traditionally, financial crisis prediction has relied on manual analysis of financial indicators. This project leverages the same financial indicator data but attempts to apply machine learning tools for more reliable crisis determination.

## Table of Contents

- [About the Project](#about-the-project)
- [Authors and Contributions](#authors-and-contributions)
- [Features](#features)
- [Methodology Overview](#methodology-overview)
- [Data](#data)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Discussion](#discussion)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Technologies and Tools](#technologies-and-tools)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

This project, developed as part of a collaborative effort in the DSCI course, explores the use of machine learning models to predict various types of financial crises. Financial crises can have a significant and enduring detrimental impact on the economy, leading to substantial growth and welfare consequences. While not wholly unforeseeable, indicators like fast credit growth and inflation of sovereign foreign debt are known precursors to impending crises. The team classified this problem as multiple binary classifications for different crisis types (currency, inflation, banking, systemic). We trained and evaluated Logistic Regression, Decision Tree, and Naive Bayes models, combining their predictions into an ensemble model. The core machine learning algorithms for this project were implemented from scratch in Python.

## Authors and Contributions

This project was a collaborative effort by the following team members:

* **R. Kishore:** Main contributor responsible for developing the Logistic Regression model, Naive Bayes model, the ensemble method, and managing the overall integration of all project components.
* **Sam (identified as `samue` in the code):** Developed the Decision Tree model implementation.
* **S. Kamien and R. Sairajath:** Contributed significantly to the project's documentation and editing of report materials.
* **C. Shashi Kumar:** Contributed to the overall project as part of the author team.

## Features

-   **Multi-Crisis Prediction:** Predicts four distinct types of financial crises: currency, inflation, banking, and systemic crises.
-   **Scratch ML Implementations:** Machine learning models (Logistic Regression, Naive Bayes, Decision Tree) are implemented from scratch, demonstrating a deep understanding of their underlying mechanics [cite: CS613project_FINAL.py].
-   **Ensemble Modeling:** Combines predictions from individual models into an ensemble for potentially improved robustness and generalizability.
-   **Performance Metrics:** Compares model performance using accuracy, precision, and recall.
-   **Data Pre-processing:** Includes steps for handling missing values, removing redundant columns, zero-meaning data, and splitting data by country for categorization [cite: 34, 35, 36, 38, CS613project_FINAL.py].
-   **Randomized Data Split:** Uses a 2/3 training, 1/3 validation split, with data shuffled and separated by country [cite: 37, 38, CS613project_FINAL.py].

## Methodology Overview

Our approach involved treating the prediction problem as multiple binary classification tasks, one for each crisis type (currency, inflation, banking, systemic). We selected Naive Bayes, Decision Tree, and Logistic Regression as our core models, which were implemented manually without relying on external ML libraries like scikit-learn [cite: 16, CS613project_FINAL.py]. The predictions from these individual models were then combined into an ensemble using a "minority vote" strategy, where a crisis is predicted if any of the base models predict one [cite: 77, CS613project_FINAL.py]. We predicted that this ensemble method would minimize false negatives, which is particularly valuable given the infrequent nature of financial crises.

The overall process for each crisis type and country involved:
1.  **Data Loading and Cleaning:** Reading the CSV data, handling "n/a" or empty values, and converting relevant columns to numerical types [cite: 34, 35, 36, CS613project_FINAL.py].
2.  **Country-wise Split:** Iterating through each country's data to train and validate models independently [cite: 38, CS613project_FINAL.py].
3.  **Train/Validation Split:** Randomly shuffling and splitting the data into a 2/3 training set and 1/3 validation set [cite: 37, CS613project_FINAL.py].
4.  **Zero-Meaning:** Normalizing the feature data by subtracting the mean [cite: 38, CS613project_FINAL.py].
5.  **Model Training and Prediction:** Training each of the three individual models and generating predictions on the validation set [cite: CS613project_FINAL.py].
6.  **Ensemble Prediction:** Combining the predictions from the individual models to form the ensemble's prediction [cite: CS613project_FINAL.py].
7.  **Measure Calculation:** Computing accuracy, precision, and recall for each model and the ensemble [cite: CS613project_FINAL.py].
8.  **Averaging Results:** Averaging these measures across all countries to get overall performance metrics [cite: 81, CS613project_FINAL.py].

## Data

The dataset employed in this project is from Carmen Reinhart et al.. It spans a broad variety of features, incorporating dates of banking crises for 70 countries, extending from 1800 to the current period. It also includes data on exchange rate crises, stock market crises, and inflation crises. The dataset used is `global_crisis_data.csv`, which is included in this repository [cite: image_149fe8].

During pre-processing, null and empty values were removed from the dataset, and columns containing only comments or redundant information were cleaned. The data was also zero-meaned before training the models, and before training, data was split by country as a means of categorizing it.

## Models Implemented

For the prediction of various crises (currency, inflation, banking, systemic), we used three different models, each implemented from scratch in Python: Logistic Regression, Decision Tree, and Naive Bayes [cite: 41, CS613project_FINAL.py].

### Logistic Regression

My scratch implementation of Logistic Regression estimates the probability of an event occurring by using a logistic function of independent variables. It employs a gradient descent approach, minimizing mean log loss. The sigmoid function, $\sigma(z) = 1 / (1 + e^{-z})$, converts a linear combination of data into probabilities between 0 and 1 [cite: 49, 50, 52, CS613project_FINAL.py]. A probability threshold of 0.5 was set to determine class labels. The model was trained over 10,000 epochs, with iterations stopping if the change in log loss fell below a tolerance of $1 \times 10^{-6}$.

### Decision Tree

The Decision Tree model uses a series of binary decisions based on feature vectors to classify data instances. To create a relatively compact tree, the feature to be evaluated at each node is selected based on its entropy [cite: 60, CS613project_FINAL.py]. Entropy for a feature is calculated as $H(p)=-\sum_{i=1}^{k}p_{i}log(p_{i})$, where $p$ is a probability distribution for that feature.

### Naive Bayes

My scratch implementation of Naive Bayes predicts class based on Bayes' Theorem, making a key assumption of independence among predictors. It computes the posterior probability $P(c|x)$ from the prior probability of class $P(c)$, the prior probability of the predictor $P(x)$, and the probability of the predictor given the class $P(x|c)$, using the equation $P(c|x)=\frac{P(x|c)P(c)}{P(x)}$ [cite: 67, 68, 69, CS613project_FINAL.py]. The feature priors are assumed independent and calculated as $P(\overline{x}|c)=\prod_{i\rightarrow D}P(x_{i}|c)$.

### Ensemble

My design and implementation of the ensemble method created multiple models and combined their predictions to improve the robustness and generalizability of the overall model. The ensemble we built checks if any of our individual models (Logistic Regression, Decision Tree, or Naive Bayes) predict a crisis, and if so, it returns a positive (crisis) prediction [cite: 77, CS613project_FINAL.py]. This approach was chosen with the aim of minimizing false negatives, considering that financial crises are relatively infrequent events.

## Results

The tables below show the accuracy, precision, and recall for all implemented models and for each type of crisis, averaged over each country for which data was available.

### Currency Crisis
| Model             | Accuracy | Precision | Recall   |
| :---------------- | :------- | :-------- | :------- |
| Decision Tree     | 0.914484 | 0.725142  | 0.180134 |
| Naive Bayes       | 0.698642 | 0.911508  | 0.161428 |
| Logistic Regression | 0.923611 | 0.862232  | 0.189818 |
| Ensemble          | 0.714690 | 0.897817  | 0.248370 |

### Inflation Crisis
| Model             | Accuracy | Precision | Recall   |
| :---------------- | :------- | :-------- | :------- |
| Decision Tree     | 0.910317 | 0.641037  | 0.364773 |
| Naive Bayes       | 0.918849 | 0.688740  | 0.314736 |
| Logistic Regression | 0.954960 | 0.888596  | 0.570598 |
| Ensemble          | 0.913294 | 0.694304  | 0.590632 |

### Banking Crisis
| Model             | Accuracy | Precision | Recall   |
| :---------------- | :------- | :-------- | :------- |
| Decision Tree     | 0.944841 | 0.745011  | 0.324059 |
| Naive Bayes       | 0.931548 | 0.704976  | 0.333445 |
| Logistic Regression | 0.944643 | 0.768469  | 0.269906 |
| Ensemble          | 0.910119 | 0.659885  | 0.395207 |

### Systemic Crisis
| Model             | Accuracy | Precision | Recall   |
| :---------------- | :------- | :-------- | :------- |
| Decision Tree     | 0.962302 | 0.788344  | 0.380254 |
| Naive Bayes       | 0.953571 | 0.801064  | 0.383840 |
| Logistic Regression | 0.962698 | 0.845476  | 0.323946 |
| Ensemble          | 0.939484 | 0.743931  | 0.426036 |

## Discussion

Examining the above tables, accuracy is consistently rather high across all models, with logistic regression usually being the highest. Precision is always above sixty percent, and logistic regression usually shows the highest precision. However, recall is frequently quite poor, never going above sixty percent and sometimes dropping as low as sixteen percent.

Looking at individual models, logistic regression frequently performs the best. Interestingly, the ensemble performs worse in most measures compared to individual models, *except* for recall, where it is consistently the highest-performing. This confirms that our intended behavior of minimizing false negatives through the ensemble occurred as planned.

Among the various crisis types, inflation crises were the easiest to consistently positively identify, as shown by the relatively high recall in that table. Conversely, currency crises were difficult to positively identify, exhibiting the lowest recall among all crisis types.

Ultimately, our classifiers performed poorly at detecting financial crises. While accuracy is high, the low recall values indicate this is mostly due to true negatives dominating the dataset.

## Future Work

This problem is far from solved. Future approaches could include incorporating more machine learning models, exploring different ensemble methods, or utilizing alternative datasets. It might also be possible to use similar data to predict other significant events, such as wars or election outcomes.

## Conclusion

In this paper, we attempted to use machine learning models to predict and identify financial crises. Ultimately, our scratch-implemented classifiers struggled at detecting financial crises. While accuracy is high, the low recall values confirm that this is primarily due to true negatives dominating the dataset. Overall, there is much room for improvement in this problem, and our approach is only one of many.

## Technologies and Tools

-   **Python 3.x:** Primary programming language.
-   **Numpy:** Used for numerical operations and array manipulation [cite: CS613project_FINAL.py].
-   **CSV module:** For reading and processing CSV data [cite: CS613project_FINAL.py].
-   **Math module:** For mathematical functions like `log` and `ceil` [cite: CS613project_FINAL.py].
-   **Custom ML Implementations:** Logistic Regression, Decision Tree, and Naive Bayes models were implemented from scratch [cite: CS613project_FINAL.py].

## Getting Started

To get a local copy of this project up and running, follow these simple steps.

### Prerequisites

-   Python 3.x installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
    (Replace `yourusername/your-repo-name` with your actual GitHub path)

2.  **Install dependencies:**
    ```bash
    pip install numpy
    ```

### Usage

1.  **Ensure Dataset Location:**
    The project expects the `global_crisis_data.csv` file to be located in the same directory as `CS613project_FINAL.py`, or within a `Downloads/` subdirectory relative to where the script is run [cite: CS613project_FINAL.py]. Since `global_crisis_data.csv` is included in this repository, ensure it is in the correct path when running the script [cite: image_149fe8].

2.  **Run the Python Script:**
    From your terminal, navigate to the project directory and execute the main Python script:
    ```bash
    python CS613project_FINAL.py
    ```

3.  **View Results:**
    The script will print the average accuracy, precision, and recall for each model (Decision Tree, Naive Bayes, Logistic Regression, and Ensemble) across the four crisis types (Currency, Inflation, Banking, Systemic) to the console [cite: CS613project_FINAL.py]. These results are averaged over all countries in the dataset [cite: 81, CS613project_FINAL.py].

## Project Structure

This project is organized with the following file structure:

* `CS613 Project Presentation.pptx` - The project's presentation slides [cite: image_149fe8].
* `CS613 Project Proposal (2).pdf` - The project proposal document [cite: image_149fe8].
* `CS613 Project Report.pdf` - The full project report detailing methodology, results, and discussion [cite: CS613 Project Report.pdf, image_149fe8].
* `CS613project_FINAL.py` - The main Python script containing the scratch implementations of ML models and the core logic for data processing and evaluation [cite: CS613project_FINAL.py, image_149fe8].
* `global_crisis_data.csv` - The raw dataset used for predicting financial crises [cite: image_149fe8].
* `readme.txt` - A text file, possibly an older README version or notes [cite: image_149fe8].
* `README.md` - This Markdown README file, providing an overview of the project.
* `~CS613 Project Presentation.pptx` - A temporary or backup version of the presentation slides [cite: image_149fe8].

## Contributing

- Ram Kishore KV- Logisitic regression, Naive Bayes, Ensemble and Integration
- San Kamien - Decision Trees

## License

[Specify your license here, e.g., MIT License, Apache 2.0, etc.]

## Contact

**Project Authors:** S. Kamien, R. Sairajath, C. Shashi Kumar, Ram Kishore KV
Project Link: [https://github.com/yourusername/your-repo-name](https://github.com/yourusername/your-repo-name)
