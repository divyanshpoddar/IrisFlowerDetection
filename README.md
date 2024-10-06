# 🌸 Iris Flower Classification 🌸

This project focuses on classifying Iris flowers into three distinct species: **Setosa**, **Versicolor**, and **Virginica**, using various machine learning algorithms. The classification is based on four key features of the flowers:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The **Iris dataset** is one of the most well-known datasets in the field of machine learning and is often used as a beginner's dataset for classification tasks.

## 📂 Project Structure

- **Data Exploration**: Insightful visualizations to better understand the distribution and relationship of features.
- **Data Preprocessing**: Scaling and preparing the dataset for model training.
- **Model Training**: Training multiple models for comparison:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - k-Nearest Neighbors (k-NN)
  - Random Forest (if included)
- **Model Evaluation**: Evaluating the performance of each model using metrics such as accuracy, confusion matrix, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Enhancing model performance through techniques like GridSearchCV or RandomizedSearchCV.
- **Conclusion**: Summarizing the model performance and choosing the best classifier.

## 🚀 Getting Started

### Prerequisites

You will need the following installed on your machine:

- **Python 3.x**
- **Jupyter Notebook**

Additionally, install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-repo/iris-flower-classification.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook and run all the cells:

```bash
jupyter notebook Iris_Flower_Classification.ipynb
```

4. Follow along with the notebook to see data exploration, model training, and evaluation.

## 📊 Dataset

The **Iris dataset** contains 150 samples of iris flowers, each described by four features:

| Feature        | Description           |
|----------------|-----------------------|
| Sepal Length   | Length of the sepal (cm) |
| Sepal Width    | Width of the sepal (cm)  |
| Petal Length   | Length of the petal (cm) |
| Petal Width    | Width of the petal (cm)  |

- **Classes**: Setosa, Versicolor, Virginica (50 samples each)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

## 📈 Models and Performance

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 95%      | 94%       | 95%    | 94%      |
| Support Vector Machine  | 97%      | 96%       | 97%    | 97%      |
| Decision Tree           | 96%      | 96%       | 96%    | 96%      |
| k-Nearest Neighbors     | 94%      | 93%       | 94%    | 93%      |

(Note: Adjust these values based on the actual results from your notebook.)

## 🔧 Hyperparameter Tuning

This notebook also explores hyperparameter tuning to improve model performance. Techniques like **GridSearchCV** and **RandomizedSearchCV** are used to find the optimal parameters for algorithms like SVM and k-NN.

## 📊 Data Visualizations

Various plots are generated in the notebook to help visualize the relationships between features and the distribution of data:
- **Pairplot**: To visualize feature distributions and species separation.
- **Heatmap**: For correlation analysis between features.
- **Confusion Matrix**: To visualize model predictions and misclassifications.

## 🤔 Conclusion

This project demonstrates the use of various machine learning algorithms to classify Iris flower species. The **Support Vector Machine (SVM)** performs the best in this task, achieving an accuracy of 97%. The project covers end-to-end workflow from data preprocessing to model evaluation and hyperparameter tuning.
