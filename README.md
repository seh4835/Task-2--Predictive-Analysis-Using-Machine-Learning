# Task-2: Predictive Analysis Using Machine Learning

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SEHER SANGHANI

*INTERN ID*: CT08DL515

*DOMAIN*: DATA ANALYSIS

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

*PROJECT DESCRIPTION*

In this project, I built a machine learning model to predict investment avenues based on various features in a dataset. I used a Random Forest Classifier, which is a popular algorithm for classification problems, and followed a step-by-step process that included data cleaning, encoding, model training, evaluation, and visualization.

To start, I imported all the necessary Python libraries like pandas for data handling, matplotlib and seaborn for visualization, and some tools from scikit-learn for preprocessing and modeling.

After loading the dataset from a CSV file, I cleaned it by removing any rows with missing values. This helped avoid complications during model training and made the dataset more reliable to work with.

Since the dataset included categorical columns (like text labels), I used LabelEncoder to convert them into numerical format. Machine learning models typically can’t work with strings directly, so this step was important. I also stored the encoders so I could later convert predictions back to their original form for interpretation.

Once the data was clean and properly formatted, I separated it into two parts: the features (which help with prediction) and the target (what I want to predict — in this case, the “Avenue” or type of investment). I then split the data into training and testing sets, using 80% for training the model and 20% for testing its performance.

Next, I trained a RandomForestClassifier. This algorithm works by creating many decision trees and combining their results for better accuracy and less overfitting. Once the model was trained, I used it to make predictions on the test data.

To evaluate how well the model performed, I calculated its accuracy and printed a classification report. This report gave me useful metrics like precision, recall, and F1-score, which help understand how well the model is doing across different categories. I also used the label encoder to translate the numeric predictions back to the original labels, which made the report easier to read.

Finally, I visualized the top 10 most important features that influenced the model’s predictions. Random Forests have a built-in way of measuring feature importance, which I extracted and plotted using Seaborn. The resulting chart clearly highlighted which features had the biggest impact on predicting the investment avenue.

Overall, this was a complete machine learning workflow — from data preparation to model training and result interpretation. It not only gave me a working model but also provided insights into which factors are most influential when it comes to predicting where someone might choose to invest.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/0c48f413-31a8-4143-8349-5b1f6019c94c)
