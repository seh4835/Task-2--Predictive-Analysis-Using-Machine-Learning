# Task-2: Predictive Analysis Using Machine Learning

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SEHER SANGHANI

*INTERN ID*: CT08DL515

*DOMAIN*: DATA ANALYSIS

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

*PROJECT DESCRIPTION*

In this project, I worked on building a machine learning model that handles both classification and regression tasks using a dataset called "Data_set 2.csv". The main objective was to predict two different types of outputs: one categorical (the type of investment avenue) and one numerical (the mutual fund score). I wanted to explore how different ML techniques perform on the same dataset but with different prediction goals.

I started by importing all the essential libraries like pandas for data handling, matplotlib and seaborn for visualizations, and several modules from scikit-learn for modeling. Once I loaded the dataset, I cleaned it up by dropping any rows with missing values using df.dropna() to avoid any issues during model training.

Since machine learning models can’t directly work with categorical data, I used LabelEncoder to convert all categorical columns into numerical form. To make this process reusable and reversible if needed, I stored each label encoder in a dictionary. This way, I can refer back to the original labels when needed—especially helpful for interpreting the classification output later.

The first ML task I tackled was classification. I wanted to predict the type of investment avenue a user might choose based on their features. I separated the features (independent variables) and the target (the "Avenue" column) and then split the data into training and testing sets (80-20 split).

For this, I used a Random Forest Classifier because it’s generally reliable, handles both categorical and numerical data well, and gives good results out of the box. After training the model, I tested it on the test set and calculated the accuracy score. I also printed a detailed classification report, including precision, recall, and F1-score. To make it more readable, I mapped the numeric predictions back to their original class labels using the saved label encoder.

I also plotted the top 10 most important features that the model considered when making predictions. This helped me understand which variables had the most influence in determining the investment avenue, which could be very useful from a business or financial planning perspective.

Next, I moved on to a regression task where the goal was to predict the Mutual Fund score, a continuous numerical value. I used the same preprocessing approach but dropped "Mutual_Funds" as the target variable this time.

For the model, I chose Linear Regression since it's a good starting point for regression tasks and helps in understanding linear relationships in the data. After training the model, I predicted the mutual fund scores for the test set and evaluated its performance using Mean Squared Error (MSE) and R² score. These metrics gave me a sense of how close my predictions were to the actual values and how well the model captured the variability in the data.

To visualize the results, I created a scatter plot comparing the actual vs. predicted mutual fund scores, along with an ideal prediction line (y = x). This really helped me see how accurate the predictions were at a glance.

*Final Thoughts*

This project gave me a hands-on understanding of how to approach both classification and regression problems using real-world data. I got to practice essential steps like cleaning data, encoding categorical variables, training models, evaluating performance, and creating visualizations to support my findings.

I also appreciated how different modeling techniques (Random Forest vs. Linear Regression) could be applied to the same dataset to answer different kinds of questions. This really reinforced how flexible machine learning can be when applied thoughtfully.

Overall, it was a solid learning experience that improved my confidence in handling supervised learning tasks and interpreting model results effectively.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/0c48f413-31a8-4143-8349-5b1f6019c94c)
