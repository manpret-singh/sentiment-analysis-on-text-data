# sentiment-analysis-on-text-data

Summary of Results:
The sentiment analysis solution was implemented using a combination of various machine learning models, including Logistic Regression, Naive Bayes, Support Vector Machine (SVM) and a Long Short-Term Memory (LSTM) neural network. The dataset was randomly split into training and testing sets (80-20), and the accuracy of the models was evaluated on the testing data.


Results:

•	Logistic Regression Accuracy: 0.81
•	Naive Bayes Accuracy: 0.71
•	SVM Accuracy: 0.82
•	Ensemble Model (Voting classifier): 0.82
•	LSTM Neural Network Accuracy: 0.83


Logic and Rationale:
The solution was designed with a thoughtful selection of models to leverage their unique strengths for text classification tasks.

Traditional Machine Learning Models:

Logistic Regression:
•	Reason: Logistic Regression is a simple yet effective model for binary classification tasks like sentiment analysis. It is well-suited for linearly separable problems and provides interpretable results.
•	Benefits: Fast training, low computational requirements, and easy to interpret.

Naive Bayes:
•	Reason: Naive Bayes is particularly useful for text classification due to its assumption of independence between features, making it suitable for high-dimensional data like TF-IDF vectors.
•	Benefits: Efficient, handles sparse data well, and performs well on text data.

Support Vector Machine (SVM):
•	Reason: SVM is a powerful model that aims to find a hyperplane that best separates different classes. It is effective for complex decision boundaries in high-dimensional spaces.
•	Benefits: Effective in high-dimensional spaces, robust against overfitting, and versatile with different kernel functions.

Ensemble Learning using a Voting Classifier:
•	Reason: Ensemble learning combines predictions from multiple models to achieve a more robust and accurate result. A Voting Classifier was employed to aggregate predictions from Logistic Regression, Naive Bayes, and SVM.
•	Benefits:
•	Diversity: Each model contributes a different perspective, capturing various aspects of sentiment in the data.
•	Improved Generalization: Ensemble models tend to generalize better, reducing the risk of overfitting.
•	Enhanced Performance: The combination of models often results in better overall performance compared to individual models.

Neural Network (LSTM):
•	Reason: The LSTM neural network was chosen to capture sequential dependencies in the text data. LSTMs are well-suited for tasks where the order of words matters, allowing the model to understand the context and dependencies between words.
•	Benefits:
•	Sequential Understanding: LSTMs excel in capturing long-term dependencies, crucial for understanding sentiment in sentences.
•	Representation Learning: The embedding layer and LSTM layers enable the model to learn meaningful representations of words and their context.


Dataset Preprocessing:
•	TF-IDF Vectorization: TF-IDF was used for traditional machine learning models to transform the text data into numerical vectors, emphasizing the importance of words based on their frequency and rarity.
•	Tokenization: Tokenization was applied to break down the text into individual words, facilitating the input representation for both machine learning models and the neural network.
•	Stemming: Stemming was applied to reduce words to their root form, consolidating variations of words and reducing feature dimensionality.
•	Data Balancing: Synthetic Minority Over-sampling Technique (SMOTE) was employed to address the class imbalance in the dataset. SMOTE generates synthetic samples for the minority class, improving the model's ability to learn from underrepresented instances.


The combination of these models, ensemble learning, and the use of a neural network ensures a comprehensive approach to sentiment analysis, considering both traditional and deep learning methodologies.
