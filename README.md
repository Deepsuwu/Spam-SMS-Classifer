# Spam-SMS-Classifer

SPAM SMS Classifier
A machine learning-based classifier designed to detect whether an SMS is spam or not. This project applies text preprocessing techniques and machine learning algorithms to classify messages into "spam" or "ham" (not spam).

Key Features:
Text Preprocessing:

Removal of non-alphabetical characters using regular expressions.
Tokenization and removal of stopwords using NLTK.
Stemming words using the PorterStemmer to standardize word forms.
Modeling:

Trained on SMS datasets with labels 'spam' and 'ham.'
Utilizes the Multinomial Naive Bayes algorithm for classification.
The model was trained and evaluated using common metrics like accuracy.
Prediction:

Function predict_spam() that takes a new SMS and predicts whether it is spam or not.
Sample test messages show correct predictions based on the trained model.
Tools and Libraries:
Python
NLTK for natural language processing tasks
Scikit-learn for machine learning modeling
Pandas and NumPy for data manipulation
Matplotlib and Seaborn for data visualization
Data:
The dataset used in this classifier is a collection of SMS messages labeled as either 'spam' or 'ham' (not spam). The dataset was preprocessed to remove duplicates and reset the index.
How to Run:
Clone the repository.
Install the required libraries using:
Copy code
pip install -r requirements.txt
Load the dataset, preprocess it, train the model, and make predictions using the provided predict_spam function.
Sample Output:
Input: "Hi! You are pre-qualified for Premium SBI Credit Card. Click for more."

Prediction: "Wait a minute, this is a SPAM!"

Input: "Your stock broker reported your fund balance Rs.1500.5."

Prediction: "Ohhh, this is a normal message."

Future Enhancements:
Deploying the model through a web interface.
Expanding the model to include other languages and larger datasets.
