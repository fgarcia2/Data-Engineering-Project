import sys
from sqlalchemy import create_engine
import pandas as pd 
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Function to extract data from SQL database. 
    Input: Filepath for SQL database
    Output: X,Y dataframes used in modeling 
    '''
    
    # Extract data and structure into X and Y dataframes. 
    engine = create_engine('sqlite:///data/emergency.db')
    df = pd.read_sql_table(database_filepath, engine) 
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    
    # Create a series with categories. 
    categories = Y.columns
    
    return X,Y, categories

def tokenize(text):
    '''
    Function to clean text from messages. 
    Input: Individual text to be tokenized
    Output: Clean tokens 
    '''
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    Function to clean text from messages. 
    Input: Individual text to be tokenized
    Output: Clean tokens 
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate and report model results
    Input: Model, test set ie X_test & y_test
    Output: Prints the Classification report
    '''
    
    # Perform predictions using model. 
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))

        
def save_model(model, model_filepath):
    '''
    Function to save model into Pickle file 
    Input: Model, path to save pickle file. 
    Output: Pickle file with ML model. 
    '''
    
    # Deploy model into a Pickle file. 
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
