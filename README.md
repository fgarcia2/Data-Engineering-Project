# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Go to app directory: cd app

3. Run your web app: python run.py

4. Navigate to the URL shown in the kernel directly in your browser to load the app. 

## Context 
In this project, a dataset containing messages that were sent during a disaster event is utilized. Subsequently, an ETL framework will be leveraged to prepare the data for model consumption. 

Using this framework, a machine learning pipeline is designed so that messages can be routed to an appropriate disaster relief agency. Visualizations of the data will also be included. 

## Content 
The primary dataset contains information about messages sent during an emergency, including id, message, original and genre. 

## Installation 
This project requires Python 3.x and the following Python libraries installed:

NumPy Pandas sqlalchemy nltk sklearn json plotly joblib pickle and flask

Python Spyder IDE has been used and is recommended for coding.

## Project Motivation 
This project will center itself on successfully classifying messages based on potential target disaster relief agencies. The structure of the folder is as follows 

1. Data: contains the data required to be extracted and that will be used for training/testing of the model. 
2. Model: contains the functions used to build and train the machine learning model.  
3. App: contains the code used for rendering the app that will display the visualization and results of the classifier model. 

## File descriptions 

Main Folder (run.py). Also supported by process_data.py and train_classifier.py files. 

## Results 
Overall, one finding from the training data is that the messages can be primarily distributed as coming from the news genre, then followed by the direct and social categories.  

A secondary visual depicts that related, aid-related and weather-related are the top three agencies to relay messages into. 

