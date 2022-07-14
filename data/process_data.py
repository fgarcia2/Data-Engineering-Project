import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the input data directly from CSV file
    Input: Path to the CSV files containing the input data
    Output: Pandas dataframe containing tabulated input data.  
    '''
    
    # Read CSV data into pandas. 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge both tables using ID as the join. 
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    '''
    Function to clean the data from table, dropping duplicates and redundant entries. 
    Input: Pandas dataframe with input data. 
    Output: Pandas dataframe containing clean data. 
    '''
    
    # Extract categories from dataframe. 
    categories = df['categories'].str.split(pat=";", expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # Drop duplicates
    df_new = df.drop_duplicates()
    
    # drop columns that do not contain binary values. 
    df_new = df_new.drop(df_new[df_new['related'] == 2].index)
    
    return df_new


def save_data(df, database_filename):
    '''
    Function to save data into a SQL database. 
    Input: Pandas dataframe with clean data, filename for SQL file. 
    Output: SQL file with clean data. 
    '''
    
    # Using Pandas functionality to create SQL file. 
    engine = create_engine('sqlite:///./data/emergency.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
