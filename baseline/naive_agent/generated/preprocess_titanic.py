import pandas as pd

# Load the dataset
df = pd.read_csv('..\\data\\sample_titanic.csv')

# Fill missing values
# Assuming 'Age' and 'Fare' are numerical columns and 'Embarked' is categorical
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Fare' in df.columns:
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical columns to numerical
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop irrelevant columns
if 'Name' in df.columns:
    df.drop(columns=['Name'], inplace=True)

if 'Ticket' in df.columns:
    df.drop(columns=['Ticket'], inplace=True)

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_titanic.csv', index=False)

# Print a short summary of the result
print("Data preprocessing completed. Here is a summary:")
print(df.describe())
print("\nColumns:", df.columns.tolist())