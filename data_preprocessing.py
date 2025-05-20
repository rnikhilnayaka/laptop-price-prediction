import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    df['Touchscreen'] = df['Touchscreen'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['IPSpanel'] = df['IPSpanel'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['RetinaDisplay'] = df['RetinaDisplay'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop(['Product', 'Screen', 'CPU_model', 'GPU_model'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df
