import pandas as pd 
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data():
    # Load true news data from both 2020 ~ 2024
    true_df_2020 = pd.read_csv('./data/True_2020.csv')[['title', 'text']]
    true_df_2022 = pd.read_csv('./data/True_2022.csv')[['title', 'text']]
    true_df_2024 = pd.read_csv('./data/True_2024.csv')[['title', 'text']]
    true_df = pd.concat([true_df_2020, true_df_2022, true_df_2024], axis=0).reset_index(drop=True)
    # Load fake news data from both 2020 ~ 2024
    fake_df_2020 = pd.read_csv('./data/Fake_2020.csv')[['title', 'text']]
    fake_df_2022 = pd.read_csv('./data/Fake_2022.csv')[['title', 'text']]
    fake_df_2024 = pd.read_csv('./data/Fake_2024.csv')[['title', 'text']]
    fake_df = pd.concat([fake_df_2020, fake_df_2022, fake_df_2024], axis=0).reset_index(drop=True)
    # Label the data
    true_df['target'] = 0
    fake_df['target'] = 1
    
    # Load additional dataset
    add_df = pd.read_csv('./data/WELFake.csv')[['title', 'text', 'label']]
    add_df = add_df.rename(columns={'label': 'target'})
    
    # Combine all datasets and Remove duplicates
    df = pd.concat([true_df, fake_df, add_df], axis=0).reset_index(drop=True)
    df = df.drop_duplicates(subset=['title', 'text'])
    return df

def plot_distribution(df):
    df.target.value_counts(normalize=True).plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.show()

def text_preprocessing(text):
    # Check if the input is a string; if not, convert it to an empty string
    if not isinstance(text, str):
        text = ''
        
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation + "–—−±×÷"), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    text = re.sub(r'reuters', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

def preprocess_data(df):
    df['title'] = df['title'].apply(text_preprocessing)
    df['text'] = df['text'].apply(text_preprocessing)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_and_save(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train_df.to_csv('./data/train.csv', index=False)
    test_df.to_csv('./data/test.csv', index=False)
    print("Train dataset size : ", len(train_df))
    print("Test dataset size : ", len(test_df))

if __name__ == "__main__":
    df = load_data()
    print("News dataset size : ", len(df), end= "\n")
    plot_distribution(df)
    df = preprocess_data(df)
    print(df.head(), end = "\n\n")
    split_and_save(df)
    print("News data preprocessing is done.")
