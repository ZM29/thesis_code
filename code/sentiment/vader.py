"""
This script implements the VADER sentiment analysis on Reddit data from multiple finance-related subreddits.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import os
import emoji
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.sentiment.util import mark_negation

#list of subreddits
subreddits = [
    'algotrading',
    'Bogleheads',
    'Daytrading',
    'dividends',
    'ETFs',
    'ExpatFIRE',
    'fatFIRE',
    'financialindependence',
    'investing',
    'investing_discussion',
    'leanfire',
    'options',
    'pennystocks',
    'realestateinvesting',
    'RealEstate',
    'SecurityAnalysis',
    'StockMarket',
    'stocks',
    'thewallstreet',
    'ValueInvesting',
    'WallStreetbetsELITE',
    'Wallstreetbetsnew',
    'wallstreetbetsOGs',
    'wallstreetbets_1',
    'wallstreetbets_2',
    'wallstreetbets_3',
    'wallstreetbets_4',
    'wallstreetbets_5',
    'wallstreetbets_6',
    'wallstreetbets_7',
    'wallstreetbets_8',
    'wallstreetbets_9',
    'Wallstreetsilver_5',
    'Wallstreetsilver_6',
    'Wallstreetsilver_7',
    'Wallstreetsilver_8' 
]

def clean_text(text):
    """
    Function to clean the text.
    """

    text = re.sub(r"<[^>]*>", "", text)  #removing HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  #removing special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  #removing URLs
    text = text.lower() #lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)) #removing punctuation

    #expanding contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ve", " have", text)
    
    #marking negations
    text = mark_negation(word_tokenize(text))
    text = ' '.join(text)

    #tokenizing, removing stop words and lemmatizing 
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    cleaned_text = ' '.join(tokens)
    return cleaned_text



def remove_emojis(text):
    #function that removes all emojis
    return emoji.replace_emoji(text, replace='')



def logic_clean(data):
    """
    Function that filters the data based on specific logic
    """

    data['created'] = pd.to_datetime(data['created']).dt.strftime('%Y-%m-%d')

    #filtering based on date (2020 and 2021)
    data = data[data['created'] < '2022-01-01']
    data = data[data['created'] > '2019-12-31']
    
    #removing all comments that are removed or deleted (either by the user or moderator)
    mask = data['title'].isna() & data['text'].astype(str).isin(['[removed]', '[deleted]'])
    data = data[~mask]
    
    #removing the engagement that are posts (submissions) without comments or comments on posts before 2020
    duplicated_ids = data['id'][data['id'].duplicated(keep=False)]
    data = data[data['id'].isin(duplicated_ids)]
    not_null_titles = data['title'].notnull()
    data = data.loc[not_null_titles.groupby(data['id']).transform('any')]

    #removing all posts that are removed or deleted (either by the user or moderator)
    mask = data['title'].astype(str).isin(['[deleted by user]']) & data['text'].astype(str).isin(['[removed]', '[deleted]'])
    data = data[~mask]
    mask = data['title'].notna() & data['text'].astype(str).isin(['[removed]', '[deleted]'])
    data = data[~mask]
    
    data = data.reset_index(drop=True)
    return data


#downloading the prerequisites
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#initializing VADER and stopwords
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))


file_path = r'datasets\reddit\reddit_csv\appended_datasets'

#looping over the subreddits
for subreddit in subreddits:  

    data = pd.read_csv(os.path.join(file_path ,f"{subreddit}.csv"))

    #filtering the data
    data = logic_clean(data)
    data["text"] = data["text"].fillna("")
    data["title"] = data["title"].fillna("")
    print(f"Processing subreddit: {subreddit}") 
    
    #cleaning the text and removing emojis
    tqdm.pandas(desc="Progress")
    data["text"] = data["text"].progress_apply(clean_text)
    data["title"] = data["title"].progress_apply(clean_text)
    data["text"] = data["text"].progress_apply(remove_emojis)
    data["title"] = data["title"].progress_apply(remove_emojis)

    #concatenating title and text
    data["concatenated"] = data["title"] + ' ' + data["text"]

    #counting the engagement per day
    comment_count = data[data['title'] == ""].groupby('created')['text'].count()
    post_count = data[data['title'] != ""].groupby('created')['title'].count()

    #performing the sentiment analysis and compute the average sentiment per day
    data['sentiment_score'] = data['concatenated'].progress_apply(lambda x: analyzer.polarity_scores(x)['compound'])
    avg_sentiment = data.groupby('created')['sentiment_score'].mean()
    avg_sentiment = pd.DataFrame(avg_sentiment)
    avg_sentiment = avg_sentiment.join(post_count, on='created', rsuffix='_posts')
    avg_sentiment = avg_sentiment.join(comment_count, on='created', rsuffix='_comments')
    avg_sentiment['engagement'] = avg_sentiment['title'].fillna(0) + avg_sentiment['text'].fillna(0)
    avg_sentiment = avg_sentiment.drop(columns=['title', 'text'])
    avg_sentiment = avg_sentiment.reset_index()
    
    print(f"Saving results for subreddit: {subreddit}")
    avg_sentiment.to_csv(f"vader_{subreddit}.csv", index=False)

