"""
This script joins all the comments and posts by using the link id. 
"""

import os
import pandas as pd

file_path = r'datasets\reddit_csv'
output_file_path = r'datasets\reddit_csv\appended_datasets'

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
    'wallstreetbets',
    'Wallstreetsilver'
]

	
for subreddit in subreddits:        
    comments_data = pd.read_csv(os.path.join(file_path, f"{subreddit}_comments.csv"))
    submissions_data = pd.read_csv(os.path.join(file_path, f"{subreddit}_submissions.csv"))

    #only the last part of the code is needed when joining comments and posts
    comments_data['id'] = comments_data['link_id'].str[3:]
    comments_data = comments_data.rename(columns = {'body': 'text'})
    merged_data = pd.concat([comments_data, submissions_data])
    merged_data = merged_data.drop('link_id', axis = 1)

    output_file = os.path.join(output_file_path, f"{subreddit}.csv")
    merged_data.to_csv(output_file, index=False)
    print(f"Done with {subreddit}")
