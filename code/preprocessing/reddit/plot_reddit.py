"""
This script plots the amount of engagement per month on the selected subreddits. 
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

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

file_path = r'datasets\reddit\reddit_csv\appended_datasets'
comment_counts = {}
post_counts = {}

for subreddit in subreddits:
    data = pd.read_csv(os.path.join(file_path, f"{subreddit}.csv"))
    data['created'] = pd.to_datetime(data['created'])

    #filtering the data and splitting into comments and posts
    data = data[data['created'] > '2010-01-01']
    comments_data = data[data['title'].isnull()]
    posts_data = data[data['title'].notnull()]

    #counting the frequencies per day
    comment_freq = comments_data['created'].dt.date.value_counts()
    post_freq = posts_data['created'].dt.date.value_counts()
    
    comment_counts[f"{subreddit}"] = comment_freq
    post_counts[f"{subreddit}"] = post_freq

#combining all comment and post counts into a Dataframe
all_comment_counts = pd.concat(comment_counts.values(), axis = 1)
all_comment_counts.columns = comment_counts.keys()
all_comment_counts.index = pd.to_datetime(all_comment_counts.index)

all_post_counts = pd.concat(post_counts.values(), axis = 1)
all_post_counts.columns = post_counts.keys()
all_post_counts.index = pd.to_datetime(all_post_counts.index)


#resampling to monthly frequency
monthly_comment_counts = all_comment_counts.resample('M').sum()
monthly_post_counts = all_post_counts.resample('M').sum()
monthly_total = monthly_post_counts + monthly_comment_counts

monthly_total = monthly_total.reset_index()
monthly_total['created'] = pd.to_datetime(monthly_total['created'])
monthly_total = monthly_total.set_index('created')

#visualizing the monthly frequency of engagement 
fig, ax = plt.subplots(figsize = (20, 10))
monthly_total.sum(axis = 1).plot(kind = 'line', ax = ax, label = 'Comments')
plt.xlabel('Year', fontsize = 24)
plt.ylabel('Frequency', fontsize = 24)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(24)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
