import pandas as pd

games_df = pd.read_csv('/content/drive/MyDrive/Capstone Project/games_filtered.csv')
tags_df = pd.read_csv('/content/drive/MyDrive/Capstone Project/filtered_tags.csv')
unique_tags_df = pd.DataFrame({'tag': tags_df['tag'].unique()})
user_library = pd.read_csv('/content/drive/MyDrive/Capstone Project/yoshie.csv')
