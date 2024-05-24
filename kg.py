import csv
import pandas as pd

def read_movies():
    # read csv with movies for budget and imdb_id
    columns_of_interest = ['budget', 'imdb_id']
    data = []
    with open('movie_data_tmbd.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            extracted_row = {col: row[col] for col in columns_of_interest}
            data.append(extracted_row)

    movies_budget_df = pd.DataFrame(data)
    movies_budget_df = movies_budget_df.fillna({
        'budget': 0,
        'imdb_id': '',
        'title': ''
    })

    # merge movie budget with id
    link_df = pd.read_csv("links.csv")
    link_df['imdbId'] = link_df['imdbId'].apply(lambda x: f'tt0{int(x)}')

    movies_id_df = pd.merge(movies_budget_df, link_df, left_on='imdb_id', right_on='imdbId', how='inner')
    movies_id_df['budget'] = pd.to_numeric(movies_id_df['budget'])
    movies_id_df = movies_id_df[movies_id_df.budget != 0]

    movies_info_df = pd.read_csv("movies.csv")
    movies_df = pd.merge(movies_id_df, movies_info_df, on="movieId", how="inner")

    ratings_df = pd.read_csv("ratings.csv")
    ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]

    return movies_df, ratings_df

movies_df, ratings_df = read_movies()
print(movies_df.head())
print(ratings_df.head())