import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_user_data(ratings_df, movies_df, train_ratio=0.8):
    # Convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # Sort ratings by timestamp
    ratings_df = ratings_df.sort_values('timestamp')
    
    # Find the timestamp for the split
    split_index = int(len(ratings_df) * train_ratio)
    split_timestamp = ratings_df.iloc[split_index]['timestamp']
    
    # Split the data
    train_ratings = ratings_df[ratings_df['timestamp'] < split_timestamp]
    test_ratings = ratings_df[ratings_df['timestamp'] >= split_timestamp]
    
    # Analyze the test set
    print(f"Analysis of test set (data after {split_timestamp}):")
    print(f"Number of ratings: {len(test_ratings)}")
    print(f"Number of unique users: {test_ratings['userId'].nunique()}")
    print(f"Number of unique movies: {test_ratings['movieId'].nunique()}")
    
    # New users and movies
    new_users = set(test_ratings['userId']) - set(train_ratings['userId'])
    new_movies = set(test_ratings['movieId']) - set(train_ratings['movieId'])
    print(f"Number of new users: {len(new_users)}")
    print(f"Number of new movies: {len(new_movies)}")
    
    # Distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(test_ratings['rating'], bins=10, kde=True)
    plt.title('Distribution of Ratings in Test Set')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('test_ratings_distribution.png')
    plt.close()
    
    # Users activity
    user_activity = test_ratings['userId'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_activity, bins=30, kde=True)
    plt.title('Distribution of User Activity in Test Set')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Count')
    plt.savefig('test_user_activity_distribution.png')
    plt.close()
    
    # Movie popularity
    movie_popularity = test_ratings['movieId'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_popularity, bins=30, kde=True)
    plt.title('Distribution of Movie Popularity in Test Set')
    plt.xlabel('Number of Ratings per Movie')
    plt.ylabel('Count')
    plt.savefig('test_movie_popularity_distribution.png')
    plt.close()
    
    # Time series of ratings
    daily_ratings = test_ratings.set_index('timestamp').resample('D').size()
    plt.figure(figsize=(12, 6))
    daily_ratings.plot()
    plt.title('Number of Ratings per Day in Test Set')
    plt.xlabel('Date')
    plt.ylabel('Number of Ratings')
    plt.savefig('test_daily_ratings.png')
    plt.close()
    
    # Genre analysis
    test_movies = movies_df[movies_df['movieId'].isin(test_ratings['movieId'])]
    genre_counts = test_movies['genres'].str.get_dummies('|').sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar')
    plt.title('Genre Distribution in Test Set')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('test_genre_distribution.png')
    plt.close()
    
    return train_ratings, test_ratings

def analyze_user_features(user_features_df):
    print("Statistical Analysis of User Features")
    print("=====================================")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(user_features_df.describe())

    # Correlation analysis
    print("\nCorrelation Matrix:")
    correlation_matrix = user_features_df.corr()
    print(correlation_matrix)

    # Top correlated features
    print("\nTop 10 Feature Correlations:")
    corr_pairs = correlation_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs != 1].sort_values(ascending=False)
    print(corr_pairs.head(10))

    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Distribution of User Features", fontsize=16)
    
    features_to_plot = ['avg_rating', 'budget_preference', 'revenue_preference', 
                        'vote_count_preference', 'Action', 'Drama']
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i // 3, i % 3]
        if feature in user_features_df.columns:
            sns.histplot(user_features_df[feature], kde=True, ax=ax)
            ax.set_title(feature)
            ax.set_xlabel('')
        else:
            ax.text(0.5, 0.5, f"Feature '{feature}' not found", ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('user_features_distribution.png')
    print("\nDistribution plot saved as 'user_features_distribution.png'")

    # Genre preference analysis
    genre_columns = [col for col in user_features_df.columns if col not in ['avg_rating', 'budget_preference', 'revenue_preference', 'vote_count_preference']]
    genre_preferences = user_features_df[genre_columns]
    
    print("\nGenre Preference Analysis:")
    print("Average genre preferences across all users:")
    print(genre_preferences.mean().sort_values(ascending=False))

    # Visualize genre preferences
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=genre_preferences)
    plt.title("Distribution of Genre Preferences")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('genre_preferences_distribution.png')
    print("\nGenre preference distribution plot saved as 'genre_preferences_distribution.png'")
