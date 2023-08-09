import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score
import pickle

class ContentFilterRecommender:
    """
    Class for generating content-based recommendations using book genres.

    Attributes:
        df (pd.DataFrame): Input DataFrame containing user-item interactions and book information.

    Methods:
        __init__(self, df: pd.DataFrame)
            Initialize the ContentFilterRecommender class.

        get_genres_for_isbn(self, isbn: str) -> list[str]
            Get the genres associated with a given ISBN.

        get_genre_vector(self) -> pd.DataFrame
            Get the vectorized genre representation for ISBN values.

        get_books_df(self) -> pd.DataFrame
            Get a DataFrame containing unique book ISBNs.

        load_isbn_map(self) -> dict[int, int]
            Load the mapping dictionary for ISBN values.

        load_isbn_df(self) -> pd.DataFrame
            Load the DataFrame containing book reference information.

        get_cs_matrix(self) -> pd.DataFrame
            Get the cosine similarity matrix based on book genres.

        get_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            Split the data into training and validation sets.

        generate_recommendations(self, user: int = 1) -> list[str]
            Generate book recommendations for a given user based on content similarity.

        get_aty_recs(self, user: int = 1) -> tuple[list[str], list[str], list[str]]
            Get authors, titles, and publication years of recommended books for a user.

        get_eval_metrics(self) -> tuple[float, float, float, float]
            Calculate evaluation metrics (MSE, accuracy, recall, precision) for the content-based recommender.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ContentFilterRecommender class.

        Args:
            df (pd.DataFrame): Input DataFrame containing user-item interactions and book information.
        """
        self.df = df
        self.books = self.get_books_df()

    def get_genres_for_isbn(self, isbn: str) -> list[str]:
        """
        Get the genres associated with a given ISBN.

        Args:
            isbn (str): ISBN value of the book.

        Returns:
            list[str]: List of genres associated with the ISBN.
        """
        api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        response = requests.get(api_url)
        data = response.json()

        if 'items' in data:
            item = data['items'][0]
            if 'volumeInfo' in item:
                volume_info = item['volumeInfo']
                if 'categories' in volume_info:
                    genres = volume_info['categories']
                    return genres
        
        return []

    def get_genre_vector(self) -> pd.DataFrame:
        """
        Get the vectorized genre representation for ISBN values.

        Returns:
            pd.DataFrame: DataFrame with vectorized genre representation.
        """
        vec = CountVectorizer()
        genres_vec = vec.fit_transform(self.books['isbn_gr'].unique())
        genres_vectorized = pd.DataFrame(genres_vec.todense(),columns=vec.get_feature_names_out(),index=self.books.isbn_gr)
        return genres_vectorized

    def get_books_df(self) -> pd.DataFrame:
        """
        Get a DataFrame containing unique book ISBNs.

        Returns:
            pd.DataFrame: DataFrame with unique book ISBNs.
        """
        books = self.df[~self.df['isbn_gr'].duplicated(keep='first')]
        return books

    def load_isbn_map(self) -> dict[int, int]:
        """
        Load the mapping dictionary for ISBN values.

        Returns:
            dict[int, int]: Mapping dictionary for ISBN values.
        """
        isbn_map_path = 'data/processed/isbn_gr_mapping.pkl'
        with open(isbn_map_path, 'rb') as file:
            isbn_map = pickle.load(file)
        isbn_map = {value: key for key, value in isbn_map.items()}
        return isbn_map

    def load_isbn_df(self) -> pd.DataFrame:
        """
        Load the DataFrame containing book reference information.

        Returns:
            pd.DataFrame: DataFrame containing book reference information.
        """
        isbn_ref_path = 'data/processed/book_reference_dataframe.pkl'
        isbn_ref = pd.read_pickle(isbn_ref_path)
        return isbn_ref

    def get_cs_matrix(self) -> pd.DataFrame:
        """
        Get the cosine similarity matrix based on book genres.

        Returns:
            pd.DataFrame: Cosine similarity matrix.
        """
        genres_vec = self.get_genre_vector()
        csmatrix = cosine_similarity(genres_vec)
        csmatrix = pd.DataFrame(csmatrix,columns=self.books.isbn_gr,index=self.books.isbn_gr)
        return csmatrix
    
    def get_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and validation sets.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple containing X_train, X_val, y_train, y_val DataFrames/Series.
        """
        X = self.df.drop(labels=['rating_gr','genres', 'rating_bx'],axis=1)
        y = self.df['rating_gr']
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.2)
        return X_train, X_val, y_train, y_val

    def generate_recommendations(self, user: int = 1) -> list[str]:
        """
        Generate book recommendations for a given user based on content similarity.

        Args:
            user (int, optional): User ID for whom recommendations are generated. Default is 1.

        Returns:
            list[str]: List of ISBN values of recommended books.
        """
        simtable = self.get_cs_matrix()
        user_ratings = self.df.loc[self.df['user_id_gr'] == user]
        user_ratings = user_ratings.sort_values(by='rating_bx', axis=0, ascending=False)
        topratedbook = user_ratings.iloc[0, :]['isbn_gr']
        sims = simtable.loc[topratedbook, :]
        mostsimilar = sims.sort_values(ascending=False).index.values
        isbns = mostsimilar[0:3]
        return isbns
    
    def get_aty_recs(self, user: int = 1) -> tuple[list[str], list[str], list[str]]:
        """
        Get authors, titles, and publication years of recommended books for a user.

        Args:
            user (int, optional): User ID for whom recommendations are generated. Default is 1.

        Returns:
            tuple[list[str], list[str], list[str]]: Tuple containing lists of authors, titles, and publication years of recommended books.
        """
        authors = []
        titles = []
        years = []

        isbns = self.generate_recommendations(user=user)
        isbn_df = self.load_isbn_df()

        for isbn in isbns:
            ref_row = isbn_df[isbn_df['isbn_gr'] == isbn].iloc[0]
            authors.append(ref_row['authors_gr'])
            titles.append(ref_row['title_gr'])
            years.append(ref_row['original_publication_year_gr'])
        
        return authors, titles, years
    
    def get_eval_metrics(self) -> tuple[float, float, float, float]:
        """
        Calculate evaluation metrics (MSE, accuracy, recall, precision) for the content-based recommender.

        Returns:
            tuple[float, float, float, float]: Tuple containing MSE, accuracy, recall, and precision metrics.
        """
        simtable = self.get_cs_matrix()
        X_train, X_val, y_train, y_val = self.get_split_data()
        
        user_books = X_train.groupby('user_id_gr')['isbn_gr'].apply(list).to_dict()

        preds = []

        for user, book in tqdm(zip(X_val['user_id_gr'], X_val['isbn_gr']), total=len(X_val)):
            books_read = user_books.get(user, [])
            if books_read:
                simtable_filtered = simtable.loc[book, books_read]
                most_similar_read = simtable_filtered.idxmax()
                idx = X_train.loc[(X_train['user_id_gr'] == user) & (X_train['isbn_gr'] == most_similar_read)].index.values
                if len(idx) > 0:
                    most_similar_rating = y_train.loc[idx[0]]
                    preds.append(most_similar_rating)
                else:
                    preds.append(None)
            else:
                preds.append(None)

        preds = np.array([x if x is not None else np.nan for x in preds])
        valid_indices = ~np.isnan(preds)

        # Calculate MSE using valid predictions
        mse = mean_squared_error(y_val[valid_indices], preds[valid_indices])
        accuracy = accuracy_score(y_val[valid_indices], preds[valid_indices])
        recall = recall_score(y_val[valid_indices], preds[valid_indices], average='micro')
        precision = precision_score(y_val[valid_indices], preds[valid_indices], average='micro')
        return mse, accuracy, recall, precision
