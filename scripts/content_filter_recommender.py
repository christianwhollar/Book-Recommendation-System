import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

class ContentFilterRecommender():

    def __init__(self, df):
        self.df = df
        self.books = self.get_books_df()

    def get_genre_vector(self):
        vec = CountVectorizer()
        genres_vec = vec.fit_transform(self.books['isbn_gr'].unique())
        genres_vectorized = pd.DataFrame(genres_vec.todense(),columns=vec.get_feature_names_out(),index=self.books.isbn_gr)
        return genres_vectorized

    def get_books_df(self):
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

    def get_cs_matrix(self):
        genres_vec = self.get_genre_vector()
        csmatrix = cosine_similarity(genres_vec)
        csmatrix = pd.DataFrame(csmatrix,columns=self.books.isbn_gr,index=self.books.isbn_gr)
        return csmatrix
    
    def get_split_data(self):
        X = self.df.drop(labels=['rating_gr','genres', 'rating_bx'],axis=1)
        y = self.df['rating_gr']
        X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)
        return X_train, X_val, y_train, y_val

    def generate_recommendations(self, user = 1): 
        simtable = self.get_cs_matrix()
        user_ratings = self.df.loc[self.df['user_id_gr']==user]
        user_ratings = user_ratings.sort_values(by='rating_bx',axis=0,ascending=False)
        topratedbook = user_ratings.iloc[0,:]['isbn_gr']
        sims = simtable.loc[topratedbook,:]
        mostsimilar = sims.sort_values(ascending=False).index.values
        isbns = mostsimilar[0:3]
        return isbns
    
    def get_aty_recs(self, user = 1):
        authors = []
        titles = []
        years = []

        isbns = self.generate_recommendations(user = user)
        isbn_df = self.load_isbn_df()

        for isbn in isbns:
            ref_row = isbn_df[isbn_df['isbn_gr'] == isbn].iloc[0]
            authors.append(ref_row['authors_gr'])
            titles.append(ref_row['title_gr'])
            years.append(ref_row['original_publication_year_gr'])
        
        return authors, titles, years