from scripts.NNHybridFilteringModule import NNHybridFiltering
import time
import pickle 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class HybridFilteringRecommender():
    """
    Class for generating recommendations using a hybrid filtering approach.

    Attributes:
        input_df (pd.DataFrame): Input DataFrame containing the data for generating recommendations.

    Methods:
        __init__(self, input_df: pd.DataFrame)
            Initialize the HybridFilteringRecommender class.

        map_columns_to_int(self, df: pd.DataFrame, column_name: str = 'isbn_gr', mapping_dir: str = None) -> pd.DataFrame
            Map the values in a specified column of the DataFrame to integer values.

        process_df(self) -> pd.DataFrame
            Process the input DataFrame by mapping relevant columns to integer values.

        predict_rating(self, model: nn.Module, userid: int, isbn: int, bxrating: float, device: torch.device) -> torch.Tensor
            Predict the rating for a given user, item, and book rating.

        generate_recommendations(self, df: pd.DataFrame, model: nn.Module, userid: int, n_recs: int, device: torch.device) -> np.ndarray
            Generate book recommendations for a given user.

        load_isbn_map(self) -> dict[int, int]
            Load the mapping dictionary for ISBN values.

        load_isbn_df(self) -> pd.DataFrame
            Load the DataFrame containing book reference information.

        get_default_books(self, model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[list[str], list[str], list[str]]
            Get default book recommendations using the trained hybrid recommendation model.
    """

    def __init__(self, input_df: pd.DataFrame):
        """
        Initialize the HybridFilteringRecommender class.

        Args:
            input_df (pd.DataFrame): Input DataFrame containing the data for generating recommendations.
        """
        self.input_df = input_df

    def map_columns_to_int(self, df: pd.DataFrame, column_name: str = 'isbn_gr', mapping_dir: str = None) -> pd.DataFrame:
        """
        Map the values in a specified column of the DataFrame to integer values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column_name (str): Name of the column to map. Default is 'isbn_gr'.
            mapping_dir (str): Directory to save the mapping file. Default is None.

        Returns:
            pd.DataFrame: DataFrame with the mapped column.
        """
        # Make a copy of the DataFrame to avoid modifying the original data
        df = df.copy()

        unique_values = df[column_name].unique()
        mapping_dict = {value: idx for idx, value in enumerate(unique_values)}

        # Replace the strings in the column with their corresponding integer values
        df[column_name] = df[column_name].map(mapping_dict)

        # Save the mapping dictionary to a file if mapping_dir is provided
        if mapping_dir is not None:
            mapping_file_path = f"{mapping_dir}/{column_name}_mapping.pkl"
            with open(mapping_file_path, 'wb') as f:
                pickle.dump(mapping_dict, f)
            print(f"Mapping for column '{column_name}' saved to {mapping_file_path}")

        return df

    def process_df(self) -> pd.DataFrame:
        """
        Process the input DataFrame by mapping relevant columns to integer values.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        df = self.map_columns_to_int(self.input_df, column_name='isbn_gr', mapping_dir='data/processed/')
        return df

    def predict_rating(self, model: nn.Module, userid: int, isbn: int, bxrating: float, device: torch.device) -> torch.Tensor:
        """
        Predict the rating for a given user, item, and book rating.

        Args:
            model (nn.Module): Trained hybrid recommendation model.
            userid (int): User ID.
            isbn (int): ISBN value.
            bxrating (float): Book rating.
            device (torch.device): Torch device for prediction.

        Returns:
            torch.Tensor: Predicted rating for the user, item, and book rating.
        """
        with torch.no_grad():
            model.eval()
            X = torch.Tensor([userid, isbn, bxrating]).long().view(1, -1)
            X = X.to(device)
            pred = model.forward(X)
            return pred
        
    def generate_recommendations(self, df: pd.DataFrame, model: nn.Module, userid: int, n_recs: int, device: torch.device) -> np.ndarray:
        """
        Generate book recommendations for a given user.

        Args:
            df (pd.DataFrame): DataFrame containing user-item interactions.
            model (nn.Module): Trained hybrid recommendation model.
            userid (int): User ID for whom recommendations are generated.
            n_recs (int): Number of recommendations to generate.
            device (torch.device): Torch device for prediction.

        Returns:
            np.ndarray: Array containing ISBN values of recommended books.
        """
        pred_ratings = []
        books_read = df.loc[df['user_id_gr'] == userid, 'isbn_gr'].values.tolist()
        isbns = [x for x in df['isbn_gr'].unique().tolist() if x not in books_read]

        for isbn in tqdm(isbns):
            bxrating = df.loc[df['isbn_gr'] == isbn, 'rating_bx'].values[0]
            pred = self.predict_rating(model, userid=userid, isbn=isbn, bxrating=bxrating, device=device)
            pred_ratings.append([isbn, pred.detach().cpu().item()])
            
        idxs = np.argsort(np.array(pred_ratings)[:, 1])
        recs = np.array(pred_ratings)[idxs][-n_recs:, 0]
        return recs

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

    def get_default_books(self, model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[list[str], list[str], list[str]]:
        """
        Get default book recommendations using the trained hybrid recommendation model.

        Args:
            model_dir (str, optional): Directory where the trained model is saved. Default is 'models/hybrid_recommender.pkl'.

        Returns:
            tuple[list[str], list[str], list[str]]: Tuple containing lists of authors, titles, and publication years of recommended books.
        """
        authors = []
        titles = []
        years = []

        df = self.process_df()
        uservcs = df['user_id_gr'].value_counts()
        lfv = uservcs.idxmin()
        min_user = df.loc[df['user_id_gr'] == lfv, 'user_id_gr']

        model = torch.load(model_dir)
        userid = min_user.iloc[0]
        n_recs = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        recs = self.generate_recommendations(df, model, userid, n_recs, device)
        isbn_map = self.load_isbn_map()
        isbn_df = self.load_isbn_df()

        for rec in recs:
            ref_row = isbn_df[isbn_df['isbn_gr'] == isbn_map[int(rec)]].iloc[0]
            authors.append(ref_row['authors_gr'])
            titles.append(ref_row['title_gr'])
            years.append(ref_row['original_publication_year_gr'])
        
        return authors, titles, years
