import pickle 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score

class HybridFilterRecommender():
    """
    Class for generating recommendations using a hybrid filtering approach.

    Attributes:
        input_df (pd.DataFrame): Input DataFrame containing the data for generating recommendations.

    Methods:
        __init__(self, input_df: pd.DataFrame)
            Initialize the HybridFilterRecommender class.

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

        get_best_user(self, selected_books:list = ["Atonement"]) -> int
            Get the user who read most of the selected books.

        get_default_books(self, selected_books:list = ["Atonement"], model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[list[str], list[str], list[str]]
            Get default book recommendations using the trained hybrid recommendation model.

        eval_model(self, model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[float, float, float, float]
            Evaluate the model's performance using mean squared error (MSE), accuracy, recall, and precision metrics.
            ...
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
    
    def get_best_user(self, selected_books:list = ["Atonement"]) -> int:
        """
        Get the user who read most of the selected books.

        Args:
            selected_books (list, optional): List of selected book titles.

        Returns:
            int: User ID who read most of the selected books.
        """
        df = self.process_df()
        isbn_map = self.load_isbn_map()
        isbn_map = {value: key for key, value in isbn_map.items()}
        isbn_df = self.load_isbn_df()

        books_mapped = []

        for book in selected_books:
            temp_isbn_df = isbn_df.loc[isbn_df['title_gr'] == book]
            book_isbn = temp_isbn_df.iloc[0]
            book_isbn = book_isbn['isbn_gr']
            books_mapped.append(isbn_map[book_isbn])

        filtered_df = df[df['isbn_gr'].isin(books_mapped)]
        user_book_counts = filtered_df['user_id_gr'].value_counts()
        selected_user = user_book_counts.idxmax()

        return selected_user


    def get_default_books(self, selected_books:list = ["Atonement"], model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[list[str], list[str], list[str]]:
        """
        Get default book recommendations using the trained hybrid recommendation model.

        Args:
            selected_books (str, optional): Books that have been read by the user.
            model_dir (str, optional): Directory where the trained model is saved. Default is 'models/hybrid_recommender.pkl'.

        Returns:
            tuple[list[str], list[str], list[str]]: Tuple containing lists of authors, titles, and publication years of recommended books.
        """
        authors = []
        titles = []
        years = []

        df = self.process_df()
        isbn_map = self.load_isbn_map()
        isbn_map = {value: key for key, value in isbn_map.items()}
        isbn_df = self.load_isbn_df()

        books_mapped = []

        for book in selected_books:
            temp_isbn_df = isbn_df.loc[isbn_df['title_gr'] == book]
            book_isbn = temp_isbn_df.iloc[0]
            book_isbn = book_isbn['isbn_gr']
            books_mapped.append(isbn_map[book_isbn])

        filtered_df = df[df['isbn_gr'].isin(books_mapped)]
        user_book_counts = filtered_df['user_id_gr'].value_counts()
        selected_user = user_book_counts.idxmax()

        model = torch.load(model_dir)
        userid = selected_user
        n_recs = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        isbn_map = {value: key for key, value in isbn_map.items()}
        recs = self.generate_recommendations(df, model, userid, n_recs, device)
        isbn_df = self.load_isbn_df()

        for rec in recs:
            curr_isbn = str(isbn_map[int(rec)])
            ref_row = isbn_df[isbn_df['isbn_gr'] == curr_isbn].iloc[0]
            authors.append(ref_row['authors_gr'])
            titles.append(ref_row['title_gr'])
            years.append(ref_row['original_publication_year_gr'])
        
        return authors, titles, years

    def eval_model(self, model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[float, float, float, float]:

        """
        Evaluate the model's performance using mean squared error (MSE), accuracy, recall, and precision metrics.

        Args:
            model_dir (str, optional): Directory where the trained model is saved. Default is 'models/hybrid_recommender.pkl'.

        Returns:
            tuple[float, float, float, float]: Tuple containing MSE, accuracy, recall, and precision metrics.
        """

        df = self.process_df()
        df.drop(columns = ['genres'], inplace = True)
        df = df.astype(int)
        X = df.loc[:, ['user_id_gr', 'isbn_gr','rating_bx']]
        y = df.loc[:, ['rating_gr']]
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.2)
        
        model = torch.load(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        preds = []
        for userid, isbn, bxrating in tqdm(zip(X_val['user_id_gr'],X_val['isbn_gr'],X_val['rating_bx']), total=len(X_val)):
            pred = self.predict_rating(model = model, userid = userid, isbn = isbn, bxrating = bxrating, device = device)
            preds.append(pred)

        pred_list = [int(tensor.item()) for tensor in preds]

        mse = mean_squared_error(y_val, pred_list)
        accuracy = accuracy_score(y_val, pred_list)
        recall = recall_score(y_val, pred_list, average = 'micro')
        precision = precision_score(y_val, pred_list, average = 'micro')
        return mse, accuracy, recall, precision