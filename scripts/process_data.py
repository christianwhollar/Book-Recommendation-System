import os
import pandas as pd
from typing import List, Set, Tuple

class ProcessData:
    """
    A class to process and analyze data from the Goodreads and bookcrossing datasets.

    This class provides methods to read data from CSV files, merge, filter, and analyze the data to generate a final DataFrame
    containing relevant information from both datasets. It also offers functionalities to rank user IDs, calculate average book
    ratings, and save DataFrames to pickle files.

    Attributes:
        None

    Methods:
        read_goodreads(data_directory='data/raw/goodreads/') -> Tuple[pd.DataFrame, pd.DataFrame]:
            Read the Goodreads data files and return two DataFrames.

        rank_gr_userid(df_gr_books: pd.DataFrame, df_gr_ratings: pd.DataFrame) -> pd.DataFrame:
            Rank the 'user_id_gr' column in the merged DataFrame 'df_gr' to consecutive integers.

        read_bookcrossing(data_directory='data/raw/bookcrossing/'):
            Read the bookcrossing data files and return three DataFrames.

        avg_bx_ratings(df_bx_books: pd.DataFrame, df_bx_ratings: pd.DataFrame, df_bx_users: pd.DataFrame) -> pd.DataFrame:
            Calculate the average book rating for each ISBN in the merged DataFrame 'df_bx'.

        filter_rows_by_column_value(data: pd.DataFrame, column_name: str, allowed_values: Set[str]) -> pd.DataFrame:
            Filter rows of the DataFrame based on the allowed values in the specified column.

        get_common_elements(dataframes: List[pd.DataFrame], column_names: List[str]) -> Set[str]:
            Get the common elements between two DataFrames in the specified columns.

        filter_dataframes(dataframes: List[pd.DataFrame], column_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
            Filter two DataFrames based on the common elements in the specified columns.

        save_dataframe_to_pickle(df: pd.DataFrame, target_dir: str, file_name: str) -> None:
            Save DataFrame to a .pkl file in the specified directory.

        get_final_dataframe() -> pd.DataFrame:
            Generate the final DataFrame by merging, filtering, and sorting the data.

    Example:
        
    """

    def read_goodreads(self, data_directory='data/raw/goodreads/') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the Goodreads data files and return two DataFrames.

        Parameters:
            data_directory (str): Path to the directory containing the Goodreads data files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: df_gr_ratings and df_gr_books.
        """
        df_gr_books = pd.read_csv(f'{data_directory}books.csv')
        df_gr_ratings = pd.read_csv(f'{data_directory}ratings.csv')
        return df_gr_ratings, df_gr_books
    
    def rank_gr_userid(self, df_gr_books: pd.DataFrame, df_gr_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Rank the 'user_id_gr' column in the merged DataFrame 'df_gr' to consecutive integers.

        Parameters:
            df_gr_books (pd.DataFrame): DataFrame containing Goodreads book data.
            df_gr_ratings (pd.DataFrame): DataFrame containing Goodreads ratings data.

        Returns:
            pd.DataFrame: DataFrame with 'user_id_gr' column ranked by consecutive integers.
        """
        df_gr = pd.merge(df_gr_ratings, df_gr_books, how='inner', on='book_id')
        df_gr = df_gr.add_suffix('_gr')
        df_gr['user_id_gr'] = df_gr['user_id_gr'].rank(method='dense').astype(int)
        df_gr = df_gr.sort_values(by='user_id_gr')
        df_gr.reset_index(drop=True, inplace=True)
        return df_gr
    
    def read_bookcrossing(self, data_directory='data/raw/bookcrossing/'):
        """
        Read the bookcrossing data files and return three DataFrames.

        Parameters:
            data_directory (str): Path to the directory containing the bookcrossing data files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames: df_bx_ratings, df_bx_books, and df_bx_users.
        """
        df_bx_ratings = pd.read_csv(f'{data_directory}BX-Book-Ratings.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')
        df_bx_books = pd.read_csv(f'{data_directory}BX-Books.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')  
        df_bx_users = pd.read_csv(f'{data_directory}BX-Users.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')  
        return df_bx_ratings, df_bx_books, df_bx_users
    
    def avg_bx_ratings(self, df_bx_books: pd.DataFrame, df_bx_ratings: pd.DataFrame, df_bx_users: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the average book rating for each ISBN in the merged DataFrame 'df_bx'.

        Parameters:
            df_bx_books (pd.DataFrame): DataFrame containing bookcrossing book data.
            df_bx_ratings (pd.DataFrame): DataFrame containing bookcrossing ratings data.
            df_bx_users (pd.DataFrame): DataFrame containing bookcrossing user data.

        Returns:
            pd.DataFrame: DataFrame with average book ratings for each ISBN.
        """
        df_bx = pd.merge(df_bx_books, df_bx_ratings, how='inner', on='ISBN')
        df_bx = pd.merge(df_bx, df_bx_users, how='inner', on='User-ID')
        df_bx = df_bx.add_suffix('_bx')
        df_bx['Book-Rating_bx'] = df_bx['Book-Rating_bx'].astype(int)
        df_bx = df_bx.groupby("ISBN_bx")["Book-Rating_bx"].mean().reset_index()
        return df_bx
    
    def filter_rows_by_column_value(self, data: pd.DataFrame, column_name: str, allowed_values: Set[str]) -> pd.DataFrame:
        """
        Filter rows of the DataFrame based on the allowed values in the specified column.

        Parameters:
            data (pd.DataFrame): DataFrame to be filtered.
            column_name (str): Column name to be used for filtering.
            allowed_values (Set[str]): Set of allowed values in the specified column.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered_data = data[data[column_name].isin(allowed_values)]
        return filtered_data

    def get_common_elements(self, dataframes: List[pd.DataFrame], column_names: List[str]) -> Set[str]:
        """
        Get the common elements between two DataFrames in the specified columns.

        Parameters:
            dataframes (List[pd.DataFrame]): List of DataFrames.
            column_names (List[str]): List of column names to check for common elements.

        Returns:
            Set[str]: Set of common elements in the specified columns.
        """
        set0 = set(dataframes[0][column_names[0]])
        set1 = set(dataframes[1][column_names[1]])
        return set0.intersection(set1)

    def filter_dataframes(self, dataframes: List[pd.DataFrame], column_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter two DataFrames based on the common elements in the specified columns.

        Parameters:
            dataframes (List[pd.DataFrame]): List of DataFrames to be filtered.
            column_names (List[str]): List of column names to check for common elements.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two filtered DataFrames.
        """
        intersection_set = self.get_common_elements(dataframes=dataframes, column_names=column_names)
        bx_filtered = self.filter_rows_by_column_value(dataframes[0], column_names[0], intersection_set)
        gr_filtered = self.filter_rows_by_column_value(dataframes[1], column_names[1], intersection_set)
        return bx_filtered, gr_filtered

    def save_dataframe_to_pickle(self, df: pd.DataFrame, target_dir: str, file_name: str) -> None:
        """
        Save DataFrame to a .pkl file in the specified directory.

        Parameters:
            df (pd.DataFrame): DataFrame to be saved.
            target_dir (str): Target directory path to save the file.
            file_name (str): Name of the .pkl file.

        Returns:
            None
        """
        # Check if the target directory exists, if not, create it
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Create the file path
        file_path = os.path.join(target_dir, file_name)

        # Save the DataFrame to the .pkl file
        df.to_pickle(file_path)

        print(f"DataFrame saved to {file_path}")

    def get_final_dataframe(self) -> pd.DataFrame:
        """
        Generate the final DataFrame by merging, filtering, and sorting the data.

        Returns:
            pd.DataFrame: Final merged and filtered DataFrame.
        """
        df_gr_ratings, df_gr_books = self.read_goodreads()
        df_gr = self.rank_gr_userid(df_gr_books, df_gr_ratings)

        df_bx_ratings, df_bx_books, df_bx_users = self.read_bookcrossing()
        df_bx = self.avg_bx_ratings(df_bx_books, df_bx_ratings, df_bx_users)

        gr_filtered, bx_filtered = self.filter_dataframes(dataframes=[df_gr, df_bx], column_names=['isbn_gr', 'ISBN_bx'])

        df_final = pd.merge(gr_filtered, bx_filtered, how='inner', left_on='isbn_gr', right_on='ISBN_bx')  
        df_final.sort_values('user_id_gr', inplace=True) 

        df_final.rename(columns={'Book-Rating_bx':'rating_bx'}, inplace=True)

        self.save_dataframe_to_pickle(df_final[['isbn_gr', 'authors_gr', 'title_gr', 'original_publication_year_gr']],
                                      target_dir='data/processed/',
                                      file_name='book_reference_dataframe.pkl'
                                      )

        self.save_dataframe_to_pickle(df_final[['user_id_gr', 'isbn_gr','rating_bx','rating_gr']],
                                      target_dir='data/processed/',
                                      file_name='final_dataframe.pkl'
                                      )
    
        return df_final
