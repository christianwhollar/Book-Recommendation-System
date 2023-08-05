import os
import pandas as pd
from typing import List, Set, Tuple

class ProcessData:
    """
    ProcessData class for handling data processing operations.

    Methods:
    read_goodreads -> Tuple[pd.DataFrame, pd.DataFrame]: Read data from the Goodreads dataset.
    read_bookcrossings -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Read data from the Bookcrossings dataset.
    filter_rows_by_column_value -> pd.DataFrame: Filter rows in a DataFrame based on a specific column value.
    get_common_elements -> Set[str]: Get common elements between two DataFrames based on specified columns.
    filter_dataframes -> Tuple[pd.DataFrame, pd.DataFrame]: Filter two DataFrames based on their common elements in a specified column.
    avg_ratings -> pd.DataFrame: Calculate the average ratings for a DataFrame.
    save_dataframe_to_pickle -> None: Save a DataFrame to a .pkl file.
    get_final_dataframe -> pd.DataFrame: Get the final processed DataFrame.
    """

    def __init__(self) -> None:
        pass

    def read_goodreads(self, data_directory='data/raw/goodreads/') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read data from the Goodreads dataset.

        Parameters:
            data_directory (str): Directory path where the Goodreads dataset is located.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing ratings and book information from Goodreads.
        """
        df_gr_ratings = pd.read_csv(f'{data_directory}books.csv')
        df_gr_books = pd.read_csv(f'{data_directory}ratings.csv')
        return df_gr_ratings, df_gr_books

    def read_bookcrossings(self, data_directory='data/raw/bookcrossing/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read data from the Bookcrossings dataset.

        Parameters:
            data_directory (str): Directory path where the Bookcrossings dataset is located.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames containing ratings, book, and user information from Bookcrossings.
        """
        df_bx_ratings = pd.read_csv(f'{data_directory}BX-Book-Ratings.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')
        df_bx_books = pd.read_csv(f'{data_directory}BX-Books.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')  
        df_bx_users = pd.read_csv(f'{data_directory}BX-Users.csv', encoding='unicode_escape', on_bad_lines='skip', sep=';', dtype='unicode')  
        return df_bx_ratings, df_bx_books, df_bx_users
    
    def filter_rows_by_column_value(self, data: pd.DataFrame, column_name: str, allowed_values: Set[str]) -> pd.DataFrame:
        """Filter rows in a DataFrame based on a specific column value.

        Parameters:
            data (pd.DataFrame): DataFrame to be filtered.
            column_name (str): Name of the column to filter on.
            allowed_values (Set[str]): Set of allowed values for the specified column.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered_data = data[data[column_name].isin(allowed_values)]
        return filtered_data

    def get_common_elements(self, dataframes: List[pd.DataFrame], column_names: List[str]) -> Set[str]:
        """Get common elements between two DataFrames based on specified columns.

        Parameters:
            dataframes (List[pd.DataFrame]): List of two DataFrames to compare.
            column_names (List[str]): List of column names from each DataFrame to use for comparison.

        Returns:
            Set[str]: Set of common elements between the two DataFrames based on the specified columns.
        """
        set0 = set(dataframes[0][column_names[0]])
        set1 = set(dataframes[1][column_names[1]])
        return set0.intersection(set1)

    def filter_dataframes(self, dataframes: List[pd.DataFrame], column_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter two DataFrames based on their common elements in a specified column.

        Parameters:
            dataframes (List[pd.DataFrame]): List of two DataFrames to filter.
            column_names (List[str]): List of column names from each DataFrame to use for filtering.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered DataFrames for both inputs.
        """
        intersection_set = self.get_common_elements(dataframes=dataframes, column_names=column_names)
        bx_filtered = self.filter_rows_by_column_value(dataframes[0], 'ISBN', intersection_set)
        gr_filtered = self.filter_rows_by_column_value(dataframes[1], 'isbn', intersection_set)
        return bx_filtered, gr_filtered
    
    def avg_ratings(self, bx_filtered: pd.DataFrame, df_bx_ratings: pd.DataFrame) -> pd.DataFrame:
        """Calculate the average ratings for a DataFrame.

        Parameters:
            bx_filtered (pd.DataFrame): DataFrame containing filtered book information.
            df_bx_ratings (pd.DataFrame): DataFrame containing book ratings.

        Returns:
            pd.DataFrame: DataFrame with the calculated average ratings.
        """
        merged_df = pd.merge(bx_filtered, df_bx_ratings, on="ISBN", how="inner")
        merged_df['Book-Rating'] = merged_df['Book-Rating'].astype(int)
        merged_df = merged_df[['ISBN', 'Book-Rating']]
        merged_df = merged_df.groupby("ISBN")["Book-Rating"].mean().reset_index()

        return merged_df

    def save_dataframe_to_pickle(self, df: pd.DataFrame, target_dir: str, file_name: str) -> None:
        """Save a DataFrame to a .pkl file.

        Parameters:
            df (pd.DataFrame): DataFrame to be saved.
            target_dir (str): Target directory path where the DataFrame will be saved.
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
        """Get the final processed DataFrame.

        Returns:
            pd.DataFrame: Final processed DataFrame.
        """
        df_gr_ratings, df_gr_books = self.read_goodreads()
        df_bx_ratings, df_bx_books, df_bx_users = self.read_bookcrossings()
        bx_filtered, gr_filtered = self.filter_dataframes(dataframes=[df_bx_books, df_gr_ratings], column_names=['ISBN', 'isbn'])

        bx_merged = self.avg_ratings(bx_filtered, df_bx_ratings)
        final_df = pd.merge(gr_filtered, bx_merged, left_on="isbn", right_on="ISBN", how="inner")
        
        self.save_dataframe_to_pickle(final_df[['ISBN', 'authors', 'title', 'original_publication_year']],
                                      target_dir='data/processed/',
                                      file_name='reference_dataframe.pkl'
                                      )

        columns_to_drop = [
            'authors',
            'best_book_id',
            'book_id',
            'goodreads_book_id',
            'image_url',
            'isbn13',
            'ISBN',
            'language_code',
            'original_title',
            'small_image_url',
            'title',
            'work_id'
        ]

        final_df.drop(columns=columns_to_drop, inplace=True)
        
        return final_df
