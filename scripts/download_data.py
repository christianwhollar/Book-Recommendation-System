import os
import io
import requests
import zipfile
from typing import Optional

class DownloadData:
    """
    A class for downloading and managing data from various sources.

    This class provides methods to download files (CSV and ZIP) from URLs and save them to specified directories.
    It also includes utilities to extract files from ZIP archives.

    Attributes:
        None

    Methods:
        get_filename(url: str) -> str:
            Get the filename from the given URL.

        download_file(url: str, save_path: str) -> None:
            Download a file from the given URL and save it to the specified path.

        download_csv_from_url(url: str, save_directory: str) -> None:
            Download a CSV file from the given URL and save it to the specified directory.

        download_goodreads_csv(data_directory: str = 'data/raw/goodreads/') -> None:
            Download Goodreads CSV files from pre-defined URLs and save them to the specified directory.

        download_and_extract_zip(zip_url: str, extract_directory: str) -> None:
            Download a zip file from the given URL, extract its contents, and save them to the target directory.

        download_bookcrossing_csv() -> None:
            Download BookCrossing CSV files and save them to the specified directory.
    """
    def __init__(self) -> None:
        pass

    def get_filename(self, url: str) -> str:
        """
        Get the filename from the given URL.

        Parameters:
            url (str): The URL from which to extract the filename.

        Returns:
            str: The extracted filename from the URL.
        """
        return os.path.basename(url)

    def download_file(self, url: str, save_path: str) -> Optional[bool]:
        """
        Download a file from the given URL and save it to the specified path.

        Parameters:
            url (str): The URL of the file to download.
            save_path (str): The path where the downloaded file should be saved.

        Returns:
            bool or None: True if the download was successful, False otherwise.
                         None if there was an error during the download.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx errors

            with open(save_path, 'wb') as file:
                file.write(response.content)

            print(f"File downloaded successfully: {save_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Failed to download file from {url}. Error: {e}")
            return False

    def download_csv_from_url(self, url: str, save_directory: str) -> Optional[bool]:
        """
        Download a CSV file from the given URL and save it to the specified directory.

        Parameters:
            url (str): The URL of the CSV file to download.
            save_directory (str): The directory where the downloaded CSV file should be saved.

        Returns:
            bool or None: True if the download was successful, False otherwise.
                         None if there was an error during the download.
        """
        filename = self.get_filename(url)
        save_path = os.path.join(save_directory, filename)

        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        return self.download_file(url, save_path)

    def download_goodreads_csv(self, data_directory: str = 'data/raw/goodreads/') -> None:
        """
        Download Goodreads CSV files from pre-defined URLs and save them to the specified directory.

        Parameters:
            data_directory (str, optional): The directory where the downloaded CSV files should be saved.
                                            Defaults to 'data/raw/goodreads/'.
        """
        raw_links = [
            'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv',
            'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv'
        ]

        for link in raw_links:
            self.download_csv_from_url(link, data_directory)

    def download_and_extract_zip(self, zip_url: str, extract_directory: str) -> Optional[bool]:
        """
        Download a zip file from the given URL, extract its contents, and save them to the target directory.

        Parameters:
            zip_url (str): The URL of the zip file to download.
            extract_directory (str): The directory where the extracted files should be saved.

        Returns:
            bool or None: True if the download and extraction were successful, False otherwise.
                         None if there was an error during the download or extraction.
        """
        try:
            response = requests.get(zip_url)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx errors

            # Create the target directory if it doesn't exist
            if not os.path.exists(extract_directory):
                os.makedirs(extract_directory)

            # Load the zip data into a ZipFile object
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as z:
                # Extract all the files from the zip archive to the target directory
                z.extractall(extract_directory)

            print(f"Zip download and extraction completed successfully: {extract_directory}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Failed to download the zip file from {zip_url}. Error: {e}")
            return False
        except zipfile.BadZipFile as e:
            print(f"Failed to extract the zip file. Error: {e}")
            return False

    def download_bookcrossing_csv(self) -> None:
        """
        Download BookCrossing CSV files and save them to the specified directory.

        Note: This method performs the download and extraction without returning a value.
        """
        zip_url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
        target_directory = 'data/raw/bookcrossing/'
        self.download_and_extract_zip(zip_url, target_directory)
