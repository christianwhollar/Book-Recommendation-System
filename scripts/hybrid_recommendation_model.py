from scripts.NNHybridFilteringModule import NNHybridFiltering
import time
import pickle 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class HybridRecommendation():
    """
    Class for building and training a hybrid recommendation system.

    Attributes:
        input_df (pd.DataFrame): Input DataFrame containing the data for the recommendation system.

    Methods:
        __init__(self, input_df: pd.DataFrame)
            Initialize the HybridRecommendation class.

        map_columns_to_int(self, df: pd.DataFrame, column_name: str = 'isbn_gr', mapping_dir: str = None) -> pd.DataFrame
            Map the values in a specified column of the DataFrame to integer values.

        process_df(self) -> pd.DataFrame
            Process the input DataFrame by mapping relevant columns to integer values.

        split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Split the processed DataFrame into training and validation sets.

        prep_dataloaders(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
            Prepare DataLoader objects for training and validation data.

        train_model(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device, num_epochs: int = 5, scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_name: str = 'hybrid_recommender.pkl') -> dict
            Train the hybrid recommendation model.

        run_train(self) -> None
            Prepare data, build the model, and train the hybrid recommendation system.
    """

    def __init__(self, input_df: pd.DataFrame):
        """
        Initialize the HybridRecommendation class.

        Args:
            input_df (pd.DataFrame): Input DataFrame containing the data for the recommendation system.
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

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the processed DataFrame into training and validation sets.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing X, y, X_train, X_val, y_train, y_val.
        """
        df = self.process_df()
        df = df.astype(int)
        X = df.loc[:, ['user_id_gr', 'isbn_gr','rating_bx']]
        y = df.loc[:, ['rating_gr']]
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.2)
        return X, y, X_train, X_val, y_train, y_val
    
    def prep_dataloaders(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Prepare DataLoader objects for training and validation data.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation target data.
            batch_size (int): Batch size for DataLoader.

        Returns:
            tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Tuple containing trainloader and valloader.
        """
        # Convert training and test data to TensorDatasets
        trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(), 
                                torch.from_numpy(np.array(y_train)).float())
        valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(), 
                                torch.from_numpy(np.array(y_val)).float())

        # Create Dataloaders for our training and test data to allow us to iterate over minibatches 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

        return trainloader, valloader

    def train_model(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device, num_epochs: int = 5, scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_name: str = 'hybrid_recommender.pkl') -> dict:
        """
        Train the hybrid recommendation model.

        Args:
            model (nn.Module): Hybrid recommendation model.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Model optimizer.
            dataloaders (dict[str, torch.utils.data.DataLoader]): Dataloader objects for training and validation data.
            device (torch.device): Torch device for training.
            num_epochs (int, optional): Number of training epochs. Default is 5.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.
            model_save_name (str, optional): Name to save the trained model. Default is 'hybrid_recommender.pkl'.

        Returns:
            dict: Dictionary containing the training and validation loss for each epoch.
        """
        model = model.to(device) # Send model to GPU if available
        since = time.time()

        costpaths = {'train':[],'val':[]}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in tqdm(['train', 'val']):
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Get the inputs and labels, and send to GPU if available
                for (inputs,labels) in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels = labels.squeeze(1)
                    # Zero the weight gradients
                    optimizer.zero_grad()

                    # Forward pass to get outputs and calculate loss
                    # Track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model.forward(inputs).view(-1)
                        loss = criterion(outputs, labels)

                        # Backpropagation to get the gradients with respect to each weight
                        # Only if in train
                        if phase == 'train':
                            loss.backward()
                            # Update the weights
                            optimizer.step()

                    # Convert loss into a scalar and add it to running_loss
                    running_loss += np.sqrt(loss.item()) * labels.size(0)

                # Step along learning rate scheduler when in train
                if (phase == 'train') and (scheduler is not None):
                    scheduler.step()

                # Calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                costpaths[phase].append(epoch_loss)
                print('{} loss: {:.4f}'.format(phase, epoch_loss))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        torch.save(model, f'models/{model_save_name}')

        return costpaths
    
    def run_train(self) -> None:
        """
        Prepare data, build the model, and train the hybrid recommendation system.
        """
        X, y, X_train, X_val, y_train, y_val = self.split_data()
        batchsize = 128
        trainloader, valloader = self.prep_dataloaders(X_train, y_train, X_val, y_val, batchsize)

        dataloaders = {'train': trainloader, 'val': valloader}

        n_users = X.loc[:, 'user_id_gr'].max() + 1
        n_isbn = X.loc[:, 'isbn_gr'].max() + 1
        n_bxrating = X.loc[:, 'rating_bx'].max() + 1

        model = NNHybridFiltering(n_users,
                            n_isbn,
                            n_bxrating,
                            embdim_users=50, 
                            embdim_isbn=50, 
                            embdim_bxrating=50,
                            n_activations = 100,
                            rating_range=[0., 4.])
        criterion = nn.MSELoss()
        lr = 0.001
        n_epochs = 14
        wd = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cost_paths = self.train_model(model, criterion, optimizer, dataloaders, device, n_epochs, scheduler=None)
