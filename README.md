# Book-Recommendation-System
## Project
Book recommendation system using a deep learning approach (Hybrid Filtering Model) and non-deep learning approach (Content Filtering Model).

Models:
* Hybrid Filter
* Content Filter

Data:
* Goodreads: https://github.com/zygmuntz/goodbooks-10k
* BookCrossing: http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip

### Business News Summarization App Instructions
To just run the Streamlit App, do the following:
1. Create a new conda environment and activate it: 
    ```
    conda create --name cv python=3.10
    conda activate cv
    ```
2. Install python package requirements:*
    ```
    pip install -r requirements.txt 
    ```
3. Run the streamlit app:
    ```
    !streamlit run app.py
    ```

#### Repository Structure
```
├── README.md                          <- description of project and how to set up and run it
├── requirements.txt                   <- requirements file to document dependencies
├── setup.py                           <- script to set up project (get data, train models, evaluate models)
├── main.ipynb                         <- main script/notebook to run model, streamlit web application from colab
├── scripts                            <- directory for pipeline scripts or utility scripts
    ├── setup.py                       <- full model pipeline
    ├── app.py                         <- Streamlit app
    ├── download_data.py               <- download csv data for good reads and book crossing
    ├── process_data.py                <- combine csv data into final dataframe for model
    ├── hybrid_recommendation_model.py <- train hybrid filter model
    ├── NNHybridFilteringModeul.py     <- torch module architecture for hybrid filter
    ├── hybrid_filter_recommender.py   <- get predictions using hybrid filter
    ├── content_filter_recommender.py  <- get predictions using content filter
├── models                             <- directory for trained models
    ├── hybrid_recommender.pkl         <- torch pkl file containing trained Hybrid Filter
├── data                               <- directory for project data
    ├── raw                            <- directory to store raw csv downloads
    ├── processed                      <- directory to store processed dataframes
├── evals                              <- directory for project data
    ├── content_filter_eval.json       <- json file for content filter eval
    ├── hybrid_filter_eval.json        <- json file for hybrid filter eval
├── .gitignore                         <- git ignore file
```

## HybridFilter Model Instructions
```
# Train Hybrid Filter Model
hr = HybridRecommendation(df_final)
hr.run_train()

# run_train will run the following methods:
# map_columns_to_int(self, df: pd.DataFrame, column_name: str = 'isbn_gr', mapping_dir: str = None) -> pd.DataFrame
#    Map the values in a specified column of the DataFrame to integer values.

# process_df(self) -> pd.DataFrame
#    Process the input DataFrame by mapping relevant columns to integer values.

# split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
#    Split the processed DataFrame into training and validation sets.

# prep_dataloaders(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
#    Prepare DataLoader objects for training and validation data.

# train_model(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device, num_epochs: int = 5, scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
# model_save_name: str = 'hybrid_recommender.pkl') -> dict
#    Train the hybrid recommendation model.

# run_train(self) -> None
#    Prepare data, build the model, and train the hybrid recommendation system.
```
## HybridFilter Interact & Eval Instructions
```
# Eval Hybrid Filter
hfr = HybridFilterRecommender(df_final)
hfr.eval_model()

# eval_model will run the following methods:
# __init__(self, input_df: pd.DataFrame)
#     Initialize the HybridFilterRecommender class.

# map_columns_to_int(self, df: pd.DataFrame, column_name: str = 'isbn_gr', mapping_dir: str = None) -> pd.DataFrame
#     Map the values in a specified column of the DataFrame to integer values.

# process_df(self) -> pd.DataFrame
#     Process the input DataFrame by mapping relevant columns to integer values.

# predict_rating(self, model: nn.Module, userid: int, isbn: int, bxrating: float, device: torch.device) -> torch.Tensor
#     Predict the rating for a given user, item, and book rating.

# generate_recommendations(self, df: pd.DataFrame, model: nn.Module, userid: int, n_recs: int, device: torch.device) -> np.ndarray
#     Generate book recommendations for a given user.

# load_isbn_map(self) -> dict[int, int]
#     Load the mapping dictionary for ISBN values.

# load_isbn_df(self) -> pd.DataFrame
#     Load the DataFrame containing book reference information.

# get_best_user(self, selected_books:list = ["Atonement"]) -> int
#     Get the user who read most of the selected books.

# get_default_books(self, selected_books:list = ["Atonement"], model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[list[str], list[str], list[str]]
#     Get default book recommendations using the trained hybrid recommendation model.

# eval_model(self, model_dir: str = 'models/hybrid_recommender.pkl') -> tuple[float, float, float, float]
#     Evaluate the model's performance using mean squared error (MSE), accuracy, recall, and precision metrics.
```
## ContentFilter Model, Interact & Eval Instructions
```
# Hybrid Content Filter
cfr = ContentFilterRecommender()
cfr.get_eval_metrics()

# get_eval_metrics will run the following methods:
# get_genres_for_isbn(self, isbn: str) -> list[str]
#     Get the genres associated with a given ISBN.

# get_genre_vector(self) -> pd.DataFrame
#     Get the vectorized genre representation for ISBN values.

# get_books_df(self) -> pd.DataFrame
#     Get a DataFrame containing unique book ISBNs.

# load_isbn_map(self) -> dict[int, int]
#     Load the mapping dictionary for ISBN values.

# load_isbn_df(self) -> pd.DataFrame
#     Load the DataFrame containing book reference information.

# get_cs_matrix(self) -> pd.DataFrame
#     Get the cosine similarity matrix based on book genres.

# get_split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
#     Split the data into training and validation sets.

# generate_recommendations(self, user: int = 1) -> list[str]
#     Generate book recommendations for a given user based on content similarity.

# get_aty_recs(self, user: int = 1) -> tuple[list[str], list[str], list[str]]
#     Get authors, titles, and publication years of recommended books for a user.

# get_eval_metrics(self) -> tuple[float, float, float, float]
#     Calculate evaluation metrics (MSE, accuracy, recall, precision) for the content-based recommender.
"""