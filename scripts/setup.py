from scripts.download_data import DownloadData
from scripts.process_data import ProcessData
from scripts.hybrid_recommendation_model import HybridRecommendation
from scripts.hybrid_filter_recommender import HybridFilterRecommender
from scripts.content_filter_recommender import ContentFilterRecommender
import pandas as pd

if __name__ == '__main__':
    # Download Data .csv
    dd = DownloadData()
    dd.download_goodreads_csv()
    dd.download_bookcrossing_csv()

    # Merge Data & Get Final DataFrame
    proc_data = ProcessData()
    df_final = proc_data.get_final_dataframe()

    # Train Hybrid Filter Model
    hr = HybridRecommendation(df_final)
    hr.run_train()

    # Eval Hybrid Filter
    hfr = HybridFilterRecommender(df_final)
    hfr.eval_model()

    # Hybrid Content Filter
    cfr = ContentFilterRecommender(df_final)
    cfr.get_eval_metrics()
