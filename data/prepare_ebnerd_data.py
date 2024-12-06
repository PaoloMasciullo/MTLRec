import sys
import os

# extend the sys.path to fix the import problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir])
from src.utils.utils import (tokenize_seq, map_feat_id_func, impute_list_with_mean, hours_date_list, weekday_date_list,
                             bin_list_values, create_reading_patterns, compute_mean)
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import gc
import argparse
import warnings
from src.utils.torch_utils import seed_everything

warnings.filterwarnings("ignore")

SEED = 42
seed_everything(SEED)

if __name__ == '__main__':
    ''' 
    Usage: 
    python prepare_data.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='small', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='The embedding size you want to reduce the initial embeddings')
    parser.add_argument('--embedding_type', type=str, default='roberta',
                        help='The embedding type you want to use')
    parser.add_argument('--neg_sampling', action="store_true", help='Use this flag to perform negative sampling')

    args = vars(parser.parse_args())

    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_size = args['embedding_size']
    embedding_type = args['embedding_type']
    dataset_version = f"Preprocessed_Ebnerd_{dataset_size}"

    dataset_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)

    MAX_SEQ_LEN = 150
    train_path = dataset_path + '/train/'
    dev_path = dataset_path + '/validation2/'
    test_path = dataset_path + '/test/'
    test2_path = dataset_path + '/test2/'

    print("Preprocess news info...")
    train_news_file = os.path.join(train_path, "articles.parquet")
    train_news = pl.scan_parquet(train_news_file)
    test_news_file = os.path.join(test_path, "articles.parquet")
    test_news = pl.scan_parquet(test_news_file)

    news = pl.concat([train_news, test_news])

    del train_news, test_news
    gc.collect()

    news = (
        news
        .unique(subset=['article_id'])
        .fill_null("")
        .with_columns(
            article_len=pl.col('title').map_elements(lambda el: len(el)) + pl.col('subtitle').map_elements(
                lambda el: len(el)) + pl.col('body').map_elements(lambda el: len(el)))
        .select(['article_id', 'published_time', 'last_modified_time', 'premium', 'article_type', 'ner_clusters',
                 'topics', 'category', 'subcategory', 'sentiment_score', 'sentiment_label', 'article_len'])
        .with_columns(subcat1=pl.col('subcategory').apply(lambda x: str(x[0]) if len(x) > 0 else ""))
        .collect()
    )

    news2cat = dict(zip(news["article_id"].cast(str), news["category"].cast(str)))
    news2subcat = dict(zip(news["article_id"].cast(str), news["subcat1"].cast(str)))
    news = tokenize_seq(news, 'ner_clusters', map_feat_id=True)
    news = tokenize_seq(news, 'topics', map_feat_id=True)
    news = tokenize_seq(news, 'subcategory', map_feat_id=False)
    news = map_feat_id_func(news, "sentiment_label")
    news = map_feat_id_func(news, "article_type")
    news2sentiment = dict(zip(news["article_id"].cast(str), news["sentiment_label"]))
    news2type = dict(zip(news["article_id"].cast(str), news["article_type"]))

    news = (
        news
        .with_columns(topic1=pl.col('topics').apply(lambda x: str(x.split("^")[0]) if len(x.split("^")) > 0 else ""))
    )
    news2topic1 = dict(zip(news["article_id"].cast(str), news["topic1"].cast(str)))

    print(news.head())
    print("Preprocess behavior data...")


    def join_data(data_path):
        history_file = os.path.join(data_path, "history.parquet")
        history_df = pl.scan_parquet(history_file)
        history_df = (
            history_df
            .rename({"article_id_fixed": "hist_id",
                     "read_time_fixed": "hist_read_time",
                     "scroll_percentage_fixed": "hist_scroll_percent"})
            # missing imputation of hist_scroll_percent, hist_read_time, hist_time
            .with_columns(
                pl.col("hist_scroll_percent").apply(impute_list_with_mean),
                pl.col("hist_read_time").apply(impute_list_with_mean),
                hist_hour=pl.col('impression_time_fixed').apply(hours_date_list).cast(pl.List(pl.Int32)),
                hist_weekday=pl.col('impression_time_fixed').apply(weekday_date_list).cast(pl.List(pl.Int32)),
            )
            .drop("impression_time_fixed")
        )

        history_df = (
            history_df
            .with_columns(
                user_mean_scroll_percentage=pl.col('hist_scroll_percent').map_elements(compute_mean),
                user_mean_read_time=pl.col('hist_read_time').map_elements(compute_mean),
            )
            .drop_nulls()
            .with_columns(
                pl.col('user_mean_scroll_percentage').cast(pl.Float32),
                pl.col('user_mean_read_time').cast(pl.Float32),
            )
        )

        news_reading_df = (
            history_df.collect()
            .explode(['hist_id', 'hist_scroll_percent', 'hist_read_time'])
            .select('hist_id', 'hist_scroll_percent', 'hist_read_time')
            .rename({'hist_id': 'article_id'})
            .drop_nulls()
            .join(
                news,
                on='article_id',
                how='left'
            )
        )

        news_reading_df = (
            news_reading_df
            .group_by('category')
            .agg(['hist_scroll_percent', 'hist_read_time'])
        )

        news_reading_df = (
            news_reading_df
            .with_columns(
                category_mean_scroll_percentage=pl.col('hist_scroll_percent').map_elements(compute_mean),
                category_mean_read_time=pl.col('hist_read_time').map_elements(compute_mean),
            )
            .drop_nulls()
            .with_columns(
                pl.col('category_mean_scroll_percentage').cast(pl.Float32),
                pl.col('category_mean_read_time').cast(pl.Float32),
            )
            .select('category', 'category_mean_scroll_percentage', 'category_mean_read_time')
        )

        # discretize read_time and scroll_percent
        # Number of bins
        tmp = history_df.select("hist_scroll_percent", "hist_read_time").explode("hist_scroll_percent",
                                                                                 "hist_read_time").collect().max()
        max_scroll, max_read = tmp["hist_scroll_percent"][0], tmp["hist_read_time"][0]
        del tmp
        gc.collect()
        bins_read_time = np.linspace(0, max_read, 51)  # 50 bins from 0 to max_read
        bins_scroll_percent = np.linspace(0, max_scroll, 51)  # 50 bins from 0 to max_scroll

        # Apply binning to each list element
        history_df = history_df.with_columns([
            pl.col("hist_read_time").apply(lambda x: bin_list_values(x, bins_read_time)),
            pl.col("hist_scroll_percent").apply(lambda x: bin_list_values(x, bins_scroll_percent)),
        ])

        # Extract unique bin intervals
        unique_bins_read_time = history_df.select("hist_read_time").explode("hist_read_time").collect()[
            "hist_read_time"].unique()
        unique_bins_scroll_percent = history_df.select("hist_scroll_percent").explode("hist_scroll_percent").collect()[
            "hist_scroll_percent"].unique()

        # Create a mapping from bin intervals to integers
        read_time_bin_mapping = {bin_interval: idx for idx, bin_interval in enumerate(unique_bins_read_time.to_list())}
        scroll_percent_bin_mapping = {bin_interval: idx for idx, bin_interval in
                                      enumerate(unique_bins_scroll_percent.to_list())}

        del unique_bins_read_time, unique_bins_scroll_percent
        gc.collect()

        # Apply the mapping to encode the bins
        history_df = history_df.with_columns([
            pl.col("hist_read_time").apply(lambda bins: [read_time_bin_mapping[b] for b in bins]),
            pl.col("hist_scroll_percent").apply(lambda bins: [scroll_percent_bin_mapping[b] for b in bins]),
        ])

        history_df = tokenize_seq(history_df, 'hist_id', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_read_time', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_scroll_percent', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_hour', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
        history_df = tokenize_seq(history_df, 'hist_weekday', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)

        history_df = history_df.with_columns(
            pl.col("hist_id").apply(lambda x: "^".join([news2cat.get(i, "") for i in x.split("^")])).alias("hist_cat"),
            pl.col("hist_id").apply(lambda x: "^".join([news2subcat.get(i, "") for i in x.split("^")])).alias(
                "hist_subcat1"),
            pl.col("hist_id").apply(lambda x: "^".join([news2topic1.get(i, "") for i in x.split("^")])).alias(
                "hist_topic1"),
            pl.col("hist_id").apply(lambda x: "^".join([news2sentiment.get(i, "") for i in x.split("^")])).alias(
                "hist_sentiment"),
            pl.col("hist_id").apply(lambda x: "^".join([news2type.get(i, "") for i in x.split("^")])).alias(
                "hist_type"),
        ).collect()
        del read_time_bin_mapping, scroll_percent_bin_mapping
        gc.collect()

        behavior_file = os.path.join(data_path, "behaviors.parquet")
        sample_df = pl.scan_parquet(behavior_file)
        if "test/" in data_path:
            sample_df = (
                sample_df
                .select('impression_id', 'session_id' 'impression_time', 'device_type', 'article_ids_inview',
                        'user_id', 'is_sso_user', 'is_subscriber')
                .rename({"article_ids_inview": "article_id"})
                .explode('article_id')
                .with_columns(
                    pl.lit(None).alias("trigger_id"),
                    pl.lit(0).alias("click")
                )
                .collect()
            )
        else:
            sample_df = (
                sample_df
                .select('impression_id', 'session_id', 'article_id', 'impression_time', 'device_type',
                        'article_ids_inview',
                        'article_ids_clicked', 'user_id', 'is_sso_user', 'is_subscriber', 'next_read_time',
                        'next_scroll_percentage')
                .with_columns(
                    length=pl.col('article_ids_clicked').map_elements(lambda x: len(x)))
                .collect()
            )
            sample_df = (
                sample_df.rename({"article_id": "trigger_id"})
                .rename({"article_ids_inview": "article_id"})
                .filter(pl.col("length") == 1)  # filter out impression where number of clicked articles are more than 1
                .explode('article_id')
                .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                .drop(["article_ids_clicked", "length"])
            )

        sample_df = (
            sample_df
            .with_columns(
                pl.when(pl.col('click') == 0).then(0).otherwise(pl.col('next_read_time')).alias('next_read_time'),
                pl.when(pl.col('click') == 0).then(0).otherwise(pl.col('next_scroll_percentage')).alias(
                    'next_scroll_percentage')
            )
        )

        sample_df = (
            sample_df
            .with_columns(
                fully_read=pl.when(pl.col("next_scroll_percentage") == 100).then(1).otherwise(0),
                over_half_read=pl.when(pl.col("next_scroll_percentage") >= 50).then(1).otherwise(0)
            )
        )

        sample_df = (
            sample_df
            .join(news, on='article_id', how="left")
            .join(history_df, on='user_id', how="left")
            .join(news_reading_df, on='category', how='left')
        )

        # Ensure the DataFrame is sorted by `impression_time` before performing any time-based calculations
        sample_df = sample_df.sort("impression_time")

        sample_df = (
            sample_df
            .with_columns(
                publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
                # days since publication
                hours_since_publish=(pl.col("impression_time") - pl.col("published_time")).dt.hours()

            )
        )
        # ============ Global Non-Personalized Interests ============
        # Cumulative Article Exposure Count
        sample_df = sample_df.with_columns(
            pl.col("impression_time")
            .cumcount()
            .over("article_id")
            .alias("cumulative_exposure_count")
        )

        # Relative Popularity Change in First 48 Hours
        sample_df = sample_df.with_columns(
            within_48_hours=(pl.col("hours_since_publish") <= 48)
        )

        # Using the cumulative exposure count, only within the first 48 hours
        sample_df = sample_df.with_columns(
            pl.when(pl.col("within_48_hours"))
            .then(pl.col("cumulative_exposure_count"))
            .otherwise(None)
            .alias("relative_popularity_change_48h")
        ).drop("within_48_hours")

        # ============ Hourly Real-Time Non-Personalized Interests ============
        # Hourly Article Exposure Count
        sample_df = sample_df.with_columns(
            pl.col("impression_time").dt.truncate("1h").alias("hour")
        )

        # Ensure correct counting for hourly exposure count; ensure `impression_time` is sorted within each hour
        sample_df = sample_df.with_columns(
            pl.col("impression_time")
            .count()
            .over(["article_id", "hour"])
            .alias("hourly_exposure_count")
        ).drop("hour")

        # ============ Session-Level Personalized Interests ============
        # Frequency of Article Appearance in Session
        sample_df = sample_df.with_columns(
            pl.col("article_id").count().over(["session_id", "impression_time"]).alias("article_frequency_in_session")
        )

        # Time Difference Between Impressions in Session
        sample_df = sample_df.with_columns(
            pl.col("impression_time").diff().over("session_id").alias("time_diff_in_session")
        )
        sample_df = sample_df.with_columns(
            pl.col("time_diff_in_session").fill_null(0)
        )

        # Deviation of Article Popularity within Session
        # Sort by `session_id` and `impression_time` before calculating `cumulative_exposure_count`
        sample_df = sample_df.sort(by=["session_id", "impression_time"])

        # Calculate cumulative exposure count within session, but only up to the current row
        sample_df = sample_df.with_columns(
            pl.col("impression_time")
            .cumcount()
            .over(["session_id", "impression_time"])
            .alias("cumulative_exposure_count_session")
        )

        # Calculate session mean exposure, and ensure it only considers previous impressions within the session
        sample_df = sample_df.with_columns(
            pl.col("cumulative_exposure_count_session").mean().over(["session_id", "impression_time"]).alias(
                "session_mean_exposure")
        )

        # Calculate deviation from session mean
        sample_df = sample_df.with_columns(
            (pl.col("cumulative_exposure_count_session") - pl.col("session_mean_exposure"))
            .alias("deviation_from_session_mean")
        ).drop(["cumulative_exposure_count_session", "session_mean_exposure"])

        # ============ Impression-Level Personalized Interests ============
        # Number of Articles Browsed by User
        # Sort the data by user and impression time to ensure we're considering past interactions only
        sample_df = sample_df.sort(by=["user_id", "impression_time"])

        sample_df = sample_df.with_columns(
            pl.col("article_id").n_unique().over(["user_id", "impression_time"]).alias("articles_browsed_by_user")
        )

        # List of numerical columns you want to discretize
        numerical_columns = [
            'publish_days', 'hours_since_publish', 'cumulative_exposure_count',
            'relative_popularity_change_48h', 'hourly_exposure_count',
            'article_frequency_in_session', 'time_diff_in_session',
            'deviation_from_session_mean', 'articles_browsed_by_user',
            'user_mean_scroll_percentage', 'user_mean_read_time', 'article_len',
            'category_mean_scroll_percentage', 'category_mean_read_time'
        ]

        # Discretizing each numerical feature into 50 bins using equal-frequency binning
        for col in numerical_columns:
            sample_df = sample_df.with_columns(
                pl.col(col)
                .cast(pl.Float64)  # Ensuring the column is treated as numeric
                .qcut(50, labels=[str(i) for i in range(50)], allow_duplicates=True)
            )

        if "test" not in data_path and args['neg_sampling']:
            # Filter negatives and positives
            negatives = sample_df.filter(pl.col("click") == 0)
            positives = sample_df.filter(pl.col("click") == 1)
            # Sample negatives with replacement
            sampled_negatives = negatives.groupby("impression_id").apply(
                lambda group: group.sample(n=4, with_replacement=False)
            )
            # Re-assemble positives and negatives
            sample_df = pl.concat([positives, sampled_negatives])

        # remove raws where next_read_time or next_scroll_percentage have null value
        sample_df = sample_df.drop_nulls(subset=["next_read_time", "next_scroll_percentage"])
        # Filter out impressions where all samples have click=0
        sample_df = sample_df.with_columns(
            total_clicks=pl.col("click").sum().over("impression_id")
        )
        sample_df = sample_df.filter(pl.col("total_clicks") > 0).drop("total_clicks")
        #sample_df = sample_df.filter(pl.col("click") == 1)
        sample_df = sample_df.select(pl.all().shuffle(SEED))
        print(sample_df.columns)
        return sample_df


    if os.path.isdir(f"{data_folder}/{dataset_version}"):
        print(f"Folder '{data_folder}/{dataset_version}' exists.")
    else:
        os.makedirs(f"{data_folder}/{dataset_version}")
        print(f"Folder '{data_folder}/{dataset_version}' has been created.")

    train_df = join_data(train_path)
    print(train_df.head())
    print("Train samples", train_df.shape)
    train_df.write_csv(f"{data_folder}/{dataset_version}/train.csv")
    del train_df
    gc.collect()

    valid_df = join_data(dev_path)
    print(valid_df.head())
    print("Validation samples", valid_df.shape)
    valid_df.write_csv(f"{data_folder}/{dataset_version}/valid.csv")
    del valid_df
    gc.collect()

    test2_df = join_data(test2_path)
    print(test2_df.head())
    print("Test2 samples", test2_df.shape)
    test2_df.write_csv(f"{data_folder}/{dataset_version}/test.csv")
    del test2_df
    gc.collect()

    if args['test']:
        test_df = join_data(test_path)
        print(test_df.head())
        print("Test samples", test_df.shape)
        test_df.write_csv(f"{data_folder}/{dataset_version}/test.csv")
        del test_df
        gc.collect()

    del news2cat, news2type, news2subcat, news2sentiment, news2topic1
    gc.collect()

    # print("Preprocess pretrained embeddings...")
    # image_emb_path = dataset_path + '/image_embeddings.parquet'
    # image_emb_df = pl.read_parquet(image_emb_path)
    # pca = PCA(n_components=embedding_size)
    # image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
    # print("image_embedding.shape", image_emb.shape)
    # item_dict = {
    #     "key": image_emb_df["article_id"].cast(str),
    #     "value": image_emb
    # }
    # print(f"Save image_emb_dim{embedding_size}.npz...")
    # np.savez(f"{data_folder}/{dataset_version}/image_emb_dim{embedding_size}.npz", **item_dict)
    # del image_emb_df, image_emb, item_dict
    # gc.collect()
    #
    # emb_path = dataset_path + f'/{embedding_type}_vector.parquet'
    # emb_df = pl.read_parquet(emb_path)
    # emb = pca.fit_transform(np.array(emb_df[emb_df.columns[-1]].to_list()))
    # print(f"{embedding_type}_emb.shape", emb.shape)
    # item_dict = {
    #     "key": emb_df["article_id"].cast(str),
    #     "value": emb
    # }
    # print(f"Save {embedding_type}_emb_dim{embedding_size}.npz...")
    # np.savez(f"{data_folder}/{dataset_version}/{embedding_type}_emb_dim{embedding_size}.npz", **item_dict)
    # del emb, item_dict
    # gc.collect()

    print("All done.")
