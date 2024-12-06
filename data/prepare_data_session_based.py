import sys
import os

from src.utils.utils import tokenize_seq, map_feat_id_func, compute_item_popularity_scores, sampling_strategy_wu2019, \
    create_binary_labels_column, exponential_decay, impute_list_with_mean, encode_date_list

# extend the sys.path to fix the import problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_two_up = os.path.dirname(os.path.dirname(current_dir))
sys.path.extend([parent_dir_two_up])
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import gc

import argparse
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ''' 
    Usage: 
    python prepare_data.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='demo', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='The embedding size you want to reduce the initial embeddings')
    parser.add_argument('--embedding_type', type=str, default='roberta',
                        help='The embedding type you want to use')
    parser.add_argument('--neg_sampling', action="store_true", help='Use this flag to perform negative sampling')

    args = vars(parser.parse_args())

    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_size = args['embedding_size']
    embedding_type = args['embedding_type']
    dataset_version = f"Session_based_Ebnerd_{dataset_size}_{embedding_size}"

    dataset_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)

    MAX_SEQ_LEN = 50
    train_path = dataset_path + '/train/'
    dev_path = dataset_path + '/validation/'
    test_path = dataset_path + '/test/'

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
        .select(['article_id', 'published_time', 'last_modified_time', 'premium', 'article_type', 'ner_clusters',
                 'topics', 'category', 'subcategory', 'sentiment_score', 'sentiment_label'])
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
    print("Save news info...")
    os.makedirs(f"{data_folder}/{dataset_version}/", exist_ok=True)
    with open(f"{data_folder}/{dataset_version}//news_info.jsonl", "w") as f:
        f.write(news.write_json(row_oriented=True, pretty=True))

    print("Preprocess behavior data...")


    def join_data(data_path):

        behavior_file = os.path.join(data_path, "behaviors.parquet")
        sample_df = pl.scan_parquet(behavior_file)
        if "test/" in data_path:
            sample_df = (
                sample_df
                .select('session_id', 'impression_id', 'impression_time', 'device_type', 'article_ids_inview',
                        'user_id', 'is_sso_user', 'is_subscriber')
                .rename({"article_ids_inview": "article_id"})
                .explode('article_id')
                .with_columns(
                    pl.lit(0).alias("click")
                )
                .collect()
            )
        else:
            sample_df = (
                sample_df
                .select('session_id', 'impression_id', 'impression_time', 'article_ids_inview',
                        'article_ids_clicked', 'user_id', 'is_sso_user', 'is_subscriber', 'device_type')
                .with_columns(
                    length=pl.col('article_ids_clicked').map_elements(lambda x: len(x)))
                .collect()
            )
            sample_df = (
                sample_df
                .rename({"article_ids_inview": "article_id"})
                .filter(pl.col("length") == 1)
                .explode('article_id')
                .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                .drop(["article_ids_clicked", "length"])
            )

        sample_df = (
            sample_df.sort(["session_id", "impression_time"])
            .groupby("session_id", maintain_order=True).apply(
                lambda group: group.with_columns(
                    pl.concat_list(
                        pl.col("article_id").cumcount().apply(lambda idx: group["article_id"].head(idx).to_list())
                    ).alias("previous_session_clicks")
                )
            )
            .with_columns(
                pl.col('previous_session_clicks').apply(lambda lst: lst[:-1])
            )
        )

        sample_df = tokenize_seq(sample_df, "previous_session_clicks", map_feat_id=False, max_seq_length=MAX_SEQ_LEN)

        sample_df = sample_df.join(news, on='article_id', how="left").drop(
                ["session_id", "impression_time", "published_time", "last_modified_time"])

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

    if args['test']:
        test_df = join_data(test_path)
        print(test_df.head())
        print("Test samples", test_df.shape)
        test_df.write_csv(f"{data_folder}/{dataset_version}/test.csv")
        del test_df
        gc.collect()

    del news2cat, news2type, news2subcat, news2sentiment, news2topic1
    gc.collect()

    print("Preprocess pretrained embeddings...")
    image_emb_path = dataset_path + '/image_embeddings.parquet'
    image_emb_df = pl.read_parquet(image_emb_path)
    pca = PCA(n_components=embedding_size)
    image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
    print("image_embedding.shape", image_emb.shape)
    item_dict = {
        "key": image_emb_df["article_id"].cast(str),
        "value": image_emb
    }
    print(f"Save image_emb_dim{embedding_size}.npz...")
    np.savez(f"{data_folder}/{dataset_version}/image_emb_dim{embedding_size}.npz", **item_dict)
    del image_emb_df, image_emb, item_dict
    gc.collect()

    emb_path = dataset_path + f'/{embedding_type}_vector.parquet'
    emb_df = pl.read_parquet(emb_path)
    emb = pca.fit_transform(np.array(emb_df[emb_df.columns[-1]].to_list()))
    print(f"{embedding_type}_emb.shape", emb.shape)
    item_dict = {
        "key": emb_df["article_id"].cast(str),
        "value": emb
    }
    print(f"Save {embedding_type}_emb_dim{embedding_size}.npz...")
    np.savez(f"{data_folder}/{dataset_version}/{embedding_type}_emb_dim{embedding_size}.npz", **item_dict)
    del emb, item_dict
    gc.collect()

    print("All done.")
