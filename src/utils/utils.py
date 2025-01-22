from typing import Iterable
import numpy as np
import polars as pl
import torch
from pandas.core.common import flatten
from collections import Counter

from src.utils.constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_LABELS_COL
from src.utils.polars_utils import _check_columns_in_df, shuffle_list_column
from src.utils.python_utils import generate_unique_name


def map_feat_id_func(df, column, seq_feat=False):
    feat_set = set(flatten(df[column].to_list()))
    map_dict = dict(zip(list(feat_set), range(1, 1 + len(feat_set))))
    if seq_feat:
        df = df.with_columns(pl.col(column).apply(lambda x: [map_dict.get(i, 0) for i in x]))
    else:
        df = df.with_columns(pl.col(column).apply(lambda x: map_dict.get(x, 0)).cast(str))
    return df


def tokenize_seq(df, column, map_feat_id=True, max_seq_length=5, sep="^"):
    df = df.with_columns(pl.col(column).apply(lambda x: x[-max_seq_length:]))
    if map_feat_id:
        df = map_feat_id_func(df, column, seq_feat=True)
    df = df.with_columns(pl.col(column).apply(lambda x: f"{sep}".join(str(i) for i in x)))
    return df


def padding_seq(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    Custom implementation of pad_sequences similar to Keras' pad_sequences.

    Parameters:
    - sequences (list of lists): A list of sequences (e.g., list of lists of integers).
    - maxlen (int, optional): The maximum length of the sequences. Sequences longer than `maxlen`
      will be truncated and sequences shorter will be padded. If None, the longest sequence will be used.
    - dtype (str, optional): The data type to return. Default is 'int32'.
    - padding (str, optional): 'pre' or 'post'. Whether to pad before or after each sequence.
    - truncating (str, optional): 'pre' or 'post'. Whether to truncate before or after each sequence.
    - value (float or int, optional): The padding value. Default is 0.

    Returns:
    - np.ndarray: The padded sequences.
    """

    # Check that sequences are non-empty
    if len(sequences) == 0:
        return np.array([], dtype=dtype)

    # Determine the maximum length if not provided
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # Create the padded sequences
    padded_sequences = []

    for seq in sequences:
        # Truncate sequences that are longer than maxlen
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]  # Keep the last 'maxlen' tokens
            elif truncating == 'post':
                seq = seq[:maxlen]  # Keep the first 'maxlen' tokens

        # Pad sequences that are shorter than maxlen
        if len(seq) < maxlen:
            pad_length = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * pad_length + seq  # Pad at the beginning
            elif padding == 'post':
                seq = seq + [value] * pad_length  # Pad at the end

        # Add the sequence to the list of padded sequences
        padded_sequences.append(seq)

    # Convert the list of padded sequences to a numpy array
    return np.array(padded_sequences, dtype=dtype)


def compute_item_popularity_scores(R: Iterable[np.ndarray]) -> dict[str, float]:
    """Compute popularity scores for items based on their occurrence in user interactions.

    This function calculates the popularity score of each item as the fraction of users who have interacted with that item.
    The popularity score, p_i, for an item is defined as the number of users who have interacted with the item divided by the
    total number of users.

    Formula:
        p_i = | {u ∈ U}, r_ui != Ø | / |U|

    where p_i is the popularity score of an item, U is the total number of users, and r_ui is the interaction of user u with item i (non-zero
    interaction implies the user has seen the item).

    Note:
        Each entry can only have the same item ones.

    Args:
        R (Iterable[np.ndarray]): An iterable of numpy arrays, where each array represents the items interacted with by a single user.
            Each element in the array should be a string identifier for an item.

    Returns:
        dict[str, float]: A dictionary where keys are item identifiers and values are their corresponding popularity scores (as floats).

    Examples:
    >>> R = [
            np.array(["item1", "item2", "item3"]),
            np.array(["item1", "item3"]),
            np.array(["item1", "item4"]),
        ]
    >>> print(compute_item_popularity_scores(R))
        {'item1': 1.0, 'item2': 0.3333333333333333, 'item3': 0.6666666666666666, 'item4': 0.3333333333333333}
    """
    U = len(R)
    R_flatten = np.concatenate(R)
    item_counts = Counter(R_flatten)
    return {item: (r_ui / U) for item, r_ui in item_counts.items()}


def sampling_strategy_wu2019(
        df: pl.DataFrame,
        npratio: int,
        shuffle: bool = False,
        with_replacement: bool = True,
        seed: int = None,
        inview_col: str = "article_id",
        clicked_col: str = "article_ids_clicked",
) -> pl.DataFrame:
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(
            remove_positives_from_inview, inview_col=inview_col, clicked_col=clicked_col
        )
        # Step 2: Explode the DataFrame based on the clicked articles column
        .explode(clicked_col)
        # Step 3: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids,
            n=npratio,
            with_replacement=with_replacement,
            seed=seed,
            inview_col=inview_col,
        )
        # Step 4: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([inview_col, clicked_col]))
        # Step 5: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(inview_col).list.tail(1).alias(clicked_col))
    )
    if shuffle:
        df = shuffle_list_column(df, inview_col, seed)
    return df


def remove_positives_from_inview(
        df: pl.DataFrame,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Examples:
    >>> from src.utils.constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                DEFAULT_CLICKED_ARTICLES_COL: [
                    [1, 2],
                    [1],
                    [3],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_inview(df)
        shape: (3, 3)
        ┌─────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ article_ids_inview │
        │ ---     ┆ ---                 ┆ ---                │
        │ i64     ┆ list[i64]           ┆ list[i64]          │
        ╞═════════╪═════════════════════╪════════════════════╡
        │ 1       ┆ [1, 2]              ┆ [3]                │
        │ 1       ┆ [1]                 ┆ [2, 3]             │
        │ 2       ┆ [3]                 ┆ [1, 2]             │
        └─────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(df[inview_col].to_list(), df[clicked_col].to_list())
    )
    return df.with_columns(pl.Series(inview_col, list(negative_article_ids)))


def sample_article_ids(
        df: pl.DataFrame,
        n: int,
        with_replacement: bool = False,
        seed: int = None,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Randomly sample article IDs from each row of a DataFrame with or without replacement

    Args:
        df: A polars DataFrame containing the column of article IDs to be sampled.
        n: The number of article IDs to sample from each list.
        with_replacement: A boolean indicating whether to sample with replacement.
            Default is False.
        seed: An optional seed to use for the random number generator.

    Returns:
        A new polars DataFrame with the same columns as `df`, but with the article
        IDs in the specified column replaced by a list of `n` sampled article IDs.

    Examples:
    >>> from src.utils.constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "clicked": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["G", "H", "I"],
                ],
                "col" : [
                    ["h"],
                    ["e"],
                    ["y"]
                ]
            }
        )
    >>> print(df)
        shape: (3, 3)
        ┌──────────────────┬─────────────────┬───────────┐
        │ list_destination ┆ article_ids     ┆ col       │
        │ ---              ┆ ---             ┆ ---       │
        │ list[i64]        ┆ list[str]       ┆ list[str] │
        ╞══════════════════╪═════════════════╪═══════════╡
        │ [1]              ┆ ["A", "B", "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "E", "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "H", "I"] ┆ ["y"]     │
        └──────────────────┴─────────────────┴───────────┘
    >>> sample_article_ids(df, n=2, seed=42)
        shape: (3, 3)
        ┌──────────────────┬─────────────┬───────────┐
        │ list_destination ┆ article_ids ┆ col       │
        │ ---              ┆ ---         ┆ ---       │
        │ list[i64]        ┆ list[str]   ┆ list[str] │
        ╞══════════════════╪═════════════╪═══════════╡
        │ [1]              ┆ ["A", "C"]  ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "F"]  ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "I"]  ┆ ["y"]     │
        └──────────────────┴─────────────┴───────────┘
    >>> sample_article_ids(df.lazy(), n=4, with_replacement=True, seed=42).collect()
        shape: (3, 3)
        ┌──────────────────┬───────────────────┬───────────┐
        │ list_destination ┆ article_ids       ┆ col       │
        │ ---              ┆ ---               ┆ ---       │
        │ list[i64]        ┆ list[str]         ┆ list[str] │
        ╞══════════════════╪═══════════════════╪═══════════╡
        │ [1]              ┆ ["A", "A", … "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "D", … "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "G", … "I"] ┆ ["y"]     │
        └──────────────────┴───────────────────┴───────────┘
    """
    _check_columns_in_df(df, [inview_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.explode(inview_col)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(inview_col).sample(n=n, with_replacement=with_replacement, seed=seed)
        )
    )
    return (
        df.drop(inview_col)
        .join(df_, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )


def create_binary_labels_column(
        df: pl.DataFrame,
        shuffle: bool = True,
        seed: int = None,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
        inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
        label_col: str = DEFAULT_LABELS_COL,
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Examples:
    >>> from src.utils.constants import (
            DEFAULT_CLICKED_ARTICLES_COL,
            DEFAULT_INVIEW_ARTICLES_COL,
            DEFAULT_LABELS_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [4, 5, 6], [7, 8]],
                DEFAULT_CLICKED_ARTICLES_COL: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [1, 2, 3]          ┆ [2, 3, 4]           ┆ [0, 1, 1] │
        │ [4, 5, 6]          ┆ [3, 5]              ┆ [0, 1, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [3, 1, 2]          ┆ [2, 3, 4]           ┆ [1, 0, 1] │
        │ [5, 6, 4]          ┆ [3, 5]              ┆ [1, 0, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[DEFAULT_LABELS_COL].to_list() == [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[DEFAULT_LABELS_COL].list.sum().to_list() == [
            2,
            1,
            0,
        ]
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_index(GROUPBY_ID)

    if shuffle:
        df = shuffle_list_column(df, column=inview_col, seed=seed)

    df_labels = (
        df.explode(inview_col)
        .with_columns(
            pl.col(inview_col).is_in(pl.col(clicked_col)).cast(pl.Int8).alias(label_col)
        )
        .group_by(GROUPBY_ID)
        .agg(label_col)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [label_col])
    )


def exponential_decay(freshness, alpha=0.1):
    return np.exp(-alpha * freshness)


def impute_list_with_mean(lst):
    non_null_values = [x for x in lst if x not in [None, "null"]]
    if non_null_values:
        mean_value = sum(non_null_values) / len(non_null_values)
        return [x if x is not None else mean_value for x in lst]
    else:
        return lst


def encode_date_list(lst):
    return [x.timestamp() for x in lst]


def hours_date_list(lst):
    return [x.hour for x in lst]


def weekday_date_list(lst):
    return [x.weekday() for x in lst]


def bin_list_values(values, bins):
    labels = np.digitize(values, bins, right=True)  # Bin indices
    intervals = []
    for i in labels:
        if i == 0:  # Below the range of bins
            intervals.append(f"(nan, {bins[0]}]")
        elif i >= len(bins):  # Above the range of bins
            intervals.append(f"({bins[-2]}, nan]")
        else:
            intervals.append(f"({bins[i - 1]}, {bins[i]}]")
    return intervals


def create_reading_patterns(row):
    next_read_time = row['next_read_time']
    next_scroll_percentage = row['next_scroll_percentage']

    # Not Engaged
    if next_read_time == 0 and next_scroll_percentage == 0:
        return 0  # Not Read
    elif next_read_time <= 29 and next_scroll_percentage <= 0.333:
        return 0  # Quick Exit

    # Skimmers
    elif next_read_time <= 29 and 0.333 <= next_scroll_percentage <= 0.667:
        return 1  # Skimming
    elif next_read_time <= 58 and next_scroll_percentage >= 0.667:
        return 1  # Fast Scrollers

    # Engaged Readers
    elif 58 <= next_read_time <= 87 and next_scroll_percentage >= 0.667:
        return 2  # Normal Read
    elif next_read_time >= 87 and next_scroll_percentage >= 0.667:
        return 2  # Deep Read

    # Selective Readers
    elif 29 <= next_read_time <= 87 and 0.333 <= next_scroll_percentage <= 0.667:
        return 3  # Selective Reading
    elif 29 <= next_read_time <= 58 and next_scroll_percentage <= 0.333:
        return 3  # Focused Start (similar to Selective)

    # Uncategorized
    return 4  # Uncategorized


def remove_outliers(lst):
    """
    Removes outliers from the list and returns the filtered data.

    Parameters:
        lst (list or np.ndarray): The list of numerical data.
    Returns:
        filtered_data (np.ndarray): The data with outliers removed.
    """
    # Convert the data to a numpy array for easier computation
    data = np.array(lst)

    # Remove null values (None or NaN)
    data = data[~np.isnan(data)]

    # If the list is empty after removing NaNs, return an empty array
    if len(data) == 0:
        return data

    # Compute the mean and standard deviation of the data
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define the upper and lower bounds for outliers
    lower_bound = mean - 1.5 * std_dev
    upper_bound = mean + 1.5 * std_dev

    # Filter data to exclude outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data


def compute_mean(lst):
    """
    Computes the mean of the list after removing outliers.

    Parameters:
        lst (list): The list of numerical data.

    Returns:
        float: The mean of the filtered data or NaN if empty.
    """
    filtered_data = remove_outliers(lst)

    # If the filtered data is empty, return NaN
    if len(filtered_data) == 0:
        return None

    return np.mean(filtered_data)


def compute_std(lst):
    """
    Computes the standard deviation of the list after removing outliers.

    Parameters:
        lst (list): The list of numerical data.

    Returns:
        float: The standard deviation of the filtered data or NaN if empty.
    """
    filtered_data = remove_outliers(lst)

    # If the filtered data is empty, return NaN
    if len(filtered_data) == 0:
        return None

    return np.std(filtered_data)


# KL Divergence for Gaussian distributions
def kl_divergence_gaussian(mu_pred, sigma_pred, mu_pos, sigma_pos):
    epsilon = 1e-8
    sigma_pred = sigma_pred + epsilon
    sigma_pos = sigma_pos + epsilon

    kl_loss = torch.log(sigma_pos / sigma_pred) + (sigma_pred ** 2 + (mu_pred - mu_pos) ** 2) / (
                2 * sigma_pos ** 2) - 0.5
    return kl_loss.mean()
