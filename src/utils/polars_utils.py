import time
import numpy as np
import random
import gc
import os
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import polars as pl
except ImportError:
    print("polars not available")

from src.utils.python_utils import generate_unique_name


def _check_columns_in_df(df: pl.DataFrame, columns: list[str]) -> None:
    """
    Checks whether all specified columns are present in a Polars DataFrame.
    Raises a ValueError if any of the specified columns are not present in the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): The names of the columns to check for.

    Returns:
        None.

    Examples:
    >>> df = pl.DataFrame({"user_id": [1], "first_name": ["J"]})
    >>> check_columns_in_df(df, columns=["user_id", "not_in"])
        ValueError: Invalid input provided. The dataframe does not contain columns ['not_in'].
    """
    columns_not_in_df = [col for col in columns if col not in df.columns]
    if columns_not_in_df:
        raise ValueError(
            f"Invalid input provided. The DataFrame does not contain columns {columns_not_in_df}."
        )


def _validate_equal_list_column_lengths(df: pl.DataFrame, col1: str, col2: str) -> bool:
    """
    Checks if the items in two list columns of a DataFrame have equal lengths.

    Args:
        df (pl.DataFrame): The DataFrame containing the list columns.
        col1 (str): The name of the first list column.
        col2 (str): The name of the second list column.

    Returns:
        bool: True if the items in the two list columns have equal lengths, False otherwise.

    Raises:
        None.

    >>> df = pl.DataFrame({
            'col1': [[1, 2, 3], [4, 5], [6]],
            'col2': [[10, 20], [30, 40, 50], [60, 70, 80]],
        })
    >>> _validate_equal_list_column_lengths(df, 'col1', 'col2')
        ValueError: Mismatch in the lengths of the number of items (row-based) between the columns: 'col1' and 'col2'. Please ensure equal lengths.
    >>> df = df.with_columns(pl.Series('col1', [[1, 2], [3, 4, 5], [6, 7, 8]]))
    >>> _validate_equal_list_column_lengths(df, 'col1', 'col2')
    """
    if not df.select(pl.col(col1).list.len() == pl.col(col2).list.len())[col1].all():
        raise ValueError(
            f"Mismatch in the lengths of the number of items (row-based) between the columns: '{col1}' and '{col2}'. Please ensure equal lengths."
        )


# def slice_join_dataframes(
#         df1: pl.DataFrame,
#         df2: pl.DataFrame,
#         on: str,
#         how: str,
#         chunk_size: int
# ) -> pl.DataFrame:
#     """
#     Join two dataframes optimized for memory efficiency.
#     """
#     result_chunks = []
#
#     for i, rows in enumerate(df1.iter_slices(chunk_size)):
#         print(f"Processing chunk {i + 1}")
#
#         # Perform the join operation
#         joined_chunk = rows.join(df2, on=on, how=how)
#
#         # Collect the result to the list
#         result_chunks.append(joined_chunk)
#
#         # Free memory by deleting the chunk and triggering garbage collection
#         del rows
#         gc.collect()
#
#     # Concatenate all chunks into a single DataFrame
#     result_df = pl.concat(result_chunks)
#
#     # Free memory by deleting the intermediate list and triggering garbage collection
#     del result_chunks
#     gc.collect()
#
#     return result_df

def slice_join_dataframes(
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        on: str,
        how: str,
        chunk_size: int,
        output_file_prefix: str
) -> list:
    """
    Join two dataframes optimized for memory efficiency, saving results to separate files.
    """
    chunk_files = []

    for i, rows in enumerate(df1.iter_slices(chunk_size)):
        print(f"Processing chunk {i + 1}")

        # Perform the join operation
        joined_chunk = rows.join(df2, on=on, how=how)

        # Save the chunk to disk
        chunk_file = f"{output_file_prefix}_chunk_{i}.parquet"
        joined_chunk.write_parquet(chunk_file)
        chunk_files.append(chunk_file)

        # Free memory by deleting the chunk and triggering garbage collection
        del rows
        del joined_chunk
        gc.collect()

    return chunk_files


def concatenate_parquet_files(chunk_files, output_file):
    """
    Concatenate multiple Parquet files into a single Parquet file using PyArrow.
    """
    # Open the first file and write its schema to the output file
    first_file = chunk_files.pop(0)
    table = pq.read_table(first_file)
    with pq.ParquetWriter(output_file, table.schema) as writer:
        writer.write_table(table)
        del table

        # Append the rest of the files
        for chunk_file in chunk_files:
            table = pq.read_table(chunk_file)
            writer.write_table(table)
            del table
            gc.collect()
            os.remove(chunk_file)


def concatenate_csv_files(chunk_files, output_file):
    """
    Concatenate multiple CSV files into a single CSV file.
    """
    # Read the first file and write it to the output file
    first_file = chunk_files.pop(0)
    with open(output_file, 'w', newline='') as fout:
        with open(first_file, 'r') as fin:
            fout.write(fin.read())
    os.remove(first_file)

    # Append the rest of the files
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as fin:
            fin.readline()  # Skip the header
            with open(output_file, 'a', newline='') as fout:
                fout.write(fin.read())
        os.remove(chunk_file)


def rename_columns(df: pl.DataFrame, map_dict: dict[str, str]) -> pl.DataFrame:
    """
    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> map_dict = {'A': 'X', 'B': 'Y'}
        >>> rename_columns(df, map_dict)
            shape: (2, 2)
            ┌─────┬─────┐
            │ X   ┆ Y   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘
        >>> rename_columns(df, {"Z" : "P"})
            shape: (2, 2)
            ┌─────┬─────┐
            │ A   ┆ B   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘
    """
    map_dict = {key: val for key, val in map_dict.items() if key in df.columns}
    if len(map_dict):
        df = df.rename(map_dict)
    return df


def from_dict_to_polars(dictionary: dict) -> pl.DataFrame:
    """
    When dealing with dictionary with intergers as keys
    Example:
    >>> dictionary = {1: "a", 2: "b"}
    >>> from_dict_to_polars(dictionary)
        shape: (2, 2)
        ┌──────┬────────┐
        │ keys ┆ values │
        │ ---  ┆ ---    │
        │ i64  ┆ str    │
        ╞══════╪════════╡
        │ 1    ┆ a      │
        │ 2    ┆ b      │
        └──────┴────────┘
    >>> pl.from_dict(dictionary)
        raise ValueError("Series name must be a string.")
            ValueError: Series name must be a string.
    """
    return pl.DataFrame(
        {"keys": list(dictionary.keys()), "values": list(dictionary.values())}
    )


def shuffle_rows(df: pl.DataFrame, seed: int = None) -> pl.DataFrame:
    """
    Shuffle the rows of a DataFrame. This methods allows for LazyFrame,
    whereas, 'df.sample(fraction=1)' is not compatible.

    Examples:
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    >>> shuffle_rows(df.lazy(), seed=123).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 3   ┆ 3   ┆ 3   │
        │ 2   ┆ 2   ┆ 2   │
        └─────┴─────┴─────┘
    >>> shuffle_rows(df.lazy(), seed=None).collect().sort("a")
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 2   ┆ 2   ┆ 2   │
        │ 3   ┆ 3   ┆ 3   │
        └─────┴─────┴─────┘

    Test_:
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df, seed=None).iter_rows()])
        True

    Note:
        Be aware that 'pl.all().shuffle()' shuffles columns-wise, i.e., with if pl.all().shuffle(None)
        each column's element are shuffled independently from each other (example might change with no seed):
    >>> df_ = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}).select(pl.all().shuffle(None)).sort("a")
    >>> df_
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 3   ┆ 1   │
        │ 2   ┆ 2   ┆ 3   │
        │ 3   ┆ 1   ┆ 2   │
        └─────┴─────┴─────┘
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df_, seed=None).iter_rows()])
        False
    """
    seed = seed if seed is not None else random.randint(1, 1_000_000)
    return df.select(pl.all().shuffle(seed))


def keep_unique_values_in_list(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Removes duplicate article IDs from the specified list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing the list column with article IDs.
        column (str): The name of the list column containing article IDs.

    Returns:
        pl.DataFrame: A new DataFrame with the same columns as the input DataFrame, but with duplicate
        article IDs removed from the specified list column.

    Example:
        >>> df = pl.DataFrame({
                "article_ids": [[1, 2, 3, 1, 2], [3, 4, 5, 3], [1, 2, 3, 1, 2, 3]],
                "hh": ["h", "e", "y"]
            })
        >>> keep_unique_values_in_list(df.lazy(), "article_ids").collect()
            shape: (3, 1)
            ┌─────────────┐
            │ article_ids │
            │ ---         │
            │ list[i64]   │
            ╞═════════════╡
            │ [1, 2, 3]   │
            │ [3, 4, 5]   │
            │ [1, 2, 3]   │
            └─────────────┘
    """
    return df.with_columns(pl.col(column).list.unique())


def filter_minimum_lengths_from_list(
        df: pl.DataFrame,
        n: int,
        column: str,
) -> pl.DataFrame:
    """Filters a DataFrame based on the minimum number of elements in an array column.

    Args:
        df (pl.DataFrame): The input DataFrame to filter.
        n (int): The minimum number of elements required in the array column.
        column (str): The name of the array column to filter on.

    Returns:
        pl.DataFrame: The filtered DataFrame.

    Example:
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "article_ids": [["a", "b", "c"], ["a", "b"], ["a"], ["a"]],
            }
        )
    >>> filter_minimum_lengths_from_list(df, n=2, column="article_ids")
        shape: (2, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        └─────────┴─────────────────┘
    >>> filter_minimum_lengths_from_list(df, n=None, column="article_ids")
        shape: (4, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        │ 3       ┆ ["a"]           │
        │ 4       ┆ ["a"]           │
        └─────────┴─────────────────┘
    """
    return (
        df.filter(pl.col(column).list.len() >= n)
        if column in df and n is not None and n > 0
        else df
    )


def filter_maximum_lengths_from_list(
        df: pl.DataFrame,
        n: int,
        column: str,
) -> pl.DataFrame:
    """Filters a DataFrame based on the maximum number of elements in an array column.

    Args:
        df (pl.DataFrame): The input DataFrame to filter.
        n (int): The maximum number of elements required in the array column.
        column (str): The name of the array column to filter on.

    Returns:
        pl.DataFrame: The filtered DataFrame.

    Example:
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "article_ids": [["a", "b", "c"], ["a", "b"], ["a"], ["a"]],
            }
        )
    >>> filter_maximum_lengths_from_list(df, n=2, column="article_ids")
        shape: (3, 2)
        ┌─────────┬─────────────┐
        │ user_id ┆ article_ids │
        │ ---     ┆ ---         │
        │ i64     ┆ list[str]   │
        ╞═════════╪═════════════╡
        │ 2       ┆ ["a", "b"]  │
        │ 3       ┆ ["a"]       │
        │ 4       ┆ ["a"]       │
        └─────────┴─────────────┘
    >>> filter_maximum_lengths_from_list(df, n=None, column="article_ids")
        shape: (4, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        │ 3       ┆ ["a"]           │
        │ 4       ┆ ["a"]           │
        └─────────┴─────────────────┘
    """
    return (
        df.filter(pl.col(column).list.len() <= n)
        if column in df and n is not None and n > 0
        else df
    )


def drop_nulls_from_list(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Drops null values from a specified column in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to drop null values from.

    Returns:
        pl.DataFrame: A new DataFrame with null values dropped from the specified column.

    Examples:
    >>> df = pl.DataFrame(
            {"user_id": [101, 102, 103], "dynamic_article_id": [[1, None, 3], None, [4, 5]]}
        )
    >>> print(df)
        shape: (3, 2)
        ┌─────────┬────────────────────┐
        │ user_id ┆ dynamic_article_id │
        │ ---     ┆ ---                │
        │ i64     ┆ list[i64]          │
        ╞═════════╪════════════════════╡
        │ 101     ┆ [1, null, 3]       │
        │ 102     ┆ null               │
        │ 103     ┆ [4, 5]             │
        └─────────┴────────────────────┘
    >>> drop_nulls_from_list(df, "dynamic_article_id")
        shape: (3, 2)
        ┌─────────┬────────────────────┐
        │ user_id ┆ dynamic_article_id │
        │ ---     ┆ ---                │
        │ i64     ┆ list[i64]          │
        ╞═════════╪════════════════════╡
        │ 101     ┆ [1, 3]             │
        │ 102     ┆ null               │
        │ 103     ┆ [4, 5]             │
        └─────────┴────────────────────┘
    """
    return df.with_columns(pl.col(column).list.eval(pl.element().drop_nulls()))


def filter_list_elements(df: pl.DataFrame, column: str, ids: list[any]) -> pl.DataFrame:
    """
    Removes list elements from a specified column in a Polars DataFrame that are not found in a given list of identifiers.

    Args:
        df (pl.DataFrame): The Polars DataFrame to process.
        column (str): The name of the column from which to remove unknown elements.
        ids (list[any]): A list of identifiers to retain in the specified column. Elements not in this list will be removed.

    Returns:
        pl.DataFrame: A new Polars DataFrame with the same structure as the input DataFrame, but with elements not found in
                    the 'ids' list removed from the specified 'column'.

    Examples:
    >>> df = pl.DataFrame({"A": [1, 2, 3, 4, 5], "B": [[1, 3], [3, 4], None, [7, 8], [9, 10]]})
    >>> ids = [1, 3, 5, 7]
    >>> filter_list_elements(df.lazy(), "B", ids).collect()
        shape: (5, 2)
        ┌─────┬───────────┐
        │ A   ┆ B         │
        │ --- ┆ ---       │
        │ i64 ┆ list[i64] │
        ╞═════╪═══════════╡
        │ 1   ┆ [1, 3]    │
        │ 2   ┆ [3]       │
        │ 3   ┆ null      │
        │ 4   ┆ [7]       │
        │ 5   ┆ null      │
        └─────┴───────────┘
    """
    GROUPBY_COL = "_groupby"
    COLUMNS = df.columns
    df = df.with_row_index(GROUPBY_COL)
    df_ = (
        df.select(pl.col(GROUPBY_COL, column))
        .drop_nulls()
        .explode(column)
        .filter(pl.col(column).is_in(ids))
        .group_by(GROUPBY_COL)
        .agg(column)
    )
    return df.drop(column).join(df_, on=GROUPBY_COL, how="left").select(COLUMNS)


def filter_elements(df: pl.DataFrame, column: str, ids: list[any]) -> pl.DataFrame:
    """
    Removes elements from a specified column in a Polars DataFrame that are not found in a given list of identifiers.

    Args:
        df (pl.DataFrame): The Polars DataFrame to process.
        column (str): The name of the column from which to remove unknown elements.
        ids (list[any]): A list of identifiers to retain in the specified column. Elements not in this list will be removed.

    Returns:
        pl.DataFrame: A new Polars DataFrame with the same structure as the input DataFrame, but with elements not found in
                    the 'ids' list removed from the specified 'column'.

    Examples:
    >>> df = pl.DataFrame({"A": [1, 2, 3, 4, 5], "B": [[1, 3], [3, 4], None, [7, 8], [9, 10]]})
        shape: (5, 2)
        ┌─────┬───────────┐
        │ A   ┆ B         │
        │ --- ┆ ---       │
        │ i64 ┆ list[i64] │
        ╞═════╪═══════════╡
        │ 1   ┆ [1, 3]    │
        │ 2   ┆ [3, 4]    │
        │ 3   ┆ null      │
        │ 4   ┆ [7, 8]    │
        │ 5   ┆ [9, 10]   │
        └─────┴───────────┘
    >>> ids = [1, 3, 5, 7]
    >>> filter_elements(df.lazy(), "A", ids).collect()
        shape: (5, 2)
        ┌──────┬───────────┐
        │ A    ┆ B         │
        │ ---  ┆ ---       │
        │ i64  ┆ list[i64] │
        ╞══════╪═══════════╡
        │ 1    ┆ [1, 3]    │
        │ null ┆ [3, 4]    │
        │ 3    ┆ null      │
        │ null ┆ [7, 8]    │
        │ 5    ┆ [9, 10]   │
        └──────┴───────────┘
    """
    GROUPBY_COL = "_groupby"
    COLUMNS = df.columns
    df = df.with_row_index(GROUPBY_COL)
    df_ = (
        df.select(pl.col(GROUPBY_COL, column))
        .drop_nulls()
        .filter(pl.col(column).is_in(ids))
    )
    return df.drop(column).join(df_, on=GROUPBY_COL, how="left").select(COLUMNS)


def concat_str_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Alice"],
                "last_name": ["Doe", "Doe", "Smith"],
            }
        )
    >>> concatenated_df, concatenated_column_name = concat_str_columns(df, columns=['first_name', 'last_name'])
    >>> concatenated_df
        shape: (3, 4)
        ┌─────┬────────────┬───────────┬──────────────────────┐
        │ id  ┆ first_name ┆ last_name ┆ first_name-last_name │
        │ --- ┆ ---        ┆ ---       ┆ ---                  │
        │ i64 ┆ str        ┆ str       ┆ str                  │
        ╞═════╪════════════╪═══════════╪══════════════════════╡
        │ 1   ┆ John       ┆ Doe       ┆ John Doe             │
        │ 2   ┆ Jane       ┆ Doe       ┆ Jane Doe             │
        │ 3   ┆ Alice      ┆ Smith     ┆ Alice Smith          │
        └─────┴────────────┴───────────┴──────────────────────┘
    """
    concat_name = "-".join(columns)
    concat_columns = df.select(pl.concat_str(columns, separator=" ").alias(concat_name))
    return df.with_columns(concat_columns), concat_name


def filter_empty_text_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Example:
    >>> df = pl.DataFrame({"Name": ["John", "Alice", "Bob", ""], "Age": [25, 28, 30, 22]})
    >>> filter_empty_text_column(df, "Name")
        shape: (3, 2)
        ┌───────┬─────┐
        │ Name  ┆ Age │
        │ ---   ┆ --- │
        │ str   ┆ i64 │
        ╞═══════╪═════╡
        │ John  ┆ 25  │
        │ Alice ┆ 28  │
        │ Bob   ┆ 30  │
        └───────┴─────┘
    """
    return df.filter(pl.col(column).str.lengths() > 0)


def shuffle_list_column(
        df: pl.DataFrame, column: str, seed: int = None
) -> pl.DataFrame:
    """Shuffles the values in a list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to shuffle.
        seed (int, optional): An optional seed value.
            Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column shuffled.

    Example:
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "list_col": [["a-", "b-", "c-"], ["a#", "b#"], ["a@", "b@", "c@"]],
                "rdn": ["h", "e", "y"],
            }
        )
    >>> shuffle_list_column(df, 'list_col', seed=1)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["c-", "b-", "a-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["b@", "c@", "a@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    No seed:
    >>> shuffle_list_column(df, 'list_col', seed=None)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["b-", "a-", "c-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["a@", "c@", "b@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )

    >>> df = pl.DataFrame({
            'id': [1, 2, 3],
            'list_col': [[6, 7, 8], [-6, -7, -8], [60, 70, 80]],
            'rdn': ['h', 'e', 'y']
        })
    >>> shuffle_list_column(df.lazy(), 'list_col', seed=2).collect()
        shape: (3, 3)
        ┌─────┬──────────────┬─────┐
        │ id  ┆ list_col     ┆ rdn │
        │ --- ┆ ---          ┆ --- │
        │ i64 ┆ list[i64]    ┆ str │
        ╞═════╪══════════════╪═════╡
        │ 1   ┆ [7, 6, 8]    ┆ h   │
        │ 2   ┆ [-8, -7, -6] ┆ e   │
        │ 3   ┆ [60, 80, 70] ┆ y   │
        └─────┴──────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )
    """
    _COLUMN_ORDER = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMN_ORDER, "_groupby_id")

    df = df.with_row_count(GROUPBY_ID)
    df_shuffle = (
        df.explode(column)
        .pipe(shuffle_rows, seed=seed)
        .group_by(GROUPBY_ID)
        .agg(column)
    )
    return (
        df.drop(column)
        .join(df_shuffle, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMN_ORDER)
    )


def split_df_in_n(df: pl.DataFrame, num_splits: int) -> list[pl.DataFrame]:
    """
    Split a DataFrame into n equal-sized splits.

    Args:
        df (pandas.DataFrame): The DataFrame to be split.
        num_splits (int): The number of splits to create.

    Returns:
        List[pandas.DataFrame]: A list of DataFrames, each representing a split.

    Examples:
        >>> df = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7], "B" : [1, 2, 3, 4, 5, 6, 7]})
        >>> splits = split_df_in_n(df, 3)
        >>> for d in splits:
                print(d)
                shape: (3, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 1   ┆ 1   │
                │ 2   ┆ 2   │
                │ 3   ┆ 3   │
                └─────┴─────┘
                shape: (3, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 4   ┆ 4   │
                │ 5   ┆ 5   │
                │ 6   ┆ 6   │
                └─────┴─────┘
                shape: (1, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 7   ┆ 7   │
                └─────┴─────┘

    """
    rows_per_split = int(np.ceil(df.shape[0] / num_splits))
    return [
        df[i * rows_per_split: (1 + i) * rows_per_split] for i in range(num_splits)
    ]


def concat_list_str(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Concatenate strings within lists for a specified column in a DataFrame.

    Args:
        df (polars.DataFrame): The input DataFrame.
        column (str): The name of the column in `df` that contains lists of strings
                        to be concatenated.

    Returns:
        polars.DataFrame: A DataFrame with the same structure as `df` but with the
                            specified column's lists of strings concatenated and
                            converted to a string instead of list.

    Examples:
        >>> df = pl.DataFrame({
                "strings": [["ab", "cd"], ["ef", "gh"], ["ij", "kl"]]
            })
        >>> concat_list_str(df, "strings")
            shape: (3, 1)
            ┌─────────┐
            │ strings │
            │ ---     │
            │ str     │
            ╞═════════╡
            │ ab cd   │
            │ ef gh   │
            │ ij kl   │
            └─────────┘
    """
    return df.with_columns(
        pl.col(column).list.eval(pl.element().str.concat(" "))
    ).explode(column)


def reduce_mem(df: pl.DataFrame) -> pl.DataFrame:
    starttime = time.time()
    numerics = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
    original_mem = df.estimated_size(unit="mb")  # Estimated size in MB

    # Iterate through columns
    for col in tqdm(df.columns):
        col_type = df.schema[col]
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min is None or c_max is None:  # Skip columns with missing values
                continue

            if col_type.is_integer():
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type.is_float():
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df = df.with_columns(df[col].cast(pl.Float32))  # Polars doesn't support Float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    df = df.with_columns(df[col].cast(pl.Float64))

    # Calculate memory reduction
    final_mem = df.estimated_size(unit="mb")
    print(
        f"-- Mem. usage decreased to {final_mem:5.2f} MB ({100 * (original_mem - final_mem) / original_mem:.1f}% reduction), "
        f"time spent: {(time.time() - starttime) / 60:2.2f} min")
    return df
