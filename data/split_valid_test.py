import os
import polars as pl
import warnings

warnings.filterwarnings("ignore")
dataset_size = "small"
data_folder = "../data"

data_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)
dev_path = data_path + '/validation/'

behavior_file = os.path.join(dev_path, "behaviors.parquet")
behavior_df = pl.scan_parquet(behavior_file)

history_file = os.path.join(dev_path, "history.parquet")
history_df = pl.scan_parquet(history_file)

median_impression_time = behavior_df.select(pl.col('impression_time')).median().collect()

valid_behavior_df = behavior_df.filter(pl.col('impression_time') <= median_impression_time.cast(pl.Datetime))
test2_behavior_df = behavior_df.filter(pl.col('impression_time') > median_impression_time.cast(pl.Datetime))

unique_user_id_valid = valid_behavior_df.select("user_id").unique().collect().to_numpy().flatten()
valid_history_df = history_df.filter(pl.col("user_id").is_in(unique_user_id_valid))

unique_user_id_test2 = test2_behavior_df.select("user_id").unique().collect().to_numpy().flatten()
test2_history_df = history_df.filter(pl.col("user_id").is_in(unique_user_id_test2))

os.makedirs(f"{data_path}/test2", exist_ok=True)
os.makedirs(f"{data_path}/validation2", exist_ok=True)

valid_behavior_df.collect().write_parquet(f"{data_path}/validation2/behaviors.parquet")
valid_history_df.collect().write_parquet(f"{data_path}/validation2/history.parquet")

test2_behavior_df.collect().write_parquet(f"{data_path}/test2/behaviors.parquet")
test2_history_df.collect().write_parquet(f"{data_path}/test2/history.parquet")
