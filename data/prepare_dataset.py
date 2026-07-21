from datasets import load_dataset
import pandas as pd
import os


# =========================
# 参数
# =========================

STATION_ID = "0001953ce171ce70"

OUTPUT_PATH = "data/pv_dataset.csv"


# =========================
# 1. 读取光伏功率
# =========================

print("Loading generation...")

generation = load_dataset(
    "EDS-lab/pv-generation",
    "generation",
    split="train"
)

generation = generation.to_pandas()


# 选择一个电站
generation = generation[
    generation["unique_id"] == STATION_ID
]


generation = generation[
    [
        "timestamp",
        "unique_id",
        "y"
    ]
]


print("Generation:")
print(generation.head())


# 处理负值
generation["y"] = generation["y"].clip(lower=0)


# =========================
# 2. 获取 location_id
# =========================

print("\nLoading metadata...")


metadata = load_dataset(
    "EDS-lab/pv-generation",
    "metadata",
    split="train"
)

metadata = metadata.to_pandas()


station_meta = metadata[
    metadata["unique_id"] == STATION_ID
]


location_id = station_meta.iloc[0]["location_id"]


print("Location:")
print(location_id)



# =========================
# 3. 读取天气
# =========================

print("\nLoading weather...")


weather = load_dataset(
    "EDS-lab/pv-generation",
    "weather",
    split="train"
)

weather = weather.to_pandas()


# 选择对应地点

weather = weather[
    weather["location_id"] == location_id
]


weather = weather[
    [
        "timestamp",
        "temperature_2m",
        "cloud_cover",
        "shortwave_radiation",
        "relative_humidity_2m"
    ]
]


print("Weather:")
print(weather.head())


# =========================
# 4. 合并
# =========================

print("\nMerging...")


df = pd.merge(
    generation,
    weather,
    on="timestamp",
    how="inner"
)


# =========================
# 5. 调整字段顺序
# =========================

df = df[
    [
        "timestamp",
        "temperature_2m",
        "cloud_cover",
        "shortwave_radiation",
        "relative_humidity_2m",
        "y"
    ]
]


# 时间排序

df = df.sort_values(
    "timestamp"
)


# 保存

os.makedirs(
    "data",
    exist_ok=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("\n====================")
print("Finished!")
print("====================")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nSaved:")
print(OUTPUT_PATH)