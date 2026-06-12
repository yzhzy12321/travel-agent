"""
数据集构建辅助脚本
放在 model/data/ 目录下运行，所有文件生成在当前目录
用法: cd model/data && python build_dataset.py --city fuzhou --poi_file fuzhou_raw.csv
输入 CSV 只需包含: name, lon, lat, district, desc
（可选列: address, rating, category，不提供则自动填充默认值）
脚本自动完成坐标转换、ID分配、周边POI检测、格式化为标准 CSV
"""
import os
import sys
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point

# 脚本所在目录就是数据目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 周边 POI 检测距离分档（米）
#   ≤200m: 步行范围 (citywalk)
# 200-500m: 骑行范围 (bike)
# 500-1000m: 短途范围 (transit)
WALK_RADIUS = 200
BIKE_RADIUS = 500
TRANSIT_RADIUS = 1000


def convert_to_mercator(lon, lat):
    """WGS-84 → EPSG:3857 墨卡托坐标"""
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_mercator = point.to_crs("EPSG:3857")
    x = point_mercator.geometry.x.iloc[0]
    y = point_mercator.geometry.y.iloc[0]
    return x, y


def format_poi_entry(names, categories, distances, indices):
    """将邻近 POI 按距离排序后格式化为字符串"""
    entries = []
    for idx in indices:
        d = int(distances[idx])
        entries.append((d, f"{names[idx]}({categories[idx]}, {d}m)"))
    entries.sort()
    return "; ".join(e[1] for e in entries) if entries else "无"


def find_nearby_pois_3level(df):
    """
    为每个 POI 找周边三个距离档位的邻近 POI：
      - walk_nearby:  ≤200m（步行/citywalk）
      - bike_nearby:  200~500m（骑行）
      - transit_nearby: 500~1000m（短途公交/骑行）

    Returns:
        tuple[list[str], list[str], list[str]]
    """
    coords = df[["x", "y"]].astype(float).to_numpy()
    dist_matrix = cdist(coords, coords, metric="euclidean")
    names = df["name"].tolist()
    categories = df["category"].tolist() if "category" in df.columns else ["景点"] * len(df)

    walk_list, bike_list, transit_list = [], [], []

    for i in range(len(df)):
        walk_idx, bike_idx, transit_idx = [], [], []
        for j in range(len(df)):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if d <= WALK_RADIUS:
                walk_idx.append(j)
            elif d <= BIKE_RADIUS:
                bike_idx.append(j)
            elif d <= TRANSIT_RADIUS:
                transit_idx.append(j)

        walk_list.append(format_poi_entry(names, categories, dist_matrix[i], walk_idx))
        bike_list.append(format_poi_entry(names, categories, dist_matrix[i], bike_idx))
        transit_list.append(format_poi_entry(names, categories, dist_matrix[i], transit_idx))

    return walk_list, bike_list, transit_list

    return nearby_list


def build_city_csv(city_name, poi_file, start_id=10001):
    """
    读取原始 POI 列表，生成标准格式 CSV

    输入 CSV (poi_file) 列:
        必填:
        name     - POI 名称
        lon      - 经度 (WGS-84)
        lat      - 纬度 (WGS-84)
        district - 所在区域（如"鼓楼区"）
        desc     - POI 描述
        可选（不提供则自动填充）:
        address  - 详细地址，默认用"福建省福州市{district}"格式
        rating   - 评分(1-5)，留空
        category - 当前景点分类，默认"景点"

    输出列（论文完整字段 + 三级周边POI）:
        id, name, x, y, lon, lat, address, rating, category,
        walk_nearby, bike_nearby, transit_nearby, context
    """
    # poi_file 支持相对路径（相对于 DATA_DIR）或绝对路径
    if not os.path.isabs(poi_file):
        poi_file = os.path.join(DATA_DIR, poi_file)
    df = pd.read_csv(poi_file)

    # 验证必需列
    required = ["name", "lon", "lat", "district", "desc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 输入文件缺少列: {missing}")
        sys.exit(1)

    # 分配 ID
    df["id"] = range(start_id, start_id + len(df))

    # 转换坐标
    print(f"🔄 转换 {len(df)} 个 POI 的坐标...")
    xy_coords = []
    for _, row in df.iterrows():
        try:
            x, y = convert_to_mercator(row["lon"], row["lat"])
            xy_coords.append((x, y))
        except Exception as e:
            print(f"⚠️ 坐标转换失败 {row['name']}: {e}")
            xy_coords.append((0, 0))
    df["x"], df["y"] = zip(*xy_coords)

    # 补全可选列: address
    if "address" in df.columns:
        print(f"ℹ️ 使用原始数据中的 address 列")
    else:
        df["address"] = df.apply(
            lambda r: f"福建省福州市{r['district']}" if city_name == "fuzhou" else r['district'], axis=1
        )
        print(f"ℹ️ 未提供 address 列，已根据 district 自动填充")

    # 补全可选列: rating（不填默认值，留空）
    if "rating" not in df.columns:
        df["rating"] = ""
        print(f"ℹ️ 未提供 rating 列，已留空")

    # 补全可选列: category（当前景点分类）
    if "category" not in df.columns:
        df["category"] = "景点"
        print(f"ℹ️ 未提供 category 列，已默认填充 '景点'")

    # 检测周边 POI（三档：步行≤200m / 骑行200-500m / 短途500-1000m）
    print(f"🔍 检测周边 POI（步行≤{WALK_RADIUS}m / 骑行{BIKE_RADIUS}m / 短途{TRANSIT_RADIUS}m）...")
    df["walk_nearby"], df["bike_nearby"], df["transit_nearby"] = find_nearby_pois_3level(df)
    for label, col in [("步行≤200m", "walk_nearby"), ("骑行200-500m", "bike_nearby"), ("短途500-1000m", "transit_nearby")]:
        count = sum(1 for v in df[col] if v != "无")
        print(f"   {label}: {count}/{len(df)} 个 POI 有关联")

    # 构建 context 列：论文格式 "名称, 区域, 描述"
    df["context"] = df.apply(
        lambda r: f"{r['name']}, {r['district']}, {r['desc']}", axis=1
    )

    # 输出完整列
    output_df = df[["id", "name", "x", "y", "lon", "lat", "address", "rating", "category",
                     "walk_nearby", "bike_nearby", "transit_nearby", "context"]]

    # 保存到 data 目录
    out_path = os.path.join(DATA_DIR, f"{city_name}_zh.csv")
    output_df.to_csv(out_path, index=False)
    print(f"✅ 已生成: {out_path} ({len(output_df)} 个 POI)")
    print(f"   ID 范围: {start_id} - {start_id + len(df) - 1}")
    print(f"   列: {list(output_df.columns)}")
    print()
    print(f"📌 下一步:")
    print(f"   运行: python main.py --city {city_name} --type zh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集构建工具")
    parser.add_argument("--city", type=str, required=True, help="城市名（拼音）")
    parser.add_argument("--poi_file", type=str, required=True, help="原始 POI CSV 文件路径（相对于 data 目录或绝对路径）")
    parser.add_argument("--start_id", type=int, default=10001, help="起始 ID（默认 10001）")
    args = parser.parse_args()

    build_city_csv(args.city, args.poi_file, args.start_id)
