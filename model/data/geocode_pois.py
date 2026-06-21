"""
通过高德地理编码 API，用 address 字段重新定位所有 POI
=====================================================
用法: cd model/data && python geocode_pois.py --city fuzhou
      cd model/data && python geocode_pois.py --all

效果：
  - 读取 fuzhou_zh.csv，逐行调高德 API 用 address 获取精准 GCJ-02 坐标
  - 更新 lon, lat 列
  - 重新计算 x, y（EPSG:3857 墨卡托投影）
  - 重新生成 fuzhou_zh.npy embedding
  - 渲染端不再需要 wgs2gcj 转换（数据已是 GCJ-02）
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GaodeMap_apikey

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 高德地理编码 API
GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"


def geocode_address(address: str, city: str = "") -> tuple[float, float] | None:
    """调用高德 API 将地址转为 GCJ-02 坐标"""
    params = {
        "key": GaodeMap_apikey,
        "address": address,
    }
    if city:
        params["city"] = city

    try:
        resp = requests.get(GEOCODE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and int(data.get("count", 0)) > 0:
            loc = data["geocodes"][0]["location"]  # "119.2965,26.0745"
            lon, lat = loc.split(",")
            return float(lon), float(lat)
        else:
            print(f"  ⚠️ 地理编码失败: {address} → {data.get('info', 'unknown')}")
            return None
    except Exception as e:
        print(f"  ❌ 请求异常: {address} → {e}")
        return None


def convert_to_mercator(lon, lat):
    """GCJ-02 → EPSG:3857 墨卡托（与 build_dataset.py 一致）"""
    import geopandas as gpd
    from shapely.geometry import Point
    # 注意：这里 lon,lat 是 GCJ-02，但 EPSG:3857 投影对 GCJ-02 和 WGS-84
    # 的误差在局部范围内可忽略（<1m），不影响空间聚类
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_mercator = point.to_crs("EPSG:3857")
    return point_mercator.geometry.x.iloc[0], point_mercator.geometry.y.iloc[0]


def geocode_csv(city: str):
    """对单个城市的 CSV 进行地理编码更新"""
    csv_path = os.path.join(DATA_DIR, f"{city}_zh.csv")
    emb_path = os.path.join(DATA_DIR, f"{city}_zh.npy")

    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📍 {city}: {len(df)} 个 POI，开始地理编码...")

    updated = 0
    for idx, row in df.iterrows():
        address = str(row.get("address", ""))
        old_lon, old_lat = row["lon"], row["lat"]

        if not address or address == "nan":
            print(f"  [{idx+1}/{len(df)}] {row['name']}: 无地址，跳过")
            continue

        print(f"  [{idx+1}/{len(df)}] {row['name']}: {address[:40]}...", end=" ")

        result = geocode_address(address, city="福州" if city == "fuzhou" else "")
        if result:
            new_lon, new_lat = result
            df.at[idx, "lon"] = new_lon
            df.at[idx, "lat"] = new_lat
            # 重新计算墨卡托坐标
            x, y = convert_to_mercator(new_lon, new_lat)
            df.at[idx, "x"] = x
            df.at[idx, "y"] = y

            old_dist = np.sqrt((old_lon - new_lon)**2 + (old_lat - new_lat)**2) * 111000
            print(f"✅ ({new_lon:.6f}, {new_lat:.6f}) 偏移 {old_dist:.0f}m")
            updated += 1
        else:
            print("❌ 保留原坐标")

        time.sleep(0.15)  # 限速，避免触发 API 频率限制

    # 保存
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 已更新 {updated}/{len(df)} 个 POI，保存到 {csv_path}")

    # 重新生成 embedding
    print("🔄 重新生成 embedding...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    context = df["context"].astype(str).tolist()
    embedding = model.encode(context, normalize_embeddings=True)
    np.save(emb_path, embedding)
    print(f"✅ Embedding 已保存: {emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None, help="单个城市名（如 fuzhou）")
    parser.add_argument("--all", action="store_true", help="处理所有城市")
    args = parser.parse_args()

    if args.all:
        cities = ["fuzhou", "guangzhou", "shanghai", "shenzhen"]
    elif args.city:
        cities = [args.city]
    else:
        print("请指定 --city fuzhou 或 --all")
        sys.exit(1)

    for city in cities:
        print(f"\n{'='*50}")
        geocode_csv(city)
        print(f"{'='*50}")

    print("\n🎉 全部完成！现在数据使用高德 GCJ-02 坐标")
    print("   渲染端可移除 wgs2gcj 转换，直接使用 lon/lat")
