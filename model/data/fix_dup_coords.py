"""
修复 Shenzhen 和 Guangzhou 数据集中重复坐标的 POI
==============================================
用法: cd model/data && python fix_dup_coords.py

原理: 对共享相同坐标的 POI，用高德 POI 搜索 API 按名称+城市
      查找精确坐标，替换到 _zh.csv 和 _raw.csv 中。
      搜索失败则跳过，保留原坐标。
"""
import os
import sys
import time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GaodeMap_apikey, CITY_ZH_MAP

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_URL = "https://restapi.amap.com/v3/place/text"


def search_poi_location(name: str, city: str) -> tuple[float, float] | None:
    """高德 POI 搜索，返回 (lon, lat) GCJ-02 坐标，失败返回 None"""
    params = {
        "key": GaodeMap_apikey,
        "keywords": name,
        "city": city,
        "extensions": "all",
        "offset": 1,
    }
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and int(data.get("count", 0)) > 0:
            poi = data["pois"][0]
            location = poi.get("location", "")
            if location and "," in location:
                lon_str, lat_str = location.split(",")
                lon, lat = float(lon_str), float(lat_str)
                if 100 <= lon <= 120 and 20 <= lat <= 40:
                    return (lon, lat)
    except Exception as e:
        print(f"    ⚠️ API异常: {e}")
    return None


def fix_city(city_en: str):
    """修复单个城市的重复坐标"""
    csv_path = os.path.join(DATA_DIR, f"{city_en}_zh.csv")
    raw_path = os.path.join(DATA_DIR, f"{city_en}_raw.csv")
    city_cn = CITY_ZH_MAP.get(city_en, city_en)

    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return

    zh = pd.read_csv(csv_path)
    raw = pd.read_csv(raw_path)

    # 找出重复坐标组
    dup_mask = zh.duplicated(subset=["lon", "lat"], keep=False)
    dup_count = dup_mask.sum()
    if dup_count == 0:
        print(f"✅ {city_cn}: 无重复坐标，跳过")
        return

    # 按坐标分组
    coord_groups = zh[dup_mask].groupby(["lon", "lat"])
    print(f"🔧 {city_cn}: {dup_count} 个重复坐标 POI，共 {len(coord_groups)} 组")
    print()

    fixed = 0
    for (lon, lat), group in coord_groups:
        names = group["name"].tolist()
        if len(names) <= 1:
            continue

        # 每组保留第一个，其余的尝试修正
        keep_name = names[0]
        for name in names[1:]:
            print(f"  🔍 {name} (原坐标 {lon:.4f},{lat:.4f})...", end=" ")
            new_loc = search_poi_location(name, city_cn)
            if new_loc is None:
                print("未找到 → 保留原坐标")
                continue

            new_lon, new_lat = new_loc
            # 检查新坐标是否和原来的不同
            if abs(new_lon - lon) < 0.0001 and abs(new_lat - lat) < 0.0001:
                print(f"坐标相同 ({new_lon:.4f},{new_lat:.4f}) → 保留")
                continue

            print(f"→ ({new_lon:.4f},{new_lat:.4f}) ✅")
            # 更新 _zh.csv
            mask_zh = zh["name"] == name
            zh.loc[mask_zh, "lon"] = new_lon
            zh.loc[mask_zh, "lat"] = new_lat

            # 更新 _raw.csv
            mask_raw = raw["name"] == name
            if mask_raw.any():
                raw.loc[mask_raw, "lon"] = new_lon
                raw.loc[mask_raw, "lat"] = new_lat

            fixed += 1
            time.sleep(0.3)  # 避免 API 限流

    # 保存
    zh.to_csv(csv_path, index=False)
    raw.to_csv(raw_path, index=False)

    # 统计修复后
    new_dup = zh.duplicated(subset=["lon", "lat"], keep=False).sum()
    print()
    print(f"✅ {city_cn}: 修复 {fixed} 个，修复后剩余 {new_dup} 个重复坐标")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None, help="指定城市 (shenzhen/guangzhou)")
    parser.add_argument("--all", action="store_true", help="修复所有城市")
    args = parser.parse_args()

    if args.all:
        for city in ["shenzhen", "guangzhou"]:
            fix_city(city)
    elif args.city:
        fix_city(args.city)
    else:
        fix_city("shenzhen")
        fix_city("guangzhou")
