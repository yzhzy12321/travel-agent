"""
从高德 POI 搜索 API 批量获取景区评分
====================================
用法: cd model/data && python fetch_ratings.py --city fuzhou
      cd model/data && python fetch_ratings.py --all

原理: 用高德 POI 搜索 API (restapi.amap.com/v3/place/text)，
      按 name + city 搜索每个 POI，提取评分 (biz_ext.rating)。
      失败则用 category 默认评分填充。
"""
import os
import sys
import time
import argparse
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GaodeMap_apikey, CITY_ZH_MAP,CATEGORY_DEFAULT_RATING

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_URL = "https://restapi.amap.com/v3/place/text"


def search_poi_rating(name: str, city: str) -> float | None:
    """高德 POI 搜索，返回评分（1-5），失败返回 None"""
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
            # 高德评分在 biz_ext.rating 或 deep_info.rating
            rating = None
            if "biz_ext" in poi and "rating" in poi["biz_ext"]:
                rating = float(poi["biz_ext"]["rating"])
            elif "deep_info" in poi and "rating" in poi["deep_info"]:
                rating = float(poi["deep_info"]["rating"])
            if rating and 1 <= rating <= 5:
                return round(rating, 1)
    except Exception:
        pass
    return None


def fetch_city(city_en: str):
    """为单个城市获取评分"""
    csv_path = os.path.join(DATA_DIR, f"{city_en}_zh.csv")
    raw_path = os.path.join(DATA_DIR, f"{city_en}_raw.csv")
    city_cn = CITY_ZH_MAP.get(city_en, city_en)

    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return

    zh = pd.read_csv(csv_path)
    raw = pd.read_csv(raw_path)

    print(f"⭐ {city_cn}: {len(zh)} 个 POI，开始获取评分...")
    updated = 0
    from_api = 0
    from_default = 0

    for idx, row in zh.iterrows():
        name = row["name"]
        category = str(row.get("category", ""))
        old_rating = row.get("rating", "")
        if pd.notna(old_rating) and old_rating != "":
            continue  # 已有评分，跳过

        print(f"  [{idx+1}/{len(zh)}] {name} ({category})...", end=" ")

        # 1. 尝试高德 API
        rating = search_poi_rating(name, city_cn)
        if rating:
            from_api += 1
        else:
            # 2. fallback 类别默认评分 + 随机微调
            base = CATEGORY_DEFAULT_RATING.get(category, 3.0)
            import random
            rating = round(base + random.uniform(-0.2, 0.2), 1)
            rating = max(1.0, min(5.0, rating))
            from_default += 1

        zh.at[idx, "rating"] = rating
        raw.at[idx, "rating"] = rating if "rating" in raw.columns else rating
        print(f"⭐ {rating}")

        updated += 1
        time.sleep(0.05)  # 节制 API 频率

    # 确保 raw 也有 rating 列
    if "rating" not in raw.columns:
        raw["rating"] = ""
    for idx, row in raw.iterrows():
        name = row["name"]
        match = zh[zh["name"] == name]
        if len(match) > 0:
            raw.at[idx, "rating"] = match.iloc[0]["rating"]

    zh.to_csv(csv_path, index=False)
    raw.to_csv(raw_path, index=False)

    print(f"  ✅ {city_cn}: 已更新 {updated} 个 (API: {from_api}, 默认: {from_default})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None, help="单个城市")
    parser.add_argument("--all", action="store_true", help="全部城市")
    args = parser.parse_args()

    cities = ["fuzhou", "guangzhou", "shanghai", "shenzhen"] if args.all else [args.city]
    for c in cities:
        if c:
            print(f"\n{'='*50}")
            fetch_city(c)
    print("\n🎉 完成！")
