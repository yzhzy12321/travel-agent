"""
batch_baseline.py — 批量运行 Baseline 实验
===========================================
补全福州、广州、深圳的 Baseline 数据（上海已有）。
输出：model/pk/baseline/{model}/{city}/q_{query_key}.json

用法：
    python model/eval/batch_baseline.py                     # 全部 3 城市
    python model/eval/batch_baseline.py --city fuzhou        # 单城市
    python model/eval/batch_baseline.py --models deepseek-v4-flash
"""
import argparse, datetime, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.utils.proxy_call import OpenaiCall
from model.baseline.agent import BaselineAgent

PK_DIR = os.path.join(os.path.dirname(__file__), "..", "pk")
BASE_DIR = os.path.join(PK_DIR, "baseline")

MODELS = {
    "deepseek-v4-flash": {
        "model": "deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "label": "DeepSeek-V4-Flash",
    },
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "api_base": "https://api.deepseek.com",
        "label": "DeepSeek-V4-Pro",
    },
}

CITIES = ["fuzhou", "guangzhou", "shenzhen"]
CITY_ZH = {"fuzhou": "福州", "guangzhou": "广州", "shenzhen": "深圳", "shanghai": "上海"}
DAYS = 3

QUERIES = {
    "A1_dist10km": (
        "我想在{city}玩{days}天。核心要求：每天的总移动距离必须控制在10公里以内，"
        "在此前提下尽量多去景点。景点之间的游览顺序必须按最短路径排列，避免来回绕路。"
    ),
    "A2_walk800m": (
        "我想在{city}玩{days}天。核心要求：优先选择步行距离在800米以内的相邻景点组成游览群，"
        "景点间超过2公里的必须依赖公共交通。请将步行可达的景点安排在一起。"
    ),
    "A3_min_transit": (
        "我想在{city}玩{days}天。核心要求：最小化总交通时间，每天交通时间控制在1小时内。"
        "宁愿少去景点也不能超交通预算。"
    ),
    "B1_museum_morning": (
        "我想在{city}玩{days}天。核心要求：博物馆和展览馆只在上午9:00-12:00安排参观，"
        "下午13:00-17:00安排自然风光和城市公园，晚上17:00-19:00安排美食街区和夜市。"
    ),
    "B2_stay_duration": (
        "我想在{city}玩{days}天。核心要求：博物馆90分钟、自然风光60分钟、购物美食45分钟。"
        "每天9:00开始18:00结束，合理分配不超时。"
    ),
    "C1_monday_no_museum": (
        "我想在{city}玩{days}天，出发日期是下周一。核心要求：周一绝大多数博物馆闭馆，"
        "请完全避开博物馆类景点，优先安排公园、自然风光、历史街区和购物商圈。"
    ),
    "C2_rain_indoor": (
        "我想在{city}玩{days}天。核心要求：请查询天气预报，如果某天有较大概率下雨，"
        "那天的行程应优先安排室内景点，晴天再安排户外自然风光和公园。"
    ),
    "D1_same_district": (
        "我想在{city}玩{days}天。核心要求：每天的景点必须集中在同一片地理区域，"
        "同一天内任意两个景点之间直线距离不超过5公里。"
    ),
    "D2_balanced_coverage": (
        "我想在{city}玩{days}天。核心要求：{days}天的行程要均衡覆盖城市的不同区域，"
        "每天一个主题片区，每天景点间平均间距控制在3公里以内。"
    ),
}


def get_api_key():
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        return key
    try:
        from model.config import DEEPSEEK_API_KEY
        return DEEPSEEK_API_KEY
    except ImportError:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None, help="单城市")
    parser.add_argument("--models", type=str, default=None, help="模型 key")
    parser.add_argument("--qkey", type=str, default=None, help="单个 query key")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("ERROR: no DEEPSEEK_API_KEY"); return

    cities = [args.city] if args.city else CITIES
    model_keys = [args.models] if args.models else list(MODELS.keys())
    query_keys = [args.qkey] if args.qkey else list(QUERIES.keys())

    total = len(cities) * len(model_keys) * len(query_keys)
    done = 0

    for city in cities:
        for mk in model_keys:
            mcfg = MODELS[mk]
            out_dir = os.path.join(BASE_DIR, mcfg["label"], city)
            os.makedirs(out_dir, exist_ok=True)

            proxy = OpenaiCall(api_key=api_key, base_url=mcfg["api_base"])

            for qk in query_keys:
                done += 1
                fpath = os.path.join(out_dir, f"q_{qk}.json")
                if os.path.exists(fpath):
                    print(f"[{done}/{total}] SKIP {CITY_ZH[city]} {mcfg['label']} {qk} (已存在)")
                    continue

                query = QUERIES[qk].format(city=CITY_ZH[city], days=DAYS)
                print(f"[{done}/{total}] {CITY_ZH[city]} {mcfg['label']} {qk} ...", end=" ", flush=True)

                try:
                    agent = BaselineAgent(
                        user_reqs=[query],
                        city=city,
                        num_days=DAYS,
                        type="zh",
                        proxy_call=proxy,
                        travel_date=datetime.date.today().strftime("%Y-%m-%d"),
                        citywalk=True,
                        model_override=mcfg["model"],
                        verbose=False,
                        output_dir=out_dir,
                    )
                    t0 = time.time()
                    result = agent.solve()
                    elapsed = time.time() - t0

                    # 保存为 q_ key 格式
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    n_pois = sum(len(d.get("itinerary","").split("->")) for d in result.get("days",[]))
                    print(f"OK {elapsed:.0f}s, {n_pois} POIs")

                except Exception as e:
                    print(f"FAIL: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n=== 完成: {done} 个任务 ===")


if __name__ == "__main__":
    main()
