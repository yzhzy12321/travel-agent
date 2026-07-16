"""
Travel Agent: 全局配置文件
"""

# ==================== LLM 配置 ====================

LLM_MODEL = "deepseek-v4-flash"

# ---- 从 api.py 导入 API Keys（文件不存在时为空）----
try:
    from model.api import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
        OPENAI_API_KEY,   OPENAI_BASE_URL,
        ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
        MINIMAX_API_KEY,   MINIMAX_BASE_URL,
        STEPFUN_API_KEY,   STEPFUN_BASE_URL,
        GEMINI_API_KEY,    GEMINI_BASE_URL,
        BAILIAN_API_KEY,   BAILIAN_BASE_URL,
        GaodeMap_apikey,
    )
except ImportError:
    
    DEEPSEEK_API_KEY = DEEPSEEK_BASE_URL = ""
    OPENAI_API_KEY   = OPENAI_BASE_URL   = ""
    ANTHROPIC_API_KEY = ANTHROPIC_BASE_URL = ""
    MINIMAX_API_KEY   = MINIMAX_BASE_URL   = ""
    STEPFUN_API_KEY   = STEPFUN_BASE_URL   = ""
    GEMINI_API_KEY    = GEMINI_BASE_URL    = ""
    BAILIAN_API_KEY   = BAILIAN_BASE_URL   = ""
    
    GaodeMap_apikey = "5e22fd9c5a1925afd65b15f27cba862f"

# ---- 兼容旧引用 ----
api_key  = DEEPSEEK_API_KEY
base_url = DEEPSEEK_BASE_URL
# ==================== 城市名称映射 ====================

# 拼音 → 中文名（命令行参数 → 展示/提示词）
CITY_ZH_MAP = {
    "shanghai": "上海", "guangzhou": "广州",
    "shenzhen": "深圳", "fuzhou": "福州",
    "beijing": "北京", "hangzhou": "杭州",
    "qingdao": "青岛", "changsha": "长沙", "wuhan": "武汉",
}

# 中文名 → 英文名（wttr.in 等 API 需要英文城市名）
CITY_EN_MAP = {
    "上海": "Shanghai", "广州": "Guangzhou",
    "深圳": "Shenzhen", "福州": "Fuzhou",
    "北京": "Beijing", "杭州": "Hangzhou",
    "成都": "Chengdu", "武汉": "Wuhan",
    "南京": "Nanjing", "重庆": "Chongqing",
    "西安": "Xian",    "长沙": "Changsha",
    "青岛": "Qingdao", "厦门": "Xiamen",
}

# ==================== 天气配置 ====================

WEATHER_TYPES = [
    "晴天☀️", "多云⛅", "阴天☁️", "小雨🌧️", "中雨🌧️",
    "大雨⛈️", "阵雨🌦️", "雾霾🌫️", "大风💨", "酷热🔥", "寒冷❄️"
]

WEATHER_ACTIVITY_MAP = {
    "晴天☀️": "非常适合户外活动，推荐公园、山景、滨水区域",
    "多云⛅": "适合大多数户外活动，光线柔和适合拍照",
    "阴天☁️": "户外活动尚可，建议备选室内景点以防下雨",
    "小雨🌧️": "建议以室内活动为主（博物馆、商场、咖啡馆），户外需带伞",
    "中雨🌧️": "强烈建议安排室内景点，户外活动体验较差",
    "大雨⛈️": "应避免户外活动，全部安排室内场所",
    "阵雨🌦️": "可安排部分户外活动，但需有室内备选方案",
    "雾霾🌫️": "不建议登山等户外运动，空气质量差，建议室内",
    "大风💨": "不适合滨水和高空活动，建议选择有遮挡的场所",
    "酷热🔥": "建议上午/傍晚户外活动，中午安排室内避暑",
    "寒冷❄️": "户外活动需保暖，优先选择有暖气的室内场所",
}

# wttr.in 英文天气描述 → 中文天气类型
WTTR_WEATHER_MAP = {
    "sunny": "晴天☀️", "clear": "晴天☀️",
    "partly cloudy": "多云⛅", "partly cloudy ": "多云⛅",
    "cloudy": "阴天☁️", "overcast": "阴天☁️",
    "mist": "雾霾🌫️", "fog": "雾霾🌫️", "haze": "雾霾🌫️",
    "patchy rain nearby": "小雨🌧️", "patchy rain possible": "小雨🌧️",
    "light drizzle": "小雨🌧️", "light rain": "小雨🌧️",
    "light rain shower": "阵雨🌦️", "patchy light rain": "阵雨🌦️",
    "patchy light drizzle": "小雨🌧️",
    "moderate rain": "中雨🌧️", "moderate rain at times": "中雨🌧️",
    "heavy rain": "大雨⛈️", "heavy rain at times": "大雨⛈️",
    "torrential rain shower": "大雨⛈️",
    "thunderstorm": "大雨⛈️", "thundery outbreaks possible": "大雨⛈️",
    "patchy light snow": "寒冷❄️", "light snow": "寒冷❄️",
    "moderate snow": "寒冷❄️", "heavy snow": "寒冷❄️",
    "blizzard": "寒冷❄️", "blowing snow": "寒冷❄️",
    "freezing fog": "雾霾🌫️", "freezing drizzle": "小雨🌧️",
    "ice pellets": "寒冷❄️",
}

# ==================== 客流时段 ====================

TIME_SLOT_CROWD = {
    "morning":       ("08:00-10:00", "客流较少，适合热门景点"),
    "late_morning":  ("10:00-12:00", "客流逐渐增多"),
    "noon":          ("12:00-14:00", "午餐高峰，美食街区人流量大"),
    "afternoon":     ("14:00-17:00", "客流高峰，热门景点可能拥挤"),
    "evening":       ("17:00-19:00", "晚高峰交通拥堵，景点客流回落"),
    "night":         ("19:00-22:00", "夜生活时段，夜景和夜市人气旺"),
}

# ==================== POI 类别常识 ====================

CATEGORY_TIPS = {
    "博物馆/纪念馆": "⏰ 通常 9:00-17:00 开放，周一可能闭馆",
    "自然风光":      "⏰ 建议上午前往，光线好且凉爽",
    "美食街区":      "🍽️ 午餐 11:30-13:30、晚餐 17:30-19:30 为高峰",
    "购物商圈":      "🛍️ 10:00-22:00 营业，周末人较多",
    "城市公园":      "🏃 晨练 6:00-8:00、傍晚 17:00-19:00 人较多",
    "宗教寺庙":      "🙏 通常 8:00-17:00 开放，注意着装",
    "历史街区":      "📸 清晨或傍晚光线最佳，适合拍照",
    "建筑地标":      "🌃 夜景通常比白天更有特色",
    "文化艺术":      "🎭 注意演出时间，部分需提前预约",
    "主题乐园":      "🎢 建议工作日前往，避开周末高峰",
}

# ==================== POI 建议停留时长（分钟） ====================
# 可在命令行用 --visit 覆盖，或直接改这里
DEFAULT_VISIT_MINUTES = 30  # 默认停留时长

CATEGORY_VISIT_MINUTES = {
    "博物馆/纪念馆":   90,
    "文化艺术":       45,
    "历史街区":       30,
    "建筑地标":       20,
    "自然风光":       120,
    "城市公园":       60,
    "美食街区":       30,
    "购物商圈":       45,
    "宗教寺庙":       30,
    "主题乐园":       180,
    "历史遗迹":       30,
    "海滨沙滩":       90,
    "特色村落":       60,
    "温泉度假":       180,
    "高校学府":       40,
    "演艺/剧院":      90,
}

# ==================== POI 类别默认评分（1-5） ====================
# 高德 API 未覆盖时使用的 fallback 值，基于大众点评/携程认知
CATEGORY_DEFAULT_RATING = {
    "博物馆/纪念馆": 3.8,
    "文化艺术":       3.6,
    "历史街区":       3.7,
    "建筑地标":       3.5,
    "自然风光":       3.9,
    "城市公园":       3.6,
    "美食街区":       3.5,
    "购物商圈":       3.4,
    "宗教寺庙":       3.7,
    "主题乐园":       3.8,
    "历史遗迹":       3.6,
    "海滨沙滩":       3.7,
    "特色村落":       3.5,
    "温泉度假":       3.7,
    "高校学府":       3.5,
    "演艺/剧院":      3.7,
}

# ==================== 每日行程时间 ====================
# 格式: "HH:MM"，可在命令行用 --start / --end 覆盖
DEFAULT_START_TIME = "08:00"   # 每天默认出发时间
DEFAULT_END_TIME   = "20:00"   # 每天默认结束时间（超过则警告）

# ==================== 多目标优化 ====================
# 距离 vs 品质 的权衡系数（0=只管距离，1=距离和品质各半，>1=品质优先）
POI_QUALITY_WEIGHT = 0.3   # 建议 0.2~0.5

# ==================== 多日行程参数 ====================
# 格式: (每天最少聚簇数, 每天最少POI数, 步行距离阈值)

MULTIDAY_TIME2NUM = {
    1: [(6, 9, 5000)],
    2: [(5, 7, 4000), (5, 7, 4000)],
    3: [(5, 7, 4000), (5, 7, 4000), (5, 7, 4000)],
    4: [(5, 7, 4000), (5, 7, 4000), (4, 5, 3000), (4, 5, 3000)],
    5: [(5, 7, 4000), (5, 7, 4000), (4, 5, 3000), (4, 5, 3000), (4, 5, 3000)],
    6: [(5, 6, 4000), (5, 6, 4000), (4, 5, 3000), (4, 5, 3000), (4, 5, 3000), (3, 4, 3000)],
    7: [(5, 6, 4000), (5, 6, 4000), (4, 5, 3000), (4, 5, 3000), (4, 5, 3000), (3, 4, 3000), (3, 4, 3000)],
}
