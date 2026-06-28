# ROAM: A Retrieval-augmented Optimization Agentic Model for Spontaneous Trip Blueprint Generation


## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **真实道路距离** | OSRM 驾车距离替代直线距离，自动缓存 |
| **时间窗约束** | 自动避开关门时间，TSP 后贪心交换修正 |
| **多目标优化** | 距离 + 评分联合优化，高分 POI 引力更大 |
| **空间智能分天** | K-Means 聚类确保每天 POI 地理内聚 |
| **LLM 精选重优化** | AI 精选后二次 TSP 排序，安全网校验 |
| **多模型支持** | DeepSeek / GPT / Claude / Gemini 等 6 厂商 |
| **流式对话** | AI 对话逐字输出，行程卡片嵌入聊天流 |
| **对话持久化** | 自动保存历史，支持恢复/删除 |
| **交互式地图** | 高德瓦片 + GCJ-02 坐标系 + 每日路线 |

## 🚀 快速开始

### 1. 安装依赖 & 构建数据

```bash
conda create -n roam python=3.10 -y
conda activate roam
pip install -r requirements.txt

cd model/data && python build_dataset.py && cd ../..
```

### 2. 配置 API Key

创建 `model/api.py`：

```python
# DeepSeek
DEEPSEEK_API_KEY  = "sk-your-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

或直接在 Web 前端「🔑 API 配置」中填入。

### 3. 启动

**Web 前端（推荐）：**
```bash
streamlit run app.py
# → 打开 http://localhost:8501
```

**命令行：**
```bash
python main.py --city fuzhou --days 3
python main.py --city shanghai --days 2 --hotel "外滩" --exclude "鼓山"
python main.py --city shenzhen --days 1 --date 2026-07-01
```

## 📁 项目结构

```
travel-agent-main/
├── app.py                 # Streamlit Web 前端
├── main.py                # CLI 入口
├── requirements.txt
├── model/
│   ├── agent.py           # TravelAgent 核心调度
│   ├── spatial.py         # 空间聚类 + TSP + 时间窗
│   ├── search.py          # BGE 语义搜索
│   ├── rag.py             # 天气/交通 RAG 上下文
│   ├── config.py          # 全局配置（可安全提交）
│   ├── api.py             # API Key
│   ├── data/              # 4 城市 POI 数据 + 预处理脚本
│   │   ├── fuzhou_raw.csv  
│   │   ├── guangzhou_raw.csv
│   │   ├── shanghai_raw.csv
│   │   ├── shenzhen_raw.csv
│   │   ├── build_dataset.py
│   │   ├── fetch_ratings.py
│   │   ├── geocode_pois.py
│   │   └── fix_dup_coords.py
│   ├── output/            # 生成的行程 JSON + HTML 地图
│   └── utils/
│       ├── proxy_call.py  # LLM API 调用（OpenAI SDK 兼容）
│       ├── all_prompts.py # Prompt 模板
└──     └── funcs.py       # 工具函数
```

## 🔧 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--city` | 城市 (fuzhou/guangzhou/shenzhen/shanghai) | shanghai |
| `--days` | 行程天数 | 2 |
| `--date` | 出发日期 YYYY-MM-DD | 明天 |
| `--hotel` | 住宿点（模糊匹配） | 无 |
| `--exclude` | 排除景点（逗号分隔） | 无 |
| `--start` | 每日出发时间 | 08:00 |
| `--end` | 每日结束时间 | 20:00 |
| `--serve` | 生成后启动本地 HTTP 服务器 | False |

## 🌐 Web 前端功能

- **AI 对话**：独立于行程规划，配好 Key 就能聊，流式输出
- **行程规划**：侧边栏设置参数 → 生成 → 卡片嵌入聊天流
- **行程查看**：独立详情页（路线 + 时间线 + 交通 + 地图）
- **对话管理**：自动保存 + 加载历史 + 删除
- **调试工具**：一键加载最近行程 JSON，加速调试
- **多模型切换**：下拉选择，Base URL 自动捆绑

## 📊 数据说明

四个城市共 515 个 POI，坐标均为高德 GCJ-02 坐标系（与高德瓦片匹配）：

| 城市 | POI 数 | 评分来源 |
|------|:------:|---------|
| 福州 | 130 | 高德 API 抓取 |
| 广州 | 132 | 高德 API 抓取 |
| 上海 | 127 | 高德 API 抓取 |
| 深圳 | 121 | 高德 API 抓取 |

