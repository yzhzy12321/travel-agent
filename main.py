"""
Travel Agent: 多日长行程 + RAG 时空感知旅游规划
=====================================================
用法:
    python main.py --city shanghai --days 2
    python main.py --city guangzhou --days 3 --use_rag True
    python main.py --city shenzhen --days 1 --date 2026-07-01
"""

import argparse
import os
from model.utils.proxy_call import OpenaiCall
from model.agent import TravelAgent


def arg_parse(parser):
    parser.add_argument('--city', type=str, default='shanghai',
                        choices=["fuzhou", "guangzhou", "shenzhen", "shanghai"],
                        help='城市名称')
    parser.add_argument('--type', type=str, default='zh',
                        choices=["zh"], help='语言类型')
    parser.add_argument('--days', type=int, default=2,
                        help='行程天数')
    parser.add_argument('--use_rag', type=bool, default=True,
                        help='是否启用 RAG 时空感知')
    parser.add_argument('--date', type=str, default=None,
                        help='出发日期 YYYY-MM-DD')
    parser.add_argument('--hotel', type=str, default=None,
                        help='住宿点名称（模糊匹配POI库，如"外滩"、"南京路"）')
    parser.add_argument('--exclude', type=str, default=None,
                        help='要排除的景点，逗号分隔（如"鼓山,西湖公园"），模糊匹配 POI 库')
    parser.add_argument('--serve', action='store_true', default=False,
                        help='生成后启动本地 HTTP 服务器（高德地图需从 http:// 访问）')
    return parser.parse_args()


def run(args) -> None:
    # 城市中文名映射（命令行用拼音，展示用中文）
    city_zh = {"shanghai": "上海", "guangzhou": "广州", "shenzhen": "深圳", "fuzhou": "福州"}.get(args.city, args.city)
    query = (f"我想在{city_zh}玩{args.days}天，"
             f"体验当地的历史文化与现代风貌，品尝地道美食，"
             f"逛逛特色街区和公园。")

    # 解析排除景点列表
    exclude_pois = None
    if args.exclude:
        exclude_pois = [name.strip() for name in args.exclude.split(',') if name.strip()]

    planner = TravelAgent(
        user_reqs=[query],
        city=args.city,
        num_days=args.days,
        type=args.type,
        proxy_call=OpenaiCall(),
        use_rag=args.use_rag,
        travel_date=args.date,
        citywalk=True,
        hotel=args.hotel,
        exclude_pois=exclude_pois
    )
    result = planner.solve_multiday()

    # 打印结果摘要
    print(f"\n{'='*30}")
    print(f"🎯 {result.get('trip_title', '行程')}")
    print(f"{'='*30}")
    if result.get('rag_context'):
        print(result['rag_context'][:500])
    print(f"\n📋 行程总览: {result.get('trip_overview', '')}")
    if result.get('hotel'):
        h = result['hotel']
        print(f"\n🏨 住宿: {h['name']} | {h['address']}")
        nearby = result.get('hotel_nearby', [])
        if nearby:
            print(f"   周边推荐: {', '.join(n['name'] for n in nearby[:5])}")
    for day in result.get('days', []):
        print(f"\n--- Day {day['day']}: {day.get('day_theme', '')} ---")
        print(f"  路线: {day.get('itinerary', '')}")
        print(f"  理由: {day.get('day_reason', '')}")
        # 交通摘要
        ts = day.get('transit_summary', '')
        if ts and '暂缺' not in ts:
            # 只打印汇总部分
            for line in ts.split('\n'):
                if '交通汇总' in line or '步行' in line or '骑行' in line or '车辆' in line or '总移动' in line or '总花费' in line or '──' in line:
                    print(f"  {line.strip()}")
        if day.get('weather_note') and day['weather_note'] != '无':
            print(f"  🌤️ 天气提示: {day['weather_note']}")

    # 高德地图需从 http:// 访问，提供一键启动本地服务器
    if args.serve:
        import http.server
        import socketserver
        import threading
        import webbrowser
        PORT = 8899
        os.chdir("./model/output")
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            url = f"http://localhost:{PORT}/"
            print(f"\n🌐 本地服务器已启动: {url}")
            print(f"   在浏览器打开上述地址，点击 HTML 文件即可查看高德地图")
            print(f"   按 Ctrl+C 停止服务器")
            webbrowser.open(url)
            httpd.serve_forever()


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    run(args)