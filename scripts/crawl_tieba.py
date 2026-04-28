# -*- coding: utf-8 -*-
"""
crawl_tieba.py — 爬取百度贴吧帖子正文与评论
输出格式：JSONL，每行一条文本，label 字段留空待人工/模型标注
用法：
    python scripts/crawl_tieba.py --kw 湖南第一师范学院 --pages 10 --out data/tieba_raw.jsonl
"""
import argparse
import json
import time
import random
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://tieba.baidu.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─── 工具函数 ────────────────────────────────────────────

def sleep():
    """随机延迟 1~3 秒，避免对服务器造成压力"""
    time.sleep(random.uniform(1.0, 3.0))


def get_soup(url: str, retries: int = 3) -> BeautifulSoup | None:
    for i in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.encoding = "utf-8"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            print(f"  [warn] HTTP {resp.status_code}: {url}")
        except Exception as e:
            print(f"  [error] 第{i+1}次请求失败: {e}")
        sleep()
    return None


def clean(text: str) -> str:
    """去除多余空白和换行"""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── 爬取帖子列表 ────────────────────────────────────────

def get_post_links(kw: str, page_count: int) -> list[str]:
    """从吧首页分页获取帖子链接"""
    links = []
    encoded_kw = quote(kw)

    for page in range(page_count):
        pn = page * 50
        url = f"https://tieba.baidu.com/f?kw={encoded_kw}&pn={pn}"
        print(f"[列表] 第 {page+1}/{page_count} 页: {url}")
        soup = get_soup(url)
        if not soup:
            continue

        for a in soup.select("a.j_th_tit"):
            href = a.get("href", "")
            if href.startswith("/p/"):
                full = "https://tieba.baidu.com" + href
                if full not in links:
                    links.append(full)

        sleep()

    print(f"[列表] 共获取 {len(links)} 个帖子链接")
    return links


# ─── 爬取单个帖子 ────────────────────────────────────────

def get_post_texts(post_url: str) -> list[str]:
    """爬取帖子所有分页的正文+评论文字"""
    texts = []
    page = 1

    while True:
        url = post_url if page == 1 else f"{post_url}?pn={page}"
        soup = get_soup(url)
        if not soup:
            break

        # 楼层内容
        for div in soup.select("div.d_post_content"):
            text = clean(div.get_text())
            if text and len(text) >= 5:
                texts.append(text)

        # 判断是否有下一页
        next_btn = soup.select_one("a.next.pagination-item")
        if not next_btn:
            break

        page += 1
        sleep()

    return texts


# ─── 主流程 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kw", type=str, default="湖南第一师范学院", help="贴吧关键词")
    parser.add_argument("--pages", type=int, default=5, help="爬取贴吧列表页数（每页约50帖）")
    parser.add_argument("--post_pages", type=int, default=3, help="每个帖子最多爬取的分页数")
    parser.add_argument("--out", type=str, default="data/tieba_raw.jsonl", help="输出文件路径")
    args = parser.parse_args()

    post_links = get_post_links(args.kw, args.pages)

    results = []
    seen = set()

    for i, link in enumerate(post_links):
        print(f"[帖子 {i+1}/{len(post_links)}] {link}")
        texts = get_post_texts(link)
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            results.append({
                "text": text,
                "source": link,
                # 标签字段留空，后续用模型或人工打标
                "ad": -1,
                "abuse": -1,
                "negative": -1,
                "misinfo": -1,
            })
        print(f"  获取 {len(texts)} 条文本，累计 {len(results)} 条")

    with open(args.out, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n完成！共 {len(results)} 条文本，已保存至 {args.out}")


if __name__ == "__main__":
    main()
