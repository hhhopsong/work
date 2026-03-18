import cdsapi
import os
import threading
from queue import Queue
import calendar
import requests
import json
import time
from typing import List, Tuple

# ============================================================
# ======================= User settings ======================
# ============================================================

# --------- Time range (inclusive) ----------
start_year, start_month = 1961, 1
end_year, end_month = 2025, 12

# CDS area order: [North, West, South, East]
# Northern Hemisphere + Eastern Hemisphere (0–90N, 0–180E)
NH_EHEM_AREA = [90, 0, 0, 180]

# Variables: Surface Pressure
# 注意：这里下载的是日平均(daily_mean)的地面气压，用于配合你之前的日平均IVT数据
VARS = [
    "2m_temperature",
]

# Dataset: ERA5 daily statistics on single levels
dataset = "derived-era5-single-levels-daily-statistics"

# Output directory (已修改文件夹名，避免与IVT混淆)
output_dir = "/Volumes/TiPlus7100/data/ERA5/daily/t2m"
os.makedirs(output_dir, exist_ok=True)

# Accounts pool
# key format: "UID:APIKEY"
ACCOUNTS = API_ACCOUNTS = [
    {
        "name": "sty",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "c1128eeb-5ffb-47c5-bf53-d9ee49bd0ee8",
    },
    {
        "name": "gmail",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "597a9d0a-c76d-455c-ba3e-fd216892f2a5",
    },
    {
        "name": "outlook",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "44c4b366-992b-4d64-a23c-762df238b256",
    },
    {
        "name": "outlook01",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "70e7cd2e-5244-49c6-bf56-daa8fea76a19",
    },
    {
        "name": "outlook02",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "e40466fc-198d-45e5-b624-3de5a7895290",
    },
    {
        "name": "IsabellPricecfry@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "4a693aa0-e52c-4db3-b867-c241883c4f4c",
    },
    {
        "name": "TiaMosciskigngp@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "d1a22e82-0d42-4bb9-b742-b4b8794f5d77",
    },
    {
        "name": "MustafaJerdecmyn@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "485386f6-bbc8-4700-a5cc-5321a91415b2",
    },
    {
        "name": "CatherineLangsr@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "f70904af-aedb-4976-9cae-551bf63111fe",
    },
    {
        "name": "WestleyMrazxzx@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "81ae194d-721c-4a7d-8f4e-0b266b314a77",
    },
    {
        "name": "AliviaWeberqs@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "ed61650f-1b8f-4cd4-a2f1-f7a9079628ba",
    },
    {
        "name": "JefferyRennerrpjj@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "2ac20345-3c6a-4e00-aabb-93f2db01b45b",
    },
    {
        "name": "SusieRiceomh@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "be50cad9-0285-4773-8860-3f680932ad22",
    },
    {
        "name": "JohnJacobsonzjnm@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "95ca9611-0e51-407c-a64e-70abdc0e17b1",
    },
    {
        "name": "BabyKubkhp@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "64c6067e-9c80-495a-9838-ecb01ab49a21",
    },
    {
        "name": "ArchibaldHillsath@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "32c310df-8eec-4438-bcb2-009b8b54cd8e",
    },
    {
        "name": "EmilieMoenef@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "000ad20e-c198-4da6-b565-cce54809c831",
    },
    {
        "name": "DarleneAltenwerthho@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "054bb282-cfd0-40d5-9b00-29973b6cec3d",
    },
    {
        "name": "ArneAltenwerthwhn@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "92f932d6-414f-489a-8205-d7c1f3dd8d16",
    },
    {
        "name": "AlfonsoJacobswc@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "be8ae80f-aad4-4a1d-aeb0-35b3c25ba945",
    },
    {
        "name": "JalonRolfsonal@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "88571325-d938-43ed-be46-570c908c77e6",
    },
    {
        "name": "CordieCasperyc@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "87d2ef9f-fa6a-4d51-826f-84dcf1a30cd3",
    },
    {
        "name": "TravonShieldsosse@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "f237c2e7-66f1-4eb0-ae1f-97575385ef98",
    },
    {
        "name": "MagdalenaSchadenosvy@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "20522dac-bb52-4eed-a28c-d5a2bd44bcc9",
    },
    {
        "name": "MurphyVeumnomjb@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "2d879c75-25cc-433a-a342-27c353217cfe",
    },
    {
        "name": "DaleBotsfordrd@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "b91d5c5c-6803-4159-ba9d-e4e09c9941f2",
    },
    {
        "name": "KevenWolfgjp@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "a99f1166-f709-465b-ac00-0e139108b6ce",
    },
    {
        "name": "ChaseRutherfordrfx@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "8a833e23-2376-455a-9a06-997699e6287e",
    },
    {
        "name": "DerrickBoganvj@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "54a29a40-9007-4b02-ad4c-5766fbe16e1f",
    },
    {
        "name": "JeramyHagenessauij@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "7c4690cb-682a-47ea-8771-b71e86a5b2bf",
    },
    {
        "name": "RyleyWalternf@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "1be5a570-fc32-4f2c-bf53-409e22fb0c23",
    },
    {
        "name": "NicholasGreen9389@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "e3498bd0-ded5-47c2-841e-de9a83110989",
    },
    {
        "name": "KarenWilkerson6715@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "03a55d2e-32f4-4cb5-9a73-b833b041fd87",
    },
    {
        "name": "LuisBryant1509@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "a93421fb-743a-466e-afaf-44592bb3d46c",
    },
    {
        "name": "AdrianCrane2402@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "ee320bd4-45f5-44b2-b458-9656b8b2aee8",
    },
    {
        "name": "RichardHoffman2974@outlook.com",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "595ca59f-272a-4256-ac8b-bd98a66c75ce",
    },

]

# Request settings
GRID = "1.0/1.0"
DAILY_STATISTIC = "daily_mean"
TIME_ZONE = "utc+00:00"
FREQUENCY = "1_hourly"
FILE_PREFIX = "ERA5_daily_"  # 修改了文件前缀


# ============================================================
# ====================== Helper functions ====================
# ============================================================

def get_days_of_month(year: int, month: int) -> List[str]:
    ndays = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, ndays + 1)]


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month) from start to end inclusive."""
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


# Task queue
task_queue = Queue()


# ============================================================
# ================= Motrix RPC download helper ===============
# ============================================================

def motrix_rpc_download(url, dir_path=None, filename=None) -> bool:
    rpc_url = "http://localhost:16800/jsonrpc"
    options = {}
    if dir_path:
        options["dir"] = dir_path
    if filename:
        options["out"] = filename

    data = {
        "jsonrpc": "2.0",
        "id": "qwer",
        "method": "aria2.addUri",
        "params": [[url], options],
    }

    try:
        response = requests.post(rpc_url, json=data, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 Motrix RPC 服务 ({rpc_url})。请确保 Motrix 正在运行并启用 RPC。")
    except requests.exceptions.Timeout:
        print("错误: 连接到 Motrix RPC 服务超时。")
    except requests.exceptions.RequestException as e:
        print(f"错误: Motrix RPC 请求失败: {e}")
    return False


# ============================================================
# ================== Download task execution =================
# ============================================================

def execute_download_task(client_instance, request_params, target_directory, output_fname, worker_id_str):
    try:
        outpath = os.path.join(target_directory, output_fname)
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            print(f"[{worker_id_str}] {output_fname} 已存在，跳过")
            return

        print(f"[{worker_id_str}] 请求：{output_fname}")

        # CDSAPI retrieve
        r = client_instance.retrieve(dataset, request_params)

        url = getattr(r, "location", None)
        if url:
            print(f"[{worker_id_str}] 获取到 URL：{url}")
            ok = motrix_rpc_download(url, dir_path=target_directory, filename=output_fname)
            if ok:
                print(f"[{worker_id_str}] 已发送到 Motrix：{output_fname}")
                return
            else:
                print(f"[{worker_id_str}] Motrix 不可用，回退为 cdsapi 直下：{output_fname}")

        r.download(outpath)
        print(f"[{worker_id_str}] cdsapi 下载完成：{outpath}")

    except Exception as e:
        print(f"[{worker_id_str}] 失败 {output_fname}: {e}")
        with open(os.path.join(target_directory, "error_log.txt"), "a", encoding="utf-8") as ef:
            ef.write(f"{worker_id_str} {output_fname}: {e}\n")


# ============================================================
# ===================== Worker thread ========================
# ============================================================

def worker_function(client, worker_name, target_directory):
    print(f"线程 {worker_name} 启动")
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            print(f"线程 {worker_name} 接收到停止信号")
            break

        req_params, fname = task
        # 这里的 client 是 cdsapi.Client 对象
        execute_download_task(client, req_params, target_directory, fname, worker_name)
        task_queue.task_done()

    print(f"线程 {worker_name} 退出")


# ============================================================
# ============================ Main ==========================
# ============================================================

def main():
    # Initialize CDS clients
    clients: List[Tuple[str, cdsapi.Client]] = []
    for acc in ACCOUNTS:
        # 注意：如果不填写 key，cdsapi 会尝试读取 ~/.cdsapirc
        if "***************" in acc["key"]:
            print(f"警告：{acc['name']} 的 API KEY 未配置，请检查代码。")
        cli = cdsapi.Client(url=acc["url"], key=acc["key"], timeout=600)
        clients.append((acc["name"], cli))

    if not clients:
        raise RuntimeError("ACCOUNTS 为空：请至少配置一个可用的 CDS API 账号。")

    # Start threads
    threads = []
    # 这里将任务分发给不同的 client 执行
    for name, cli in clients:
        t = threading.Thread(target=worker_function, args=(cli, name, output_dir))
        t.daemon = True
        t.start()
        threads.append(t)

    # Build monthly tasks
    print("正在生成任务队列...")
    for y, m in iter_months(start_year, start_month, end_year, end_month):
        year_str = str(y)
        mon_str = f"{m:02d}"
        days = get_days_of_month(y, m)

        fname = f"{FILE_PREFIX}"+VARS[0]+f"_{year_str}{mon_str}.nc"
        outpath = os.path.join(output_dir, fname)

        # 简单检查文件是否存在
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            print(f"{fname} 已存在，跳过")
            continue

        # ============================================================
        # ========================== 请求参数 =========================
        # ============================================================
        req = {
            "product_type": "reanalysis",
            "variable": VARS,
            "year": year_str,
            "month": mon_str,
            "day": days,
            "daily_statistic": DAILY_STATISTIC,
            "time_zone": TIME_ZONE,
            "frequency": FREQUENCY,
            "grid": GRID,
            "format": "netcdf",
        }

        # 将 (参数, 文件名) 放入队列
        task_queue.put((req, fname))

    print(f"任务队列生成完毕，共 {task_queue.qsize()} 个任务，开始下载...")

    # Send stop signals (每个线程一个 None)
    for _ in threads:
        task_queue.put(None)

    # Wait for completion
    # task_queue.join() 会阻塞直到所有任务被处理完
    task_queue.join()

    # 等待线程真正结束
    for t in threads:
        t.join()

    print("所有 SP 下载任务完成")


if __name__ == "__main__":
    main()
