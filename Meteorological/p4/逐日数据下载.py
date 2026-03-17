import os
import time
import random
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

# -----------------------------
# Config
# -----------------------------
DATASET = "derived-era5-single-levels-daily-statistics"
OUTDIR = "/Volumes/TiPlus7100/data/ERA5/daily/t2m"
START_YEAR = 1961
END_YEAR = 2025

# 并发线程数
MAX_WORKERS = 4

# 每个任务失败后的最大重试次数
MAX_RETRIES = 5

# 多个 CDS API 账号
# 去 https://cds.climate.copernicus.eu/ 拿你的 uid:key
# url 一般固定；key 格式通常是 "uid:apikey"
API_ACCOUNTS = [
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
    }
]

os.makedirs(OUTDIR, exist_ok=True)

months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
var = ["2m_temperature"]

# 账号池
account_queue = Queue()
for acc in API_ACCOUNTS:
    account_queue.put(acc)

print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def build_request(year: int):
    return {
        "product_type": "reanalysis",
        "variable": var,
        "year": [str(year)],
        "month": months,
        "day": days,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "3_hourly",
        # 如需 netcdf 可打开
        # "data_format": "netcdf",
        # 如需限定区域可打开
        # "area": [north, west, south, east],
    }


def get_client(account: dict):
    # 每次线程里单独创建 client，避免线程间共用对象
    return cdsapi.Client(
        url=account["url"],
        key=account["key"],
        quiet=True,
        debug=False,
    )


def download_one_year(year: int):
    target = os.path.join(OUTDIR, f"era5_t2m_daily_{year}.zip")
    tmp_target = target + ".part"

    if os.path.exists(target):
        log(f"[Skip] {year}: already exists -> {target}")
        return year, "skip"

    request = build_request(year)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        account = None
        try:
            # 从账号池拿一个账号
            account = account_queue.get(timeout=60)
            acc_name = account["name"]

            log(f"[Start] {year} | attempt={attempt} | account={acc_name}")

            client = get_client(account)

            # retrieve + download 到临时文件
            result = client.retrieve(DATASET, request)
            result.download(tmp_target)

            # 下载成功后再改名，避免中断留下坏文件
            os.replace(tmp_target, target)

            log(f"[Done]  {year} -> {target} | account={acc_name}")
            return year, "ok"

        except Exception as e:
            last_err = e
            log(f"[Error] {year} | attempt={attempt} | err={repr(e)}")

            # 清理临时文件
            if os.path.exists(tmp_target):
                try:
                    os.remove(tmp_target)
                except Exception:
                    pass

            # 指数退避 + 随机抖动
            sleep_s = min(60, 2 ** attempt + random.uniform(0.5, 3.0))
            time.sleep(sleep_s)

        finally:
            if account is not None:
                account_queue.put(account)

    log(f"[Fail]  {year} after {MAX_RETRIES} retries | last_err={repr(last_err)}")
    return year, "fail"


def main():
    years = list(range(START_YEAR, END_YEAR + 1))

    ok_count = 0
    skip_count = 0
    fail_count = 0
    failed_years = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one_year, year): year for year in years}

        for future in as_completed(futures):
            year = futures[future]
            try:
                _, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                    failed_years.append(year)
            except Exception as e:
                fail_count += 1
                failed_years.append(year)
                log(f"[FutureError] {year} -> {repr(e)}")

    log("======================================")
    log(f"Finished. ok={ok_count}, skip={skip_count}, fail={fail_count}")
    if failed_years:
        log(f"Failed years: {failed_years}")


if __name__ == "__main__":
    main()
