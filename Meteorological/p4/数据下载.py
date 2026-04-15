import cdsapi
import os
import threading
from queue import Queue
import calendar
import requests
import json
import time
import hashlib                          # [NEW]
from typing import List, Tuple, Optional

# ============================================================
# ======================= User settings ======================
# ============================================================

start_year, start_month = 1961, 1
end_year,   end_month   = 2022, 12

NH_EHEM_AREA = [90, 0, 0, 180]
#dataset = "derived-era5-single-levels-daily-statistics" # 单层
dataset    = "derived-era5-pressure-levels-daily-statistics"
output_dir = "/Volumes/TiPlus7100/data/ERA5/daily/uvwztSh"
os.makedirs(output_dir, exist_ok=True)

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

GRID            = "1.0/1.0"
DAILY_STATISTIC = "daily_mean"
TIME_ZONE       = "utc+00:00"
FREQUENCY       = "1_hourly"
FILE_PREFIX     = "ERA5_daily_"
VARS            = ["geopotential", "specific_humidity", "temperature",
                   "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]
p_level         = ["200", "500", "850", "925"]

# [NEW] 缓存文件路径
CACHE_FILE = os.path.join(output_dir, "submitted_jobs_cache.json")

# ============================================================
# =================== [NEW] Job Cache ========================
# ============================================================

class JobCache:
    """
    线程安全的本地缓存。
    记录已提交任务的 request_id，程序重启后可复用，避免重复提交。
    使用原子写入（先写 .tmp 再 os.replace），防止崩溃时缓存损坏。
    """
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self._lock = threading.Lock()
        self._data: dict = self._load()
        print(f"[Cache] 加载缓存：{len(self._data)} 条记录  →  {cache_file}")

    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Cache] 警告：读取缓存失败（{e}），从空缓存启动")
        return {}

    def _save(self):
        """调用方须持有 _lock。原子写入，防止写到一半时崩溃导致 JSON 损坏。"""
        tmp = self.cache_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.cache_file)

    def get(self, fp: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(fp)

    def set(self, fp: str, info: dict):
        with self._lock:
            self._data[fp] = info
            self._save()

    def remove(self, fp: str):
        with self._lock:
            if fp in self._data:
                del self._data[fp]
                self._save()


# ============================================================
# ============= [NEW] 请求指纹 + CDS REST 辅助 ===============
# ============================================================

def compute_fingerprint(dataset_name: str, req: dict) -> str:
    """对 (数据集名, 请求参数) 计算 MD5，用于唯一标识一个任务。"""
    canonical = json.dumps({"dataset": dataset_name, **req},
                           sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


# ============================================================
# ============= [FIX] CDS API v2 REST 辅助函数 ==============
# ============================================================

def _make_session(account: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "PRIVATE-TOKEN": account["key"],
        "Content-Type": "application/json",
    })
    return s


def _submit_job(account: dict, dataset_name: str, req: dict) -> str:
    """
    CDS API v2 提交端点：
      POST {base}/retrieve/v1/processes/{dataset}/execution
      Body: {"inputs": req}
    立即返回 job_id，不阻塞等待。
    """
    url  = f"{account['url']}/retrieve/v1/processes/{dataset_name}/execution"
    resp = _make_session(account).post(url, json={"inputs": req}, timeout=120)

    if resp.status_code == 404:
        raise RuntimeError(
            f"404：数据集路径不存在，请检查 dataset 名称是否正确。\n"
            f"  URL  = {url}\n"
            f"  账号 = {account['name']}"
        )
    resp.raise_for_status()

    data   = resp.json()
    job_id = data.get("jobID") or data.get("id") or data.get("request_id")
    if not job_id:
        raise RuntimeError(f"CDS 未返回 jobID，完整响应：{data}")
    return job_id


def _check_job(account: dict, job_id: str) -> Optional[dict]:
    """
    CDS API v2 查询端点：
      GET {base}/retrieve/v1/jobs/{job_id}
    任务不存在（404）时返回 None。
    """
    url = f"{account['url']}/retrieve/v1/jobs/{job_id}"
    try:
        resp = _make_session(account).get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [CDS] 查询任务状态异常 {job_id}: {e}")
        return None

# ============================================================
# ===== [FIX 1] 哨兵文件：代替"靠 Motrix 返回值判断完成" =====
# ============================================================

def _pending_path(outpath: str) -> str:
    """下载中标记文件，存在即表示该文件正在/曾经被 Motrix 接管。"""
    return outpath + ".pending"

def _download_url(url: str, target_directory: str,
                  output_fname: str, worker_id: str) -> str:
    """
    返回值改为三种字符串状态：
      "done"   — 直接下载完成，文件已落盘
      "motrix" — 已交给 Motrix，文件尚未落盘
      "fail"   — 下载失败
    """
    outpath = os.path.join(target_directory, output_fname)
    pending = _pending_path(outpath)

    ok = motrix_rpc_download(url, dir_path=target_directory, filename=output_fname)
    if ok:
        # 写哨兵文件，记录"Motrix 已接管但尚未确认完成"
        try:
            with open(pending, "w") as f:
                f.write(f"{url}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        except Exception:
            pass
        print(f"[{worker_id}] 已发送 Motrix（哨兵已写）：{output_fname}")
        return "motrix"

    # Motrix 不可用 → 直接流式下载
    print(f"[{worker_id}] Motrix 不可用，直接下载：{output_fname}")
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if os.path.exists(pending):
            os.remove(pending)
        print(f"[{worker_id}] 直接下载完成：{outpath}")
        return "done"
    except Exception as e:
        print(f"[{worker_id}] 直接下载失败：{e}")
        if os.path.exists(outpath):
            os.remove(outpath)
        return "fail"

def _get_job_results(account: dict, job_id: str) -> Optional[str]:
    """
    CDS API v2 结果端点：
      GET {base}/retrieve/v1/jobs/{job_id}/results
    返回下载 URL（asset href），失败返回 None。
    """
    url = f"{account['url']}/retrieve/v1/jobs/{job_id}/results"
    try:
        resp = _make_session(account).get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 响应结构：{"asset": {"value": {"href": "..."}}}
        href = (data.get("asset", {})
                    .get("value", {})
                    .get("href"))
        # 部分版本直接放在顶层
        if not href:
            href = data.get("location") or data.get("url")
        return href
    except Exception as e:
        print(f"  [CDS] 获取结果链接失败 {job_id}: {e}")
        return None


# ============================================================
# ===== [FIX 3] _wait_and_download 返回值与 _download_url 对齐
# ============================================================

def _wait_and_download(account: dict, request_id: str,
                       target_directory: str, output_fname: str,
                       worker_id: str,
                       poll_interval: int = 60,
                       max_wait_hours: int = 48):
    """
    返回 "done" / "motrix" / "fail"，与 _download_url 一致。
    """
    max_polls = int(max_wait_hours * 3600 / poll_interval)
    for i in range(max_polls):
        info = _check_job(account, request_id)
        if info is None:
            print(f"[{worker_id}] 任务 {request_id} 已不存在于 CDS")
            return "fail"

        status = info.get("status", "unknown")
        if i % 5 == 0 or status in ("successful", "failed"):
            print(f"[{worker_id}] {output_fname}  status={status}  "
                  f"已等待 {i * poll_interval // 60} min")

        if status == "successful":
            location = _get_job_results(account, request_id)
            if not location:
                print(f"[{worker_id}] 任务完成但取不到下载链接，job_id={request_id}")
                return "fail"
            return _download_url(location, target_directory, output_fname, worker_id)

        if status == "failed":
            reason = info.get("message") or info.get("detail") or str(info)
            print(f"[{worker_id}] 任务 {request_id} 失败：{reason}")
            return "fail"

        time.sleep(poll_interval)

    print(f"[{worker_id}] 等待超时，job_id={request_id}")
    return "fail"

# ============================================================
# ===== [FIX 4] main()：启动时扫描哨兵文件 ===================
# ============================================================

def _cleanup_finished_motrix(output_dir: str, job_cache: JobCache):
    """
    启动时检查哨兵文件：
    - 若对应 .nc 文件已存在且完整 → 删哨兵 + 删缓存条目（Motrix 已完成）
    - 否则保留哨兵，走正常的任务队列流程继续处理
    """
    pending_files = [f for f in os.listdir(output_dir) if f.endswith(".nc.pending")]
    if not pending_files:
        return
    print(f"[Startup] 发现 {len(pending_files)} 个哨兵文件，开始扫描...")
    for pf in pending_files:
        nc_name = pf[:-len(".pending")]
        nc_path = os.path.join(output_dir, nc_name)
        p_path  = os.path.join(output_dir, pf)
        if os.path.exists(nc_path) and os.path.getsize(nc_path) > 0:
            print(f"[Startup] Motrix 已完成：{nc_name}，清理哨兵")
            os.remove(p_path)
            # 缓存条目在 execute_download_task 开头的文件检查处会被清理
        else:
            print(f"[Startup] Motrix 未完成：{nc_name}，保留哨兵等待重处理")


# ============================================================
# ================= Motrix RPC（保持原逻辑）=================
# ============================================================

def motrix_rpc_download(url, dir_path=None, filename=None) -> bool:
    rpc_url = "http://localhost:16800/jsonrpc"
    options = {}
    if dir_path:  options["dir"] = dir_path
    if filename:  options["out"] = filename
    data = {"jsonrpc": "2.0", "id": "qwer",
            "method": "aria2.addUri", "params": [[url], options]}
    try:
        response = requests.post(rpc_url, json=data, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接 Motrix RPC ({rpc_url})")
    except requests.exceptions.Timeout:
        print("错误：连接 Motrix RPC 超时")
    except requests.exceptions.RequestException as e:
        print(f"错误：Motrix RPC 请求失败: {e}")
    return False


# ============================================================
# ========= [CHANGED] execute_download_task =================
# ============================================================

# ============================================================
# ===== [FIX 2] execute_download_task：修复两处漏洞 ==========
# ============================================================

# 进程级锁，防止多线程同时对相同 fingerprint 执行 get→submit→set
_submit_locks: dict = {}
_submit_locks_meta = threading.Lock()

def _get_submit_lock(fp: str) -> threading.Lock:
    with _submit_locks_meta:
        if fp not in _submit_locks:
            _submit_locks[fp] = threading.Lock()
        return _submit_locks[fp]

def _log_error(target_directory: str, worker_id: str, output_fname: str, e: Exception):
    with open(os.path.join(target_directory, "error_log.txt"), "a", encoding="utf-8") as ef:
        ef.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{worker_id}] {output_fname}: {e}\n")

def execute_download_task(account_info: dict, request_params: dict,
                          target_directory: str, output_fname: str,
                          worker_id: str, job_cache: JobCache):
    outpath = os.path.join(target_directory, output_fname)
    pending = _pending_path(outpath)

    # ① 文件已完整落盘
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        if os.path.exists(pending):
            os.remove(pending)
        job_cache.remove(compute_fingerprint(dataset, request_params))
        print(f"[{worker_id}] {output_fname} 已存在，跳过")
        return

    # ② [NEW] 运行时防重：若已有线程在处理此文件，直接跳过
    with _in_progress_lock:
        if output_fname in _in_progress:
            print(f"[{worker_id}] {output_fname} 已由其他线程处理中，跳过")
            return
        _in_progress.add(output_fname)

    try:
        _do_download(account_info, request_params, target_directory,
                     output_fname, worker_id, job_cache, outpath, pending)
    finally:
        # 无论成功/失败/异常，都释放占用标记
        with _in_progress_lock:
            _in_progress.discard(output_fname)

def _do_download(account_info, request_params, target_directory,
                 output_fname, worker_id, job_cache, outpath, pending):
    """实际下载逻辑，由 execute_download_task 在持有 _in_progress 标记时调用。"""
    fingerprint = compute_fingerprint(dataset, request_params)

    if os.path.exists(pending):
        print(f"[{worker_id}] 检测到哨兵文件，重新处理：{output_fname}")

    fp_lock = _get_submit_lock(fingerprint)
    with fp_lock:
        # 锁内双重检查文件
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            if os.path.exists(pending):
                os.remove(pending)
            job_cache.remove(fingerprint)
            print(f"[{worker_id}] {output_fname} 已存在（锁内），跳过")
            return

        cached = job_cache.get(fingerprint)
        if cached:
            rid = cached["request_id"]
            print(f"[{worker_id}] 缓存命中 job_id={rid}，复用：{output_fname}")
        else:
            try:
                print(f"[{worker_id}] 提交新任务：{output_fname}")
                rid = _submit_job(account_info, dataset, request_params)
                print(f"[{worker_id}] 已提交 job_id={rid}")
                job_cache.set(fingerprint, {
                    "request_id":   rid,
                    "fname":        output_fname,
                    "account":      account_info,
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                cached = job_cache.get(fingerprint)
            except Exception as e:
                print(f"[{worker_id}] 提交失败 {output_fname}: {e}")
                _log_error(target_directory, worker_id, output_fname, e)
                return

    # 锁外轮询（耗时，不占锁）
    result = _wait_and_download(
        account=cached["account"],
        request_id=rid,
        target_directory=target_directory,
        output_fname=output_fname,
        worker_id=worker_id,
    )

    if result == "done":
        if os.path.exists(pending):
            os.remove(pending)
        job_cache.remove(fingerprint)
    elif result == "motrix":
        print(f"[{worker_id}] Motrix 接管，保留缓存：{output_fname}")
    else:
        print(f"[{worker_id}] 任务失败，缓存已暂存：{output_fname}")

# ============================================================
# ========= [CHANGED] worker_function =======================
# ============================================================

def worker_function(account_info: dict, worker_name: str,
                    target_directory: str, job_cache: JobCache):
    print(f"线程 {worker_name} 启动")
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            print(f"线程 {worker_name} 收到停止信号，退出")
            break
        req_params, fname = task
        execute_download_task(account_info, req_params,
                              target_directory, fname, worker_name, job_cache)
        task_queue.task_done()
    print(f"线程 {worker_name} 退出")


# ============================================================
# ========================== Main ============================
# ============================================================

task_queue = Queue()

def get_days_of_month(year: int, month: int) -> List[str]:
    ndays = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, ndays + 1)]

def iter_months(sy, sm, ey, em):
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    # [NEW] 共享缓存实例（所有线程共用同一个 JobCache 对象）
    job_cache = JobCache(CACHE_FILE)

    # 启动时先清理已完成的 Motrix 任务
    _cleanup_finished_motrix(output_dir, job_cache)

    # 启动工作线程（每个账号一个线程）
    threads = []
    for acc in ACCOUNTS:
        t = threading.Thread(
            target=worker_function,
            args=(acc, acc["name"], output_dir, job_cache),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # 构建任务队列
    print("正在生成任务队列...")
    total = 0
    queued_fnames = set()  # [NEW] 队列级去重，防止多次重启累积重复条目

    for y, m in iter_months(start_year, start_month, end_year, end_month):
        year_str = str(y)
        mon_str  = f"{m:02d}"
        days     = get_days_of_month(y, m)

        if 'pressure-levels' in dataset:
            for plev in p_level:
                fname = f"{FILE_PREFIX}" + "uvwztSh" + f"_{plev}_{year_str}{mon_str}.nc"
                outpath = os.path.join(output_dir, fname)
                # [FIXED] 逐文件检查，不再只检查 VARS[0]
                if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                    print(f"  {fname} 已存在，跳过")
                    continue
                if fname in queued_fnames:  # [NEW]
                    continue
                queued_fnames.add(fname)  # [NEW]
                req = {
                    "product_type":    "reanalysis",
                    "variable":        VARS,
                    "year":            year_str,
                    "month":           mon_str,
                    "day":             days,
                    "pressure_level":  [plev],
                    "daily_statistic": DAILY_STATISTIC,
                    "time_zone":       TIME_ZONE,
                    "frequency":       FREQUENCY,
                    "grid":            GRID,
                    "format":          "netcdf",
                }
                task_queue.put((req, fname))
                total += 1

        elif 'single' in dataset:
            fname = f"{FILE_PREFIX}"+VARS[0]+f"_{year_str}{mon_str}.nc"
            outpath = os.path.join(output_dir, fname)
            # [FIXED] 逐文件检查，不再只检查 VARS[0]
            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                print(f"  {fname} 已存在，跳过")
                continue
            if fname in queued_fnames:          # [NEW]
                continue
            queued_fnames.add(fname)            # [NEW]
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
            total += 1

    print(f"任务队列就绪：{total} 个任务，开始处理...")

    for _ in threads:
        task_queue.put(None)

    task_queue.join()
    for t in threads:
        t.join()

    print("全部任务完成")

# 全局：记录"当前正在被某线程处理"的文件名
_in_progress: set = set()
_in_progress_lock = threading.Lock()


if __name__ == "__main__":
    main()
