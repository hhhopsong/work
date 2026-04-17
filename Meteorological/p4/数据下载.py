import os
import json
import time
import queue
import hashlib
import threading
import calendar
import heapq
from typing import List, Optional, Dict, Tuple

import requests


# ============================================================
# ======================= User settings ======================
# ============================================================

start_year, start_month = 1961, 1
end_year,   end_month   = 2022, 12

NH_EHEM_AREA = [90, 0, 0, 180]

dataset = "derived-era5-single-levels-daily-statistics"
# dataset = "derived-era5-pressure-levels-daily-statistics"

output_dir = "/Volumes/TiPlus7100/data/ERA5/daily/slp_tpp_rad_tcc_eva_peva"
# output_dir = "/Volumes/TiPlus7100/data/ERA5/daily/uvwztSh"
os.makedirs(output_dir, exist_ok=True)

ACCOUNTS_FILE = "cds_accounts.json"

GRID            = "1.0/1.0"
DAILY_STATISTIC = "daily_mean"
TIME_ZONE       = "utc+00:00"
FREQUENCY       = "1_hourly"
FILE_PREFIX     = "ERA5_daily_"

# VARS = [
#     "geopotential",
#     "specific_humidity",
#     "temperature",
#     "u_component_of_wind",
#     "v_component_of_wind",
#     "vertical_velocity",
# ]
# p_level = ["200", "500", "850", "925"]

VARS = [
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "top_net_solar_radiation",
    "top_net_thermal_radiation",
    "total_cloud_cover",
    "evaporation",
    "potential_evaporation"
]
VARS_fname = "slp_tpp_rad_tcc_eva_peva"

CACHE_FILE = os.path.join(output_dir, "submitted_jobs_cache.json")

# -----------------------------
# 提交与下载参数
# -----------------------------
SUBMIT_SLEEP_SECONDS = 1
DOWNLOAD_WORKERS = 6

# 非阻塞轮询参数
DEFAULT_RECHECK_SECONDS = 60
NETWORK_ERROR_RECHECK_SECONDS = 120
SUBMIT_RETRY_DELAY_SECONDS = 180
ACCOUNT_FULL_RETRY_SECONDS = 30
MAX_JOB_AGE_HOURS = 72

# 每个账号最多同时挂起多少个活跃请求
MAX_ACTIVE_JOBS_PER_ACCOUNT = 10

# 是否在 fail_job 时自动重新提交
AUTO_RESUBMIT_ON_FAIL_JOB = True


# ============================================================
# ======================= Load accounts ======================
# ============================================================

def load_accounts(accounts_file: str) -> List[dict]:
    if not os.path.exists(accounts_file):
        raise FileNotFoundError(
            f"账号文件不存在：{accounts_file}\n"
            f"请创建该文件，并写入账号列表。"
        )

    with open(accounts_file, "r", encoding="utf-8") as f:
        accounts = json.load(f)

    if not isinstance(accounts, list) or not accounts:
        raise ValueError(f"账号文件格式错误：{accounts_file}")

    required_keys = {"name", "url", "key"}
    for i, acc in enumerate(accounts):
        if not isinstance(acc, dict):
            raise ValueError(f"第 {i} 个账号不是对象")
        missing = required_keys - set(acc.keys())
        if missing:
            raise ValueError(f"第 {i} 个账号缺少字段：{missing}")

    names = [a["name"] for a in accounts]
    if len(names) != len(set(names)):
        raise ValueError("账号 name 存在重复，请保证每个账号 name 唯一")

    return accounts


ACCOUNTS = load_accounts(ACCOUNTS_FILE)


# ============================================================
# ================= Global queues & controls =================
# ============================================================

submit_queue = queue.Queue()

_submit_locks: Dict[str, threading.Lock] = {}
_submit_locks_meta = threading.Lock()

_active_downloads = set()
_active_downloads_lock = threading.Lock()

_done_submitting = False
_done_submitting_lock = threading.Lock()

shutdown_event = threading.Event()


def mark_submit_done():
    global _done_submitting
    with _done_submitting_lock:
        _done_submitting = True


def is_submit_done() -> bool:
    with _done_submitting_lock:
        return _done_submitting


def _get_submit_lock(fp: str) -> threading.Lock:
    with _submit_locks_meta:
        if fp not in _submit_locks:
            _submit_locks[fp] = threading.Lock()
        return _submit_locks[fp]


# ============================================================
# ================= Account slot limiter =====================
# ============================================================

account_slots: Dict[str, threading.BoundedSemaphore] = {}
account_slot_usage_lock = threading.Lock()
account_slot_usage: Dict[str, int] = {}


def init_account_slots():
    global account_slots, account_slot_usage
    account_slots = {
        acc["name"]: threading.BoundedSemaphore(MAX_ACTIVE_JOBS_PER_ACCOUNT)
        for acc in ACCOUNTS
    }
    account_slot_usage = {
        acc["name"]: 0
        for acc in ACCOUNTS
    }


def acquire_account_slot(account_name: str, blocking: bool = False, timeout: Optional[float] = None) -> bool:
    sem = account_slots.get(account_name)
    if sem is None:
        raise KeyError(f"未知账号：{account_name}")

    if blocking:
        ok = sem.acquire(timeout=timeout)
    else:
        ok = sem.acquire(blocking=False)

    if ok:
        with account_slot_usage_lock:
            account_slot_usage[account_name] = account_slot_usage.get(account_name, 0) + 1
    return ok


def release_account_slot(account_name: str, worker_name: str, fname: str):
    sem = account_slots.get(account_name)
    if sem is None:
        return

    try:
        sem.release()
        with account_slot_usage_lock:
            cur = account_slot_usage.get(account_name, 0)
            account_slot_usage[account_name] = max(0, cur - 1)
        print(f"[{worker_name}] 已释放账号槽位：{account_name} -> {fname}")
    except ValueError:
        print(f"[{worker_name}] 警告：账号槽位重复释放：{account_name} -> {fname}")


def release_slot_if_needed(cached: Optional[dict], worker_name: str, fname: str, job_cache=None, fingerprint=None):
    if not cached:
        return
    if not cached.get("slot_acquired"):
        return

    acc_name = cached.get("account_name")
    if not acc_name:
        return

    release_account_slot(acc_name, worker_name, fname)

    if job_cache is not None and fingerprint is not None:
        try:
            job_cache.update_fields(fingerprint, slot_acquired=False)
        except Exception:
            pass


def rebuild_account_slots_from_cache(job_cache):
    init_account_slots()

    counts = {acc["name"]: 0 for acc in ACCOUNTS}

    for _, info in job_cache.items_snapshot():
        if not info.get("slot_acquired"):
            continue

        status = info.get("status", "submitted")
        if status in ("submitted", "queued", "running", "accepted", "pending", "unknown", "network_error"):
            acc_name = info.get("account_name")
            if acc_name in counts:
                counts[acc_name] += 1

    for acc_name, used in counts.items():
        used = min(used, MAX_ACTIVE_JOBS_PER_ACCOUNT)
        for _ in range(used):
            ok = acquire_account_slot(acc_name, blocking=False)
            if not ok:
                break

    print(f"[Startup] 已按缓存恢复账号并发占用：{counts}")


def account_slot_snapshot() -> Dict[str, int]:
    with account_slot_usage_lock:
        return dict(account_slot_usage)


# ============================================================
# =================== Download scheduler =====================
# ============================================================

class DownloadScheduler:
    def __init__(self):
        self._cv = threading.Condition()
        self._heap = []
        self._seq = 0
        self._unfinished = 0
        self._scheduled_keys = set()
        self._closed = False

    def _make_key(self, fingerprint: str, fname: str) -> Tuple[str, str]:
        return fingerprint, fname

    def schedule_at(self, run_at: float, fingerprint: str, fname: str) -> bool:
        key = self._make_key(fingerprint, fname)
        with self._cv:
            if self._closed:
                return False
            if key in self._scheduled_keys:
                return False
            self._seq += 1
            heapq.heappush(self._heap, (run_at, self._seq, fingerprint, fname))
            self._scheduled_keys.add(key)
            self._unfinished += 1
            self._cv.notify_all()
            return True

    def schedule_now(self, fingerprint: str, fname: str) -> bool:
        return self.schedule_at(time.time(), fingerprint, fname)

    def schedule_after(self, delay_seconds: int, fingerprint: str, fname: str) -> bool:
        return self.schedule_at(time.time() + max(0, delay_seconds), fingerprint, fname)

    def get_due_task(self, timeout: float = 1.0):
        with self._cv:
            while True:
                if self._closed and not self._heap:
                    return None

                now = time.time()
                if self._heap:
                    run_at, seq, fp, fname = self._heap[0]
                    wait_time = run_at - now
                    if wait_time <= 0:
                        heapq.heappop(self._heap)
                        self._scheduled_keys.discard((fp, fname))
                        return fp, fname
                    self._cv.wait(timeout=min(wait_time, timeout))
                else:
                    self._cv.wait(timeout=timeout)
                    if self._closed and not self._heap:
                        return None

    def task_done(self):
        with self._cv:
            self._unfinished -= 1
            if self._unfinished <= 0:
                self._cv.notify_all()

    def join(self):
        with self._cv:
            while self._unfinished > 0:
                self._cv.wait()

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def unfinished_count(self) -> int:
        with self._cv:
            return self._unfinished


download_scheduler = DownloadScheduler()


# ============================================================
# =================== Job Cache ==============================
# ============================================================

class JobCache:
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
        tmp = self.cache_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.cache_file)
        except Exception as e:
            print(f"[Cache] 警告：缓存写入磁盘失败（{e}），内存缓存仍有效")

    def get(self, fp: str) -> Optional[dict]:
        with self._lock:
            v = self._data.get(fp)
            return dict(v) if isinstance(v, dict) else v

    def set(self, fp: str, info: dict):
        with self._lock:
            self._data[fp] = info
            self._save()

    def update_fields(self, fp: str, **kwargs):
        with self._lock:
            if fp in self._data:
                self._data[fp].update(kwargs)
                self._save()

    def remove(self, fp: str):
        with self._lock:
            if fp in self._data:
                del self._data[fp]
                self._save()

    def items_snapshot(self) -> List[Tuple[str, dict]]:
        with self._lock:
            return [(k, dict(v) if isinstance(v, dict) else v) for k, v in self._data.items()]


# ============================================================
# ============= Request fingerprint + CDS REST ===============
# ============================================================

def compute_fingerprint(dataset_name: str, req: dict) -> str:
    canonical = json.dumps(
        {"dataset": dataset_name, **req},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _make_session(account: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "PRIVATE-TOKEN": account["key"],
        "Content-Type": "application/json",
    })
    return s


def _submit_job(account: dict, dataset_name: str, req: dict) -> str:
    url = f"{account['url']}/retrieve/v1/processes/{dataset_name}/execution"
    resp = _make_session(account).post(url, json={"inputs": req}, timeout=120)

    if resp.status_code == 404:
        raise RuntimeError(f"404：数据集路径不存在。URL={url} 账号={account['name']}")

    resp.raise_for_status()
    data = resp.json()

    job_id = data.get("jobID") or data.get("id") or data.get("request_id")
    if not job_id:
        raise RuntimeError(f"CDS 未返回 jobID，完整响应：{data}")

    return job_id


def _check_job(account: dict, job_id: str) -> Optional[dict]:
    url = f"{account['url']}/retrieve/v1/jobs/{job_id}"
    try:
        resp = _make_session(account).get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.SSLError as e:
        print(f"  [CDS] SSL异常（网络抖动），本轮跳过 {job_id}: {e}")
        return {"status": "network_error"}

    except Exception as e:
        print(f"  [CDS] 查询任务状态异常 {job_id}: {e}")
        return {"status": "network_error"}


def _get_job_results(account: dict, job_id: str) -> Optional[str]:
    url = f"{account['url']}/retrieve/v1/jobs/{job_id}/results"
    try:
        resp = _make_session(account).get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        href = data.get("asset", {}).get("value", {}).get("href")
        if not href:
            href = data.get("location") or data.get("url")

        return href

    except Exception as e:
        print(f"  [CDS] 获取结果链接失败 {job_id}: {e}")
        return None


# ============================================================
# ====================== Download utils ======================
# ============================================================

def _pending_path(outpath: str) -> str:
    return outpath + ".pending"


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
        print(f"错误：无法连接 Motrix RPC ({rpc_url})")
    except requests.exceptions.Timeout:
        print("错误：连接 Motrix RPC 超时")
    except requests.exceptions.RequestException as e:
        print(f"错误：Motrix RPC 请求失败: {e}")

    return False


def _download_url(url: str, target_directory: str, output_fname: str, worker_id: str) -> str:
    outpath = os.path.join(target_directory, output_fname)
    pending = _pending_path(outpath)

    ok = motrix_rpc_download(url, dir_path=target_directory, filename=output_fname)
    if ok:
        try:
            with open(pending, "w", encoding="utf-8") as f:
                f.write(f"{url}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        except Exception:
            pass

        print(f"[{worker_id}] 已发送 Motrix（哨兵已写）：{output_fname}")
        return "motrix"

    print(f"[{worker_id}] Motrix 不可用，直接下载：{output_fname}")
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        with open(outpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        if os.path.exists(pending):
            try:
                os.remove(pending)
            except Exception:
                pass

        print(f"[{worker_id}] 直接下载完成：{outpath}")
        return "done"

    except Exception as e:
        print(f"[{worker_id}] 直接下载失败：{e}")
        if os.path.exists(outpath):
            try:
                os.remove(outpath)
            except Exception:
                pass
        return "fail_download"


def _cleanup_finished_motrix(output_dir: str, job_cache: JobCache):
    pending_files = [f for f in os.listdir(output_dir) if f.endswith(".nc.pending")]
    if not pending_files:
        return

    print(f"[Startup] 发现 {len(pending_files)} 个哨兵文件，开始扫描...")
    for pf in pending_files:
        nc_name = pf[:-len(".pending")]
        nc_path = os.path.join(output_dir, nc_name)
        p_path = os.path.join(output_dir, pf)

        if os.path.exists(nc_path) and os.path.getsize(nc_path) > 0:
            print(f"[Startup] Motrix 已完成：{nc_name}，清理哨兵")
            try:
                os.remove(p_path)
            except Exception:
                pass
        else:
            print(f"[Startup] Motrix 未完成：{nc_name}，保留哨兵等待重处理")


def _log_error(target_directory: str, worker_id: str, output_fname: str, e: Exception):
    with open(os.path.join(target_directory, "error_log.txt"), "a", encoding="utf-8") as ef:
        ef.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{worker_id}] {output_fname}: {e}\n")


# ============================================================
# ==================== Task builders =========================
# ============================================================

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


def build_request_and_fname(year_str: str, mon_str: str, days: List[str], plev: Optional[str] = None):
    if "pressure-levels" in dataset:
        if plev is None:
            raise ValueError("pressure-levels 数据集必须提供 plev")

        fname = f"{FILE_PREFIX}uvwztSh_{plev}_{year_str}{mon_str}.nc"
        req = {
            "product_type": "reanalysis",
            "variable": VARS,
            "year": year_str,
            "month": mon_str,
            "day": days,
            "pressure_level": [plev],
            "daily_statistic": DAILY_STATISTIC,
            "time_zone": TIME_ZONE,
            "frequency": FREQUENCY,
            "grid": GRID,
            "format": "netcdf",
        }
        return req, fname

    if "single" in dataset:
        fname = f"{FILE_PREFIX}_{VARS_fname}_{year_str}{mon_str}.nc"
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
        return req, fname

    raise ValueError(f"不支持的 dataset：{dataset}")


# ============================================================
# ==================== Resubmit helper =======================
# ============================================================

def _rebuild_request_from_fname(fname: str) -> Tuple[dict, str]:
    basename = os.path.basename(fname)

    if "pressure-levels" in dataset:
        stem = basename.replace(".nc", "")
        parts = stem.split("_")
        plev = parts[-2]
        yyyymm = parts[-1]
    else:
        stem = basename.replace(".nc", "")
        parts = stem.split("_")
        plev = None
        yyyymm = parts[-1]

    year_str = yyyymm[:4]
    mon_str = yyyymm[4:6]
    days = get_days_of_month(int(year_str), int(mon_str))
    req, rebuilt_fname = build_request_and_fname(year_str, mon_str, days, plev=plev)
    return req, rebuilt_fname


def resubmit_task_with_snapshot(cached_snapshot: Optional[dict], fname: str, reason: str):
    if cached_snapshot and isinstance(cached_snapshot.get("request_params"), dict):
        req = cached_snapshot["request_params"]
        actual_fname = cached_snapshot.get("fname", fname)
        print(f"[Resubmit] 因 {reason}，使用缓存 request_params 重新提交：{actual_fname}")
        submit_queue.put((req, actual_fname))
        return

    try:
        req, rebuilt_fname = _rebuild_request_from_fname(fname)
        if rebuilt_fname != fname:
            print(f"[Resubmit] 文件名反推不一致，跳过重提：{fname} -> {rebuilt_fname}")
            return
        print(f"[Resubmit] 因 {reason}，使用文件名反推重新提交：{fname}")
        submit_queue.put((req, fname))
    except Exception as e:
        print(f"[Resubmit] 重新构造请求失败：{fname} -> {e}")


# ============================================================
# ==================== Submit stage ==========================
# ============================================================

def try_submit_task(account_info: dict, request_params: dict,
                    output_fname: str, job_cache: JobCache, worker_name: str):
    outpath = os.path.join(output_dir, output_fname)
    pending = _pending_path(outpath)
    fingerprint = compute_fingerprint(dataset, request_params)

    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        if os.path.exists(pending):
            try:
                os.remove(pending)
            except Exception:
                pass
        job_cache.remove(fingerprint)
        print(f"[{worker_name}] {output_fname} 已存在，跳过提交")
        return

    fp_lock = _get_submit_lock(fingerprint)
    with fp_lock:
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            if os.path.exists(pending):
                try:
                    os.remove(pending)
                except Exception:
                    pass
            job_cache.remove(fingerprint)
            print(f"[{worker_name}] {output_fname} 已存在（锁内），跳过提交")
            return

        cached = job_cache.get(fingerprint)
        if cached and cached.get("request_id"):
            print(f"[{worker_name}] 已有缓存 job_id={cached['request_id']}，不重复提交：{output_fname}")
            download_scheduler.schedule_now(fingerprint, output_fname)
            return

        acc_name = account_info["name"]
        got_slot = acquire_account_slot(acc_name, blocking=False)
        if not got_slot:
            print(f"[{worker_name}] 账号 {acc_name} 已满 {MAX_ACTIVE_JOBS_PER_ACCOUNT} 个活跃任务，稍后重试：{output_fname}")
            download_scheduler.schedule_after(ACCOUNT_FULL_RETRY_SECONDS, fingerprint, output_fname)
            return

        try:
            print(f"[{worker_name}] 提交新任务：{output_fname}  fingerprint={fingerprint}")
            rid = _submit_job(account_info, dataset, request_params)
            print(f"[{worker_name}] 已提交 job_id={rid}")
        except Exception as e:
            release_account_slot(acc_name, worker_name, output_fname)
            print(f"[{worker_name}] 提交CDS失败 {output_fname}: {e}")
            _log_error(output_dir, worker_name, output_fname, e)
            download_scheduler.schedule_after(SUBMIT_RETRY_DELAY_SECONDS, fingerprint, output_fname)
            return

        cache_entry = {
            "request_id": rid,
            "fname": output_fname,
            "request_params": request_params,
            "dataset": dataset,
            "account_name": account_info["name"],
            "account_url": account_info["url"],
            "account_key": account_info["key"],
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_check_at": None,
            "status": "submitted",
            "slot_acquired": True,
        }

        try:
            job_cache.set(fingerprint, cache_entry)
        except Exception as e:
            print(f"[{worker_name}] 缓存写入失败（{e}），本次仍继续下载阶段")

    download_scheduler.schedule_now(fingerprint, output_fname)

    if SUBMIT_SLEEP_SECONDS > 0:
        time.sleep(SUBMIT_SLEEP_SECONDS)


def submit_worker(account_info: dict, worker_name: str, job_cache: JobCache):
    print(f"提交线程 {worker_name} 启动")

    while True:
        task = submit_queue.get()
        if task is None:
            submit_queue.task_done()
            print(f"提交线程 {worker_name} 收到停止信号，退出")
            break

        req_params, fname = task
        try_submit_task(account_info, req_params, fname, job_cache, worker_name)
        submit_queue.task_done()

    print(f"提交线程 {worker_name} 退出")


# ============================================================
# =================== Download stage =========================
# ============================================================

def _job_too_old(cached: dict) -> bool:
    submitted_at = cached.get("submitted_at")
    if not submitted_at:
        return False
    try:
        ts = time.mktime(time.strptime(submitted_at, "%Y-%m-%dT%H:%M:%S"))
        return (time.time() - ts) > MAX_JOB_AGE_HOURS * 3600
    except Exception:
        return False


def try_download_task_nonblocking(fingerprint: str, output_fname: str,
                                  target_directory: str, job_cache: JobCache, worker_name: str):
    outpath = os.path.join(target_directory, output_fname)
    pending = _pending_path(outpath)

    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        cached = job_cache.get(fingerprint)
        if os.path.exists(pending):
            try:
                os.remove(pending)
            except Exception:
                pass
        release_slot_if_needed(cached, worker_name, output_fname, job_cache, fingerprint)
        job_cache.remove(fingerprint)
        print(f"[{worker_name}] {output_fname} 已存在，跳过下载")
        return

    with _active_downloads_lock:
        if output_fname in _active_downloads:
            print(f"[{worker_name}] {output_fname} 已由其他线程处理中，跳过")
            return
        _active_downloads.add(output_fname)

    try:
        cached = job_cache.get(fingerprint)
        if not cached:
            if not is_submit_done():
                print(f"[{worker_name}] 无缓存，稍后再试：{output_fname}")
                download_scheduler.schedule_after(SUBMIT_RETRY_DELAY_SECONDS, fingerprint, output_fname)
            else:
                print(f"[{worker_name}] 提交阶段已结束且无缓存，尝试自动重提：{output_fname}")
                if AUTO_RESUBMIT_ON_FAIL_JOB:
                    resubmit_task_with_snapshot(None, output_fname, "cache_missing")
            return

        rid = cached.get("request_id")
        if not rid:
            print(f"[{worker_name}] 缓存中无 request_id：{output_fname}")
            snapshot = dict(cached)
            release_slot_if_needed(snapshot, worker_name, output_fname, job_cache, fingerprint)
            job_cache.remove(fingerprint)
            if AUTO_RESUBMIT_ON_FAIL_JOB:
                resubmit_task_with_snapshot(snapshot, output_fname, "request_id_missing")
            return

        if _job_too_old(cached):
            print(f"[{worker_name}] job 已过老，自动重提：{output_fname}")
            snapshot = dict(cached)
            release_slot_if_needed(snapshot, worker_name, output_fname, job_cache, fingerprint)
            job_cache.remove(fingerprint)
            if AUTO_RESUBMIT_ON_FAIL_JOB:
                resubmit_task_with_snapshot(snapshot, output_fname, "job_too_old")
            return

        account_info = {
            "name": cached["account_name"],
            "url": cached["account_url"],
            "key": cached["account_key"],
        }

        info = _check_job(account_info, rid)

        if info is None:
            print(f"[{worker_name}] job 不存在，自动重提：{output_fname}")
            snapshot = dict(cached)
            release_slot_if_needed(snapshot, worker_name, output_fname, job_cache, fingerprint)
            job_cache.remove(fingerprint)
            if AUTO_RESUBMIT_ON_FAIL_JOB:
                resubmit_task_with_snapshot(snapshot, output_fname, "job_missing")
            return

        status = info.get("status", "unknown")
        job_cache.update_fields(
            fingerprint,
            status=status,
            last_check_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        print(f"[{worker_name}] {output_fname} 当前状态：{status}")

        if status == "network_error":
            download_scheduler.schedule_after(NETWORK_ERROR_RECHECK_SECONDS, fingerprint, output_fname)
            return

        if status in ("queued", "running", "accepted", "pending", "unknown"):
            download_scheduler.schedule_after(DEFAULT_RECHECK_SECONDS, fingerprint, output_fname)
            return

        if status == "failed":
            reason = info.get("message") or info.get("detail") or str(info)
            print(f"[{worker_name}] CDS任务失败：{output_fname} -> {reason}")
            snapshot = dict(cached)
            release_slot_if_needed(snapshot, worker_name, output_fname, job_cache, fingerprint)
            job_cache.remove(fingerprint)
            if AUTO_RESUBMIT_ON_FAIL_JOB:
                resubmit_task_with_snapshot(snapshot, output_fname, "job_failed")
            return

        if status == "successful":
            location = _get_job_results(account_info, rid)
            if not location:
                print(f"[{worker_name}] 取不到结果链接，稍后再试：{output_fname}")
                download_scheduler.schedule_after(DEFAULT_RECHECK_SECONDS, fingerprint, output_fname)
                return

            result = _download_url(location, target_directory, output_fname, worker_name)

            if result == "done":
                if os.path.exists(pending):
                    try:
                        os.remove(pending)
                    except Exception:
                        pass
                release_slot_if_needed(cached, worker_name, output_fname, job_cache, fingerprint)
                job_cache.remove(fingerprint)
                print(f"[{worker_name}] 下载完成，缓存已清除：{output_fname}")
                return

            if result == "motrix":
                release_slot_if_needed(cached, worker_name, output_fname, job_cache, fingerprint)
                job_cache.update_fields(fingerprint, status="motrix")
                print(f"[{worker_name}] Motrix 接管，已释放账号槽位：{output_fname}")
                return

            if result == "fail_download":
                print(f"[{worker_name}] 下载失败，稍后重试：{output_fname}")
                download_scheduler.schedule_after(DEFAULT_RECHECK_SECONDS, fingerprint, output_fname)
                return

        download_scheduler.schedule_after(DEFAULT_RECHECK_SECONDS, fingerprint, output_fname)

    finally:
        with _active_downloads_lock:
            _active_downloads.discard(output_fname)


def download_worker(worker_name: str, target_directory: str, job_cache: JobCache):
    print(f"下载线程 {worker_name} 启动")

    while not shutdown_event.is_set():
        task = download_scheduler.get_due_task(timeout=1.0)
        if task is None:
            if shutdown_event.is_set():
                break
            if is_submit_done() and download_scheduler.unfinished_count() == 0:
                break
            continue

        fp, fname = task
        try:
            try_download_task_nonblocking(fp, fname, target_directory, job_cache, worker_name)
        finally:
            download_scheduler.task_done()

    print(f"下载线程 {worker_name} 退出")


# ============================================================
# ==================== Startup recovery ======================
# ============================================================

def enqueue_cached_jobs(job_cache: JobCache):
    count = 0
    for fp, info in job_cache.items_snapshot():
        fname = info.get("fname")
        rid = info.get("request_id")

        if not fname or not rid:
            continue

        outpath = os.path.join(output_dir, fname)
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            continue

        download_scheduler.schedule_now(fp, fname)
        count += 1

    print(f"[Startup] 已恢复 {count} 个缓存任务到下载调度器")


# ============================================================
# ========================== Main ============================
# ============================================================

def main():
    job_cache = JobCache(CACHE_FILE)

    _cleanup_finished_motrix(output_dir, job_cache)
    rebuild_account_slots_from_cache(job_cache)
    enqueue_cached_jobs(job_cache)

    submit_threads = []
    for acc in ACCOUNTS:
        t = threading.Thread(
            target=submit_worker,
            args=(acc, acc["name"], job_cache),
            daemon=True,
        )
        t.start()
        submit_threads.append(t)

    download_threads = []
    for i in range(max(1, DOWNLOAD_WORKERS)):
        t = threading.Thread(
            target=download_worker,
            args=(f"downloader-{i + 1}", output_dir, job_cache),
            daemon=True,
        )
        t.start()
        download_threads.append(t)

    print("正在生成提交任务队列...")
    total = 0
    queued_fnames = set()

    for y, m in iter_months(start_year, start_month, end_year, end_month):
        year_str = str(y)
        mon_str = f"{m:02d}"
        days = get_days_of_month(y, m)

        if "pressure-levels" in dataset:
            for plev in p_level:
                req, fname = build_request_and_fname(year_str, mon_str, days, plev=plev)
                outpath = os.path.join(output_dir, fname)

                if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                    print(f"  {fname} 已存在，跳过")
                    continue

                if fname in queued_fnames:
                    continue
                queued_fnames.add(fname)

                submit_queue.put((req, fname))
                total += 1

        elif "single" in dataset:
            req, fname = build_request_and_fname(year_str, mon_str, days)
            outpath = os.path.join(output_dir, fname)

            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                print(f"  {fname} 已存在，跳过")
                continue

            if fname in queued_fnames:
                continue
            queued_fnames.add(fname)

            submit_queue.put((req, fname))
            total += 1

    print(f"提交队列就绪：{total} 个任务")
    print(f"[Startup] 当前账号占用：{account_slot_snapshot()}")

    submit_queue.join()

    for _ in submit_threads:
        submit_queue.put(None)
    for t in submit_threads:
        t.join()

    mark_submit_done()
    print("提交阶段完成，等待下载阶段结束...")

    download_scheduler.join()
    download_scheduler.close()

    shutdown_event.set()
    for t in download_threads:
        t.join()

    print("全部任务完成")


if __name__ == "__main__":
    main()
