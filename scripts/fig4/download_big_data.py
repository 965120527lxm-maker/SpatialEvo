#!/usr/bin/env python3
"""Download helpers for Fig.4 Big Xenium data with China-friendly mirrors.

S3 us-west-2 (10x full outs.zip) is often ~KB/s from CN servers. This script:
  1. Probes S3 vs HF mirror speeds and prints a recommendation.
  2. Fast path: combined h5ad (~887 MB) via hf-mirror.com (~MB/s).
  3. Slow path: full 10x outs.zip via aria2/wget (only if speed OK, proxy set, or --force).
  4. Local path: reuse an existing copy on this machine if present.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S3_BIG1 = (
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.0.2/"
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_1/Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs.zip"
)
S3_BIG2 = (
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.0.2/"
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_2/Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs.zip"
)
HF_H5AD_REPO = "theislab/SpatialCorpus-110M"
HF_H5AD_FILE = "10xgenomics_xenium_breast_cancer_big.h5ad"
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]
LOCAL_H5AD_CANDIDATES = [
    "/root/autodl-tmp/lsy/nicheformer/data/data_unprocessed/10xgenomics_xenium_breast_cancer_big.h5ad",
    os.path.join(PROJECT_ROOT, "data", "xenium_big", HF_H5AD_FILE),
]

MIN_S3_KBPS = 200  # abort raw zip if below this after probe unless --force or proxy set
DEFAULT_PROXY = "http://127.0.0.1:7890"
CLASH_BIN = "/root/autodl-tmp/lsy/clash-for-AutoDL/bin/clash-linux-amd64"
CLASH_DIR = "/root/autodl-tmp/lsy/clash-for-AutoDL/conf"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_proxy_env() -> str | None:
    """Ensure HTTPS_PROXY is set; start local clash if needed."""
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if proxy:
        return proxy
    # Try default clash port
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 7890), timeout=1):
            proxy = DEFAULT_PROXY
    except OSError:
        if os.path.isfile(CLASH_BIN):
            log("Starting local clash proxy on 127.0.0.1:7890 ...")
            subprocess.Popen(
                [CLASH_BIN, "-d", CLASH_DIR],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            try:
                import socket
                with socket.create_connection(("127.0.0.1", 7890), timeout=2):
                    proxy = DEFAULT_PROXY
            except OSError:
                return None
        else:
            return None
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = proxy
    return proxy


def probe_url(url: str, nbytes: int = 5 * 1024 * 1024, timeout: float = 25.0) -> float:
    """Return download speed in KB/s for first `nbytes` bytes (HTTP Range)."""
    ensure_proxy_env()
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        got = 0
        while got < nbytes:
            chunk = resp.read(min(256 * 1024, nbytes - got))
            if not chunk:
                break
            got += len(chunk)
    elapsed = max(time.time() - t0, 1e-6)
    return got / 1024 / elapsed


def cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_probe() -> dict[str, float]:
    log("=== Network probe (5 MB sample each) ===")
    results: dict[str, float] = {}
    try:
        results["s3_big1"] = probe_url(S3_BIG1)
        log(f"S3 Big1 zip:        {results['s3_big1']:.1f} KB/s")
    except Exception as e:
        results["s3_big1"] = 0.0
        log(f"S3 Big1 zip:        FAILED ({e})")

    for mirror in HF_MIRRORS:
        url = f"{mirror}/datasets/{HF_H5AD_REPO}/resolve/main/{HF_H5AD_FILE}"
        label = mirror.replace("https://", "")
        try:
            spd = probe_url(url)
            results[f"hf:{label}"] = spd
            log(f"HF {label}:  {spd:.1f} KB/s")
        except Exception as e:
            results[f"hf:{label}"] = 0.0
            log(f"HF {label}:  FAILED ({e})")

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        log(f"HTTPS_PROXY set: {proxy}")
    else:
        log("HTTPS_PROXY: not set (set it to accelerate S3 if you have a proxy)")

    best_hf = max((v for k, v in results.items() if k.startswith("hf:")), default=0.0)
    if results.get("s3_big1", 0) < MIN_S3_KBPS and best_hf > MIN_S3_KBPS:
        log("\nRecommendation: use `h5ad` mode (HF mirror). Avoid full S3 zip on this link.")
    elif results.get("s3_big1", 0) >= MIN_S3_KBPS:
        log("\nRecommendation: S3 speed OK; `raw` mode is feasible.")
    else:
        log("\nRecommendation: download raw zip on another machine and upload, or set HTTPS_PROXY.")
    return results


def best_hf_mirror(probed: dict[str, float] | None = None) -> str:
    if probed:
        return max(HF_MIRRORS, key=lambda m: probed.get(m, probed.get(f"hf:{m.replace('https://', '')}", 0.0)))
    speeds = {}
    for mirror in HF_MIRRORS:
        url = f"{mirror}/datasets/{HF_H5AD_REPO}/resolve/main/{HF_H5AD_FILE}"
        try:
            speeds[mirror] = probe_url(url, nbytes=2 * 1024 * 1024, timeout=15)
        except Exception:
            speeds[mirror] = 0.0
    return max(HF_MIRRORS, key=lambda m: speeds.get(m, 0.0))
def copy_local_h5ad(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, HF_H5AD_FILE)
    for src in LOCAL_H5AD_CANDIDATES:
        if src != dst and os.path.isfile(src):
            log(f"Copy local h5ad: {src} -> {dst}")
            shutil.copy2(src, dst)
            return dst
    raise FileNotFoundError("No local combined Big h5ad found; run `h5ad` mode.")


def download_h5ad(out_dir: str, mirror: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, HF_H5AD_FILE)
    if os.path.isfile(dst) and os.path.getsize(dst) > 800_000_000:
        log(f"Already present: {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
        return dst

    for src in LOCAL_H5AD_CANDIDATES:
        if src != dst and os.path.isfile(src) and os.path.getsize(src) > 800_000_000:
            log(f"Reuse local copy: {src}")
            shutil.copy2(src, dst)
            return dst

    mirror = mirror or best_hf_mirror()
    url = f"{mirror}/datasets/{HF_H5AD_REPO}/resolve/main/{HF_H5AD_FILE}"
    log(f"Downloading via {mirror} (~887 MB, ETA a few minutes on hf-mirror)...")

    os.environ["HF_ENDPOINT"] = mirror
    if cmd_exists("huggingface-cli"):
        subprocess.check_call([
            "huggingface-cli", "download", "--repo-type", "dataset",
            "--resume-download", HF_H5AD_REPO, "--include", HF_H5AD_FILE,
            "--local-dir", out_dir, "--local-dir-use-symlinks", "False",
        ])
        downloaded = os.path.join(out_dir, HF_H5AD_FILE)
        if os.path.isfile(downloaded):
            return downloaded

    # curl fallback with resume
    partial = dst + ".part"
    curl_cmd = ["curl", "-L", "--fail", "--continue-at", "-", "-o", dst, url]
    subprocess.check_call(curl_cmd)
    if os.path.isfile(partial):
        os.replace(partial, dst)
    return dst


def download_raw_zip(slice_idx: int, out_dir: str, force: bool = False) -> str:
    proxy = ensure_proxy_env()
    url = S3_BIG1 if slice_idx == 1 else S3_BIG2
    name = f"Xenium_V1_FFPE_Human_Breast_IDC_Big_{slice_idx}_outs.zip"
    dst = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    if not force:
        try:
            spd = probe_url(url, nbytes=2 * 1024 * 1024, timeout=30)
            log(f"S3 probe Big{slice_idx}: {spd:.1f} KB/s")
            if spd < MIN_S3_KBPS and not proxy:
                eta_days = (50e9 / 1024) / max(spd, 1) / 86400
                raise RuntimeError(
                    f"S3 too slow ({spd:.1f} KB/s < {MIN_S3_KBPS} KB/s); "
                    f"estimated ~{eta_days:.0f} days for 50GB. "
                    "Run `proxy_on` / start clash, or pass --force."
                )
        except RuntimeError:
            raise
        except Exception as e:
            log(f"S3 probe failed ({e}); continuing because probe errors are non-fatal")

    log(f"Downloading raw zip Big{slice_idx} -> {dst}")
    if proxy:
        log(f"Using proxy: {proxy}")
    env = os.environ.copy()
    if cmd_exists("aria2c"):
        cmd = [
            "aria2c", "-c", "-x", "4", "-s", "4", "-k", "1M",
            "-d", out_dir, "-o", name, url,
        ]
        if proxy:
            cmd.insert(1, f"--all-proxy={proxy}")
        subprocess.check_call(cmd, env=env)
    else:
        subprocess.check_call(["wget", "-c", "-O", dst, url], env=env)
    return dst


def parse_args():
    p = argparse.ArgumentParser(description="Fig.4 Big data download with mirror probe")
    p.add_argument("mode", choices=["probe", "h5ad", "raw", "local"],
                   help="probe speeds | h5ad via HF mirror | raw S3 zip | copy local h5ad")
    p.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data", "xenium_big"))
    p.add_argument("--slice", type=int, choices=[1, 2], default=1, help="For raw mode")
    p.add_argument("--mirror", default=None, help="HF mirror base URL")
    p.add_argument("--force", action="store_true", help="Allow slow S3 raw download")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "probe":
        run_probe()
        return
    if args.mode == "local":
        path = copy_local_h5ad(args.out_dir)
        log(f"Done: {path}")
        return
    if args.mode == "h5ad":
        path = download_h5ad(args.out_dir, mirror=args.mirror)
        log(f"Done: {path}")
        log("Note: this combined h5ad is for barcode/probe only; official Fig.4 still needs raw Read_Xenium.")
        return
    if args.mode == "raw":
        path = download_raw_zip(args.slice, args.out_dir, force=args.force)
        log(f"Done: {path}")
        log("Unzip then use cell_feature_matrix.h5 + cells.csv with Tutorial 3 Read_Xenium.")


if __name__ == "__main__":
    main()
