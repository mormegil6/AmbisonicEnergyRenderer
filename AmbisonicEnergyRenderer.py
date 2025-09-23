#!/usr/bin/env python3
"""
AmbisonicEnergyRenderer.py

Fast Ambisonic energy visualization to MP4:
- Precomputed k-NN IDW gridding with azimuth wrap for artifact-free seams
- CPU multiprocessing with per-worker globals
- Direct ffmpeg streaming (portable encoder selection)
- 10-second debug timing checkpoints + maxRMSdB print in debug
"""

import argparse
from pathlib import Path
import multiprocessing as mp
import subprocess
from datetime import datetime
import sys

import numpy as np
import wavio
import spaudiopy as spa
from scipy.spatial import cKDTree
import matplotlib 


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fast Ambisonic energy video via k-NN IDW gridding + multiprocessing + ffmpeg."
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Input Ambisonics WAV (ACN/SN3D)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output MP4 file (default: input with .mp4)")
    p.add_argument("-t", "--tdesign", type=Path, default=None, help="Design_5200_100_random.dat (default: script dir)")
    p.add_argument("-d", "--dynamic-db", type=float, default=20.0, help="Dynamic range in dB (default: 20)")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name (e.g., inferno, magma, coolwarm, twilight_r). Default: viridis")
    p.add_argument("-n", "--frames", type=int, default=None, help="Number of frames to render (default: full length)")
    p.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    p.add_argument("--grid-res", type=float, default=(1.0/6.0), help="Grid step in degrees (default: 1/6 → ~2160x1080)")
    p.add_argument("--knn", type=int, default=3, help="k-NN for IDW (default: 3)")
    p.add_argument("--idw-pow", type=float, default=2.0, help="IDW power (default: 2.0)")
    p.add_argument("--encoder", type=str, default="libx264",
                   choices=[
                       "libx264",
                       "h264_videotoolbox", "hevc_videotoolbox",   # macOS
                       "h264_nvenc", "hevc_nvenc",                 # NVIDIA (Windows/Linux)
                       "h264_qsv", "hevc_qsv",                     # Intel Quick Sync
                       "h264_amf", "hevc_amf"                      # AMD AMF (Windows)
                   ],
                   help="Video encoder (default: libx264; choose *_videotoolbox on macOS, *_nvenc on NVIDIA/Windows)")
    p.add_argument("--crf", type=str, default="20", help="CRF for libx264 (software)")
    p.add_argument("--bitrate", type=str, default="10M", help="Target bitrate for hardware encoders (NVENC/VT/QSV/AMF)")
    p.add_argument("--preset", type=str, default="ultrafast", help="libx264 preset (default: ultrafast)")
    p.add_argument("--ffmpeg", type=str, default="ffmpeg", help="Path to ffmpeg executable (default: ffmpeg in PATH)")
    p.add_argument("--debug", action="store_true", default=True, help="Print timing/progress (default: on)")
    return p.parse_args()


# -------------------- Globals for worker processes --------------------

G = {}  # set by pool initializer


def _pool_init(dataNorm, weights, Yt, idx_map, w_map, grid_shape, numSmp, frameLen, winLenSmp, dyn_db, maxRMSdB):
    G["dataNorm"] = dataNorm
    G["weights"] = weights
    G["Yt"] = Yt
    G["idx_map"] = idx_map
    G["w_map"] = w_map
    G["H"], G["W"] = grid_shape
    G["numSmp"] = numSmp
    G["frameLen"] = frameLen
    G["winLenSmp"] = winLenSmp
    G["dyn_db"] = dyn_db
    G["maxRMSdB"] = maxRMSdB


def _compute_frame(frame_idx):
    dataNorm = G["dataNorm"]
    weights = G["weights"]
    Yt = G["Yt"]
    idx_map = G["idx_map"]
    w_map = G["w_map"]
    H, W = G["H"], G["W"]
    numSmp = G["numSmp"]
    frameLen = G["frameLen"]
    winLenSmp = G["winLenSmp"]
    dyn_db = G["dyn_db"]
    maxRMSdB = G["maxRMSdB"]

    s0 = frame_idx * frameLen
    s1 = min(s0 + winLenSmp, numSmp)

    # Decode windowed energy to t-design directions
    curr = (dataNorm[s0:s1, :].astype(np.float32) * weights[np.newaxis, :].astype(np.float32))
    tdGains = np.dot(Yt, curr.T)  # [Ndirs x Nwin]
    tdRMS = np.sqrt(np.mean(tdGains * tdGains, axis=1, dtype=np.float32)).astype(np.float32)
    tdRMS = np.maximum(tdRMS, 1e-12)  # avoid log(0)
    tdDB = (10.0 * np.log10(tdRMS) - maxRMSdB).astype(np.float32)
    td01 = np.clip(tdDB / dyn_db + 1.0, 0.0, 1.0).astype(np.float32)

    # Fast k-NN IDW mapping to grid (precomputed indices+weights)
    gather = td01[idx_map]                          # [M x k]
    grid_flat = np.sum(gather * w_map, axis=1, dtype=np.float32)  # [M]
    grid_img = grid_flat.reshape(H, W)

    # Quantize 0..255 for LUT colormap in main process
    return (np.clip(grid_img, 0.0, 1.0) * 255.0).astype(np.uint8)


# -------------------- Mapping builder with azimuth seam wrap --------------------

def build_periodic_knn_mapping(tdAzimDeg, tdZenDeg, xgv, ygv, k=3, p=2.0, workers=-1):
    """
    Build k-NN IDW mapping from scattered (azimuth, zenith) to regular (xgv, ygv),
    duplicating points at ±360° azimuth to avoid seam artifacts.
    """
    pts_base = np.column_stack([tdAzimDeg, tdZenDeg]).astype(np.float32)  # [N x 2]
    N = pts_base.shape[0]
    pts_ext = np.vstack([
        pts_base,
        pts_base + np.array([-360.0, 0.0], dtype=np.float32),
        pts_base + np.array([+360.0, 0.0], dtype=np.float32),
    ])
    orig_idx_ext = np.concatenate([np.arange(N, dtype=np.int32)] * 3)

    H = len(ygv)
    W = len(xgv)
    xGrid, yGrid = np.meshgrid(xgv, ygv)  # [H x W]
    grid_pts = np.column_stack([xGrid.ravel(), yGrid.ravel()]).astype(np.float32)  # [M x 2]

    tree = cKDTree(pts_ext)
    try:
        dists, nn_idx = tree.query(grid_pts, k=k, workers=workers)  # newer SciPy
    except TypeError:
        dists, nn_idx = tree.query(grid_pts, k=k)  # older SciPy

    if k == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]

    idx_map = orig_idx_ext[nn_idx]  # [M x k] original indices

    eps = 1e-6
    exact = dists <= eps
    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(dists + eps, p, dtype=np.float32)
    if np.any(exact):
        w[exact] = 0.0
        rows = np.where(exact.any(axis=1))[0]
        first = exact[rows].argmax(axis=1)
        w[rows, first] = 1.0
    w_sum = np.sum(w, axis=1, keepdims=True, dtype=np.float32)
    w_map = (w / np.maximum(w_sum, eps)).astype(np.float32)

    return idx_map.astype(np.int32), w_map, H, W


# -------------------- Main --------------------

def main():
    args = parse_args()
    in_path = args.input.resolve()
    out_path = (args.output or in_path.with_suffix(".mp4")).resolve()
    tdesign_path = args.tdesign or (Path(__file__).parent / "Design_5200_100_random.dat")

    # Read audio
    wav = wavio.read(str(in_path))
    data = wav.data
    fs = wav.rate
    numCh = data.shape[1]
    numSmp = data.shape[0]
    order = int(np.sqrt(numCh) - 1)

    # Normalize to float32
    scale = float(2 ** (int(wav.sampwidth) * 8 - 1))
    dataNorm = (data.astype(np.float32) / scale).astype(np.float32)

    # Max-rE weights (ACN/SN3D)
    sn3d2n3d = np.array([
        1.0000000, 1.7320508, 1.7320508, 1.7320508,
        2.2360680, 2.2360680, 2.2360680, 2.2360680, 2.2360680,
        2.6457512, 2.6457512, 2.6457512, 2.6457512, 2.6457512, 2.6457512, 2.6457512,
        3.0000000, 3.0000000, 3.0000000, 3.0000000, 3.0000000, 3.0000000, 3.0000000, 3.0000000, 3.0000000,
        3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249, 3.3166249,
        3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512, 3.6055512,
        3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832, 3.8729832
    ], dtype=np.float32)[:numCh]
    maxre = spa.sph.max_rE_weights(order).astype(np.float32)
    weights = np.zeros((numCh,), dtype=np.float32)
    idx = 0
    for o in range(order + 1):
        for _m in range(-o, o + 1):
            weights[idx] = sn3d2n3d[idx] * maxre[o]
            idx += 1

    # T-design and SH
    tdesign = np.loadtxt(str(tdesign_path))
    tdAzim = tdesign[:, 0].astype(np.float32)
    tdZen = tdesign[:, 1].astype(np.float32)
    tdAzim[tdAzim > np.pi] = -2.0 * np.pi + tdAzim[tdAzim > np.pi]
    Yt = spa.sph.sh_matrix(order, tdAzim, tdZen, "real").astype(np.float32)
    tdAzimDeg = tdAzim * (180.0 / np.pi)
    tdZenDeg = tdZen * (180.0 / np.pi)

    # Grid (~2160x1080 default)
    grid_res = float(args.grid_res)
    xgv = np.linspace(-180.0, 180.0, int(round(360.0 / grid_res)), dtype=np.float32)
    ygv = np.linspace(0.0, 180.0, int(round(180.0 / grid_res)), dtype=np.float32)

    # Precompute k-NN IDW mapping (periodic in azimuth)
    idx_map, w_map, H, W = build_periodic_knn_mapping(
        tdAzimDeg, tdZenDeg, xgv, ygv, k=int(args.knn), p=float(args.idw_pow), workers=-1
    )

    # Timing / scheduling
    fps = int(args.fps)
    total_frames = int(numSmp / fs * fps)
    numFrames = min(args.frames if args.frames is not None else total_frames, total_frames)
    winLenSmp = int(0.2 * fs)
    frameLen = int(fs / fps)

    # Normalization anchor
    maxIdx = int(np.argmax(np.sum(np.abs(data), axis=1)))
    s0 = max(maxIdx - winLenSmp // 2, 0)
    s1 = min(maxIdx + winLenSmp // 2, numSmp)
    ref = (dataNorm[s0:s1, :].astype(np.float32) * weights[np.newaxis, :].astype(np.float32))
    refG = np.dot(Yt, ref.T)
    refRMS = np.sqrt(np.mean(refG * refG, axis=1, dtype=np.float32))
    refRMS = np.maximum(refRMS, 1e-12)
    maxRMSdB = float(10.0 * np.log10(np.max(refRMS)))
    if args.debug:
        print(f"[DEBUG] maxRMSdB (ref window): {maxRMSdB:.2f} dB")

    # ffmpeg encoder command (portable)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff = args.ffmpeg
    ff_args = [
        ff, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-an",
    ]
    enc = args.encoder
    if enc == "libx264":
        ff_args += [
            "-vcodec", "libx264",
            "-preset", str(args.preset),
            "-crf", str(args.crf),
            "-pix_fmt", "yuv420p",
            "-threads", "0",
            "-movflags", "faststart",
            str(out_path),
        ]
    else:
        # Hardware encoders prefer bitrate over CRF
        ff_args += [
            "-vcodec", enc,
            "-b:v", str(args.bitrate),
            "-pix_fmt", "yuv420p",
            "-threads", "0",
            "-movflags", "faststart",
            str(out_path),
        ]
    proc = subprocess.Popen(ff_args, stdin=subprocess.PIPE)

    try:
        cmap_obj = matplotlib.colormaps[args.cmap]  # raises KeyError if unknown
    except KeyError as e:
        raise ValueError(f"Unknown colormap '{args.cmap}'. "
                         f"Try one of: {', '.join(list(matplotlib.colormaps.keys())[:15])} ...") from e
    lut = (cmap_obj(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)

    if args.debug:
        print(f"[DEBUG] Computing {numFrames} frames at {fps} fps, grid {W}x{H}, writing to {out_path.resolve()} ...")

    old_time = datetime.now()

    # Multiprocessing pool (ordered streaming to ffmpeg)
    cpu = mp.cpu_count()
    chunksize = max(8, numFrames // (cpu * 8)) if numFrames > 0 else 8
    with mp.Pool(
        processes=cpu,
        initializer=_pool_init,
        initargs=(dataNorm, weights, Yt, idx_map, w_map, (H, W), numSmp, frameLen, winLenSmp, float(args.dynamic_db), maxRMSdB),
    ) as pool:
        for fidx, gray in enumerate(pool.imap(_compute_frame, range(numFrames), chunksize=chunksize)):
            # Debug checkpoint every 10 seconds of frames
            if args.debug and (fidx % (10 * fps) == 0):
                now = datetime.now()
                dt = now - old_time
                print(f"[DEBUG] Starting frame {fidx}, dt/10s: {dt}")
                old_time = now

            # Gray HxW -> RGB HxWx3 via LUT
            rgb = lut[gray]  # uint8
            try:
                proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                if args.debug:
                    print("[DEBUG] FFmpeg ended early (BrokenPipe).")
                break

    proc.stdin.close()
    proc.wait()
    print(f"Saved energy video to: {out_path.resolve()}")


if __name__ == "__main__":
    # Safer cross-platform start method (mitigates some IPC warnings on macOS/Linux)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
