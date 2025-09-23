[![PyPI](https://img.shields.io/badge/python-3.11-blue.svg)]() [![CC BY SA 4.0][cc-by-sa-shield]][cc-by-sa]

# AmbisonicEnergyRenderer.py
Generates an energy visualization video from an Ambisonic (ACN/SN3D) WAV file using precomputed k-NN inverse-distance weighting (IDW) onto a regular 2D grid, multiprocessing for per-frame computation, and direct ffmpeg streaming for efficient encoding.

The script supports high resolutions (e.g., 2160×1080), optional hardware encoders on macOS and Windows/NVIDIA, and periodic debug timing checkpoints every 10 seconds of frames to track throughput.

## Features
- Ambisonic order is detected from the input channel count and used to build spherical-harmonic matrices and *max*-rE weights automatically, so higher orders are supported when more channels are present.
- Fast gridding via precomputed cKDTree k-NN mapping with azimuth wrap-around duplication to avoid vertical seam artifacts at −180°/180° longitudes, replacing slow per-frame scattered interpolation.
- Multiprocessing with per-worker globals minimizes inter-process overhead by sharing large read-only arrays and streaming frames in order directly into ffmpeg, avoiding large intermediate files and peak memory spikes.
- Matplotlib colormap API is optionally selectable with `--cmap` to choose any registered colormap name.
- Optional hardware-accelerated video encoding on macOS (VideoToolbox) and Windows/NVIDIA (NVENC), with libx264 as the portable default fallback when hardware codecs are not available.

## Requirements
- Python 3.9+ with NumPy, SciPy, Matplotlib, spaudiopy, and wavio installed, as the script relies on array math, spherical-harmonic utilities, and WAV I/O.
- FFmpeg available on the system PATH (or specify via `–-ffmpeg`), since frames are streamed as raw RGB for encoding to MP4, and encoders must exist in the local ffmpeg build for the selected `–-encoder`.
- T-design file `Design_5200_100_random.dat` placed next to the script by default (or a custom path via `–-tdesign`), providing the sampling directions for directional energy computation on the sphere.

## Install dependencies with pip:
```
python -m pip install numpy scipy matplotlib spaudiopy wavio
```
FFmpeg should be a recent build with encoders for the chosen backend (libx264, h264_videotoolbox on macOS, h264_nvenc on NVIDIA, etc.), which can be verified via `ffmpeg -encoders` on each platform.

## Hardware acceleration notes
- macOS: use `–-encoder h264_videotoolbox` or `–-encoder hevc_videotoolbox`, providing a target bitrate via `–-bitrate` to leverage Apple VideoToolbox hardware encoding on supported Macs.
- Windows/NVIDIA (and Linux with NVIDIA): use `–-encoder h264_nvenc` or `–-encoder hevc_nvenc` with `–-bitrate` to use NVENC on RTX GPUs, provided the FFmpeg build includes NVENC support and drivers are installed.
- Generic fallback: use `–-encoder libx264` with `–-crf` and `–-preset` controls for a portable software path that works without hardware encoders on all platforms.

## Usage
Basic invocation:
```
python AmbisonicEnergyRenderer.py -i path/to/input.wav
```
This creates an MP4 next to the input with the same basename, using libx264, CRF=20, “ultrafast” preset, 25 fps, and a ~2160×1080 grid by default via `–-grid-res 1/6`.

Full CLI:
```
-i, --input PATH              Input Ambisonics WAV (ACN/SN3D) [required]
-o, --output PATH             Output MP4 (default: input with .mp4)
-t, --tdesign PATH            Path to Design_5200_100_random.dat (default: script dir)
-d, --dynamic-db FLOAT        Dynamic range in dB for normalization (default: 20.0)
-n, --frames INT              Number of frames to render (default: full length)
--fps INT                     Frames per second (default: 25)
--grid-res FLOAT              Grid step in degrees (default: 1/6 → ~2160×1080)
--knn INT                     k-NN for IDW (default: 3)
--idw-pow FLOAT               IDW power parameter p (default: 2.0)
-–cmap STR                    Matplotlib colormap name (e.g., inferno, magma, coolwarm, twilight_r; default: viridis)
--encoder STR                 One of: libx264 (default), h264_videotoolbox, hevc_videotoolbox,
                              h264_nvenc, hevc_nvenc, h264_qsv, hevc_qsv, h264_amf, hevc_amf
--crf STR                     CRF for libx264 (default: 20); ignored by hardware encoders
--bitrate STR                 Target bitrate (e.g., 10M) for hardware encoders; ignored by libx264
--preset STR                  libx264 preset (default: ultrafast); hardware encoders ignore this
--ffmpeg STR                  Path to ffmpeg executable (default: ffmpeg on PATH)
--debug                       Print timing/progress every 10 seconds of frames (default: True)
```
These flags match the script's argparse interface and encoder handling, allowing platform-independent selection of software or hardware encoding and control of spatial grid, frame rate, dynamic range normalization, and any registered Matplotlib colormap.

## Colormaps
- Choose any registered colormap via `--cmap`, for example: inferno, viridis, plasma, magma, cividis, coolwarm, RdYlBu, twilight, twilight_r, tab10, and all `_r` reversed variants.
- Matplotlib's universal registry is accessible through `matplotlib.colormaps[name]`, which this script uses to build a 256‑entry RGB lookup table.
- Perceptually uniform sequential colormaps (viridis, magma, inferno, cividis) are recommended for scalar magnitude; diverging maps (e.g., coolwarm) are best around a meaningful midpoint; cyclic maps (e.g., twilight) are appropriate when endpoints should meet.

List available colormaps known to the local Matplotlib installation:
```
python -c "import matplotlib; print(list(matplotlib.colormaps.keys()))"
```

## Examples
- Portable software encode (CRF based):
```
python AmbisonicEnergyRenderer.py -i input.wav -o out.mp4 --encoder libx264 --crf 20 --preset ultrafast
```

- macOS VideoToolbox hardware encode:
```
python AmbisonicEnergyRenderer.py -i input.wav -o out.mp4 --encoder h264_videotoolbox --bitrate 12M
```

- Windows/NVIDIA NVENC hardware encode:
```
python AmbisonicEnergyRenderer.py -i input.wav -o out.mp4 --encoder h264_nvenc --bitrate 12M
```

- Explicit ffmpeg path on Windows:
```
python AmbisonicEnergyRenderer.py -i input.wav -o out.mp4 --ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"
```

## Performance tuning
- Resolution: `-–grid-res` controls spatial resolution; the default 1/6° produces ~2160×1080, and larger values (e.g., 0.5°) reduce pixels and speed up computation and encoding.
- Interpolation smoothness/detail: `–-knn` and `–-idw-pow` adjust IDW interpolation; increasing k smooths the map while increasing idw-pow sharpens local detail, with runtime and memory roughly scaling linearly with k due to the gather-and-weight step.
- Encoder: hardware encoders (`-–encoder h264_videotoolbox` on macOS or `–-encoder h264_nvenc` on NVIDIA) can substantially reduce total time-to-video compared to libx264, especially at larger resolutions, assuming the local ffmpeg build supports them.
- Frames per second and window length: `–-fps` controls frame count per second, while the script uses a 200 ms energy window; fewer frames or a smaller grid reduce total compute and bitrate while maintaining intelligible motion for many use cases.

## How it works
- The script reads the WAV, infers the Ambisonic order from the channel count, computes max-rE weights, and applies spherical harmonics at t-design directions to obtain directional energy over a sliding window per frame, normalized to a chosen dynamic range.
- A periodic cKDTree mapping is built once to map from the scattered spherical sample points to a uniform azimuth×zenith grid using k-NN IDW with azimuth seam duplication, and each frame applies a vectorized gather-and-weight to produce a grid without triangulation overhead or seam artifacts.
- Frames are colormapped via a 256‑entry LUT using `matplotlib.colormaps[name]`, and raw RGB frames are streamed into ffmpeg for near real‑time encoding with the chosen encoder.

## Troubleshooting
- Hardware encoder not found: run `ffmpeg -encoders` to verify availability, and choose `--encoder libx264` if the desired hardware encoder is not compiled in or drivers are missing, or point `--ffmpeg` to a different ffmpeg binary that includes the encoder.
- Multiprocessing semaphore warnings at shutdown: these often indicate the resource tracker cleaned up IPC objects on exit and are generally harmless; the script uses a with Pool context to cleanly close and join processes, and the warning can be ignored if outputs are correct.

## Acknowledgments
This animation approach builds on the original logic and implementation created by [Thomas Deppisch](https://github.com/thomasdeppisch) - the first author of the earlier script, whose work on spherical sampling, directional energy mapping, and visualization inspired this ffmpeg‑based version. Thank you for the foundational idea and methodology that made this tool possible.

## Support
All questions, comments and insights please address to me via e-mail: bartlomiej.mroz@pg.edu.pl

## License
This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

This means you are free to:

-   **Share**: Copy and redistribute the material in any medium or format.
-   **Adapt**: Remix, transform, and build upon the material.

Under the following terms:

-   **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
-   **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

[![CC BY SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: https://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://i.creativecommons.org/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY%20SA%204.0-lightgrey.svg