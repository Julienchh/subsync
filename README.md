# subsync

Command-line tool and local web UI to automatically download and synchronize subtitles (`.srt`, `.ass`, `.ssa`, `.vtt`) to video audio tracks.

## Background & Credits

This repository is based on [sc0ty/subsync](https://github.com/sc0ty/subsync) created by Michał Szymaniak (`@sc0ty`), licensed under the GPLv3.

The original project relied on a C++ extension (`gizmo`) and `Pocketsphinx`, which has become difficult to build on modern systems due to FFmpeg and Python API changes. This fork removes the C++ extension and replaces it with a Python implementation built on `ffmpeg`, `Silero VAD` (ONNX), and `numpy`.

## Main Changes

- **Python + Silero VAD + FFT alignment engine**:
  Audio is decoded with `ffmpeg` (extracting the center channel on 5.1/7.1 tracks to isolate dialogue from background music) and passed through **Silero VAD** to get a 100 Hz speech probability envelope (10 ms resolution).
- **Scene-by-scene alignment (commercial breaks & recaps)**:
  TV broadcast subtitles often have different commercial break timings than BluRay or WEB-DL releases, or include a *"Previously on..."* recap that is not in the video (or vice versa). The aligner splits subtitles into scenes at silence gaps, finds the offset for each scene, and drops orphan recap lines that fall before `00:00:00`.
- **Linear drift compensation**:
  When a video and subtitle track come from cuts of different lengths (for example an extended/uncut video vs. a TV broadcast subtitle that drifts by 1–2 seconds per minute), the aligner estimates and compensates for the linear slope before aligning individual scenes.
- **Post-sync verification and duration trimming**:
  After applying the alignment formula, a verification pass checks each scene window against the voice envelope to fix any residual offset, then adjusts individual subtitle start/end times to match speech boundaries (preventing subtitles from appearing early during silence or lingering after a sentence ends).
- **Series subtitle downloader (`subsync series`)**:
  Searches and downloads subtitles from Addic7ed, Gestdown, SubDL, or OpenSubtitles, normalizes character encodings to `UTF-8`, strips credit/watermark lines, and writes output files using Jellyfin / Plex naming conventions (`VideoName.fr.srt`).
- **Multi-core processing (`--workers`)**:
  Processes multiple episodes in parallel across CPU cores using `ProcessPoolExecutor`.

## Installation

Requirements: Python 3.8+ and `ffmpeg` installed on your system (`sudo apt install ffmpeg`).

```bash
git clone git@github.com:Julienchh/subsync.git
cd subsync
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Download and synchronize a TV series (`series`)

```bash
# Synchronize a season (with Jellyfin / Plex file naming)
./bin/subsync series "/path/to/Show/S01" --lang fr --jellyfin

# Force re-synchronization even if .srt files already exist, using 8 CPU workers
./bin/subsync series "/path/to/Show/S01" --lang fr --jellyfin --force --workers 8

# Scan all seasons recursively
./bin/subsync series "/path/to/Show" --lang fr --jellyfin --recursive --force

# Process specific episodes only
./bin/subsync series "/path/to/Show/S01" --lang fr --jellyfin --force --episodes 4,5
```

At the end of a run, a summary table shows the selected subtitle release, offset progression (start / middle / end), and the correlation score before and after synchronization.

### 2. Synchronize existing local subtitles (`batch`)

If `.srt` files are already present on disk next to the video files:

```bash
./bin/subsync batch "/path/to/folder" --in-place --backup --recursive
```

### 3. Synchronize a single file

From the command line:
```bash
./bin/subsync --cli \
  --sub "/path/to/episode.srt" \
  --ref "/path/to/episode.mkv" \
  --out "/path/to/episode.fr.srt"
```

From Python:
```python
from subsync.synchro.modern_engine import synchronize_pair

res = synchronize_pair(
    sub_path="episode.srt",
    ref_path="episode.mkv",
    out_path="episode.fr.srt",
)
print(f"Offset: {res.offset_s:+.2f}s | Correlation: {res.raw_factor*100:.1f}% -> {res.post_factor*100:.1f}%")
```

### 4. Local Web UI

```bash
./bin/subsync-gui
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

GNU General Public License v3.0 (GPLv3) — see [LICENSE](LICENSE).
- Original project: [Michał Szymaniak (`sc0ty/subsync`)](https://github.com/sc0ty/subsync)
