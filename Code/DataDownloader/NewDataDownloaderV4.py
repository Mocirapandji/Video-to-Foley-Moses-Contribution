import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import yt_dlp
    import subprocess
    from pathlib import Path
    import librosa
    import soundfile as sf
    import numpy as np
    import cv2
    from ultralytics import YOLO 
    from scipy.signal import find_peaks
    import json
    import shutil
    import IPython.display as ipd
    import random
    import matplotlib.pyplot as plt 
    import time
    import os

    return (
        Path,
        YOLO,
        cv2,
        find_peaks,
        json,
        librosa,
        mo,
        random,
        sf,
        shutil,
        subprocess,
        yt_dlp,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Preparing model
    """)
    return


@app.cell
def _(Path, YOLO, mo):
    # --- CONFIGURATION ---
    # Removed 'SEM' so it goes straight to your desktop folder
    PROJECT_ROOT = Path.home() / "Desktop/foley-to-sound"
    DATA_DIR = PROJECT_ROOT / "FinalDataset"

    # Clean folder structure for easy combining later
    DIRS = {
        "video": DATA_DIR / "raw_downloads/video",     # Stores the full 10-min YouTube videos
        "audio": DATA_DIR / "raw_downloads/audio",     # Stores the full extracted WAVs
        "out_audio": DATA_DIR / "clips/audio",         # ALL your final sliced audio sequences go here
        "out_video": DATA_DIR / "clips/video",         # ALL your final sliced video sequences go here
        "metadata": DATA_DIR / "metadata",             # JSON files tracking your data
    }

    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Initialize YOLO
    model = YOLO('yolov8n-pose.pt') 

    mo.md(r"""
    # 🦶 Smart Foley Generator (YOLOv8 Edition)
    **Status:** Using YOLOv8 Pose for robust human tracking. Saving to FinalDataset.
    """)
    return DATA_DIR, DIRS, PROJECT_ROOT, model


@app.cell
def _(mo):
    mo.md(r"""
    # Downloads the Code
    """)
    return


@app.cell
def _(DIRS, PROJECT_ROOT, cv2, find_peaks, librosa, model, subprocess, yt_dlp):
    def download_data(url, dataset_id):
        """Downloads video and extracts audio track."""
        print(f"⬇️ Processing: {dataset_id}")

        video_out = DIRS["video"] / f"{dataset_id}.mp4"
        audio_out = DIRS["audio"] / f"{dataset_id}.wav"

        if video_out.exists() and audio_out.exists():
            print("   Cache hit! Files already exist.")
            return video_out, audio_out

        # 1. Download Video
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': str(video_out),
            'quiet': True,
            'cookiefile': str(PROJECT_ROOT / 'cookies.txt'), 
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])



        # 2. Extract WAV Audio (High Quality)
        cmd = [
            'ffmpeg', '-i', str(video_out),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', # Mono 44.1kHz
            '-y', str(audio_out)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return video_out, audio_out

    def analyze_footsteps(video_path, audio_path, dataset_id, threshold=0.3):
        print(f"🧠 Optimized Analysis for {dataset_id}...")

        # 1. VISUAL TRACKING (Optimized YOLO)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamps = []
        ankle_y_positions = [] 

        frame_count = 0
        skip_frames = 3 

        print(f"   -> Video Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
        print(f"   -> Processing every {skip_frames}th frame...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Only process every Nth frame
            if frame_count % skip_frames == 0:
                if (frame_count // skip_frames) % 100 == 0:
                    print(f"      Processed {frame_count}/{total_frames} frames...", end='\r')

                results = model.predict(frame, imgsz=320, verbose=False, conf=0.5, classes=[0])

                current_y = 0.0
                found = False

                for r in results:
                    if r.keypoints is not None and len(r.keypoints.xyn) > 0:
                        kpts = r.keypoints.xyn[0].cpu().numpy()
                        if len(kpts) >= 17:
                            l_y = kpts[15][1]
                            r_y = kpts[16][1]
                            if l_y > 0 or r_y > 0:
                                current_y = max(l_y, r_y)
                                found = True

                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                timestamps.append(ts)
                ankle_y_positions.append(current_y if found else 0.0)

            frame_count += 1

        cap.release()
        print(f"\n   -> Visual tracking complete.")

        if not ankle_y_positions:
            print("❌ No people detected in video.")
            return [], None, None

        # 2. DETECT VISUAL STRIKES
        eff_fps = fps / skip_frames

        visual_peaks, _ = find_peaks(ankle_y_positions, distance=int(eff_fps*0.35), prominence=0.01)
        visual_times = [timestamps[i] for i in visual_peaks]

        print(f"   -> Found {len(visual_times)} visual steps")

        # 3. AUDIO ANALYSIS
        y, sr = librosa.load(audio_path, sr=44100)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        audio_onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', delta=threshold)

        # 4. SYNC
        valid_events = []
        for a_time in audio_onsets:
            match = min(visual_times, key=lambda x: abs(x - a_time), default=None)
            if match and abs(match - a_time) <= 0.3:
                valid_events.append({"time": a_time, "visual_time": match})

        print(f"✅ MATCHED: {len(valid_events)} verified footsteps")
        return valid_events, y, sr

    return analyze_footsteps, download_data


@app.cell
def _(mo):
    mo.md(r"""
    # Slices it into overlapping pairs
    """)
    return


@app.cell
def _(DIRS, sf, subprocess):
    def slice_and_save(events, video_path, audio_data, sr, dataset_id, min_steps=2, clip_duration=3.0, slide_step=2.5):
        """
        Slices the video into perfectly uniform lengths using a sliding window.
        - min_steps: Minimum footsteps required inside the window to save it.
        - clip_duration: EXACT length of every exported clip (in seconds).
        - slide_step: Minimum time to move forward before starting a new clip (prevents exact duplicates).
        """
        if not events:
            return []

        results = []
        saved_count = 0
        last_saved_start = -999.0  # Tracks the start time of the last saved clip

        total_audio_duration = len(audio_data) / sr

        print(f"🔄 Slicing into consistent {clip_duration}s windows...")

        for i, event in enumerate(events):
            # 1. Define the uniform window (starting 0.3s before the footstep)
            start_t = max(0.0, event['time'] - 0.3)
            end_t = start_t + clip_duration

            # 2. Safety checks
            if end_t > total_audio_duration:
                continue # Skip if it goes past the end of the video

            if start_t < last_saved_start + slide_step:
                continue # Skip if we just made a clip right here (prevents spam)

            # 3. Count how many footsteps fall inside this exact 3-second window
            steps_in_window = [e for e in events if start_t <= e['time'] <= end_t]

            # 4. Save if it meets our minimum step requirement
            if len(steps_in_window) >= min_steps:

                # Save Audio
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                audio_clip = audio_data[start_sample:end_sample]

                a_name = f"{dataset_id}_win_{saved_count:03d}_{len(steps_in_window)}steps.wav"
                sf.write(DIRS["out_audio"] / a_name, audio_clip, sr)

                # Save Video
                v_name = f"{dataset_id}_win_{saved_count:03d}_{len(steps_in_window)}steps.mp4"
                cmd = [
                    'ffmpeg', '-ss', str(start_t), '-i', str(video_path),
                    '-t', str(clip_duration), '-c:v', 'libx264', '-c:a', 'aac',
                    '-loglevel', 'error', '-y', 
                    str(DIRS["out_video"] / v_name)
                ]
                subprocess.run(cmd, check=False)

                # Log the metadata
                results.append({
                    "id": f"{dataset_id}_win_{saved_count}",
                    "filename": v_name,
                    "start": start_t,
                    "end": end_t,
                    "step_count": len(steps_in_window),
                    "duration": clip_duration
                })

                saved_count += 1
                last_saved_start = start_t  # Update tracker so we slide forward properly

        print(f"✨ Saved {saved_count} perfectly uniform {clip_duration}s clips!")
        return results

    return (slice_and_save,)


@app.cell
def _(DATA_DIR, DIRS, analyze_footsteps, download_data, json, slice_and_save):
    # --- RUN PIPELINE ---

    VIDEO_URL = "https://www.youtube.com/watch?v=FmQepeckYIM"
    DATA_ID = "gravel_walk_08" 

    # 2. Download
    v_path, a_path = download_data(VIDEO_URL, DATA_ID)

    # 3. Analyze (Lowered threshold from 0.3 to 0.15 to hear leaves better)
    events, y, sr = analyze_footsteps(v_path, a_path, DATA_ID, threshold=0.25)

    # 4. Slice
    if len(events) > 0:
        # Lowered min_steps to 2 so we don't throw away good, short clips
        metadata = slice_and_save(events, v_path, y, sr, DATA_ID, min_steps=2)

        # Save Manifest
        with open(DIRS["metadata"] / f"{DATA_ID}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n🎉 Done! Created {len(metadata)} paired clips.")
        print(f"📂 Check your folders at: {DATA_DIR.resolve()}")
    else:
        print("\n⚠️ No verified footsteps found. Try adjusting thresholds.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dont run this cell or else it will reset everyhting (i hashed it out just in case)
    """)
    return


@app.cell
def _(DATA_DIR, DIRS, mo, random, shutil):
    mo.md(r"""
    # ✂️ Train / Test Splitter
    Run this ONLY when you are finished generating all your clips. It will safely copy 80% to train and 20% to test.
    """)

    def split_dataset(train_ratio=0.8):
        print("🚀 Starting Train/Test Split...")

        # Define our source folders
        src_audio_dir = DIRS["out_audio"]
        src_video_dir = DIRS["out_video"]

        # Define our new split folders
        split_base = DATA_DIR / "dataset_split"

        dest_dirs = {
            "train_audio": split_base / "train/audio",
            "train_video": split_base / "train/video",
            "test_audio": split_base / "test/audio",
            "test_video": split_base / "test/video",
        }

        # 👇 THE FIX: Completely delete the old split folder to prevent duplicates/leakage
        if split_base.exists():
            print("🧹 Cleaning up old split data...")
            shutil.rmtree(split_base)

        # Create the new directories safely
        for d in dest_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Get all the audio files we generated
        audio_files = list(src_audio_dir.glob("*.wav"))

        if not audio_files:
            print("⚠️ No clips found in your 'clips/audio' folder! Generate some data first.")
            return

        # Match the audio files to their corresponding video files
        paired_data = []
        for audio_path in audio_files:
            base_name = audio_path.stem 
            video_path = src_video_dir / f"{base_name}.mp4"

            if video_path.exists():
                paired_data.append((audio_path, video_path))

        print(f"✅ Found {len(paired_data)} perfectly matched Audio/Video pairs.")

        # Randomly shuffle the data
        random.shuffle(paired_data)

        # Calculate where to slice the list
        split_index = int(len(paired_data) * train_ratio)

        train_pairs = paired_data[:split_index]
        test_pairs = paired_data[split_index:]

        print(f"📂 Copying {len(train_pairs)} pairs to TRAIN...")
        for a_path, v_path in train_pairs:
            shutil.copy2(a_path, dest_dirs["train_audio"] / a_path.name)
            shutil.copy2(v_path, dest_dirs["train_video"] / v_path.name)

        print(f"📂 Copying {len(test_pairs)} pairs to TEST...")
        for a_path, v_path in test_pairs:
            shutil.copy2(a_path, dest_dirs["test_audio"] / a_path.name)
            shutil.copy2(v_path, dest_dirs["test_video"] / v_path.name)

        print(f"\n🎉 Split Complete! Check your new folder: {split_base.resolve()}")

    # Run the split! 
    # split_dataset(train_ratio=0.8)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
