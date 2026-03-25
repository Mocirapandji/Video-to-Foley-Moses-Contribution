import marimo

__generated_with = "0.19.7"
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
        sf,
        subprocess,
        yt_dlp,
    )


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
    return DATA_DIR, DIRS, model


@app.cell
def _(DIRS, cv2, find_peaks, librosa, model, subprocess, yt_dlp):
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
def _(DIRS, sf, subprocess):
    def slice_and_save(events, video_path, audio_data, sr, dataset_id, min_steps=3, gap_tolerance=1.5):
        """
        Groups verified footsteps into sequences.
        """
        if not events:
            return []

        # 1. Group events into sequences
        sequences = []
        current_seq = [events[0]]
    
        for i in range(1, len(events)):
            prev_time = current_seq[-1]['time']
            curr_time = events[i]['time']
        
            if (curr_time - prev_time) <= gap_tolerance:
                current_seq.append(events[i])
            else:
                sequences.append(current_seq)
                current_seq = [events[i]]
            
        if current_seq:
            sequences.append(current_seq)

        # 2. Process and Save Sequences
        results = []
        print(f"🔄 Found {len(sequences)} potential walking sequences...")

        saved_count = 0
        for i, seq in enumerate(sequences):
            if len(seq) < min_steps:
                continue
            
            first_step = seq[0]['time']
            last_step = seq[-1]['time']
        
            start_t = max(0, first_step - 0.5)
            end_t = last_step + 0.8
            duration = end_t - start_t
        
            if duration > 20.0:
                print(f"   Skipping sequence {i} (too long: {duration:.1f}s)")
                continue

            # Save Audio
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            audio_clip = audio_data[start_sample:end_sample]
        
            a_name = f"{dataset_id}_seq_{saved_count:03d}_{len(seq)}steps.wav"
            sf.write(DIRS["out_audio"] / a_name, audio_clip, sr)

            # Save Video
            v_name = f"{dataset_id}_seq_{saved_count:03d}_{len(seq)}steps.mp4"
            cmd = [
                'ffmpeg', '-ss', str(start_t), '-i', str(video_path),
                '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
                '-loglevel', 'error', '-y', 
                str(DIRS["out_video"] / v_name)
            ]
            subprocess.run(cmd, check=False)
        
            results.append({
                "id": f"{dataset_id}_seq_{saved_count}",
                "filename": v_name,
                "start": first_step,
                "end": last_step,
                "step_count": len(seq),
                "duration": duration
            })
            saved_count += 1

        print(f"✨ Saved {saved_count} sequences with {min_steps}+ steps.")
        return results
    return (slice_and_save,)


@app.cell
def _(DATA_DIR, DIRS, analyze_footsteps, download_data, json, slice_and_save):
    # --- RUN PIPELINE ---

    VIDEO_URL = "" 
    DATA_ID = "crunchyleaves_walk_01" 

    # 2. Download
    v_path, a_path = download_data(VIDEO_URL, DATA_ID)

    # 3. Analyze (Lowered threshold from 0.3 to 0.15 to hear leaves better)
    events, y, sr = analyze_footsteps(v_path, a_path, DATA_ID, threshold=0.15)

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
