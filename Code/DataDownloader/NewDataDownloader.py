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
    from ultralytics import YOLO # <--- The new, more robust library
    from scipy.signal import find_peaks
    import json
    import shutil
    import IPython.display as ipd
    import random
    import matplotlib.pyplot as plt # Ensure plt is imported

    import time
    import os


    # --- CONFIGURATION ---
    PROJECT_ROOT = Path.home() / "Desktop/SEM/foley-to-sound"
    DATA_DIR = PROJECT_ROOT / "data"

    # Create folders
    DIRS = {
        "video": DATA_DIR / "raw_videos",
        "audio": DATA_DIR / "raw_audio",
        "out_audio": DATA_DIR / "isolated_footsteps/audio",
        "out_video": DATA_DIR / "isolated_footsteps/video",
        "metadata": DATA_DIR / "metadata",
    }
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Initialize YOLO (Downloads a small model automatically)
    # This model is great at finding "person" and "keypoints" (ankles/knees)
    model = YOLO('yolov8n-pose.pt') 

    mo.md(r"""
    # 🦶 Smart Foley Generator (YOLOv8 Edition)
    **Status:** Using YOLOv8 Pose for robust human tracking.
    """)
    return (
        DIRS,
        cv2,
        find_peaks,
        ipd,
        json,
        librosa,
        mo,
        model,
        np,
        os,
        plt,
        random,
        sf,
        subprocess,
        time,
        yt_dlp,
    )


@app.cell
def _(mo):
    mo.md(r"""
    #First Cell
    """)
    return


@app.cell
def _():
    # # Setup directory structure
    # PROJECT_ROOT = Path.home() / "Desktop/SEM/foley-to-sound"
    # DATA_DIR = PROJECT_ROOT / "data"

    # # Subdirectories
    # DIRS = {
    #     "video": DATA_DIR / "raw_videos",
    #     "audio": DATA_DIR / "raw_audio",
    #     "clips_audio": DATA_DIR / "isolated_footsteps_audio",
    #     "clips_video": DATA_DIR / "isolated_footsteps_video",
    #     "metadata": DATA_DIR / "metadata",
    #     "debug": DATA_DIR / "debug_plots" # New: for checking our detection quality
    # }

    # # Create all directories
    # for d in DIRS.values():
    #     d.mkdir(parents=True, exist_ok=True)

    # print("✅ Directory structure ready.")
    return


@app.cell
def _(DIRS, subprocess, yt_dlp):
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
    return (download_data,)


@app.cell
def _(mo):
    mo.md(r"""
    #Second Cell
    """)
    return


@app.cell
def _(cv2, find_peaks, librosa, model):
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
                # Progress indicator every 100 processed frames
                if (frame_count // skip_frames) % 100 == 0:
                    print(f"      Processed {frame_count}/{total_frames} frames...", end='\r')

                # Run YOLO (Fastest settings)
                results = model.predict(frame, imgsz=320, verbose=False, conf=0.5, classes=[0]) # classes=[0] detects only people
            
                current_y = 0.0
                found = False
            
                for r in results:
                    if r.keypoints is not None and len(r.keypoints.xyn) > 0:
                        kpts = r.keypoints.xyn[0].cpu().numpy()
                        if len(kpts) >= 17:
                            # Index 15/16 are ankles
                            l_y = kpts[15][1]
                            r_y = kpts[16][1]
                            if l_y > 0 or r_y > 0:
                                current_y = max(l_y, r_y)
                                found = True
            
                # Record data ONLY for processed frames
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                timestamps.append(ts)
                ankle_y_positions.append(current_y if found else 0.0)
        
            frame_count += 1
            
        cap.release()
        print(f"\n   -> Visual tracking complete.")

        # If the list is empty because no one was found, stop early
        if not ankle_y_positions:
            print("❌ No people detected in video.")
            return [], None, None

        # 2. DETECT VISUAL STRIKES
        # Adjust distance based on the fact we skipped frames
        # effectively fps for our data is (fps / skip_frames)
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
    return (analyze_footsteps,)


@app.cell
def _(mo):
    mo.md(r"""
    #Third Cell
    """)
    return


@app.cell
def _():
    # def slice_and_save(events, video_path, audio_data, sr, dataset_id, pad=0.5):
    #     results = []

    #     for i, event in enumerate(events):
    #         center = event['time']
    #         start = max(0, center - pad)
    #         end = center + pad

    #         # 1. Save Audio Clip
    #         start_sample = int(start * sr)
    #         end_sample = int(end * sr)
    #         audio_clip = audio_data[start_sample:end_sample]

    #         a_name = f"{dataset_id}_step_{i:03d}.wav"
    #         # FIX: use 'out_audio' instead of 'clips_audio'
    #         sf.write(DIRS["out_audio"] / a_name, audio_clip, sr)

    #         # 2. Save Video Clip
    #         v_name = f"{dataset_id}_step_{i:03d}.mp4"
    #         cmd = [
    #             'ffmpeg', '-ss', str(start), '-i', str(video_path),
    #             '-t', str(end - start), '-c:v', 'libx264', '-c:a', 'aac',
    #             '-loglevel', 'error', '-y', 
    #             # FIX: use 'out_video' instead of 'clips_video'
    #             str(DIRS["out_video"] / v_name)
    #         ]
    #         subprocess.run(cmd, check=False)

    #         results.append({
    #             "id": f"{dataset_id}_{i}",
    #             "audio_file": a_name,
    #             "video_file": v_name,
    #             "timestamp": center
    #         })

    #     return results
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Third Cell V2
    """)
    return


@app.cell
def _(DIRS, sf, subprocess):
    def slice_and_save(events, video_path, audio_data, sr, dataset_id, min_steps=3, gap_tolerance=1.5):
        """
        Groups verified footsteps into sequences.
        - min_steps: Minimum number of footsteps required to save a clip (e.g., 3 or 4).
        - gap_tolerance: If steps are further apart than this (seconds), start a new sequence.
        """
    
        if not events:
            return []

        # 1. Group events into sequences
        sequences = []
        current_seq = [events[0]]
    
        for i in range(1, len(events)):
            prev_time = current_seq[-1]['time']
            curr_time = events[i]['time']
        
            # If the next step is close enough, add to current sequence
            if (curr_time - prev_time) <= gap_tolerance:
                current_seq.append(events[i])
            else:
                # Gap is too big -> End current sequence
                sequences.append(current_seq)
                current_seq = [events[i]]
            
        # Append the final sequence
        if current_seq:
            sequences.append(current_seq)

        # 2. Process and Save Sequences
        results = []
        print(f"🔄 Found {len(sequences)} potential walking sequences...")

        saved_count = 0
        for i, seq in enumerate(sequences):
            # Filter: Only save if it has enough steps
            if len(seq) < min_steps:
                continue
            
            # Define start and end times with some buffer
            # Start 0.5s before the first step, End 0.8s after the last step
            first_step = seq[0]['time']
            last_step = seq[-1]['time']
        
            start_t = max(0, first_step - 0.5)
            end_t = last_step + 0.8
            duration = end_t - start_t
        
            # Sanity check: Don't save if clip is too long (e.g. > 20s) or too short
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
def _(mo):
    mo.md(r"""
    # Fourth Cell
    """)
    return


@app.cell
def _(DIRS, analyze_footsteps, download_data, json, slice_and_save):
    # --- RUN PIPELINE ---

    # 1. Define Source
    VIDEO_URL = "https://www.youtube.com/watch?v=XxmZxM8AtUc" # Replace with your URL
    DATA_ID = "gravel_walk_01"

    # 2. Download
    v_path, a_path = download_data(VIDEO_URL, DATA_ID)

    # 3. Analyze
    events, y, sr = analyze_footsteps(v_path, a_path, DATA_ID)

    # 4. Slice
    if len(events) > 0:
        metadata = slice_and_save(events, v_path, y, sr, DATA_ID)
    # 
        # Save Manifest
        with open(DIRS["metadata"] / f"{DATA_ID}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n🎉 Done! Created {len(metadata)} paired clips.")
    else:
        print("\n⚠️ No verified footsteps found. Try adjusting thresholds.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Fifth Cell
    """)
    return


@app.cell
def _():

    # # Get list of generated videos
    # video_files = list(DIRS["out_video"].glob("*.mp4"))

    # if video_files:
    #     # Pick a random video to check
    #     preview_file = random.choice(video_files)
    #     print(f"🎥 Previewing: {preview_file.name}")
    
    #     # Show video
    #     # Note: In some notebook environments, you might need to use str(preview_file)
    #     ipd.display(ipd.Video(str(preview_file), width=400, embed=True))
    
    #     # Load audio with UNIQUE variable names
    #     audio_file_path = DIRS["out_audio"] / preview_file.with_suffix(".wav").name
    #     preview_y, preview_sr = librosa.load(str(audio_file_path), sr=None)
    
    #     # Plot waveform
    #     plt.figure(figsize=(10, 2))
    #     # Create time axis for the plot
    #     time_axis = np.linspace(0, len(preview_y) / preview_sr, len(preview_y))
    #     plt.plot(time_axis, preview_y)
    #     plt.title(f"Audio Waveform: {preview_file.name}")
    #     plt.xlabel("Time (seconds)")
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("⚠️ No videos found. Check your directories.")
    return


@app.cell
def _(DIRS, ipd, librosa, np, plt, random):
    def preview_random_clip():
        """
        Picks a random clip and plays it without caching issues.
        """
        # 1. Get list of generated videos
        # We use a local list to avoid conflict
        all_videos = list(DIRS["out_video"].glob("*.mp4"))

        if not all_videos:
            print("⚠️ No videos found. Check your directories.")
            return

        # 2. Pick a random video
        selected_video = random.choice(all_videos)
        print(f"🎥 Previewing: {selected_video.name}")
    
        # 3. FORCE RELOAD: Read bytes directly
        # 'vid_file' is a unique name now
        with open(selected_video, "rb") as vid_file:
            preview_bytes = vid_file.read()
        
        # Display Video
        ipd.display(ipd.Video(data=preview_bytes, embed=True, width=400, mimetype="video/mp4"))
    
        # 4. Load and Plot Audio
        # Find matching audio file
        target_audio = DIRS["out_audio"] / selected_video.with_suffix(".wav").name
    
        if target_audio.exists():
            # Load with unique local variables
            y_local, sr_local = librosa.load(str(target_audio), sr=None)
        
            plt.figure(figsize=(10, 2))
            t_axis = np.linspace(0, len(y_local) / sr_local, len(y_local))
        
            plt.plot(t_axis, y_local, color="#4CAF50", alpha=0.8) # Green for visibility
            plt.title(f"Audio Waveform: {selected_video.name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amp")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Corresponding audio file missing.")

    # Run the function
    preview_random_clip()
    return


@app.cell
def _(DIRS, ipd, librosa, np, os, plt, random, time):


    def preview_fresh_clip(minutes=5):
        """
        Looks for videos created in the last 'minutes' and picks a RANDOM one.
        """
        # 1. Get all videos
        all_videos = list(DIRS["out_video"].glob("*.mp4"))
    
        if not all_videos:
            print("⚠️ No videos found.")
            return

        # 2. Filter: Keep only files modified recently (e.g., last 5 mins)
        current_time = time.time()
        fresh_videos = []
    
        for vid in all_videos:
            mod_time = os.path.getmtime(vid)
            # Check if file is younger than 'minutes'
            if (current_time - mod_time) < (minutes * 60):
                fresh_videos.append(vid)

        # Fallback: If no recent videos, just use all videos
        if not fresh_videos:
            print(f"⚠️ No clips found from the last {minutes} mins. Picking from ALL clips.")
            candidates = all_videos
        else:
            print(f"✅ Found {len(fresh_videos)} fresh clips from your recent run.")
            candidates = fresh_videos

        # 3. Pick RANDOM from the candidates
        selected_video = random.choice(candidates)
        print(f"🎥 Previewing: {selected_video.name}")

        # 4. Force Reload (Read Bytes)
        with open(selected_video, "rb") as vid_file:
            vid_bytes = vid_file.read()
        
        ipd.display(ipd.Video(data=vid_bytes, embed=True, width=400, mimetype="video/mp4"))

        # 5. Show Audio
        target_audio = DIRS["out_audio"] / selected_video.with_suffix(".wav").name
        if target_audio.exists():
            y_loc, sr_loc = librosa.load(str(target_audio), sr=None)
            plt.figure(figsize=(10, 2))
            plt.plot(np.linspace(0, len(y_loc)/sr_loc, len(y_loc)), y_loc, color="#00bcd4", alpha=0.8)
            plt.title(f"Audio: {selected_video.name}")
            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.show()

    # Run it
    preview_fresh_clip()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
