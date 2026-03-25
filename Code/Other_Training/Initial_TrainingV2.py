import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import cv2
    import torch
    import numpy as np
    import librosa
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim

    return DataLoader, Dataset, cv2, librosa, nn, np, optim, os, torch


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 1
    """)
    return


app._unparsable_cell(
    r"""


    def filter_and_clean_dataset(video_dir, audio_dir, dry_run=True):
        print(f"Scanning directories:\nVideo: {video_dir}\nAudio: {audio_dir}\n")

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

        # --- STEP 1: DELETE NON-GRAVEL FILES ---
        print("--- STEP 1: Filtering for 'gravel' only ---")
        non_gravel_count = 0

        for f in video_files:
            if 'gravel' not in f.lower():
                file_path = os.path.join(video_dir, f)
                if dry_run:
                    print(f"[DRY RUN] Would delete non-gravel video: {f}")
                else:sorry before i copy this wil this mean i have to reset my training in cell 3 and 4? like reset the weights?
                    os.remove(file_path)
                non_gravel_count += 1

        for f in audio_files:
            if 'gravel' not in f.lower():
                file_path = os.path.join(audio_dir, f)
                if dry_run:
                    print(f"[DRY RUN] Would delete non-gravel audio: {f}")
                else:
                    os.remove(file_path)
                non_gravel_count += 1

        print(f"Found {non_gravel_count} non-gravel files to remove.\n")

        # --- STEP 2: CLEAN MISMATCHED PAIRS ---
        # We must re-read the folders to see what survived Step 1
        if not dry_run:
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

        print("--- STEP 2: Checking for mismatched pairs ---")
        video_dict = {os.path.splitext(f)[0]: f for f in video_files}
        audio_dict = {os.path.splitext(f)[0]: f for f in audio_files}

        video_bases = set(video_dict.keys())
        audio_bases = set(audio_dict.keys())

        orphaned_videos = video_bases - audio_bases
        orphaned_audios = audio_bases - video_bases

        for base_name in orphaned_videos:
            file_to_delete = os.path.join(video_dir, video_dict[base_name])
            if dry_run:
                print(f"[DRY RUN] Would delete orphaned video: {video_dict[base_name]}")
            else:
                os.remove(file_to_delete)

        for base_name in orphaned_audios:
            file_to_delete = os.path.join(audio_dir, audio_dict[base_name])
            if dry_run:
                print(f"[DRY RUN] Would delete orphaned audio: {audio_dict[base_name]}")
            else:
                os.remove(file_to_delete)

        print(f"Found {len(orphaned_videos)} orphaned videos and {len(orphaned_audios)} orphaned audios.")

        if dry_run:
            print("\n⚠️ This was a DRY RUN. No files were actually deleted.")
            print("Change `dry_run=False` in the code below to delete them permanently!")
        else:
            print("\n✅ MEGA-CLEANUP COMPLETE! You now have a pure, perfectly paired gravel dataset.")

    # --- Try it out! ---
    v2_video_dir = 'FinalDatasetV2/dataset_split/train/video/'
    v2_audio_dir = 'FinalDatasetV2/dataset_split/train/audio/'

    # Run this with dry_run=True first to see what it wants to delete!
    filter_and_clean_dataset(v2_video_dir, v2_audio_dir, dry_run=False)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell
def _(DataLoader, Dataset, cv2, librosa, mo, np, os, torch):
    mo.md(r"""
    # Cell 2
    """)

    # --- 1. THE HELPER FUNCTION ---
    # CHEAT #3: Drop sample rate to 16000 to save GPU brainpower!
    def audio_to_melspectrogram(audio_path, target_sr=16000):
        """Converts audio to a Mel-spectrogram (Decibels)."""
        waveform, sr = librosa.load(audio_path, sr=target_sr)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=1024, hop_length=256, n_mels=80 
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db, sr

    # --- 2. THE DATASET CLASS ---
    class FoleyDataset(Dataset):
        def __init__(self, video_dir, audio_dir, max_frames=30, max_spec_time=128, image_size=(112, 112)):
            self.video_dir = video_dir
            self.audio_dir = audio_dir
            self.max_frames = max_frames
            self.max_spec_time = max_spec_time
            self.image_size = image_size

            # Grab exactly 20 files for the overfitting test!
            self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:20]
            self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])[:20]

            assert len(self.video_files) == len(self.audio_files), "Mismatch in V2 dataset!"

        def __len__(self):
            return len(self.video_files)

        def _process_video(self, video_path):
            video = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < self.max_frames:
                success, frame = video.read()
                if not success: break
                frame = cv2.resize(frame, self.image_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            video.release()

            frames = np.array(frames)
            if len(frames) < self.max_frames:
                padding = self.max_frames - len(frames)
                frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')

            frames = np.transpose(frames, (0, 3, 1, 2))
            return torch.FloatTensor(frames) / 255.0

        def _process_audio(self, audio_path):
            mel_spec_db, _ = audio_to_melspectrogram(audio_path)
            current_time_steps = mel_spec_db.shape[1]

            if current_time_steps < self.max_spec_time:
                padding = self.max_spec_time - current_time_steps
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant', constant_values=mel_spec_db.min())
            else:
                mel_spec_db = mel_spec_db[:, :self.max_spec_time]

            mel_spec_db = np.expand_dims(mel_spec_db, axis=0) 
            return torch.FloatTensor(mel_spec_db)

        def __getitem__(self, idx):
            vid_path = os.path.join(self.video_dir, self.video_files[idx])
            aud_path = os.path.join(self.audio_dir, self.audio_files[idx])
            return self._process_video(vid_path), self._process_audio(aud_path)

    # --- 3. LOAD THE V2 DATA ---
    final_train_video_dir = 'FinalDatasetV2/dataset_split/train/video/'
    final_train_audio_dir = 'FinalDatasetV2/dataset_split/train/audio/'

    v2_dataset = FoleyDataset(final_train_video_dir, final_train_audio_dir)
    v2_dataloader = DataLoader(v2_dataset, batch_size=4, shuffle=True)

    for videos, spectrograms in v2_dataloader:
        print("✅ V2 Dataset successfully loaded a batch!")
        break

    print("\nThe 20 videos used for the overfitting test are:")
    for i, vid_name in enumerate(v2_dataset.video_files):
        print(f"{i+1}. {vid_name}")
    return (v2_dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 3
    """)
    return


@app.cell
def _(nn, optim, torch):
    # --- 1. THE NEURAL NETWORK (V3 WITH TANH HACK) ---
    class FoleyGenerator(nn.Module):
        def __init__(self):
            super(FoleyGenerator, self).__init__()
            self.video_encoder = nn.Sequential(
                nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2), 
            
                nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2),
            
                nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2), 
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 3 * 14 * 14, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * 10 * 16), 
                nn.ReLU()
            )
        
            self.audio_decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
                # THE HACK: Forces active painting of frequencies
                nn.Tanh() 
            )

        def forward(self, x):
            x = x.permute(0, 2, 1, 3, 4)
            x = self.video_encoder(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
            x = x.view(-1, 128, 10, 16) 
            spectrogram = self.audio_decoder(x)
            return spectrogram

    # --- 2. THE TRAINING LOOP (STAYS THE SAME) ---
    def train_foley_model(model, dataloader, epochs=30, learning_rate=0.0001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        model = model.to(device)
        criterion = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        best_loss = float('inf') 
    
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (videos, true_spectrograms) in enumerate(dataloader):
                videos, true_spectrograms = videos.to(device), true_spectrograms.to(device)
                optimizer.zero_grad()
                generated_spectrograms = model(videos)
                loss = criterion(generated_spectrograms, true_spectrograms)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            avg_epoch_loss = running_loss / len(dataloader)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "best_foley_generator_weights_v3.pth")
        return model

    return FoleyGenerator, train_foley_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 4
    """)
    return


@app.cell
def _(FoleyGenerator, mo, os, torch, train_foley_model, v2_dataloader):
    mo.md(r"""
    # Cell 4
    """)

    # --- 3. KICK OFF (AND RESUME) TRAINING ---
    gravel_model = FoleyGenerator()
    # CHANGED TO V3! 
    save_file = "best_foley_generator_weights_v3.pth"

    if os.path.exists(save_file):
        print(f"✅ Found existing brain: {save_file}")
        print("Loading weights to RESUME training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gravel_model.load_state_dict(torch.load(save_file, map_location=device, weights_only=True))
    else:
        print("⚠️ No saved V3 weights found. Starting training of the NEW architecture from scratch!")

    # Let it cook!
    trained_gravel_model = train_foley_model(gravel_model, v2_dataloader, epochs=100, learning_rate=0.0001)
    return (trained_gravel_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 5
    """)
    return


@app.cell
def _(
    AudioFileClip,
    VideoFileClip,
    cv2,
    librosa,
    mo,
    np,
    os,
    sf,
    torch,
    trained_gravel_model,
):
    mo.md(r"""
    # Cell 5: The "Overfit Killer" Inference
    """)

    def generate_full_video_foley(silent_video_path, model_to_use, output_filename="final_gravel_fixed.mp4"):
        print(f"Applying final noise-reduction cheats to: {silent_video_path}")

        chunk_size, step_size, video_fps, image_size = 30, 15, 30, (112, 112)
        video = cv2.VideoCapture(silent_video_path)
        all_frames = []
        while True:
            success, frame = video.read()
            if not success: break
            all_frames.append(cv2.cvtColor(cv2.resize(frame, image_size), cv2.COLOR_BGR2RGB))
        video.release()

        model_to_use.eval()
        device = next(model_to_use.parameters()).device
        sr, hop_length = 16000, 256
        total_samples = int((len(all_frames) / video_fps) * sr) + (sr * 2)
        final_audio_waveform = np.zeros(total_samples)

        print("Generating and surgical noise-cleaning...")
        with torch.no_grad(): 
            for i in range(0, len(all_frames), step_size):
                chunk = all_frames[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0), (0, 0), (0, 0)), mode='constant')

                video_tensor = torch.FloatTensor(np.transpose(np.array(chunk), (0, 3, 1, 2))).unsqueeze(0).to(device) / 255.0
                generated_spec_db = model_to_use(video_tensor).squeeze().cpu().numpy()
            
                # --- SURGICAL CHEAT: THE NOISE FLOOR ---
                # Anything quieter than -45dB is definitely static. Kill it.
                # This turns the "gray" static into "black" silence.
                noise_floor = -45
                generated_spec_db[generated_spec_db < noise_floor] = -80
            
                # Inverse conversion with high iterations to smooth the 'crunch'
                chunk_audio = librosa.feature.inverse.mel_to_audio(
                    librosa.db_to_power(generated_spec_db), 
                    sr=sr, n_fft=1024, hop_length=hop_length, n_iter=100 # Super smooth phase
                )
            
                # Apply fade and add to timeline
                dynamic_window = np.hanning(len(chunk_audio))
                sample_offset = int((i / video_fps) * sr)
                final_audio_waveform[sample_offset : sample_offset + len(chunk_audio)] += chunk_audio * dynamic_window

        # --- FINAL CLEANUP ---
        print("Normalizing final waveform...")
        if np.max(np.abs(final_audio_waveform)) > 0:
            final_audio_waveform = final_audio_waveform / np.max(np.abs(final_audio_waveform))

        temp_audio_path = "temp_fixed_audio.wav"
        sf.write(temp_audio_path, final_audio_waveform, sr)

        video_clip = VideoFileClip(silent_video_path)
        audio_clip = AudioFileClip(temp_audio_path).subclipped(0, video_clip.duration)
        video_clip.with_audio(audio_clip).write_videofile(output_filename, codec="libx264", audio_codec="aac", logger=None)
    
        video_clip.close()
        audio_clip.close()
        os.remove(temp_audio_path)
        print(f"✅ DONE! Check: {output_filename}")

    # Run it on one of your 20 training videos
    test_video_path = 'FinalDatasetV2/dataset_split/train/video/gravel_walk_01_win_009_5steps.mp4'
    generate_full_video_foley(test_video_path, trained_gravel_model, output_filename="AI_Overfit_FINAL_CLEAN.mp4")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 6
    """)
    return


@app.cell
def _(VideoFileClip, librosa, mo, np, plt):
    mo.md(r"""
    # Cell 6: Audio Diagnostic Visualizer
    """)

    def visualize_generated_audio(video_file="AI_Overfit_Test2.mp4", output_image="audio_diagnosis2.png"):
        print(f"Extracting audio from {video_file} for diagnosis...")
    
        try:
            clip = VideoFileClip(video_file)
            sr = 16000 # CHEAT #3 Update
            audio_array = clip.audio.to_soundarray(fps=sr)
        
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0]
            
            clip.close()
        except Exception as e:
            print(f"Error reading video: {e}")
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_array, sr=sr, color='blue')
        plt.title('Waveform (Overlap-Add should have erased the boundary spikes!)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram (No more checkerboard static!)')

        plt.tight_layout()
        plt.savefig(output_image)
        plt.close('all')
    
        print(f"✅ Diagnosis complete! Open '{output_image}' to see the results.")

    visualize_generated_audio(video_file="AI_Overfit_Test2.mp4", output_image="audio_diagnosis2.png")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
