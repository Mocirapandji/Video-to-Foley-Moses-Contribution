import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import cv2
    import os
    import numpy as np

    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    import torch.nn as nn
    import torch.optim as optim

    return DataLoader, Dataset, cv2, librosa, nn, np, optim, os, plt, torch


@app.cell
def _(cv2, np, os):
    def extract_video_frames(video_path, output_folder=None):
        """
        Reads a video and extracts its frames as NumPy arrays.
        Optionally saves the frames as images to a folder.
        """
        # Open the video file
        video = cv2.VideoCapture(video_path)
    
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        frames = []
        frame_count = 0
    
        # Create output folder if you want to save the images to disk
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        while True:
            # Read one frame at a time
            success, frame = video.read()
        
            # If success is False, the video is over
            if not success:
                break
            
            # OpenCV reads images in BGR format, convert to standard RGB for ML
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
            # Save the frame as a .jpg to visually check it (optional)
            if output_folder:
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                # Convert back to BGR just for saving the image correctly
                cv2.imwrite(frame_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            frame_count += 1

        video.release()
        print(f"Extracted {frame_count} frames from {video_path}")
    
        # Return a single massive 4D NumPy array: (Frame, Height, Width, Colors)
        return np.array(frames)

    # --- Try it out! ---
    video_data = extract_video_frames('FinalDataset/processed_dataset/train/video_silent/gravel_walk_08_win_326_3steps.mp4', 'output_frames/')
    return


@app.cell
def _(librosa, np, plt):

    def audio_to_melspectrogram(audio_path, target_sr=22050):
        """
        Loads an audio file, converts it to a Mel-spectrogram, 
        and scales it to Decibels (dB) for machine learning.
        """
        # 1. Load the audio
        # target_sr ensures all audio files are forced to the exact same sample rate
        waveform, sr = librosa.load(audio_path, sr=target_sr)
    
        # 2. Create the Mel-spectrogram
        # n_fft: size of the window looking at the audio
        # hop_length: how far the window slides forward
        # n_mels: the "height" of our final image (number of frequency bands)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=sr, 
            n_fft=1024, 
            hop_length=256, 
            n_mels=80 
        )
    
        # 3. Convert power (amplitude squared) to Decibels (dB)
        # This is crucial so the model isn't overwhelmed by loud spikes
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
        return mel_spec_db, sr

    def visualize_spectrogram(mel_spec_db, sr, hop_length=256):
        """A handy helper to actually look at what the math generated."""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram (Decibels)')
        plt.tight_layout()
        plt.show()

    # --- Try it out! ---
    spectrogram, sample_rate = audio_to_melspectrogram('FinalDataset/processed_dataset/train/audio_enhanced/heel_walk_01_win_006_3steps.wav')
    print(f"Spectrogram shape (Frequencies x Time steps): {spectrogram.shape}")
    visualize_spectrogram(spectrogram, sample_rate)
    return (audio_to_melspectrogram,)


@app.cell
def _(DataLoader, Dataset, audio_to_melspectrogram, cv2, np, os, torch):


    class FoleyDataset(Dataset):
        def __init__(self, video_dir, audio_dir, max_frames=30, max_spec_time=128, image_size=(112, 112)):
            """
            Args:
                video_dir: Path to the silent videos.
                audio_dir: Path to the enhanced audio spectrograms.
                max_frames: Force all videos to this many frames (pad or trim).
                max_spec_time: Force all spectrograms to this "time" width.
                image_size: Resize video frames to (width, height) to save memory.
            """
            self.video_dir = video_dir
            self.audio_dir = audio_dir
            self.max_frames = max_frames
            self.max_spec_time = max_spec_time
            self.image_size = image_size
        
            # Get all files and sort them alphabetically so video_01 matches audio_01
            self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
            self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        
            # Quick sanity check
            assert len(self.video_files) == len(self.audio_files), "Mismatch in number of video and audio files!"

        def __len__(self):
            return len(self.video_files)

        def _process_video(self, video_path):
            """Extracts, resizes, and pads/trims video frames."""
            video = cv2.VideoCapture(video_path)
            frames = []
        
            while len(frames) < self.max_frames:
                success, frame = video.read()
                if not success:
                    break # Video ended before max_frames
            
                # Resize and convert color
                frame = cv2.resize(frame, self.image_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            video.release()
        
            # Convert to numpy
            frames = np.array(frames)
        
            # Pad with black frames if video was too short
            if len(frames) < self.max_frames:
                padding = self.max_frames - len(frames)
                # Pad format: ((before_frames, after_frames), (before_H, after_H), ...)
                frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
            
            # PyTorch expects channels first for images: (Frames, Channels, Height, Width)
            frames = np.transpose(frames, (0, 3, 1, 2))
        
            # Normalize pixel values between 0 and 1
            return torch.FloatTensor(frames) / 255.0

        def _process_audio(self, audio_path):
            """Generates, and pads/trims the Mel-spectrogram."""
            # Using the function we wrote earlier!
            mel_spec_db, _ = audio_to_melspectrogram(audio_path)
        
            # mel_spec_db shape is (Frequencies=80, TimeSteps)
            current_time_steps = mel_spec_db.shape[1]
        
            if current_time_steps < self.max_spec_time:
                # Pad with minimum dB value (-80 usually) if too short
                padding = self.max_spec_time - current_time_steps
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant', constant_values=mel_spec_db.min())
            else:
                # Trim if too long
                mel_spec_db = mel_spec_db[:, :self.max_spec_time]
            
            # Add a "channel" dimension so PyTorch treats it like a 1-channel grayscale image
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0) 
        
            return torch.FloatTensor(mel_spec_db)

        def __getitem__(self, idx):
            """This is called automatically by PyTorch to grab one pair of data."""
            vid_path = os.path.join(self.video_dir, self.video_files[idx])
            aud_path = os.path.join(self.audio_dir, self.audio_files[idx])
        
            video_tensor = self._process_video(vid_path)
            audio_tensor = self._process_audio(aud_path)
        
            return video_tensor, audio_tensor

    # --- Try it out! ---
    # Point these to your train directories
    train_video_dir = 'FinalDataset/processed_dataset/train/video_silent/'
    train_audio_dir = 'FinalDataset/processed_dataset/train/audio_enhanced/'

    # Create the dataset
    foley_dataset = FoleyDataset(train_video_dir, train_audio_dir)

    # Create the DataLoader (this shuffles the data and groups it into batches of 4)
    dataloader = DataLoader(foley_dataset, batch_size=4, shuffle=True)

    # Test grabbing exactly one batch
    for videos, spectrograms in dataloader:
        print("Successfully loaded a batch!")
        print(f"Video batch shape: {videos.shape} -> (Batch, Frames, Channels, Height, Width)")
        print(f"Audio batch shape: {spectrograms.shape} -> (Batch, Channels, Mels, TimeSteps)")
        break # We only want to test one batch
    return dataloader, spectrograms, videos


@app.cell
def _(nn, spectrograms, torch, videos):


    class FoleyGenerator(nn.Module):
        def __init__(self):
            super(FoleyGenerator, self).__init__()
        
            # --- 1. VIDEO ENCODER (3D CNN) ---
            # Compresses the video into a small, dense representation
            self.video_encoder = nn.Sequential(
                # Layer 1: Looks for basic edges and motion
                nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2), # Cuts dimensions in half
            
                # Layer 2: Looks for complex shapes (like a foot)
                nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2),
            
                # Layer 3: Looks for high-level concepts (stepping down)
                nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2), 
            )
        
            # --- 2. THE BOTTLENECK ---
            # Maps the compressed video to the starting point of the audio spectrogram
            # The math here comes from how much the 3 MaxPool layers shrunk the video
            self.fc = nn.Sequential(
                nn.Linear(128 * 3 * 14 * 14, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * 10 * 16), # Maps to a tiny 10x16 starting "image"
                nn.ReLU()
            )
        
            # --- 3. AUDIO DECODER (2D CNN Generator) ---
            # Transposed Convolutions expand the data instead of shrinking it
            self.audio_decoder = nn.Sequential(
                # Upsample 1: 10x16 -> 20x32
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            
                # Upsample 2: 20x32 -> 40x64
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            
                # Upsample 3: 40x64 -> 80x128 (Our exact target Spectrogram size!)
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
                # Notice there is NO Activation function here at the end. 
                # This is because our dB values are negative (-80 to 0).
            )

        def forward(self, x):
            """This defines the path data takes when you pass a video into the model."""
        
            # FIXING THE SHAPE: 
            # Your Dataset gives: (Batch, Frames, Channels, Height, Width) -> [4, 30, 3, 112, 112]
            # PyTorch Conv3d expects: (Batch, Channels, Frames, Height, Width) -> [4, 3, 30, 112, 112]
            # .permute() shifts the order of our dimensions to make PyTorch happy.
            x = x.permute(0, 2, 1, 3, 4)
        
            # 1. Encode the video
            x = self.video_encoder(x)
        
            # 2. Flatten and pass through bottleneck
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
        
            # 3. Reshape the 1D line back into a 2D grid for the audio decoder
            # -1 automatically fills in your Batch Size
            x = x.view(-1, 128, 10, 16) 
        
            # 4. Decode into a spectrogram
            spectrogram = self.audio_decoder(x)
        
            return spectrogram

    # --- Try it out! ---
    # Let's test the model to see if it outputs the exact same shape as your actual audio batch!

    model = FoleyGenerator()
    print("Model created!")

    # Pass the 'videos' tensor you generated in the previous step through the model
    generated_spectrogram = model(videos)

    print(f"Target audio shape:    {spectrograms.shape}")
    print(f"Generated audio shape: {generated_spectrogram.shape}")
    return (model,)


@app.cell
def _(dataloader, model, nn, optim, torch):
    def train_foley_model(model, dataloader, epochs=5, learning_rate=0.0001):
        # 1. Hardware setup: Use GPU if available (3D CNNs are slow on CPU!)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
    
        # Move the model to the GPU/CPU
        model = model.to(device)
    
        # 2. Define our "Grader" and "Adjuster"
        criterion = nn.MSELoss() # Mean Squared Error is standard for image/spectrogram regression
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        # Put the model in training mode
        model.train()
    
        print("Starting training...\n")
    
        # 3. The Outer Loop (Epochs - going through the whole dataset X times)
        for epoch in range(epochs):
            running_loss = 0.0
        
            # 4. The Inner Loop (Batches - looking at 4 videos at a time)
            for i, (videos, true_spectrograms) in enumerate(dataloader):
            
                # Move our data to the same device as the model
                videos = videos.to(device)
                true_spectrograms = true_spectrograms.to(device)
            
                # --- THE 5-STEP PYTORCH DANCE ---
            
                # Step 1: Clear out the gradients from the last batch
                optimizer.zero_grad()
            
                # Step 2: Forward Pass (Have the model guess the sound)
                generated_spectrograms = model(videos)
            
                # Step 3: Calculate the Loss (How wrong was the guess?)
                loss = criterion(generated_spectrograms, true_spectrograms)
            
                # Step 4: Backward Pass (Calculate how to fix the network)
                loss.backward()
            
                # Step 5: Optimize (Apply the fixes to the weights)
                optimizer.step()
            
                # --------------------------------
            
                running_loss += loss.item()
            
                # Print an update every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
            # Print average loss for the whole epoch
            avg_epoch_loss = running_loss / len(dataloader)
            print(f"--- End of Epoch {epoch+1} | Average Loss: {avg_epoch_loss:.4f} ---\n")

        print("Training done")
    

        torch.save(model.state_dict(), "foley_generator_weights.pth")
        print("Saved model weights to 'foley_generator_weights.pth'")
    
        return model


    trained_model = train_foley_model(model, dataloader, epochs=250)
    return (trained_model,)


@app.cell
def _(cv2, librosa, np, os, torch, trained_model):

    from moviepy import VideoFileClip, AudioFileClip

    def generate_and_sync_foley(silent_video_path, trained_model, output_filename="final_synced_video.mp4"):
        print(f"Processing video: {silent_video_path}")
    
        # --- 1. PREP THE VIDEO (Same as Training) ---
        max_frames = 30
        image_size = (112, 112)
    
        video = cv2.VideoCapture(silent_video_path)
        frames = []
    
        while len(frames) < max_frames:
            success, frame = video.read()
            if not success:
                break
            frame = cv2.resize(frame, image_size)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        video.release()
    
        frames = np.array(frames)
        if len(frames) < max_frames:
            padding = max_frames - len(frames)
            frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
        
        frames = np.transpose(frames, (0, 3, 1, 2))
        video_tensor = torch.FloatTensor(frames) / 255.0
    
        # FIX: Only unsqueeze to add the Batch=1 dimension.
        # The model's forward() function will handle the .permute()!
        video_tensor = video_tensor.unsqueeze(0)
    
        # --- 2. GENERATE THE SPECTROGRAM ---
        trained_model.eval() 
        with torch.no_grad(): 
            device = next(trained_model.parameters()).device
            video_tensor = video_tensor.to(device)
            generated_spec_tensor = trained_model(video_tensor)
        
        generated_spec_db = generated_spec_tensor.squeeze().cpu().numpy()
    
        # --- 3. CONVERT IMAGE BACK TO AUDIO (Griffin-Lim) ---
        print("Converting generated spectrogram back to audio...")
    
        generated_spec_power = librosa.db_to_power(generated_spec_db)
    
        sr = 22050
        hop_length = 256
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(
            generated_spec_power, 
            sr=sr, 
            n_fft=1024, 
            hop_length=hop_length, 
            n_iter=32 
        )
    
        temp_audio_path = "temp_generated_audio.wav"
        import soundfile as sf  # <--- Add this line right here!
        sf.write(temp_audio_path, reconstructed_audio, sr)
    
        # --- 4. STITCH IT TOGETHER ---
        print("Stitching generated audio to the original video...")
    
        video_clip = VideoFileClip(silent_video_path)
        audio_clip = AudioFileClip(temp_audio_path)
    
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        final_clip = video_clip.with_audio(audio_clip) 
    
        final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", logger=None)
    
        video_clip.close()
        audio_clip.close()
        os.remove(temp_audio_path)
    
        print(f"\n✅ Success! Check your folder for: {output_filename}")

    # --- Try it out! ---
    test_video_path = 'FinalDataset/processed_dataset/test/video_silent/gravel_walk_05_win_938_12steps.mp4'

    # Assuming your trained model is still loaded in memory as 'trained_model'
    generate_and_sync_foley(test_video_path, trained_model, output_filename="AI_Foley_Result3.mp4")
    return


@app.cell
def _():
    # def clean_mismatched_pairs(video_dir, audio_dir, dry_run=False):
    #     """
    #     Finds and deletes files in video_dir and audio_dir that do not have a matching pair.
    #     """
    #     # 1. Get all files
    #     video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    #     audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    #     # 2. Extract just the base names (e.g., 'barefoot_walk_01' from 'barefoot_walk_01.mp4')
    #     # We use dictionaries to map the base name back to the actual filename
    #     video_dict = {os.path.splitext(f)[0]: f for f in video_files}
    #     audio_dict = {os.path.splitext(f)[0]: f for f in audio_files}
    
    #     # 3. Find the differences using Python Sets
    #     video_bases = set(video_dict.keys())
    #     audio_bases = set(audio_dict.keys())
    
    #     orphaned_videos = video_bases - audio_bases
    #     orphaned_audios = audio_bases - video_bases
    
    #     print(f"Found {len(orphaned_videos)} videos without matching audio.")
    #     print(f"Found {len(orphaned_audios)} audio clips without matching video.\n")
    
    #     # 4. Delete orphaned videos
    #     for base_name in orphaned_videos:
    #         file_to_delete = os.path.join(video_dir, video_dict[base_name])
    #         if dry_run:
    #             print(f"[DRY RUN] Would delete video: {file_to_delete}")
    #         else:
    #             os.remove(file_to_delete)
    #             print(f"Deleted video: {file_to_delete}")
            
    #     # 5. Delete orphaned audios
    #     for base_name in orphaned_audios:
    #         file_to_delete = os.path.join(audio_dir, audio_dict[base_name])
    #         if dry_run:
    #             print(f"[DRY RUN] Would delete audio: {file_to_delete}")
    #         else:
    #             os.remove(file_to_delete)
    #             print(f"Deleted audio: {file_to_delete}")

    #     if dry_run:
    #         print("\n⚠️ This was a DRY RUN. No files were actually deleted.")
    #         print("Change `dry_run=False` in the function call to delete the files permanently.")
    #     else:
    #         print("\n✅ Cleanup complete! Both folders should now match perfectly.")

    # # --- Try it out! ---
    # # Renamed variables to avoid Marimo notebook cell conflicts!
    # cleanup_video_dir = 'FinalDataset/processed_dataset/train/video_silent/'
    # cleanup_audio_dir = 'FinalDataset/processed_dataset/train/audio_enhanced/'

    # # Run it first with dry_run=True. 
    # # If the printed list looks correct, change it to dry_run=False and run it again!
    # clean_mismatched_pairs(cleanup_video_dir, cleanup_audio_dir, dry_run=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
