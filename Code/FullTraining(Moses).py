import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import cv2
    import librosa
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    import torch
    import torchaudio
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torchvision.models as models
    import torch.optim as optim
    import time
    import io

    return (
        DataLoader,
        Dataset,
        Path,
        cv2,
        io,
        librosa,
        models,
        nn,
        np,
        optim,
        os,
        pd,
        plt,
        time,
        torch,
        torchaudio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 1
    """)
    return


@app.cell
def _(Path):
    # .parent moves up to foley-to-sound
    repo_root = Path.cwd()

    dataset_path = repo_root/"Dataset" # if you have different naming convention, change this part
    train_path = dataset_path/"Dataset_train"
    test_path = dataset_path/"Dataset_test"

    print(f"Dataset location: {dataset_path}")
    return repo_root, train_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 2
    """)
    return


@app.cell
def _(os, torchaudio, train_df):
    def check_audio_info(audio_path):
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return

        try:
            metadata = torchaudio.info(audio_path)
            sr = metadata.sample_rate
            frames = metadata.num_frames
            channels = metadata.num_channels
        except (AttributeError, RuntimeError):
            # manually check the shape
            waveform, sr = torchaudio.load(audio_path)
            channels, frames = waveform.shape

        duration = frames / sr

        print(f"Audio name: {os.path.basename(audio_path)}")
        print(f"Sample Rate: {sr} Hz")
        print(f"Channels: {channels}")
        print(f"Total Samples: {frames}")
        print(f"Duration: {duration:.2f} seconds")

    check_audio_info(train_df['audio_path'].iloc[144])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 3
    """)
    return


@app.cell
def _(cv2, train_df):
    def check_video_info(video_path):
        cap = cv2.VideoCapture(video_path)

        # Get properties
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # Float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Float
        fps    = cap.get(cv2.CAP_PROP_FPS)           # Frames per second
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

        duration = total_frames / fps if fps > 0 else 0

        print(f"Video name: {video_path.split('/')[-1]} ---")
        print(f"Resolution: {int(width)}x{int(height)}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {int(total_frames)}")
        print(f"Duration: {duration:.2f} seconds")

        cap.release()

    check_video_info(train_df['video_path'].iloc[11])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 4
    """)
    return


@app.cell
def _(os, pd, repo_root):
    def get_fixed_dataset(split_name):
        log_path = repo_root/"Dataset"/split_name/"logs"/"process_log.csv"

        # Load the CSV
        df = pd.read_csv(log_path)

        # Filter out rejected files to ensure high quality
        df = df[df['rejected'] == 'no'].copy()

        df['youtube_id'] = df['base_id']
        df['seg_start'] = 0.0
        df['seg_end'] = 3.0
        df['source'] = 'local'

        # Define the data folders
        video_dir = repo_root/"Dataset"/split_name/"video_silent"
        audio_dir = repo_root/"Dataset"/split_name/"audio_enhanced"

        # Reconstruct the paths using the base_id
        df['video_path'] = df['base_id'].apply(lambda x: str(video_dir/f"{x}.mp4"))
        df['audio_path'] = df['base_id'].apply(lambda x: str(audio_dir/f"{x}.wav"))

        # --- THE NEW FIX ---
        # Only keep the row if BOTH the video and audio file actually exist!
        df = df[df['video_path'].apply(os.path.exists) & df['audio_path'].apply(os.path.exists)]

        return df

    train_df = get_fixed_dataset("Dataset_train")
    # If want to do the model on narrowed dataset
    # train_df = train_df_all[train_df_all['base_id'].str.contains('gravel_walk', case=False, na=False)]
    test_df = get_fixed_dataset("Dataset_test")

    print(f"Train samples found: {len(train_df)}")
    print(f"Test samples found: {len(test_df)}")
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Process Videos and Audios
    ## Cell 5
    """)
    return


@app.cell
def _(cv2, np, torch):
    def process_video(mp4_path, target_frames=90, resize=(112, 112)):
        # mp4 to tensor
        cap = cv2.VideoCapture(mp4_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to standard CNN input size
            frame = cv2.resize(frame, resize)
            # Normalize pixel values between 0 and 1
            frame = frame/255.0
            frames.append(frame)

            # stop when reach 90, that's the limited frame size
            if len(frames) == target_frames:
                break

        cap.release()

        if len(frames) == 0:
                print(f"Warning: Could not read any frames from {mp4_path}. Returning zeros.")
                # Return a tensor of zeros, and the training loop continues
                return torch.zeros((target_frames, 3, resize[0], resize[1]), dtype=torch.float32)

        # padding when too short, not enough for 3 sec
        while len(frames) < target_frames:
            frames.append(frames[-1])
        cap.release()

        # list - numpy array - tensor
        frames_np = np.array(frames) 

        # PyTorch expects shape: (Time, Channels, Height, Width)
        # Current shape is (Time, Height, Width, Channels)
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2)

        return frames_tensor

    return (process_video,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 6
    """)
    return


@app.cell
def _(librosa, np, torch):
    def process_audio(wav_path, target_frames=90, sr=22050, n_mels=80):
        # wav to tensor
        y, _ = librosa.load(wav_path, sr=sr) # sample rate, in Hz

        # --- THE GRAVEL CRUNCH FIX (Pre-emphasis) ---
        # This boosts sharp high frequencies (the "crunch") and kills 
        # the low-frequency "engine rumble" before the AI even sees it.
        y = librosa.effects.preemphasis(y)

        #hop_length to sync with video frames
        hop_length = int(sr/30) # fps=30

        # Mel-spectrogram - pitch, volume, timing
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            hop_length=hop_length, 
            n_mels=n_mels,
            n_fft=2048,
            fmax=sr/2 # Force it to look at the absolute highest, crispiest frequencies
        )

        # to Log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = log_mel_spec[:, :target_frames]

        # --- THE CLEAN TEACHER FIX ---
        # Force absolute silence between real footsteps in the training data.
        # If a sound is quieter than -35dB, flatten it to -80dB (pure silence).
        log_mel_spec[log_mel_spec < -35] = -80

        # in case the audio is too short
        if log_mel_spec.shape[1] < target_frames:
            pad_width = target_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='edge')

        log_mel_spec = (log_mel_spec + 80) / 80

        # Returns shape: (n_mels, Time)
        return torch.tensor(log_mel_spec, dtype=torch.float32)

    return (process_audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load the Dataset
    ## Cell 7
    """)
    return


@app.cell
def _(Dataset, process_audio, process_video, test_df, train_df):
    class LocalFoleyDataset(Dataset):
        def __init__(self, df, clip_seconds=3.0):
            self.df = df
            self.clip_seconds = clip_seconds

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # Video - Tensor
            video_tensor = process_video(row['video_path'], target_frames=90)
            # Audio - Mel-Spectrogram Tensor
            audio_tensor = process_audio(row['audio_path'], target_frames=90)

            return video_tensor, audio_tensor, row['base_id']

    train_ds = LocalFoleyDataset(train_df, clip_seconds=3.0)
    test_ds = LocalFoleyDataset(test_df, clip_seconds=3.0)
    return test_ds, train_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 8
    """)
    return


@app.cell
def _(DataLoader, test_ds, train_ds):
    # DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2 # Uses CPU to load videos while GPU trains
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2
    )

    print(f"Number of training batches: {len(train_loader)}")
    return (train_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 9
    """)
    return


@app.cell
def _(train_ds):
    video_tensor, audio_tensor, meta = train_ds[1]

    print(f"Video Shape: {video_tensor.shape}") 
    print(f"Audio Shape: {audio_tensor.shape}") 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 10
    """)
    return


@app.cell
def _(process_audio, process_video, train_df):
    process_video_tensor = process_video(train_df['video_path'].iloc[0])
    process_audio_tensor = process_audio(train_df['audio_path'].iloc[0])

    print(f"Video frames: {process_video_tensor.shape[0]}")
    print(f"Audio columns: {process_audio_tensor.shape[1]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Extractor - ResNet
    ## Cell 11
    """)
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(device, models, nn):
    class VideoFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            # pre-trained ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # replace the fully connected layer with an Identity layer
            # the network stops one step early to get output of the raw feature vector
            resnet.fc = nn.Identity()

            self.extractor = resnet

            # Freeze - prevent training
            for param in self.extractor.parameters():
                param.requires_grad = False

        def forward(self, video_tensor):
            features = self.extractor(video_tensor)

            # Output shape: (Time, Feature_Size)
            return features

    visual_encoder = VideoFeatureExtractor().to(device)
    return (visual_encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 12
    """)
    return


@app.cell
def _(process_audio, process_video, train_df):
    v_tensor = process_video(train_df['video_path'].iloc[0])
    a_tensor = process_audio(train_df['audio_path'].iloc[0])

    print(f"Video Tensor Shape: {v_tensor.shape}") 
    # (Frames, Channels, Height, Width)

    print(f"Audio Tensor Shape: {a_tensor.shape}") 
    # (Mels, Time_Steps)
    return (v_tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visual Encoder - LSTM
    ## Cell 13
    """)
    return


@app.cell
def _(device, nn):
    ## feature to mel-spectrogram
    class FoleyGenModel(nn.Module):
        def __init__(self, visual_dim=2048, hidden_dim=256, audio_dim=80):
            super(FoleyGenModel, self).__init__()

            self.lstm = nn.LSTM(
                input_size=visual_dim, 
                hidden_size=hidden_dim, 
                num_layers=2, 
                batch_first=True,
                bidirectional=True # Looks forward and backward in time
            )

            # output layer - hidden_dim * 2 - LSTM is bidirectional
            self.fc = nn.Linear(hidden_dim * 2, audio_dim)
            self.activation = nn.Sigmoid() # Maybe ReLU?

        def forward(self, x):
            # x(Batch, 90, 2048)
            # (Batch, 90, hidden_dim * 2)
            lstm_out, _ = self.lstm(x)
            # (Batch, 90, 128)
            audio_out = self.fc(lstm_out)
            audio_out = self.activation(audio_out)

            return audio_out.permute(0, 2, 1)

    model = FoleyGenModel().to(device)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    # Cell 14
    """)
    return


@app.cell
def _(model, nn, optim):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return criterion, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 15
    """)
    return


@app.cell
def _(device, model, torch, v_tensor, visual_encoder):
    # fake batch of 1
    v_input = v_tensor.unsqueeze(0).to(device) # Shape: [1, 90, 3, 224, 224]

    with torch.no_grad():
        # Get visual features
        b, t, c, h, w = v_input.shape
        v_reshaped = v_input.view(b * t, c, h, w)
        feats = visual_encoder(v_reshaped).view(b, t, -1)

        # Predict Audio
        preds = model(feats)

    print(f"Final Prediction Shape: {preds.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    ## Cell 16
    """)
    return


@app.cell
def _(torch):
    def train_one_epoch(model, visual_encoder, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        batch_losses = []

        for batch_idx, (video, audio, _) in enumerate(dataloader):

            video = video.to(device)
            audio = audio.to(device)

            # Extract Features
            with torch.no_grad():
                b, t, c, h, w = video.shape
                v_flat = video.view(b * t, c, h, w)
                v_feats = visual_encoder(v_flat)
                v_feats = v_feats.view(b, t, -1) # [Batch, 90, 2048]

            # Forward Pass
            optimizer.zero_grad()
            output = model(v_feats)

            # Loss, Backprop
            loss = criterion(output, audio)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            batch_losses.append(current_loss)
            total_loss += current_loss

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        return avg_loss, batch_losses

        return (train_one_epoch,)

    return (train_one_epoch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 17
    """)
    return


@app.cell
def _(train_loader):
    # Grab one batch from your loader
    vi, au, _ = next(iter(train_loader))
    print(f"Audio Max: {au.max().item():.2f}")
    print(f"Audio Min: {au.min().item():.2f}")
    print(f"Audio Mean: {au.mean().item():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 18
    """)
    return


@app.cell
def _(
    criterion,
    device,
    model,
    optimizer,
    time,
    torch,
    train_loader,
    train_one_epoch,
    visual_encoder,
):
    all_epoch_losses = []
    all_batch_losses = []
    num_epochs = 200 # I usually use 3, but did 15 when overfitting for the small specific dataset
    best_loss = float('inf')

    print(f"Start Training on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        avg_loss, epoch_batch_losses = train_one_epoch(model, visual_encoder, train_loader, optimizer, criterion, device)

        # Record
        all_epoch_losses.append(avg_loss)
        all_batch_losses.extend(epoch_batch_losses)

        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] finished in {duration:.1f}s - Avg Loss: {avg_loss:.4f}")

        # Save the best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_foley_model_mosesv2.pth") # pls change the name of the saved model so that each of us won't overwrite others' model 
            print("Model saved")
    return all_batch_losses, all_epoch_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 19
    """)
    return


@app.cell
def _(all_batch_losses, all_epoch_losses, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(all_batch_losses, label='Batch Loss')

    ax1.set_title('Batch Loss')
    ax1.set_xlabel('Total Batches Processed')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", alpha=0.3, ls="-")
    ax1.legend()

    epochs = range(1, len(all_epoch_losses) + 1)
    ax2.plot(epochs, all_epoch_losses, label='Epoch Average')

    ax2.set_title('Average Epoch Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('$MSE$ Loss')
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generation
    ## Cell 20
    """)
    return


@app.cell
def _(device, librosa, model, torch, visual_encoder):
    def generate_foley(video_tensor):
        model.eval()
        visual_encoder.eval()

        sr = 22050

        v_input = video_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            # Extract Features
            b, t, c, h, w = v_input.shape
            v_flat = v_input.view(b * t, c, h, w)
            v_feats = visual_encoder(v_flat).view(b, t, -1)
            # Predict Spectrogram
            pred_spec = model(v_feats).squeeze(0).cpu().numpy()

        # Un-normalize (Reverse /80)
        pred_spec_db = (pred_spec * 80) - 80

        # 1. MORE AGGRESSIVE NOISE GATE
        # Raised from -40 to -30. Anything that isn't a distinct hit gets silenced.
        pred_spec_db[pred_spec_db < -30] = -80 

        pred_power = librosa.db_to_power(pred_spec_db)

        # 2. THE SPIKE EXPANDER
        # Mathematically stretch the dynamic range. 
        # Quiet sounds get squashed, loud spikes get exponentially louder!
        pred_power = pred_power ** 1.5 

        # 3. BETTER AUDIO RECONSTRUCTION
        # Added n_iter=60. Forces the algorithm to spend twice as much time 
        # making the gravel sound crunchy instead of bubbly.
        audio_out = librosa.feature.inverse.mel_to_audio(
            pred_power, sr=22050, n_fft=2048, hop_length=int(sr/30), n_iter=60
        )
        return audio_out

    return (generate_foley,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preview Generated Audio
    pls note this section only for preview but not for downloading.
    video preview is not working btw

    ## Cell 21
    """)
    return


@app.cell
def _(generate_foley, io, mo, model, np, test_ds, torch):
    from scipy.io import wavfile
    def to_widgets(audio_array):
        # Normalize and convert to 16-bit PCM
        audio_flat = audio_array.flatten()
        max_val = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1
        audio_int = (audio_array / max_val * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        wavfile.write(byte_io, 22050, audio_int)
        return mo.audio(byte_io.getvalue())

    # Load the trained brain
    model.load_state_dict(torch.load("best_foley_model_mosesv2.pth", weights_only=True)) # change this to the model name you saved
    model.eval()

    # Safely grab the first, a middle-ish, and the very last available video
    max_idx = len(test_ds) - 1
    test_indices = [0, min(689, max_idx // 2), max_idx]

    rows = []

    for idx in test_indices:
        v, a, vid_id = test_ds[idx]
        gen_a = generate_foley(v)

        video_path = f"data/videos/{vid_id}.mp4"

        rows.append(mo.md(f"---"))
        rows.append(mo.md(f"### Video: {vid_id}"))

        item = mo.hstack([
            # Left side: The Video
            mo.vstack([
                mo.md("**Reference Video**"),
                mo.video(video_path, width=300)
            ]),
            # Right side: The Audio comparison
            mo.vstack([
                mo.md("**AI Prediction**"), 
                to_widgets(gen_a),
                mo.md("**Real Sound**"), 
                to_widgets(a.numpy())
            ])
        ], gap=2)

        rows.append(item)

    mo.vstack(rows)
    return (wavfile,)


@app.cell
def _():
    # from scipy.io import wavfile
    # def to_widgets(audio_array):
    #     # Normalize and convert to 16-bit PCM
    #     audio_flat = audio_array.flatten()
    #     max_val = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1
    #     audio_int = (audio_array / max_val * 32767).astype(np.int16)

    #     byte_io = io.BytesIO()
    #     wavfile.write(byte_io, 22050, audio_int)
    #     return mo.audio(byte_io.getvalue())

    # # Load the trained brain
    # model.load_state_dict(torch.load("best_foley_model_mosesv2.pth", weights_only=True))

    # model.eval()
    # visual_encoder.eval()

    # test_indices = [711,689] # select any numbers of videos from the test folder. With index range [0,713]

    # rows = []

    # for idx in test_indices:
    #     v, a, vid_id = test_ds[idx]
    #     gen_a = generate_foley(v)

    #     video_path = str(repo_root / "Dataset" / "Dataset_test" / "video_silent" / f"{vid_id}.mp4")
    #     audio_real_path = str(repo_root / "Dataset" / "Dataset_test" / "audio_enhanced" / f"{vid_id}.wav")

    #     if os.path.exists(audio_real_path):
    #         real_audio_np, _ = librosa.load(audio_real_path, sr=22050)
    #     else:
    #         real_audio_np = np.zeros(22050)
    #         print(f"⚠️ Warning: Missing audio file at {audio_real_path}")

    #     rows.append(mo.md(f"---"))
    #     rows.append(mo.md(f"### Video: {vid_id}"))

    #     item = mo.hstack([
    #         # Left side: The Video
    #         mo.vstack([
    #             mo.md("**Reference Video**"),
    #             mo.video(video_path, width=300)
    #         ], align="center"),
    #         # Right side: The Audio comparison
    #         mo.vstack([
    #             mo.md("**AI Prediction**"), 
    #             to_widgets(gen_a),
    #             mo.md("**Real Sound**"), 
    #             to_widgets(real_audio_np)
    #         ])
    #     ], gap=2, justify="center")

    #     rows.append(item)

    # mo.vstack(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download the Outcome
    Syncing Audio with the Video

    ## Cell 21
    """)
    return


@app.cell
def _():
    # Modern MoviePy 2.0 syntax
    import moviepy as mp
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    import tempfile

    return AudioFileClip, VideoFileClip, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 22
    """)
    return


@app.cell
def _(AudioFileClip, VideoFileClip, mo, np, os, repo_root, tempfile, wavfile):
    def export_foley_video(vid_id, gen_audio_array, split_name="Dataset_test"):
        # Create folder
        os.makedirs("exports", exist_ok=True)
        output_name = f"exports/foley_{vid_id}.mp4"

        # Path to the silent video
        video_path = str(repo_root / "Dataset" / split_name / "video_silent" / f"{vid_id}.mp4") # Change the naming if needed

        if not os.path.exists(video_path):
            return mo.md(f"X **Error:** Video not found at {video_path}")

        # status
        with mo.status.spinner(title="Merging AI Audio with Video..."):

            # Create temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_flat = gen_audio_array.flatten()
                max_val = np.max(np.abs(audio_flat)) if np.max(np.abs(audio_flat)) > 0 else 1
                audio_int = (audio_flat / max_val * 32767).astype(np.int16)
                wavfile.write(tmp_audio.name, 22050, audio_int)
                tmp_audio_path = tmp_audio.name

            try:
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(tmp_audio_path)
                final_clip = video_clip.with_audio(audio_clip)

                # Write file, takes time
                final_clip.write_videofile(output_name, codec="libx264", audio_codec="aac")

                # Cleanup clips
                audio_clip.close()
                video_clip.close()
            finally:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

        # RETURN a success message and the video player
        return mo.vstack([
            mo.md(f"√ **Export Successful!** Saved to: `{output_name}`"),
            mo.video(output_name, width=400),
            mo.download(data=open(output_name, "rb"), filename=f"foley_{vid_id}.mp4", label="Download Video")
        ])

    return (export_foley_video,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 23
    """)
    return


@app.cell
def _(export_foley_video, generate_foley, test_ds):
    import random

    # Calculate the maximum safe index
    _max_idx = len(test_ds) - 1

    # Pick a completely random number between 0 and the max index
    random_idx = random.randint(0, _max_idx)

    # Grab that random video from the dataset
    v6, a6, vid_id6 = test_ds[random_idx] 
    gen_a6 = generate_foley(v6)

    # Calling the function (the output will appear below this cell)
    export_foley_video(vid_id6, gen_a6)
    return (random,)


@app.cell
def _():
    # # Pick a sample from your filtered gravel walk data
    # v6, a6, vid_id6 = test_ds[711] # With the same index range [0,713]
    # gen_a6 = generate_foley(v6)

    # # Calling the function (the output will appear below this cell)
    # export_foley_video(vid_id6, gen_a6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gravel only
    """)
    return


@app.cell
def _(
    generate_foley,
    io,
    mo,
    model,
    np,
    random,
    repo_root,
    test_ds,
    torch,
    wavfile,
):


    def _to_widgets(audio_array):
        # Normalize and convert to 16-bit PCM
        audio_flat = audio_array.flatten()
        max_val = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1
        audio_int = (audio_array / max_val * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        wavfile.write(byte_io, 22050, audio_int)
        return mo.audio(byte_io.getvalue())

    # Load the trained brain
    model.load_state_dict(torch.load("best_foley_model_mosesv2.pth", weights_only=True)) 
    model.eval()

    # --- GRAVEL ONLY FINDER ---
    _preview_gravel_indices = [i for i in range(len(test_ds)) if 'gravel' in test_ds.df.iloc[i]['base_id'].lower()]

    _rows = []

    if not _preview_gravel_indices:
        _rows.append(mo.md("❌ **Error:** No gravel videos found in the test dataset!"))
    else:
        # Safely grab up to 3 completely random gravel videos
        _num_samples = min(3, len(_preview_gravel_indices))
        _preview_test_indices = random.sample(_preview_gravel_indices, _num_samples)

        for _idx in _preview_test_indices:
            _v, _a, _vid_id = test_ds[_idx]
            _gen_a = generate_foley(_v)

            _video_path = str(repo_root / "Dataset" / "Dataset_test" / "video_silent" / f"{_vid_id}.mp4")

            _rows.append(mo.md(f"---"))
            _rows.append(mo.md(f"### Video: {_vid_id}"))

            _item = mo.hstack([
                # Left side: The Video
                mo.vstack([
                    mo.md("**Reference Video**"),
                    mo.video(_video_path, width=300)
                ]),
                # Right side: The Audio comparison
                mo.vstack([
                    mo.md("**AI Prediction**"), 
                    _to_widgets(_gen_a),
                    mo.md("**Real Sound**"), 
                    _to_widgets(_a.numpy())
                ])
            ], gap=2)

            _rows.append(_item)

    # Marimo will display whatever is on the last line of the cell
    mo.vstack(_rows)
    return


@app.cell
def _(export_foley_video, generate_foley, mo, random, test_ds):


    # --- GRAVEL ONLY FINDER ---
    _export_gravel_indices = [i for i in range(len(test_ds)) if 'gravel' in test_ds.df.iloc[i]['base_id'].lower()]

    if not _export_gravel_indices:
        _export_ui = mo.md("❌ **Error:** No gravel videos found to export!")
    else:
        # Pick ONE completely random gravel video from the list
        _random_idx = random.choice(_export_gravel_indices)

        # Grab that specific gravel video from the dataset
        _v_exp, _a_exp, _vid_id_exp = test_ds[_random_idx] 
        _gen_a_exp = generate_foley(_v_exp)

        # Save the output UI to a private variable
        _export_ui = export_foley_video(_vid_id_exp, _gen_a_exp)

    # Drop the variable on the last line so Marimo renders the video player!
    _export_ui
    return


@app.cell
def _(mo, test_df, train_df):
    # Count the gravel files in the training set
    _train_gravel = train_df[train_df['base_id'].str.contains('gravel', case=False, na=False)]

    # Count the gravel files in the testing set
    _test_gravel = test_df[test_df['base_id'].str.contains('gravel', case=False, na=False)]

    # Display the results nicely in Marimo!
    mo.md(f"""
    ### 🪨 Gravel Dataset Count:
    * **Training samples:** {len(_train_gravel)}
    * **Testing samples:** {len(_test_gravel)}
    * **Total Gravel Videos:** {len(_train_gravel) + len(_test_gravel)}
    """)
    return


@app.cell
def _(
    AudioFileClip,
    VideoFileClip,
    generate_foley,
    mo,
    model,
    np,
    os,
    process_video,
    tempfile,
    torch,
    wavfile,
):


    # 1. Put the exact name of your custom video here!
    _custom_video_path = "test_moses_video2.mov" 

    # Load the trained brain just to be safe
    model.load_state_dict(torch.load("best_foley_model_moses.pthv2", weights_only=True)) 
    model.eval()

    if not os.path.exists(_custom_video_path):
        _custom_ui = mo.md(f"❌ **Error:** Could not find `{_custom_video_path}`. Make sure it is in your foley-to-sound folder!")
    else:
        # 2. Process the wild video into the exact mathematical shape the AI expects
        # (This uses your existing process_video function)
        _v_custom = process_video(_custom_video_path)

        # 3. Ask the AI to predict the audio!
        _gen_a_custom = generate_foley(_v_custom)

        # 4. Custom Export Logic for standalone files
        os.makedirs("exports", exist_ok=True)
        _output_name = "exports/foley_CUSTOM_TEST.mp4"

        with mo.status.spinner(title="Merging AI Audio with Custom Video..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp_audio:
                _audio_flat = _gen_a_custom.flatten()
                _max_val = np.max(np.abs(_audio_flat)) if np.max(np.abs(_audio_flat)) > 0 else 1
                _audio_int = (_audio_flat / _max_val * 32767).astype(np.int16)
                wavfile.write(_tmp_audio.name, 22050, _audio_int)
                _tmp_audio_path = _tmp_audio.name

            try:
                _video_clip = VideoFileClip(_custom_video_path)
                _audio_clip = AudioFileClip(_tmp_audio_path)

                # Since the AI generates exactly 3 seconds of audio, trim the video to match
                if _video_clip.duration > 3.0:
                    _video_clip = _video_clip.subclipped(0, 3.0)

                _final_clip = _video_clip.with_audio(_audio_clip)

                # Write file (takes time)
                _final_clip.write_videofile(_output_name, codec="libx264", audio_codec="aac")

                _audio_clip.close()
                _video_clip.close()
            finally:
                if os.path.exists(_tmp_audio_path):
                    os.remove(_tmp_audio_path)

        # 5. Build the UI
        _custom_ui = mo.vstack([
            mo.md(f"### 🎬 Wild Custom Video Test: `{_custom_video_path}`"),
            mo.md(f"√ **Export Successful!** Saved to: `{_output_name}`"),
            mo.hstack([
                mo.vstack([
                    mo.md("**Silent Original**"),
                    mo.video(_custom_video_path, width=300)
                ]),
                mo.vstack([
                    mo.md("**AI Generated Foley**"),
                    mo.video(_output_name, width=300),
                    mo.download(data=open(_output_name, "rb"), filename="foley_moses_custom.mp4", label="Download Video")
                ])
            ], gap=4)
        ])

    # Drop the UI variable so Marimo renders it!
    _custom_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Output other videos
    """)
    return


@app.cell
def _(
    AudioFileClip,
    VideoFileClip,
    generate_foley,
    mo,
    model,
    np,
    os,
    random,
    repo_root,
    tempfile,
    torch,
    train_ds,
    train_path,
    wavfile,
):


    # 1. Setup the new output folder
    _v2_output_dir = repo_root / "v2output"
    _v2_output_dir.mkdir(exist_ok=True)

    # 2. Load your new v2 brain
    model.load_state_dict(torch.load("best_foley_model_mosesv2.pth", weights_only=True))
    model.eval()

    # 3. Define the exact materials you want to test
    _target_materials = ['barefoot', 'gravel', 'thinice', 'snow', 'heel']
    _v2_ui_rows = [mo.md("# 🎬 V2 Model Multi-Material Test (`Dataset_train`)")]

    for _material in _target_materials:
        # Find all indices in the TRAINING dataset that match the material
        _mat_indices = [i for i in range(len(train_ds)) if _material in train_ds.df.iloc[i]['base_id'].lower()]

        if not _mat_indices:
            _v2_ui_rows.append(mo.md(f"❌ **{_material.capitalize()}**: No videos found in the training dataset!"))
            continue

        # Pick 1 random video for this specific material
        _random_idx = random.choice(_mat_indices)
        _v, _a, _vid_id = train_ds[_random_idx]

        # Generate the Foley audio!
        _gen_a = generate_foley(_v)

        # --- Custom Export Logic for the v2output folder ---
        # Point directly to the Dataset_train folder
        _silent_video_path = str(train_path / "video_silent" / f"{_vid_id}.mp4")
        _output_name = str(_v2_output_dir / f"foley_{_material}_{_vid_id}.mp4")

        if not os.path.exists(_silent_video_path):
            _v2_ui_rows.append(mo.md(f"❌ **{_material.capitalize()}**: Silent video not found at `{_silent_video_path}`"))
            continue

        # Show a loading spinner so you know which material is processing
        with mo.status.spinner(title=f"Merging {_material.capitalize()} video..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp_audio:
                _audio_flat = _gen_a.flatten()
                _max_val = np.max(np.abs(_audio_flat)) if np.max(np.abs(_audio_flat)) > 0 else 1
                _audio_int = (_audio_flat / _max_val * 32767).astype(np.int16)
                wavfile.write(_tmp_audio.name, 22050, _audio_int)
                _tmp_audio_path = _tmp_audio.name

            try:
                _video_clip = VideoFileClip(_silent_video_path)
                _audio_clip = AudioFileClip(_tmp_audio_path)

                # Sync duration (forces the video to match the 3-second audio perfectly)
                if _video_clip.duration > 3.0:
                    _video_clip = _video_clip.subclipped(0, 3.0)

                _final_clip = _video_clip.with_audio(_audio_clip)

                # Write file without spamming the console logs
                _final_clip.write_videofile(_output_name, codec="libx264", audio_codec="aac", logger=None)

                _audio_clip.close()
                _video_clip.close()
            finally:
                if os.path.exists(_tmp_audio_path):
                    os.remove(_tmp_audio_path)

        # Build the visual UI block for this specific material
        _v2_ui_rows.append(mo.md(f"---"))
        _v2_ui_rows.append(mo.md(f"### 🥾 Material: **{_material.capitalize()}** | Video ID: `{_vid_id}`"))
        _v2_ui_rows.append(mo.md(f"Saved to: `{_output_name}`"))
        _v2_ui_rows.append(mo.video(_output_name, width=400))

    # Display the final stack of videos!
    _v2_output_ui = mo.vstack(_v2_ui_rows)
    _v2_output_ui
    return


@app.cell
def _(
    AudioFileClip,
    VideoFileClip,
    generate_foley,
    mo,
    model,
    np,
    os,
    process_video,
    repo_root,
    tempfile,
    torch,
    wavfile,
):


    # 1. Setup the input and output folders
    _batch_input_dir = repo_root / "TestVideos" / "Video_Silent" 
    _batch_output_dir = repo_root / "TestVideos" / "V2_Test_Results"
    _batch_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load your new v2 brain
    model.load_state_dict(torch.load("best_foley_model_mosesv2.pth", weights_only=True))
    model.eval()

    _batch_ui_rows = [mo.md(f"# 🎬 V2 Model Smart Batch Test (`{_batch_input_dir.name}`)")]

    # Find all mp4 and mov files in the folder!
    _test_videos = list(_batch_input_dir.glob("*.mp4")) + list(_batch_input_dir.glob("*.mov")) + list(_batch_input_dir.glob("*.MOV"))

    if not _test_videos:
        _batch_ui_rows.append(mo.md(f"❌ **Error:** No videos found in `{_batch_input_dir}`!"))
    else:
        for _vid_path in _test_videos:
            _vid_name = _vid_path.name
            _vid_stem = _vid_path.stem # This gets the name WITHOUT the .mp4 or .mov ending

            # Force the output to always be a safe .mp4 file
            _output_name = str(_batch_output_dir / f"AI_{_vid_stem}.mp4")

            # --- THE SKIP LOGIC ---
            # If the file already exists in the output folder, skip the heavy math!
            if os.path.exists(_output_name):
                _batch_ui_rows.append(mo.md("---"))
                _batch_ui_rows.append(mo.md(f"### ⏭️ Skipped (Already Processed): `{_vid_name}`"))
            else:
                _batch_ui_rows.append(mo.md("---"))
                _batch_ui_rows.append(mo.md(f"### ✨ Newly Processed: `{_vid_name}`"))

                # 1. Process the video into the math tensor the AI needs
                _v_custom = process_video(str(_vid_path))

                # 2. Ask the AI to predict the audio!
                _gen_a_custom = generate_foley(_v_custom)

                # 3. Stitch them together
                with mo.status.spinner(title=f"Processing new video: {_vid_name}..."):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp_audio:
                        _audio_flat = _gen_a_custom.flatten()
                        _max_val = np.max(np.abs(_audio_flat)) if np.max(np.abs(_audio_flat)) > 0 else 1
                        _audio_int = (_audio_flat / _max_val * 32767).astype(np.int16)
                        wavfile.write(_tmp_audio.name, 22050, _audio_int)
                        _tmp_audio_path = _tmp_audio.name

                    try:
                        _video_clip = VideoFileClip(str(_vid_path))
                        _audio_clip = AudioFileClip(_tmp_audio_path)

                        # Trim to 3 seconds if the video is too long
                        if _video_clip.duration > 3.0:
                            _video_clip = _video_clip.subclipped(0, 3.0)

                        _final_clip = _video_clip.with_audio(_audio_clip)
                        _final_clip.write_videofile(_output_name, codec="libx264", audio_codec="aac", logger=None)

                        _audio_clip.close()
                        _video_clip.close()
                    finally:
                        if os.path.exists(_tmp_audio_path):
                            os.remove(_tmp_audio_path)

            # 4. Add the results to the Marimo UI (Shows both skipped and newly processed ones!)
            _item = mo.hstack([
                mo.vstack([
                    mo.md("**Silent Input Video**"),
                    mo.video(str(_vid_path), width=300)
                ]),
                mo.vstack([
                    mo.md("**AI Generated Output**"),
                    mo.video(_output_name, width=300),
                    mo.download(data=open(_output_name, "rb"), filename=f"AI_{_vid_stem}.mp4", label="Download Result")
                ])
            ], gap=4)
            _batch_ui_rows.append(_item)

    # Display everything!
    _batch_output_ui = mo.vstack(_batch_ui_rows)
    _batch_output_ui
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
