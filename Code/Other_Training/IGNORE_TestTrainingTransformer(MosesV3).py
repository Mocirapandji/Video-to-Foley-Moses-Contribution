import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    #import os
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
    import torch.nn.functional as F

    return (
        DataLoader,
        Dataset,
        Path,
        cv2,
        librosa,
        models,
        nn,
        np,
        optim,
        pd,
        plt,
        time,
        torch,
        torchaudio,
    )


@app.cell
def _(mo):
    mo.md(r"""
    #Cell 1
    """)
    return


@app.cell
def _(Path):
    # .parent moves up to foley-to-sound
    repo_root = Path.cwd() 

    dataset_path = repo_root/"Dataset" 
    train_path = dataset_path/"Dataset_train"
    test_path = dataset_path/"Dataset_test"

    print(f"Dataset location: {dataset_path}")
    return (repo_root,)


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

        # --- THE SAFETY NET FIX ---
        # Only keep the row if BOTH the video and audio file physically exist!
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
    #Cell 5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Process Videos and Audios
    ## Cell 6
    """)
    return


@app.cell
def _(cv2, np, torch):
    def process_video(mp4_path, target_frames=90, resize=(224, 224)):
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
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
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
    # Cell 7
    """)
    return


@app.cell
def _(librosa, np, torch):
    def process_audio(wav_path, target_frames=90, sr=22050, n_mels=80):
        y, _ = librosa.load(wav_path, sr=sr)
        yt = librosa.effects.preemphasis(y)
        hop_length = int(sr / 30)

        mel_spec = librosa.feature.melspectrogram(
            y=yt, sr=sr, hop_length=hop_length,
            n_mels=n_mels, n_fft=1024, fmin=0.0, fmax=8000.0
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = log_mel_spec[:, :target_frames]

        if log_mel_spec.shape[1] < target_frames:
            pad_width = target_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='edge')

        log_mel_spec = np.clip(log_mel_spec, -80, 0)
        log_mel_spec = (log_mel_spec + 80) / 80

        return torch.tensor(log_mel_spec, dtype=torch.float32)

    return (process_audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load the Dataset
    ## Cell 8
    """)
    return


@app.cell
def _(Dataset, process_audio, process_video, test_df, train_df):
    class LocalFoleyDataset(Dataset):
        def __init__(self, df, clip_seconds=3.0):
            self.df = df.reset_index(drop=True)
            self.clip_seconds = clip_seconds

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            video_tensor = process_video(row['video_path'], target_frames=90)
            audio_tensor = process_audio(row['audio_path'], target_frames=90)
            return video_tensor, audio_tensor, row['base_id']

    train_ds = LocalFoleyDataset(train_df, clip_seconds=3.0)
    test_ds = LocalFoleyDataset(test_df, clip_seconds=3.0)
    return test_ds, train_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 9
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
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 10
    """)
    return


@app.cell
def _(train_ds, train_loader):
    print(f"Number of training batches: {len(train_loader)}")

    # FIX: Added 'surface_label' to catch the 4th item returned by the dataset!
    video_tensor, audio_tensor, surface_label, meta = train_ds[1]

    print(f"Video Shape: {video_tensor.shape}") 
    print(f"Audio Shape: {audio_tensor.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 11
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
    ## Cell 12
    """)
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 13
    """)
    return


@app.cell
def _(device, models, nn):
    class VideoFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet.fc = nn.Identity()
            self.extractor = resnet

            for name, param in self.extractor.named_parameters():
                if "layer4" in name or "layer3" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        def forward(self, video_tensor):
            return self.extractor(video_tensor)

    visual_encoder = VideoFeatureExtractor().to(device)
    return (visual_encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 13.5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 14
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visual Encoder - LSTM
    #Cell 15
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(SURFACE_TYPES, device, nn, torch):
    import math

    class TransformerFoleyGen(nn.Module):
        def __init__(self, visual_dim=2048, d_model=512, nhead=16, num_layers=6, 
                     audio_dim=80, num_surfaces=8):
            super().__init__()
            self.input_projection = nn.Linear(visual_dim, d_model)
            self.surface_embedding = nn.Embedding(num_surfaces, d_model)
            pe = torch.zeros(90, d_model)
            position = torch.arange(0, 90, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                batch_first=True, activation='gelu'
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.fc_out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(d_model // 2, audio_dim),
                nn.Sigmoid()
            )

        def forward(self, x, surface_labels):
            x = self.input_projection(x) + self.pe
            surf_emb = self.surface_embedding(surface_labels).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + surf_emb
            x = self.transformer_encoder(x)
            return self.fc_out(x).permute(0, 2, 1)

    model = TransformerFoleyGen(audio_dim=80, num_surfaces=len(SURFACE_TYPES)).to(device)

    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 16
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 17
    """)
    return


@app.cell
def _(model, nn, optim, visual_encoder):
    class CrunchyFoleyLoss(nn.Module):
        """
        Improved loss function to reduce 'jelly' artifacts by:
        1. Removing temporal smoothing (preserving sharp transients).
        2. Adding high-frequency emphasis for gravel textures.
        3. Using dynamic weighting to prioritize loud footstep peaks.
        """
        def __init__(self, footstep_weight=30.0):
            super().__init__()
            self.l1 = nn.L1Loss(reduction='none')
            self.weight_scale = footstep_weight

        def forward(self, pred, target):
            # 1. Base L1 Loss (Pixel-to-pixel Mel comparison)
            # We use reduction='none' so we can apply weights before averaging
            base_loss = self.l1(pred, target)

            # 2. High-Frequency Emphasis (The 'Crunch')
            # Gravel 'click' sounds live in the upper Mel bins (index 64 to 128)
            hf_pred = pred[:, 64:, :]
            hf_target = target[:, 64:, :]
            hf_loss = self.l1(hf_pred, hf_target).mean()

            # 3. Dynamic Weighting (Focus on the Peaks)
            # This makes the model care more about the loud footstep 
            # and less about the quiet background noise.
            weight = 1.0 + (target * self.weight_scale)

            # Combine: Weighted base loss + High-frequency boost
            total_loss = (base_loss * weight).mean() + (hf_loss * 2.0)

            return total_loss

    criterion = CrunchyFoleyLoss(footstep_weight=30.0)

    # We keep the optimizers as they were, but these will now 
    # backpropagate the 'crunchier' gradients.
    optimizer = optim.Adam([
        {'params': visual_encoder.parameters(), 'lr': 1e-6},
        {'params': model.parameters(), 'lr': 1e-4}
    ])
    return criterion, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    ## Cell 18
    """)
    return


@app.function
def train_one_epoch(model, visual_encoder, dataloader, optimizer, criterion, device):
    model.train()
    visual_encoder.train()
    total_loss = 0
    batch_losses = []

    for batch_idx, (video, audio, _) in enumerate(dataloader):
        video = video.to(device)
        audio = audio.to(device)

        b, t, c, h, w = video.shape
        v_feats = visual_encoder(video.view(b * t, c, h, w)).view(b, t, -1)

        optimizer.zero_grad()
        output = model(v_feats)
        loss = criterion(output, audio)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), batch_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 19
    """)
    return


@app.cell
def _(torch, train_loader):
    def validate(model, visual_encoder, dataloader, criterion, device):
        model.eval()
        visual_encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for video, audio, _ in dataloader:
                video = video.to(device)
                audio = audio.to(device)

                b, t, c, h, w = video.shape
                v_feats = visual_encoder(video.view(b * t, c, h, w)).view(b, t, -1)
                output = model(v_feats)
                loss = criterion(output, audio)
                val_loss += loss.item()

        return val_loss / len(dataloader)

    vi, au, _ = next(iter(train_loader))
    print(f"Audio Max: {au.max().item():.2f}")
    print(f"Audio Min: {au.min().item():.2f}")
    print(f"Audio Mean: {au.mean().item():.2f}")
    return (validate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 20
    """)
    return


@app.cell
def _(train_loader):
    # Grab one batch from your loader
    vi, au, sl, _ = next(iter(train_loader))
    print(f"Audio Max: {au.max().item():.2f}")
    print(f"Audio Min: {au.min().item():.2f}")
    print(f"Audio Mean: {au.mean().item():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 21
    """)
    return


@app.cell
def _(
    criterion,
    device,
    model,
    optimizer,
    test_loader,
    time,
    torch,
    train_loader,
    validate,
    visual_encoder,
):
    num_epochs = 40
    patience = 5    
    best_val_loss = float('inf')
    counter = 0

    all_epoch_losses = []
    all_batch_losses = []
    val_losses = []

    print(f"Starting Training on {device}...")
    for epoch in range(num_epochs):
        start_time = time.time()

        avg_train_loss, _ = train_one_epoch(
            model, visual_encoder, train_loader, optimizer, criterion, device
        )

        avg_val_loss = validate(model, visual_encoder, test_loader, criterion, device)

        all_epoch_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save with Transformer naming
            torch.save(model.state_dict(), "Moses_Transformer_v2_best.pth")
            torch.save(visual_encoder.state_dict(), "Moses_ResNet_Encoder_v2.pth")
            print("  --> Model Saved!")
            counter = 0 
        else:
            counter += 1
            if counter >= patience:
                break
    return all_batch_losses, all_epoch_losses, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 22
    """)
    return


@app.cell
def _(all_batch_losses, all_epoch_losses, plt, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left Plot: Batch-level loss (shows noise and convergence)
    ax1.plot(all_batch_losses, label='Batch Loss', color='blue', alpha=0.6)
    ax1.set_title('Batch Training Loss (L1)')
    ax1.set_xlabel('Total Batches')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Right Plot: Epoch-level comparison
    epochs = range(1, len(all_epoch_losses) + 1)
    ax2.plot(epochs, all_epoch_losses, label='Train Loss', marker='o')
    ax2.plot(epochs, val_losses, label='Val Loss', marker='x')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    # In marimo, use mo.as_html(plt.gcf()) or simply return the figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generation
    #Cell 23
    """)
    return


@app.cell
def _(device, librosa, model, np, torch, visual_encoder):
    def generate_foley(video_tensor):
        model.eval()
        visual_encoder.eval()
        sr = 22050
        hop_length = sr // 60
        v_input = video_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            b, t, c, h, w = v_input.shape
            v_feats = visual_encoder(v_input.view(b * t, c, h, w)).view(b, t, -1)
            pred_spec = model(v_feats).squeeze(0).cpu().numpy()

        # Un-normalize using -80dB range to match process_audio
        pred_spec_db = (pred_spec * 80) - 80

        # Noise gate - kill the engine rumble zone
        pred_spec_db = np.where(pred_spec_db < -50, -80, pred_spec_db)
        pred_power = librosa.db_to_power(pred_spec_db)
        pred_power = np.where(pred_power < 0.0001, 0, pred_power)
        pred_power = pred_power ** 2.0

        # Mute the quietest 25% of frames entirely
        frame_energy = np.mean(pred_power, axis=0)
        energy_threshold = np.percentile(frame_energy, 25)
        pred_power[:, frame_energy < energy_threshold] = 0

        audio_out = librosa.feature.inverse.mel_to_audio(
            pred_power, sr=sr,
            n_fft=2048,
            hop_length=hop_length,
            n_iter=200,
            fmax=8000.0
        )
        return audio_out

    return (generate_foley,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preview Generated Audio
    pls note this section only for preview but not for downloading.
    video preview is not working btw

    #Cell 24
    """)
    return


@app.cell
def _(np):
    import scipy.signal as signal

    def de_jellify_audio(audio_array, sr=22050):
        sos = signal.butter(6, 300, 'hp', fs=sr, output='sos')
        filtered_audio = signal.sosfilt(sos, audio_array)

        sos_notch = signal.butter(2, [200, 500], 'bandstop', fs=sr, output='sos')
        filtered_audio = signal.sosfilt(sos_notch, filtered_audio)

        sign = np.sign(filtered_audio)
        expanded = (np.abs(filtered_audio) ** 1.2) * sign

        clipped = np.tanh(expanded * 1.2)

        max_val = np.max(np.abs(clipped))
        if max_val > 0:
            clipped = clipped / max_val

        return clipped

    return (de_jellify_audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 25
    """)
    return


@app.cell
def _(
    de_jellify_audio,
    generate_foley,
    mo,
    model,
    np,
    os,
    repo_root,
    test_ds,
    torch,
    visual_encoder,
):
    from scipy.io import wavfile
    import shutil
    import tempfile
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip

    preview_dir = repo_root / "diagnostic_previews"
    os.makedirs(preview_dir, exist_ok=True)

    model.load_state_dict(torch.load("Moses_Transformer_v2_best.pth", weights_only=True))
    visual_encoder.load_state_dict(torch.load("Moses_ResNet_Encoder_v2.pth", weights_only=True))
    model.eval()
    visual_encoder.eval()

    test_indices = [20, 40]
    rows = []

    for idx in test_indices:
        preview_v, preview_a, preview_vid_id = test_ds[idx]

        raw_gen_a = generate_foley(preview_v)
        gen_a = de_jellify_audio(raw_gen_a)

        gen_audio_path = str(preview_dir / f"pred_{preview_vid_id}.wav")
        gen_flat = gen_a.flatten()
        max_val = np.max(np.abs(gen_flat)) if np.max(np.abs(gen_flat)) > 0 else 1
        gen_int = (gen_flat / max_val * 32767).astype(np.int16)
        wavfile.write(gen_audio_path, 22050, gen_int)

        video_path = str(repo_root / "Dataset" / "Dataset_test" / "video_silent" / f"{preview_vid_id}.mp4")
        audio_real_path = str(repo_root / "Dataset" / "Dataset_test" / "audio_enhanced" / f"{preview_vid_id}.wav")
        preview_vid_path = str(preview_dir / f"ref_{preview_vid_id}.mp4")
        preview_real_audio_path = str(preview_dir / f"real_{preview_vid_id}.wav")

        if os.path.exists(video_path):
            shutil.copy2(video_path, preview_vid_path)
        if os.path.exists(audio_real_path):
            shutil.copy2(audio_real_path, preview_real_audio_path)

        combined_path = str(preview_dir / f"combined_{preview_vid_id}.mp4")
        if os.path.exists(video_path):
            try:
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(gen_audio_path)
                final_clip = video_clip.with_audio(audio_clip)
                final_clip.write_videofile(combined_path, codec="libx264", audio_codec="aac", logger=None)
                audio_clip.close()
                video_clip.close()
            except Exception as e:
                print(f"Could not merge: {e}")

        rows.append(mo.md(f"---\n### {preview_vid_id}"))
        rows.append(mo.hstack([
            mo.vstack([
                mo.md("**AI Audio + Video**"),
                mo.video(combined_path, width=300) if os.path.exists(combined_path) else mo.md("merge failed"),
            ]),
            mo.vstack([
                mo.md("**Reference Video**"),
                mo.video(preview_vid_path, width=300) if os.path.exists(preview_vid_path) else mo.md("no video"),
                mo.md("**Real Audio**"),
                mo.audio(open(preview_real_audio_path, "rb").read()) if os.path.exists(preview_real_audio_path) else mo.md("no audio"),
            ]),
        ], gap=2))

    mo.vstack(rows)
    return AudioFileClip, VideoFileClip, tempfile, wavfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download the Outcome
    Syncing Audio with the Video
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


@app.cell
def _(AudioFileClip, VideoFileClip, mo, np, os, repo_root, tempfile, wavfile):
    def export_foley_video(vid_id, gen_audio_array, split_name="Dataset_test"):
        # Create folder
        os.makedirs("qianciexports", exist_ok=True)
        output_name = f"qianciexports/foley_{vid_id}.mp4"

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


app._unparsable_cell(
    """
    def _():
        from torchvision import transforms
        from scipy.io import wavfile
        def load_manual_video(path, num_frames=150):
            \"\"\"Converts a raw MP4 into the Tensor format the AI was trained on.\"\"\"
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            cap = cv2.VideoCapture(path)
            frames = []

            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV reads BGR; AI needs RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(frame))
            cap.release()

            # Pad with zeros if video is shorter than 5 seconds (150 frames)
            while len(frames) < num_frames:
                frames.append(torch.zeros(3, 224, 224))

            return torch.stack(frames)

        # 2. THE EXPORTER
        def export_foley_video(vid_id, gen_audio_array, split_name=\"Test\"):
            \"\"\"Merges predicted audio back into the original MP4.\"\"\"
            os.makedirs(\"qianciexports\", exist_ok=True)

            # Clean output name to avoid issues with web players
            clean_id = str(vid_id).replace(\" \", \"_\").replace(\"(\", \"\").replace(\")\", \"\")
            output_name = f\"qianciexports/foley_{clean_id}.mp4\"

            # Locate original video
            video_path = str(repo_root / \"Dataset\" / split_name / f\"{vid_id}.mp4\")

            if not os.path.exists(video_path):
                return f\"❌ Error: Video not found at {video_path}\"

            # Save audio to temporary WAV
            with tempfile.NamedTemporaryFile(suffix=\".wav\", delete=False) as tmp_audio:
                audio_flat = gen_audio_array.flatten()
                max_val = np.max(np.abs(audio_flat)) if np.max(np.abs(audio_flat)) > 0 else 1
                audio_int = (audio_flat / max_val * 32767).astype(np.int16)
                wavfile.write(tmp_audio.name, 22050, audio_int)
                tmp_audio_path = tmp_audio.name

            try:
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(tmp_audio_path)
                final_clip = video_clip.with_audio(audio_clip)

                # Write final video
                final_clip.write_videofile(output_name, codec=\"libx264\", audio_codec=\"aac\", logger=None)

                audio_clip.close()
                video_clip.close()
                return output_name
            finally:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

        # 3. THE EXECUTION
        target_name = \"IMG_0051 (1)\"
        manual_path = str(repo_root / \"Dataset\" / \"Test\" / f\"{target_name}.mp4\")

        if os.path.exists(manual_path):
            print(f\"🚀 Found file! Loading pixels from: {manual_path}\")

            # Convert MP4 -> Tensor
            v_manual = load_manual_video(manual_path)

            # Run through your Model
        return print(\"🧠 AI is analyzing the movement and generating sound...\")
            gen_audio_manual = generate_foley(v_manual)

            # Merge and Export
            final_video_path = export_foley_video(target_name, gen_audio_manual, split_name=\"Test\")

            # Display in Marimo
            result = mo.vstack([
                mo.md(f\"✅ **Generation Complete!**\"),
                mo.video(final_video_path, width=500),
                mo.download(data=open(final_video_path, \"rb\"), filename=f\"foley_{target_name}.mp4\")
            ])
        else:
            result = mo.md(f\"❌ **Error:** File NOT found at `{manual_path}`\")


    _()
    """,
    name="_"
)


@app.cell
def _(export_foley_video, generate_foley, test_ds):
    # Pick a sample from your filtered gravel walk data
    v6, a6, vid_id6 = test_ds[25] # With the same index range [0,713]
    gen_a6 = generate_foley(v6)

    # Calling the function (the output will appear below this cell)
    export_foley_video(vid_id6, gen_a6)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
