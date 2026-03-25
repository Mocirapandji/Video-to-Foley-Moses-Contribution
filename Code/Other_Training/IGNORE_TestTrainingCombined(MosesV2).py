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
    # Cell 1
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
    return (repo_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 2
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
    # Cell 3
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
    # Cell 4
    """)
    return


@app.cell
def _(mo, os, pd, repo_root):
    mo.md(r"""
    # Cell 4
    """)

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
    # Cell 6
    """)
    return


@app.cell
def _(librosa, mo, np, torch):
    mo.md(r"""
    # Cell 6
    """)

    def process_audio(wav_path, target_frames=90, sr=22050, n_mels=80):
        y, _ = librosa.load(wav_path, sr=sr) 

        # --- MOSES'S SECRET WEAPON: PRE-EMPHASIS ---
        # This kills the blurry low-end drone and violently sharpens the footsteps
        y = librosa.effects.preemphasis(y)

        hop_length = int(sr/30) 

        # HiFi-GAN standard math
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            hop_length=hop_length, 
            n_mels=n_mels,
            n_fft=1024,      
            fmin=0.0,        
            fmax=8000.0      
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = log_mel_spec[:, :target_frames]

        if log_mel_spec.shape[1] < target_frames:
            pad_width = target_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='edge')

        # Force absolute silence!
        log_mel_spec = np.clip(log_mel_spec, -80, 0)
        log_mel_spec = (log_mel_spec + 80) / 80

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
    # Cell 8
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
    # Cell 9
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
    # Cell 10
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 12
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

            # Freeze the early layers, but unfreeze the last layer (layer4) - learns the most abstract concepts
            # so it can learn specific textures like snow, leaves, and gravel.
            for name, param in self.extractor.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        def forward(self, video_tensor):
            features = self.extractor(video_tensor)
            return features

    visual_encoder = VideoFeatureExtractor().to(device)
    return (visual_encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 13
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
    ## Cell 14
    """)
    return


@app.cell
def _(device, nn):
    # feature to mel-spectrogram
    class FoleyGenModel(nn.Module):
        def __init__(self, visual_dim=2048, hidden_dim=256, audio_dim=80, dropout_rate=0.3):
            super(FoleyGenModel, self).__init__()

            self.lstm = nn.LSTM(
                input_size=visual_dim, 
                hidden_size=hidden_dim, 
                num_layers=2, 
                batch_first=True,
                bidirectional=True # Looks forward and backward in time
            )

            # Dropout to prevent overfitting to one material
            self.dropout = nn.Dropout(dropout_rate)
            # Adding one more layer - not directly into the size of the spectrogram
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) # only compress the size of the data without changing the shape 1024-128
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, audio_dim)
            self.activation = nn.Sigmoid() # when process_audio, data range 0-1

        def forward(self, x):
            # x(Batch, 90, 2048)
            # (Batch, 90, hidden_dim * 2)
            lstm_out, _ = self.lstm(x)
            # (Batch, 90, 128)
            x = self.dropout(lstm_out) # randomly 30% off, prevent overfitting on one
            x = self.relu(self.fc1(x))
            audio_out = self.activation(self.fc2(x))

            return audio_out.permute(0, 2, 1)

    model = FoleyGenModel().to(device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 15
    """)
    return


@app.cell
def _(model, nn, optim, visual_encoder):
    class WeightedFoleyLoss(nn.Module):
        def __init__(self, footstep_weight=8):
            super().__init__()
            self.l1 = nn.L1Loss(reduction='none') # Don't average yet
            self.weight_scale = footstep_weight

        def forward(self, pred, target):
            # Calculate the standard L1 error
            base_loss = self.l1(pred, target)

            # Weight map based on the ground truth intensity
            # target is 0.0 to 1.0, this makes footsteps heavier.
            # Silence (0) gets weight 1.0; Loud peak (1.0) gets weight 16.0 - punish more on missing one footstep than missing the silence
            weight = 1.0 + (target * self.weight_scale)

            # Apply the weight, return the mean
            return (base_loss * weight).mean()

    # Define the new criterion
    criterion = WeightedFoleyLoss(footstep_weight=8) 

    # Lower lr for ResNet, prevent destroying its existing knowledge.
    optimizer = optim.Adam([
        {'params': visual_encoder.parameters(), 'lr': 5e-6}, 
        {'params': model.parameters(), 'lr': 5e-5}
    ])
    return criterion, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 16
    """)
    return


@app.cell
def _(device, model, torch, v_tensor, visual_encoder):
    # fake batch of 1
    v_input = v_tensor.unsqueeze(0).to(device) # Shape: [1, 90, 3, 224, 224]

    # Create a fake label for the test (e.g., label 0 for gravel)
    fake_label = torch.tensor([0], dtype=torch.long).to(device)

    with torch.no_grad():
        # Get visual features
        b, t, c, h, w = v_input.shape
        v_reshaped = v_input.view(b * t, c, h, w)
        feats = visual_encoder(v_reshaped).view(b, t, -1)

        # Predict Audio (Now passing BOTH features and the label!)
        preds = model(feats, fake_label)

    print(f"Final Prediction Shape: {preds.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    ## Cell 17
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

        # Extract Features
        b, t, c, h, w = video.shape # batch, time, channel, height of the image, width
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 18
    """)
    return


@app.cell
def _(torch):
    def validate(model, visual_encoder, dataloader, criterion, device):
        model.eval()
        visual_encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for video, audio, _ in dataloader:
                video, audio = video.to(device), audio.to(device)

                # Extract Visual Features (standardize shape for ResNet)
                b, t, c, h, w = video.shape
                v_feats = visual_encoder(video.view(b * t, c, h, w)).view(b, t, -1)

                # Predict and Calculate Loss
                output = model(v_feats)
                loss = criterion(output, audio)
                val_loss += loss.item()

        return val_loss / len(dataloader)

    return (validate,)


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
    # Cell 19
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
    num_epochs = 200
    patience = 200    
    best_val_loss = float('inf')
    counter = 0

    all_epoch_losses = []
    all_batch_losses = []
    val_losses = []

    print(f"Starting Generalist Training on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        avg_train_loss, epoch_batch_losses = train_one_epoch(
            model, visual_encoder, train_loader, optimizer, criterion, device
        )

        # Validate
        avg_val_loss = validate(model, visual_encoder, test_loader, criterion, device)

        # Store data for plotting
        all_epoch_losses.append(avg_train_loss)
        all_batch_losses.extend(epoch_batch_losses)
        val_losses.append(avg_val_loss)

        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] finished in {duration:.1f}s")
        print(f"  Train L1: {avg_train_loss:.4f} | Val L1: {avg_val_loss:.4f}")

        # Early Stopping & Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "Hifigan_LSTM_v2.pth")
            torch.save(visual_encoder.state_dict(), "Hifigan_ResNet_v2.pth")
            print("  --> New best model saved!")
            counter = 0 
        else:
            counter += 1
            print(f"  (No improvement for {counter}/{patience} epochs)")
            if counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break
    return all_batch_losses, all_epoch_losses, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell 20
    """)
    return


@app.cell
def _(all_batch_losses, all_epoch_losses, plt, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(all_batch_losses, label='Batch Loss', color='blue')
    ax1.set_title('Batch Training Loss (L1)')
    ax1.set_xlabel('Total Batches')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    epochs = range(1, len(all_epoch_losses) + 1)
    ax2.plot(epochs, all_epoch_losses, label='Train Loss', marker='o')
    ax2.plot(epochs, val_losses, label='Val Loss', marker='x')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generation
    ## Cell 21
    """)
    return


@app.cell
def _(device, mo, model, torch, torchaudio, visual_encoder):
    mo.md(r"""
    # Generation
    ## Cell 21
    """)

    import torch.nn.functional as F

    # --- THE MONKEY PATCH FIX ---
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    # --- THE CIRCULAR IMPORT FIX ---
    import speechbrain
    import speechbrain.utils
    from speechbrain.inference.vocoders import HIFIGAN

    # 1. Download and load the pre-trained Vocoder into memory
    print("Downloading/Loading HiFi-GAN Neural Vocoder...")
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", 
        savedir="pretrained_models/tts-hifigan-ljspeech",
        run_opts={"device": str(device)}
    )
    print("Vocoder loaded successfully!")

    def generate_foley(video_tensor):
        model.eval()
        visual_encoder.eval()

        v_input = video_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            # 1. Extract Visual Features
            b, t, c, h, w = v_input.shape
            v_flat = v_input.view(b * t, c, h, w)
            v_feats = visual_encoder(v_flat).view(b, t, -1)

            # 2. Predict Spectrogram
            pred_spec = model(v_feats) # Shape: (Batch, 80 Mels, Time)

            # 3. Un-normalize back to decibels
            pred_spec_db = (pred_spec * 80) - 80

            # Apply your aggressive noise gate for pure silence between steps
            pred_spec_db = torch.where(pred_spec_db < -45, torch.tensor(-80.0).to(device), pred_spec_db)

            # --- THE CHIPMUNK FIX (Corrected) ---

            # Stretch the time! HiFi-GAN's math requires ~258 frames for 3 seconds of audio.
            # This cures the high-pitched chipmunk screech.
            stretched_spec = F.interpolate(pred_spec_db, size=258, mode='linear', align_corners=False)

            # Scale it for the Vocoder
            hifi_input = stretched_spec / 20.0 

            # (The bad transpose line has been completely removed!)

            # 5. Generate Audio with the Neural Vocoder!
            waveforms = hifi_gan.decode_batch(hifi_input)

            # Convert back to standard numpy array for the video exporter
            audio_out = waveforms.squeeze().cpu().numpy()

        return audio_out

    return F, HIFIGAN, generate_foley


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Cell 21.5
    """)
    return


@app.cell
def _(
    F,
    HIFIGAN,
    device,
    io,
    librosa,
    mo,
    model,
    np,
    os,
    repo_root,
    test_ds,
    torch,
    torchaudio,
    visual_encoder,
    wavfile,
):
    mo.md(r"""
    # Generation & Diagnostic Preview
    ## Cell 21.5
    """)


    # --- 1. THE MONKEY PATCH ---
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


    # --- 2. LOAD HI-FI GAN ---
    print("Loading HiFi-GAN Neural Vocoder for Diagnostics...")
    hifi_gan_diag = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", 
        savedir="pretrained_models/tts-hifigan-ljspeech",
        run_opts={"device": str(device)}
    )
    print("Diagnostic Vocoder loaded successfully!")

    # --- 3. LOAD YOUR NEW TRAINED MODEL ---
    # Using the new Hifigan_LSTM files you just generated!
    model.load_state_dict(torch.load("Hifigan_LSTM_v2.pth", weights_only=True))
    visual_encoder.load_state_dict(torch.load("Hifigan_ResNet_v2.pth", weights_only=True))
    model.eval()
    visual_encoder.eval()

    # --- 4. GENERATOR A: THE OLD LIBROSA METHOD ---
    def generate_foley_librosa_diag(video_tensor):
        sr = 22050
        v_input = video_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            b, t, c, h, w = v_input.shape
            v_feats = visual_encoder(v_input.view(b * t, c, h, w)).view(b, t, -1)
            pred_spec = model(v_feats).squeeze(0).cpu().numpy()

        pred_spec_db = (pred_spec * 80) - 80
        pred_spec_db = np.where(pred_spec_db < -45, -80, pred_spec_db)
        pred_power = librosa.db_to_power(pred_spec_db)
        pred_power = np.where(pred_power < 0.0005, 0, pred_power) 
        pred_power = pred_power ** 1.5 

        audio_out = librosa.feature.inverse.mel_to_audio(
            pred_power, sr=sr, n_fft=2048, hop_length=int(sr/30), n_iter=100
        )
        return audio_out

    # --- 5. GENERATOR B: THE NEW HIFI-GAN METHOD ---
    def generate_foley_hifigan_diag(video_tensor):
        v_input = video_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            b, t, c, h, w = v_input.shape
            v_feats = visual_encoder(v_input.view(b * t, c, h, w)).view(b, t, -1)
            pred_spec = model(v_feats) 
            pred_spec_db = (pred_spec * 80) - 80
            pred_spec_db = torch.where(pred_spec_db < -45, torch.tensor(-80.0).to(device), pred_spec_db)

            # Time stretch for HiFi-GAN
            stretched_spec = F.interpolate(pred_spec_db, size=258, mode='linear', align_corners=False)
            hifi_input = stretched_spec / 20.0 

            waveforms = hifi_gan_diag.decode_batch(hifi_input)
            audio_out = waveforms.squeeze().cpu().numpy()
        return audio_out

    # --- 6. BUILD THE UI (Quarantined in a function for Marimo) ---
    def render_diagnostic_ui():
        def to_widgets_diag(audio_array):
            audio_flat = audio_array.flatten()
            max_val = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1
            audio_int = (audio_array / max_val * 32767).astype(np.int16)
            byte_io = io.BytesIO()
            wavfile.write(byte_io, 22050, audio_int)
            return mo.audio(byte_io.getvalue())

        diag_indices = [20, 45] 
        diag_rows = []

        for idx in diag_indices:
            diag_v, _, diag_vid_id = test_ds[idx]

            gen_a_lib = generate_foley_librosa_diag(diag_v)
            gen_a_hfg = generate_foley_hifigan_diag(diag_v)

            diag_video_path = str(repo_root / "Dataset" / "Dataset_test" / "video_silent" / f"{diag_vid_id}.mp4")
            diag_audio_path = str(repo_root / "Dataset" / "Dataset_test" / "audio_enhanced" / f"{diag_vid_id}.wav")

            if os.path.exists(diag_audio_path):
                diag_real_audio, _ = librosa.load(diag_audio_path, sr=22050)
            else:
                diag_real_audio = np.zeros(22050)

            diag_rows.append(mo.md(f"---"))
            diag_rows.append(mo.md(f"### Video: {diag_vid_id}"))

            item = mo.hstack([
                mo.vstack([
                    mo.md("**Reference Video**"),
                    mo.video(diag_video_path, width=300)
                ], align="center"),
                mo.vstack([
                    mo.md("**1. Librosa (Griffin-Lim)**"), 
                    to_widgets_diag(gen_a_lib),
                    mo.md("**2. HiFi-GAN**"), 
                    to_widgets_diag(gen_a_hfg),
                    mo.md("**3. Ground Truth**"), 
                    to_widgets_diag(diag_real_audio)
                ])
            ], gap=2, justify="center")

            diag_rows.append(item)

        return mo.vstack(diag_rows)

    # Execute the UI render
    render_diagnostic_ui()
    return


@app.cell
def _(
    device,
    librosa,
    mo,
    model,
    np,
    os,
    plt,
    repo_root,
    test_ds,
    torch,
    visual_encoder,
):
    mo.md(r"""
    # Spectrogram Visualizer
    ## Cell 22
    """)


    def visualize_spectrograms(idx=20):
        # 1. Get the data
        v_vis, _, vid_id_vis = test_ds[idx]

        # 2. Predict with your LSTM
        v_input_vis = v_vis.unsqueeze(0).to(device)
        with torch.no_grad():
            b, t, c, h, w = v_input_vis.shape
            v_feats_vis = visual_encoder(v_input_vis.view(b * t, c, h, w)).view(b, t, -1)
            pred_spec_vis = model(v_feats_vis).squeeze(0).cpu().numpy()

        # Un-normalize the prediction to Decibels
        pred_spec_db_vis = (pred_spec_vis * 80) - 80

        # 3. Get the Ground Truth Audio
        audio_real_path_vis = str(repo_root / "Dataset" / "Dataset_test" / "audio_enhanced" / f"{vid_id_vis}.wav")
        if os.path.exists(audio_real_path_vis):
            real_audio_vis, _ = librosa.load(audio_real_path_vis, sr=22050)
            # Convert real audio to the exact same Mel-Spectrogram math
            real_spec = librosa.feature.melspectrogram(y=real_audio_vis, sr=22050, hop_length=int(22050/30), n_mels=80, n_fft=2048)
            real_spec_db_vis = librosa.power_to_db(real_spec, ref=np.max)
        else:
            real_spec_db_vis = np.zeros((80, 90))

        # 4. Draw the pictures!
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        img1 = ax1.imshow(real_spec_db_vis, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=0)
        ax1.set_title(f'Ground Truth Spectrogram ({vid_id_vis})')
        ax1.set_ylabel('Mel Frequency Channels')
        ax1.set_xlabel('Time Frames')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

        img2 = ax2.imshow(pred_spec_db_vis, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=0)
        ax2.set_title('AI Prediction Spectrogram (Before Vocoder)')
        ax2.set_xlabel('Time Frames')
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()
        return fig

    # Show the visualizer for video index 20 (you can change this number!)
    visualize_spectrograms(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preview Generated Audio
    pls note this section only for preview but not for downloading.
    video preview is not working btw
    """)
    return


@app.cell
def _(
    generate_foley,
    io,
    librosa,
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
    def to_widgets(audio_array):
        # Normalize and convert to 16-bit PCM
        audio_flat = audio_array.flatten()
        max_val = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1
        audio_int = (audio_array / max_val * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        wavfile.write(byte_io, 22050, audio_int)
        return mo.audio(byte_io.getvalue())

    # Load the trained brain
    model.load_state_dict(torch.load("RZfoley_model_general_LSTM_NewLoss.pth", weights_only=True))
    visual_encoder.load_state_dict(torch.load("RZfoley_model_general_ResNet_NewLoss.pth", weights_only=True))

    model.eval()
    visual_encoder.eval()

    test_indices = [2,54] # select any numbers of videos from the test folder. With index range [0,713]

    rows = []

    for idx in test_indices:
        v, a, vid_id = test_ds[idx]
        gen_a = generate_foley(v)

        video_path = str(repo_root / "Dataset" / "Dataset_test" / "video_silent" / f"{vid_id}.mp4")
        audio_real_path = str(repo_root / "Dataset" / "Dataset_test" / "audio_enhanced" / f"{vid_id}.wav")

        if os.path.exists(audio_real_path):
            real_audio_np, _ = librosa.load(audio_real_path, sr=22050)
        else:
            real_audio_np = np.zeros(22050)
            print(f"⚠️ Warning: Missing audio file at {audio_real_path}")

        rows.append(mo.md(f"---"))
        rows.append(mo.md(f"### Video: {vid_id}"))

        item = mo.hstack([
            # Left side: The Video
            mo.vstack([
                mo.md("**Reference Video**"),
                mo.video(video_path, width=300)
            ], align="center"),
            # Right side: The Audio comparison
            mo.vstack([
                mo.md("**AI Prediction**"), 
                to_widgets(gen_a),
                mo.md("**Real Sound**"), 
                to_widgets(real_audio_np)
            ])
        ], gap=2, justify="center")

        rows.append(item)

    mo.vstack(rows)
    return (wavfile,)


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
        os.makedirs("exportsRz", exist_ok=True)
        output_name = f"exportsRz/foley_{vid_id}.mp4"

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


@app.cell
def _():
    # from torchvision import transforms
    # from scipy.io import wavfile
    # def load_manual_video(path, num_frames=150):
    #     """Converts a raw MP4 into the Tensor format the AI was trained on."""
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    #     cap = cv2.VideoCapture(path)
    #     frames = []

    #     while len(frames) < num_frames:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         # OpenCV reads BGR; AI needs RGB
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frames.append(transform(frame))
    #     cap.release()

    #     # Pad with zeros if video is shorter than 5 seconds (150 frames)
    #     while len(frames) < num_frames:
    #         frames.append(torch.zeros(3, 224, 224))

    #     return torch.stack(frames)

    # # 2. THE EXPORTER
    # def export_foley_video(vid_id, gen_audio_array, split_name="Test"):
    #     """Merges predicted audio back into the original MP4."""
    #     os.makedirs("exportsRz", exist_ok=True)

    #     # Clean output name to avoid issues with web players
    #     clean_id = str(vid_id).replace(" ", "_").replace("(", "").replace(")", "")
    #     output_name = f"exportsRz/foley_{clean_id}.mp4"

    #     # Locate original video
    #     video_path = str(repo_root / "Dataset" / split_name / f"{vid_id}.mp4")

    #     if not os.path.exists(video_path):
    #         return f"❌ Error: Video not found at {video_path}"

    #     # Save audio to temporary WAV
    #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
    #         audio_flat = gen_audio_array.flatten()
    #         max_val = np.max(np.abs(audio_flat)) if np.max(np.abs(audio_flat)) > 0 else 1
    #         audio_int = (audio_flat / max_val * 32767).astype(np.int16)
    #         wavfile.write(tmp_audio.name, 22050, audio_int)
    #         tmp_audio_path = tmp_audio.name

    #     try:
    #         video_clip = VideoFileClip(video_path)
    #         audio_clip = AudioFileClip(tmp_audio_path)
    #         final_clip = video_clip.with_audio(audio_clip)

    #         # Write final video
    #         final_clip.write_videofile(output_name, codec="libx264", audio_codec="aac", logger=None)

    #         audio_clip.close()
    #         video_clip.close()
    #         return output_name
    #     finally:
    #         if os.path.exists(tmp_audio_path):
    #             os.remove(tmp_audio_path)

    # # 3. THE EXECUTION
    # target_name = "IMG_0051 (1)"
    # manual_path = str(repo_root / "Dataset" / "Test" / f"{target_name}.mp4")

    # if os.path.exists(manual_path):
    #     print(f"🚀 Found file! Loading pixels from: {manual_path}")

    #     # Convert MP4 -> Tensor
    #     v_manual = load_manual_video(manual_path)

    #     # Run through your Model
    #     print("🧠 AI is analyzing the movement and generating sound...")
    #     gen_audio_manual = generate_foley(v_manual)

    #     # Merge and Export
    #     final_video_path = export_foley_video(target_name, gen_audio_manual, split_name="Test")

    #     # Display in Marimo
    #     result = mo.vstack([
    #         mo.md(f"✅ **Generation Complete!**"),
    #         mo.video(final_video_path, width=500),
    #         mo.download(data=open(final_video_path, "rb"), filename=f"foley_{target_name}.mp4")
    #     ])
    # else:
    #     result = mo.md(f"❌ **Error:** File NOT found at `{manual_path}`")
    return


@app.cell
def _(export_foley_video, generate_foley, test_ds):
    # Pick a sample from your filtered gravel walk data
    v6, a6, vid_id6 = test_ds[78] # With the same index range [0,713]
    gen_a6 = generate_foley(v6)

    # Calling the function (the output will appear below this cell)
    export_foley_video(vid_id6, gen_a6)
    return


if __name__ == "__main__":
    app.run()
