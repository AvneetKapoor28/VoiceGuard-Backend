import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt
import librosa.display
from model import LCNN

# -------- Parameters (match training) --------
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
LABELS = {0: "bonafide", 1: "spoof"}

# -------- Load model once at import --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LCNN(num_classes=2).to(device)
state = torch.load("lcnn_weights.pth", map_location=device)
model.load_state_dict(state)
model.eval()


def _trim_initial_silence(y, sr=SR, silence_thresh_db=20, min_silence_sec=0.5):
    energy = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=HOP_LENGTH)
    # energy is amplitude RMS -> convert to dB power scale for thresholding
    energy_db = librosa.power_to_db(energy**2, ref=np.max)
    idx = np.where(energy_db > -silence_thresh_db)[0]
    if len(idx) == 0:
        return y
    first_t = times[idx[0]]
    if first_t >= min_silence_sec:
        return y[int(first_t * sr):]
    return y


def preprocess_to_logmel(
    path_or_fileobj,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    trim=True
):
    # librosa.load can take a path; for file-like objects we pass the object directly
    if trim:
        y, sr = librosa.load(path_or_fileobj, sr=sr)
        y = _trim_initial_silence(y, sr=sr)
    else:
        y, sr = librosa.load(path_or_fileobj, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel, sr


def logmel_to_tensor(log_mel: np.ndarray) -> torch.Tensor:
    # shape: (mels, T) -> (1, 1, mels, T)
    t = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return t


def predict_from_logmel(log_mel: np.ndarray):
    x = logmel_to_tensor(log_mel).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
    return {
        "pred_label": LABELS[pred_id],
        "pred_id": pred_id,
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)}
    }


def make_spectrogram_png_b64(log_mel: np.ndarray, sr=SR, hop_length=HOP_LENGTH) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', fmax=sr//2,
        ax=ax
    )
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format='%0.2f')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def run_inference(path_or_fileobj, return_array=False):
    log_mel, sr = preprocess_to_logmel(path_or_fileobj)
    out = predict_from_logmel(log_mel)
    png_b64 = make_spectrogram_png_b64(log_mel, sr=sr)
    result = {
        "prediction": out["pred_label"],
        "probabilities": out["probs"],
        "sr": int(sr),
        "n_mels": int(N_MELS),
        "hop_length": int(HOP_LENGTH),
        "spectrogram_png": png_b64
    }
    if return_array:
        # WARNING: this can be large; consider downsampling before returning in prod
        result["log_mel"] = log_mel.tolist()
    return result