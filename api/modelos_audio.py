# api/modelos_audio.py
import torch
import torch.nn as nn
import numpy as np
import librosa

CLASES_AUDIO = [
    "Aire acondicionado", "Bocina de auto", "Niños jugando", "Ladrido de perro",
    "Taladro", "Motor en marcha", "Disparo", "Martillo neumático", "Sirena", "Música callejera"
]

class CNNAudioUrbano(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        self.bloque1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.bloque2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.bloque3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16 * 21, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_clases))

    def forward(self, x):
        return self.clasificador(self.bloque3(self.bloque2(self.bloque1(x))))

def cargar_modelo_cnn(ruta):
    modelo = CNNAudioUrbano(num_clases=10)
    state_dict = torch.load(str(ruta), map_location="cpu", weights_only=False)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo

def clasificar_audio(modelo_cnn, audio_array, sr=22050):
    try:
        audio_mono = np.mean(audio_array, axis=1) if len(audio_array.shape) > 1 else audio_array
        audio_mono = audio_mono.astype(np.float32)
        target_len = sr * 4
        if len(audio_mono) < target_len:
            audio_mono = np.pad(audio_mono, (0, target_len - len(audio_mono)))
        else:
            audio_mono = audio_mono[:target_len]
        
        mel = librosa.feature.melspectrogram(y=audio_mono, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] > 173: mel_db = mel_db[:, :173]
        elif mel_db.shape[1] < 173: mel_db = np.pad(mel_db, ((0, 0), (0, 173 - mel_db.shape[1])))
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            salida = modelo_cnn(tensor)
            probabilidades = torch.softmax(salida, dim=1)
            confianza, clase_idx = torch.max(probabilidades, dim=1)
            
        return CLASES_AUDIO[clase_idx.item()], confianza.item()
    except Exception as e:
        return "Sin clasificar", 0.0