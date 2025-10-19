# Creator Boost – AutoCut + AutoCaption
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational)](#)
[![FFmpeg](https://img.shields.io/badge/Requires-FFmpeg-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/LucasMilanez/creator-boost-autocut/ci.yml?label=CI)](https://github.com/LucasMilanez/creator-boost-autocut/actions)

Corta silêncios do vídeo, gera legendas (Vosk/Faster-Whisper) e opcionalmente **queima** as legendas no vídeo final. Focado em fluxo **Windows-friendly** e otimizado para **GPU AMD (RX 580) com AMF (h264_amf)**, com fallback automático para x264. Concat robusto via `filter_complex` e cortes precisos quando solicitado.

> Script principal: `app.py` • Requisitos: **FFmpeg** e **Python 3.9+**

---

## ✨ Recursos
- **VAD (webrtcvad)** para detectar fala e cortar silêncios.
- **STT**: `vosk` (leve/offline) **ou** `faster-whisper` (qualidade).
- **Concat** robusto via `filter_complex` no Windows (evita bugs do demuxer).
- **Encoder inteligente**: usa AMF (`h264_amf`) no Windows quando disponível; senão, x264.
- **Burn-in** de SRT com fonte, cor, contorno, alinhamento e margem.
- **Denoise** leve opcional antes do STT.
- **SRT** com tempos monotônicos (evita overlaps).

---

## 📦 Instalação

1) **FFmpeg**
- **Windows**: `winget install Gyan.FFmpeg` ou `choco install ffmpeg` e garanta que `ffmpeg` está no `PATH`.
- **macOS**: `brew install ffmpeg`
- **Linux (Debian/Ubuntu)**: `sudo apt install ffmpeg`

2) **Python e dependências**
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
