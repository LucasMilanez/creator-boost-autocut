#!/usr/bin/env python3
r"""
Creator Boost – AutoCut + AutoCaption (v1.6 / Windows)

Foco desta versão:
  • Windows-friendly: caminhos, concat robusto e quebras de linha OK
  • Fallback automático para x264 caso AMF não esteja presente
  • Concat via filter_complex por padrão no Windows (evita bugs do demuxer)
  • Whisper no Windows: CPU otimizada (int8), threads configuráveis
  • Correções: escape de paths no filtro subtitles, SRT monotônico, denoise opcional

Requisitos (Windows):
- Python 3.9+
- FFmpeg **com AMF** (h264_amf). Dica: build do Gyan.dev/winget/choco geralmente tem AMF.
- pip install webrtcvad numpy tqdm
- STT offline leve: pip install vosk (e baixe um modelo pt-BR)
- STT mais preciso: pip install faster-whisper

Exemplos (PowerShell):
  # Whisper (recomendado p/ qualidade), usa CPU int8 + AMF para reencode
  python .\app.py -i .\(((NOME DO ARQUIVO .MP4))) -o .\out --stt whisper --whisper-model medium --whisper-device cpu --whisper-cpu-threads 8 --precise-cut --encoder auto --bitrate 8M --language pt --denoise 
  
  
    # Queima legendas com estilo personalizado
  python app.py -i ./video.mp4 -o ./out --stt whisper --burn  --font-name Arial --font-size 36 --font-color "#FFFFFF" --outline 2.0 --outline-color "#000000" --align bottom-center --margin-v 40 --precise-cut --encoder auto --bitrate 8M --language pt --denoise


  # Vosk (100% offline, muito leve), com reencode na GPU via AMF se disponível
  python .\app.py -i .\(((NOME DO ARQUIVO .MP4))) -o .\out --stt vosk --vosk-model C:\models\vosk-model-small-pt-0.3 --precise-cut --encoder auto --bitrate 8M --language pt --denoise

"""

from __future__ import annotations
import argparse
import contextlib
import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import List, Tuple, Optional

import numpy as np
import webrtcvad
from tqdm import tqdm

# ------------------------------- Utilidades FFmpeg -------------------------------

def run(cmd: List[str]) -> None:
    """Executa um comando com impressão amigável e checagem de erro."""
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Comando falhou: {' '.join(cmd)}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg não encontrado no PATH. Instale-o e tente novamente.")


def extract_wav_mono_16k(input_video: str, out_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav
    ]
    run(cmd)


def escape_path_for_subtitles_filter(path: str) -> str:
    """Escapa caminho para o filtro subtitles (Windows friendly)."""
    p = path.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return p

# ------------------------------- VAD (webrtcvad) -------------------------------

@dataclasses.dataclass
class Segment:
    start: float
    end: float


def read_wave_info(path: str) -> Tuple[int, float]:
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / float(sr)
        return sr, duration


def read_frames(path: str, frame_ms: int = 30) -> Tuple[List[bytes], int]:
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        sample_rate = wf.getframerate()
        frame_size = int(sample_rate * (frame_ms / 1000.0))
        frames = []
        audio = wf.readframes(wf.getnframes())
        for i in range(0, len(audio), frame_size * 2):
            chunk = audio[i:i + frame_size * 2]
            if len(chunk) == frame_size * 2:
                frames.append(chunk)
        return frames, sample_rate


def vad_segments(wav_path: str,
                 aggressiveness: int = 2,
                 frame_ms: int = 30,
                 pad_ms: int = 80,
                 min_seg_ms: int = 400,
                 merge_gap_ms: int = 250) -> List[Segment]:
    frames, sr = read_frames(wav_path, frame_ms)
    vad = webrtcvad.Vad(aggressiveness)

    is_speech = [vad.is_speech(f, sr) for f in frames]
    hop = frame_ms / 1000.0

    segs: List[Segment] = []
    i = 0
    while i < len(is_speech):
        if is_speech[i]:
            start = i * hop
            j = i + 1
            while j < len(is_speech) and is_speech[j]:
                j += 1
            end = j * hop
            segs.append(Segment(start, end))
            i = j
        else:
            i += 1

    merged: List[Segment] = []
    for s in segs:
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            if s.start - prev.end <= (merge_gap_ms / 1000.0):
                merged[-1] = Segment(prev.start, s.end)
            else:
                merged.append(s)

    _, total_dur = read_wave_info(wav_path)
    pad = pad_ms / 1000.0
    min_len = min_seg_ms / 1000.0
    padded: List[Segment] = []
    for s in merged:
        start = max(0.0, s.start - pad)
        end = min(total_dur, s.end + pad)
        if end - start >= min_len:
            padded.append(Segment(start, end))

    return padded

# ------------------------------- Encoder helpers -------------------------------

_FFMPEG_ENCODER_CACHE: dict[str, bool] = {}

def ffmpeg_has_encoder(name: str) -> bool:
    """Verifica se o FFmpeg suporta um encoder pelo nome (cacheado)."""
    if name in _FFMPEG_ENCODER_CACHE:
        return _FFMPEG_ENCODER_CACHE[name]
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ok = (proc.returncode == 0) and (name in proc.stdout)
    except Exception:
        ok = False
    _FFMPEG_ENCODER_CACHE[name] = ok
    return ok

def encoder_args_x264(crf: int, preset: str) -> List[str]:
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def encoder_args_amf(bitrate: str) -> List[str]:
    """Config simples para AMF (h264_amf) com VBR. RX 580 manda bem aqui."""
    return [
        "-c:v", "h264_amf",
        "-rc", "vbr",
        "-b:v", bitrate,
        "-quality", "quality",
        "-usage", "transcoding",
        "-pix_fmt", "yuv420p",
    ]


def pick_encoder_args(prefer: str, is_windows: bool, bitrate: str, crf: int, preset: str) -> Tuple[List[str], str]:
    """Escolhe encoder e retorna (args, nome) com fallback para x264 se AMF falhar."""
    if prefer == "x264":
        return encoder_args_x264(crf, preset), "x264"
    if prefer == "amf" or (prefer == "auto" and is_windows):
        if ffmpeg_has_encoder("h264_amf"):
            return encoder_args_amf(bitrate), "amf"
        else:
            print("Aviso: FFmpeg sem h264_amf; usando x264.")
            return encoder_args_x264(crf, preset), "x264"
    return encoder_args_x264(crf, preset), "x264"

# ------------------------------- Corte + Concat -------------------------------

def cut_and_concat(input_video: str, segments: List[Segment], out_video: str, workdir: str,
                   precise: bool=False, concat_mode: str="auto",
                   encoder_preference: str="auto", bitrate: str="8M",
                   x264_crf: int=18, x264_preset: str="veryfast") -> None:
    if not segments:
        raise RuntimeError("Nenhum segmento de fala encontrado – verifique parâmetros do VAD.")

    parts_dir = os.path.join(workdir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    is_windows = platform.system().lower().startswith("win")
    if concat_mode == "auto":
        concat_mode = "filter" if is_windows else "demuxer"

    # 1) Gera partes
    part_paths = []
    for idx, seg in enumerate(tqdm(segments, desc="Gerando cortes")):
        start = max(0.0, seg.start)
        dur = max(0.0, seg.end - seg.start)
        part_path = os.path.join(parts_dir, f"part_{idx:04d}.mp4")

        enc_args, enc_name = pick_encoder_args(
            encoder_preference, is_windows, bitrate, x264_crf, x264_preset
        )
        if precise:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-ss", f"{start:.3f}",
                "-t", f"{dur:.3f}",
                "-avoid_negative_ts", "make_zero",
                "-reset_timestamps", "1",
                "-fflags", "+genpts",
                "-map", "0:v:0?", "-map", "0:a:0?",
            ] + enc_args + [
                "-c:a", "aac", "-b:a", "160k",
                part_path,
            ]
            try:
                run(cmd)
            except RuntimeError:
                if enc_name == "amf":
                    # Fallback para x264
                    # Inclui completamente os argumentos de mapeamento (-map 0:v:0? -map 0:a:0?)
                    run(cmd[:18] + encoder_args_x264(x264_crf, x264_preset) + ["-c:a", "aac", "-b:a", "160k", part_path])
                else:
                    raise
        else:
            # Rápido (copy). Mantém container, mas pode cortar em keyframe.
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", input_video,
                "-t", f"{dur:.3f}",
                "-avoid_negative_ts", "make_zero",
                "-reset_timestamps", "1",
                "-fflags", "+genpts",
                "-c", "copy",
                part_path,
            ]
            run(cmd)
        part_paths.append(part_path)

    # 2) Concatena
    if concat_mode == "demuxer":
        concat_list = os.path.join(workdir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8", newline="") as f:
            for p in part_paths:
                f.write(f"file '{p}'\r\n")
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            out_video,
        ]
        run(cmd_concat)
    else:
        # Concat robusto (reencoda APENAS o arquivo final)
        cmd = ["ffmpeg", "-y"]
        for p in part_paths:
            cmd += ["-i", p]

        streams = "".join([f"[{i}:v:0][{i}:a:0]" for i in range(len(part_paths))])
        filtergraph = f"{streams}concat=n={len(part_paths)}:v=1:a=1[v][a]"

        enc_args, enc_name = pick_encoder_args(
            encoder_preference, is_windows, bitrate, x264_crf, x264_preset
        )
        cmd += [
            "-filter_complex", filtergraph,
            "-map", "[v]", "-map", "[a]",
        ] + enc_args + [
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            out_video,
        ]
        try:
            run(cmd)
        except RuntimeError:
            if enc_name == "amf":
                # Fallback para x264 no final
                cmd_fallback = cmd[:]
                # troca args de vídeo por x264
                idx_v = cmd_fallback.index("-map")  # antes de -map [v] -map [a]
                # Reconstroi após -map [v] -map [a]
                cmd_fallback = cmd_fallback[:idx_v+4] + encoder_args_x264(x264_crf, x264_preset) + [
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_video
                ]
                run(cmd_fallback)
            else:
                raise

# ------------------------------- STT -> SRT -------------------------------

def extract_wav_for_stt(media_path: str, out_wav: str, sr: int = 16000, denoise: bool=False) -> None:
    if denoise:
        af = "highpass=f=80,lowpass=f=12000,afftdn=nf=-28"
        cmd = [
            "ffmpeg", "-y", "-i", media_path,
            "-vn", "-ac", "1", "-ar", str(sr),
            "-af", af,
            "-f", "wav", out_wav,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", media_path,
            "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_wav
        ]
    run(cmd)


def write_srt(segments: List[Tuple[float, float, str]], out_path: str) -> None:
    def srt_ts(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_ts(start)} --> {srt_ts(end)}\n")
            f.write(text.strip() + "\n\n")


def fix_srt_monotonic(segments: List[Tuple[float, float, str]], min_gap: float = 0.06) -> List[Tuple[float, float, str]]:
    segs = sorted(segments, key=lambda x: (x[0], x[1]))
    fixed: List[Tuple[float, float, str]] = []
    last_end = -1e9
    for st, en, tx in segs:
        st = max(st, 0.0)
        en = max(en, st + 0.001)
        if st < last_end + min_gap:
            st = last_end + min_gap
        if en <= st:
            en = st + min_gap
        fixed.append((st, en, tx))
        last_end = en
    return fixed


def stt_vosk(wav_path: str, model_dir: str, lang_hint: Optional[str] = None,
             max_line_chars: int = 70, max_segment_s: float = 5.0) -> List[Tuple[float, float, str]]:
    try:
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError("Vosk não instalado. pip install vosk") from e

    print(f"Carregando modelo Vosk: {model_dir}")
    model = Model(model_dir)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))

    words = []
    for r in results:
        if 'result' in r:
            for w in r['result']:
                start = float(w.get('start', 0.0))
                end = float(w.get('end', start))
                word = (w.get('word', '') or '').strip()
                if end >= start:
                    words.append((start, end, word))

    words.sort(key=lambda x: (x[0], x[1]))

    srt_segments: List[Tuple[float, float, str]] = []
    if not words:
        text = " ".join([r.get('text', '') for r in results]).strip()
        if text:
            srt_segments.append((0.0, max_segment_s, text))
        return srt_segments

    cur_start, cur_end, cur_text = words[0]

    for i in range(1, len(words)):
        w_start, w_end, w = words[i]
        propose = (cur_text + ' ' + w).strip()
        too_long = len(propose) > max_line_chars
        too_slow = (w_end - cur_start) >= max_segment_s
        long_gap = (w_start - cur_end) > 0.6
        if too_long or too_slow or long_gap:
            srt_segments.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = w_start, w_end, w
        else:
            cur_end = max(cur_end, w_end)
            cur_text = propose

    srt_segments.append((cur_start, cur_end, cur_text))
    return srt_segments


def stt_whisper(wav_path: str, model_size: str = "small", device: Optional[str] = None,
                compute_type: str = "int8", language: Optional[str] = None,
                max_line_chars: int = 70, max_segment_s: float = 5.0,
                cpu_threads: Optional[int] = None) -> List[Tuple[float, float, str]]:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("faster-whisper não instalado. pip install faster-whisper") from e

    print(f"Carregando Faster-Whisper: {model_size}")
    model = WhisperModel(model_size, device=device or "cpu", compute_type=compute_type,
                         cpu_threads=cpu_threads or 0)

    segments, _ = model.transcribe(
        wav_path,
        language=language,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    srt_segments: List[Tuple[float, float, str]] = []
    buffer_text: List[str] = []
    cur_start: Optional[float] = None
    last_end: Optional[float] = None
    for seg in segments:
        st = float(seg.start)
        en = float(seg.end)
        tx = seg.text.strip()
        if cur_start is None:
            cur_start, last_end, buffer_text = st, en, [tx]
        else:
            candidate = (" ".join(buffer_text + [tx])).strip()
            too_long = len(candidate) > max_line_chars
            too_slow = (en - cur_start) >= max_segment_s
            if too_long or too_slow:
                srt_segments.append((cur_start, last_end, " ".join(buffer_text).strip()))
                cur_start, buffer_text = st, [tx]
            else:
                buffer_text.append(tx)
            last_end = en
    if buffer_text:
        srt_segments.append((cur_start, last_end, " ".join(buffer_text).strip()))

    return srt_segments

# ------------------------------- Burn-in ---------------------------------------

def ass_color_from_hex(hex_rgb: str, a: str = "00") -> str:
    s = hex_rgb.strip().lstrip('#')
    if len(s) != 6:
        raise ValueError(f"Cor inválida (use #RRGGBB): {hex_rgb}")
    rr, gg, bb = s[0:2], s[2:4], s[4:6]
    return f"&H{a}{bb}{gg}{rr}&"


def resolve_align(align: str) -> int:
    align = str(align).strip().lower()
    if align.isdigit() and 1 <= int(align) <= 9:
        return int(align)
    vert, _, hor = align.partition('-')
    vert_map = {'top': 8, 'middle': 5, 'center': 5, 'bottom': 2}
    hor_map = {'left': -1, 'center': 0, 'right': 1}
    v = vert_map.get(vert, 2)
    h = hor_map.get(hor or 'center', 0)
    base_row = {8: 7, 5: 4, 2: 1}[v]
    return base_row + (h + 1)


def build_force_style(style: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in style.items())


def burn_subtitles(input_video: str, srt_path: str, out_video: str, style: Optional[dict] = None) -> None:
    srt_escaped = escape_path_for_subtitles_filter(os.path.abspath(srt_path))
    if style:
        force = build_force_style(style)
        vf = f"subtitles='{srt_escaped}':force_style='{force}'"
    else:
        vf = f"subtitles='{srt_escaped}'"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", vf,
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        out_video,
    ]
    run(cmd)

# ------------------------------- CLI ------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AutoCut + AutoCaption – Windows/RX580 ready")
    parser.add_argument("-i", "--input", required=True, help="Vídeo de entrada (mp4, mov, mkv, etc.)")
    parser.add_argument("-o", "--output", default="output", help="Pasta de saída (será criada se não existir)")

    # VAD
    parser.add_argument("--vad-aggr", type=int, default=2, choices=[0,1,2,3], help="Agressividade do VAD (0-3). 2 é equilíbrio")
    parser.add_argument("--vad-frame-ms", type=int, default=30, choices=[10,20,30], help="Frame ms para VAD")
    parser.add_argument("--vad-pad-ms", type=int, default=80, help="Padding em ms nas bordas dos segmentos")
    parser.add_argument("--vad-min-seg-ms", type=int, default=400, help="Duração mínima de segmento (ms)")
    parser.add_argument("--vad-merge-gap-ms", type=int, default=250, help="Une segmentos com gaps menores que (ms)")

    # STT
    parser.add_argument("--stt", choices=["vosk", "whisper"], default="vosk", help="Motor de legendas")
    parser.add_argument("--vosk-model", type=str, default=None, help="Pasta do modelo Vosk (ex.: C:\\models\\vosk-model-small-pt-0.3)")
    parser.add_argument("--whisper-model", type=str, default="small", help="Tamanho do Faster-Whisper (tiny|base|small|medium|large)")
    parser.add_argument("--whisper-device", type=str, default="cpu", help="cpu|cuda (Windows/AMD use cpu)")
    parser.add_argument("--whisper-compute-type", type=str, default="int8", help="int8|int8_float16|float16|float32")
    parser.add_argument("--whisper-cpu-threads", type=int, default=0, help="Threads p/ CPU (0=auto)")
    parser.add_argument("--language", type=str, default=None, help="Dica de idioma (pt, en, es, etc.)")

    # Pós / estilo
    parser.add_argument("--burn", action="store_true", help="Queima as legendas no vídeo final")
    parser.add_argument("--font-name", type=str, default="Arial", help="Fonte das legendas (ASS FontName)")
    parser.add_argument("--font-size", type=int, default=36, help="Tamanho da fonte")
    parser.add_argument("--font-color", type=str, default="#FFFFFF", help="Cor do texto (#RRGGBB)")
    parser.add_argument("--outline", type=float, default=2.0, help="Espessura do contorno (0-10)")
    parser.add_argument("--outline-color", type=str, default="#000000", help="Cor do contorno (#RRGGBB)")
    parser.add_argument("--shadow", type=float, default=0.0, help="Sombra (0-10)")
    parser.add_argument("--align", type=str, default="bottom-center", help="Alinhamento 1-9 ou ex.: top-left")
    parser.add_argument("--margin-v", type=int, default=40, help="Margem vertical (px)")

    # Estabilidade e performance
    parser.add_argument("--precise-cut", action="store_true", help="Reencoda cortes para precisão e sincronismo")
    parser.add_argument("--denoise", action="store_true", help="Aplica denoise leve no áudio antes do STT")
    parser.add_argument("--srt-max-line-chars", type=int, default=70, help="Máx. caracteres por legenda")
    parser.add_argument("--srt-max-seg-s", type=float, default=5.0, help="Máx. duração de um bloco de legenda (s)")
    parser.add_argument("--srt-min-gap-ms", type=int, default=60, help="Gap mínimo entre legendas (ms)")

    # Concat e Encoder
    parser.add_argument("--concat-mode", choices=["auto","demuxer","filter"], default="auto", help="Modo de concat.")
    parser.add_argument("--encoder", choices=["auto","x264","amf"], default="auto", help="Preferência de encoder de vídeo")
    parser.add_argument("--bitrate", type=str, default="8M", help="Bitrate alvo para AMF (ex.: 8M)")
    parser.add_argument("--x264-crf", type=int, default=18, help="Qualidade CRF do x264")
    parser.add_argument("--x264-preset", type=str, default="veryfast", help="Preset do x264")

    args = parser.parse_args()

    ensure_ffmpeg()

    in_path = os.path.abspath(args.input)
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    edited_video = os.path.join(out_dir, f"{base}_editado.mp4")
    srt_path = os.path.join(out_dir, f"{base}_editado.srt")
    burned_video = os.path.join(out_dir, f"{base}_final_legendado.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Extrai WAV 16 kHz mono para VAD
        wav_vad = os.path.join(tmp, "audio_vad.wav")
        extract_wav_mono_16k(in_path, wav_vad)

        # 2) Gera segmentos de fala
        segs = vad_segments(
            wav_vad,
            aggressiveness=args.vad_aggr,
            frame_ms=args.vad_frame_ms,
            pad_ms=args.vad_pad_ms,
            min_seg_ms=args.vad_min_seg_ms,
            merge_gap_ms=args.vad_merge_gap_ms,
        )
        if not segs:
            raise RuntimeError("Nenhum segmento de fala encontrado. Tente reduzir --vad-aggr ou --vad-min-seg-ms.")

        # Salva debug
        dbg_json = os.path.join(out_dir, f"{base}_segments.json")
        with open(dbg_json, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(s) for s in segs], f, ensure_ascii=False, indent=2)
        print(f"Segmentos salvos em: {dbg_json}")

        # 3) Corta e concatena (concat filter no Windows por padrão)
        cut_and_concat(
            in_path, segs, edited_video, tmp,
            precise=args.precise_cut,
            concat_mode=args.concat_mode,
            encoder_preference=args.encoder,
            bitrate=args.bitrate,
            x264_crf=args.x264_crf,
            x264_preset=args.x264_preset,
        )
        print(f"Vídeo sem silêncios: {edited_video}")

        # 4) STT no vídeo editado -> SRT
        wav_stt = os.path.join(tmp, "audio_stt.wav")
        extract_wav_for_stt(edited_video, wav_stt, sr=16000, denoise=args.denoise)

        if args.stt == "vosk":
            if not args.vosk_model:
                raise RuntimeError("Para usar Vosk, informe --vosk-model apontando para a pasta do modelo.")
            raw_segs = stt_vosk(
                wav_stt,
                args.vosk_model,
                lang_hint=args.language,
                max_line_chars=args.srt_max_line_chars,
                max_segment_s=args.srt_max_seg_s,
            )
        else:
            raw_segs = stt_whisper(
                wav_stt,
                model_size=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                language=args.language,
                max_line_chars=args.srt_max_line_chars,
                max_segment_s=args.srt_max_seg_s,
                cpu_threads=(args.whisper_cpu_threads or 0),
            )

        # 4.1) Corrige ordem/overlap
        srt_segs = fix_srt_monotonic(raw_segs, min_gap=args.srt_min_gap_ms/1000.0)

        write_srt(srt_segs, srt_path)
        print(f"Legenda gerada: {srt_path}")

        # 5) (Opcional) Burn-in
        if args.burn:
            style = {
                "FontName": args.font_name,
                "FontSize": args.font_size,
                "PrimaryColour": ass_color_from_hex(args.font_color),
                "OutlineColour": ass_color_from_hex(args.outline_color),
                "Outline": args.outline,
                "Shadow": args.shadow,
                "Alignment": resolve_align(args.align),
                "MarginV": args.margin_v,
            }
            burn_subtitles(edited_video, srt_path, burned_video, style=style)
            print(f"Vídeo final legendado: {burned_video}")

    print("\nVídeo finalizado com sucesso!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print("Erro:", e)
        sys.exit(1)
