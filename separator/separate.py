#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
    raise SystemExit(
        "SAM-Audio dependencies are most reliable with Python 3.11. "
        "Create a Python 3.11 virtualenv and run this script from there."
    )

# SAM-Audio's DAC/VAE path can hit Conv1d kernels that MPS does not support.
# PyTorch reads this before import time, though this script defaults to CPU below.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchaudio
from safetensors.torch import load_file as load_safetensors


def install_torchcodec_audio_decoder_compat() -> None:
    """Provide AudioDecoder for torchcodec 0.1, which only exposes video decoding."""
    import torchcodec.decoders as torchcodec_decoders

    if hasattr(torchcodec_decoders, "AudioDecoder"):
        return

    class AudioSamples:
        def __init__(self, data: torch.Tensor):
            self.data = data

    class AudioDecoder:
        def __init__(
            self,
            source: str,
            *,
            sample_rate: int | None = None,
            num_channels: int | None = None,
            **_: Any,
        ):
            waveform, loaded_sample_rate = torchaudio.load(source)

            if sample_rate is not None and loaded_sample_rate != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=loaded_sample_rate,
                    new_freq=sample_rate,
                )

            if num_channels == 1 and waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif num_channels is not None and waveform.shape[0] == 1 and num_channels > 1:
                waveform = waveform.repeat(num_channels, 1)
            elif num_channels is not None and waveform.shape[0] != num_channels:
                waveform = waveform[:num_channels]

            self._data = waveform.contiguous()

        def get_all_samples(self) -> AudioSamples:
            return AudioSamples(self._data)

    torchcodec_decoders.AudioDecoder = AudioDecoder


install_torchcodec_audio_decoder_compat()
from sam_audio import SAMAudio, SAMAudioProcessor


MODEL_ID = "facebook/sam-audio-small"


@dataclass(frozen=True)
class DrumChannel:
    label: str
    prompt: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.prompt.lower()).strip("_")


CHANNELS = [
    DrumChannel("Bass drum", "bass drum"),
    DrumChannel("Snare drum", "snare drum"),
    DrumChannel("Hi-hat", "hi-hat"),
    DrumChannel("Ride cymbal", "ride cymbal"),
    DrumChannel("Crash cymbal", "crash cymbal"),
    DrumChannel("High tom", "high tom"),
    DrumChannel("Mid tom", "mid tom"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a song into drum-kit channels with SAM-Audio residual chaining."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../input/song.mp3"),
        help="Input song path, usually ../input/song.mp3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../web/public/stems"),
        help="Directory for generated WAV stems and manifest.json",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"SAM-Audio model id or local checkpoint. Default: {MODEL_ID}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--predict-spans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let SAM-Audio predict useful temporal spans for each prompt.",
    )
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Higher values can improve quality but cost more time and memory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stem files.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Process audio in chunks of this length to limit GPU memory. "
            "Defaults to 30 on CUDA, 0 (full file) elsewhere."
        ),
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Crossfade overlap between chunks. Default: 1.0",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_sample_rate(processor: SAMAudioProcessor) -> int:
    sample_rate = getattr(processor, "audio_sampling_rate", None)
    if sample_rate is None:
        sample_rate = getattr(processor, "sampling_rate", None)
    if sample_rate is None:
        sample_rate = 44_100
    return int(sample_rate)


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def is_mps_out_of_memory(exc: RuntimeError) -> bool:
    return "mps" in str(exc).lower() and "out of memory" in str(exc).lower()


def is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudamalloc" in message)


def resolve_chunk_seconds(requested: float | None, device: torch.device) -> float:
    if requested is not None:
        return max(0.0, requested)
    if device.type == "cuda":
        return 30.0
    return 0.0


def load_mono_waveform(audio_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, loaded_sample_rate = torchaudio.load(str(audio_path))
    if loaded_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=loaded_sample_rate,
            new_freq=sample_rate,
        )
    return as_channels_first(waveform.mean(dim=0, keepdim=True))


def iter_audio_chunks(
    waveform: torch.Tensor,
    chunk_samples: int,
    overlap_samples: int,
) -> list[torch.Tensor]:
    total_samples = waveform.shape[-1]
    if total_samples <= chunk_samples:
        return [waveform]

    step = max(1, chunk_samples - overlap_samples)
    chunks: list[torch.Tensor] = []
    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunks.append(waveform[..., start:end].contiguous())
        if end >= total_samples:
            break
    return chunks


def crossfade_concat(chunks: list[torch.Tensor], overlap_samples: int) -> torch.Tensor:
    if not chunks:
        raise ValueError("Expected at least one audio chunk.")
    if len(chunks) == 1 or overlap_samples <= 0:
        return torch.cat(chunks, dim=-1)

    result = chunks[0]
    fade_out = torch.linspace(1.0, 0.0, overlap_samples)
    fade_in = 1.0 - fade_out
    for chunk in chunks[1:]:
        overlap = min(overlap_samples, result.shape[-1], chunk.shape[-1])
        if overlap <= 1:
            result = torch.cat([result, chunk], dim=-1)
            continue

        tail = result[..., -overlap:]
        head = chunk[..., :overlap]
        blended = tail * fade_out[:overlap] + head * fade_in[:overlap]
        result = torch.cat([result[..., :-overlap], blended, chunk[..., overlap:]], dim=-1)
    return result.contiguous()


def resolve_model_path(model_arg: str) -> str:
    model_path = Path(model_arg).expanduser()

    if model_path.exists():
        if model_path.is_file():
            if model_path.suffix != ".safetensors":
                raise SystemExit(
                    f"Local model path is a file, but only .safetensors files are supported: {model_path}"
                )
            if not (model_path.parent / "config.json").exists():
                raise SystemExit(
                    f"Local safetensors model needs a config.json next to it: {model_path.parent}"
                )
            return str(model_path.parent.resolve())

        if not (model_path / "config.json").exists():
            raise SystemExit(f"Local model directory is missing config.json: {model_path}")
        return str(model_path.resolve())

    return model_arg


def get_ranker_config_overrides(reranking_candidates: int) -> dict[str, Any]:
    # This script is audio-only, so SAM-Audio's video/ImageBind ranker is never used.
    overrides: dict[str, Any] = {"visual_ranker": None}
    if reranking_candidates <= 1:
        overrides["text_ranker"] = None
    return overrides


def sam_audio_from_pretrained(model_name_or_path: str, **model_kwargs: Any) -> SAMAudio:
    return SAMAudio._from_pretrained(
        model_id=model_name_or_path,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        **model_kwargs,
    )


def load_sam_audio_model(
    model_name_or_path: str,
    device: torch.device,
    reranking_candidates: int,
) -> SAMAudio:
    ranker_overrides = get_ranker_config_overrides(reranking_candidates)
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return sam_audio_from_pretrained(model_name_or_path, **ranker_overrides).to(device).eval()

    checkpoint_path = model_path / "checkpoint.pt"
    if checkpoint_path.exists():
        return sam_audio_from_pretrained(str(model_path), **ranker_overrides).to(device).eval()

    safetensors_paths = sorted(model_path.glob("*.safetensors"))
    if not safetensors_paths:
        raise SystemExit(
            f"Local model directory needs checkpoint.pt or a .safetensors file: {model_path}"
        )
    if len(safetensors_paths) > 1:
        names = ", ".join(path.name for path in safetensors_paths)
        raise SystemExit(f"Local model directory has multiple .safetensors files: {names}")

    config_dict = json.loads((model_path / "config.json").read_text())
    config_dict.update(ranker_overrides)
    config = SAMAudio.config_cls(**config_dict)
    model = SAMAudio(config)
    state_dict = {
        name: value
        for name, value in load_safetensors(str(safetensors_paths[0]), device="cpu").items()
        if not name.endswith("._scale")
    }
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    return model.to(device).eval()


def get_audio_field(result: Any, name: str) -> torch.Tensor:
    if hasattr(result, name):
        value = getattr(result, name)
    elif isinstance(result, dict) and name in result:
        value = result[name]
    else:
        raise AttributeError(f"SAM-Audio result did not include {name!r}.")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def as_channels_first(waveform: torch.Tensor) -> torch.Tensor:
    audio = waveform.detach().float().cpu()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio tensor with 1 or 2 dimensions, got {tuple(audio.shape)}")
    if audio.shape[0] > audio.shape[1]:
        audio = audio.transpose(0, 1)
    return audio.contiguous()


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.98) -> torch.Tensor:
    peak = waveform.abs().max().clamp_min(1e-8)
    if peak <= target_peak:
        return waveform
    return waveform * (target_peak / peak)


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int, normalize: bool) -> None:
    audio = as_channels_first(waveform)
    if normalize:
        audio = peak_normalize(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate)


def separate_waveform(
    *,
    model: SAMAudio,
    processor: SAMAudioProcessor,
    waveform: torch.Tensor,
    prompt: str,
    device: torch.device,
    predict_spans: bool,
    reranking_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = processor(audios=[waveform], descriptions=[prompt])
    if hasattr(batch, "to"):
        batch = batch.to(device)

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

    target = as_channels_first(get_audio_field(result, "target"))
    residual = as_channels_first(get_audio_field(result, "residual"))
    del batch, result
    clear_device_cache(device)
    return target, residual


def separate_channel(
    *,
    model: SAMAudio,
    processor: SAMAudioProcessor,
    audio_path: Path,
    prompt: str,
    device: torch.device,
    predict_spans: bool,
    reranking_candidates: int,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_seconds <= 0:
        return separate_waveform(
            model=model,
            processor=processor,
            waveform=load_mono_waveform(audio_path, sample_rate),
            prompt=prompt,
            device=device,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

    waveform = load_mono_waveform(audio_path, sample_rate)
    chunk_samples = max(1, int(chunk_seconds * sample_rate))
    overlap_samples = max(0, int(overlap_seconds * sample_rate))
    chunks = iter_audio_chunks(waveform, chunk_samples, overlap_samples)
    if len(chunks) == 1:
        return separate_waveform(
            model=model,
            processor=processor,
            waveform=chunks[0],
            prompt=prompt,
            device=device,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

    print(
        f"  Processing {len(chunks)} chunks "
        f"({chunk_seconds:.1f}s each, {overlap_seconds:.1f}s overlap)..."
    )
    target_chunks: list[torch.Tensor] = []
    residual_chunks: list[torch.Tensor] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"    chunk {index}/{len(chunks)}")
        target, residual = separate_waveform(
            model=model,
            processor=processor,
            waveform=chunk,
            prompt=prompt,
            device=device,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )
        target_chunks.append(target)
        residual_chunks.append(residual)

    return (
        crossfade_concat(target_chunks, overlap_samples),
        crossfade_concat(residual_chunks, overlap_samples),
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if output_dir.exists() and not args.overwrite:
        existing = [output_dir / f"{channel.slug}.wav" for channel in CHANNELS]
        if any(path.exists() for path in existing):
            raise SystemExit(
                f"Stem files already exist in {output_dir}. "
                "Pass --overwrite to replace them."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model_name_or_path = resolve_model_path(args.model)
    print(f"Loading {model_name_or_path} on {device}...")
    processor = SAMAudioProcessor.from_pretrained(model_name_or_path)
    try:
        model = load_sam_audio_model(model_name_or_path, device, args.reranking_candidates)
    except RuntimeError as exc:
        if device.type == "mps" and is_mps_out_of_memory(exc):
            clear_device_cache(device)
            if args.device == "auto":
                device = torch.device("cpu")
                print("MPS ran out of memory while loading the model; retrying on cpu...")
                model = load_sam_audio_model(
                    model_name_or_path,
                    device,
                    args.reranking_candidates,
                )
            else:
                raise SystemExit(
                    "MPS ran out of memory while loading the model. Retry with "
                    "`--device cpu`, or start Python with "
                    "`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` if you accept the "
                    "risk of macOS instability under memory pressure."
                ) from exc
        else:
            raise
    sample_rate = get_sample_rate(processor)
    chunk_seconds = resolve_chunk_seconds(args.chunk_seconds, device)
    overlap_seconds = max(0.0, args.overlap_seconds)
    if chunk_seconds > 0:
        print(
            f"Using chunked separation: {chunk_seconds:.1f}s chunks "
            f"with {overlap_seconds:.1f}s overlap."
        )

    manifest_channels = []
    current_audio_path = input_path

    with tempfile.TemporaryDirectory(prefix="sam-audio-residuals-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for index, channel in enumerate(CHANNELS, start=1):
            print(f"[{index}/{len(CHANNELS)}] Separating {channel.label} ({channel.prompt!r})...")
            try:
                target, residual = separate_channel(
                    model=model,
                    processor=processor,
                    audio_path=current_audio_path,
                    prompt=channel.prompt,
                    device=device,
                    predict_spans=args.predict_spans,
                    reranking_candidates=args.reranking_candidates,
                    sample_rate=sample_rate,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                )
            except RuntimeError as exc:
                if device.type == "mps" and is_mps_out_of_memory(exc):
                    raise SystemExit(
                        "MPS ran out of memory during separation. Retry with "
                        "`--device cpu` for the most reliable run, or try "
                        "`--no-predict-spans` to reduce peak memory on MPS. "
                        "As a last resort, start Python with "
                        "`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, but that can "
                        "make macOS unstable if physical memory is exhausted."
                    ) from exc
                if device.type == "cuda" and is_cuda_out_of_memory(exc):
                    raise SystemExit(
                        "CUDA ran out of memory during separation. Retry with a "
                        "smaller `--chunk-seconds` value (for example 15), "
                        "`--no-predict-spans`, or `--reranking-candidates 1`."
                    ) from exc
                raise

            target_audio = as_channels_first(target)
            output_path = output_dir / f"{channel.slug}.wav"
            save_wav(output_path, target_audio, sample_rate, normalize=True)

            duration_seconds = target_audio.shape[-1] / sample_rate
            manifest_channels.append(
                {
                    "id": channel.slug,
                    "label": channel.label,
                    "prompt": channel.prompt,
                    "file": f"/stems/{channel.slug}.wav",
                    "duration": duration_seconds,
                    "sampleRate": sample_rate,
                }
            )

            residual_path = temp_dir / f"{index:02d}_{channel.slug}_residual.wav"
            save_wav(residual_path, residual, sample_rate, normalize=False)
            current_audio_path = residual_path
            del target, residual, target_audio
            clear_device_cache(device)

    manifest = {
        "model": model_name_or_path,
        "source": str(input_path),
        "strategy": "residual-chaining",
        "sampleRate": sample_rate,
        "channels": manifest_channels,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Wrote {len(manifest_channels)} stems and {manifest_path}.")


if __name__ == "__main__":
    main()
