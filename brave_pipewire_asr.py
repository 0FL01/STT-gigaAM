#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from collections import deque

import numpy as np
import onnx_asr


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def robust_recognize(model, audio: np.ndarray, sample_rate: int) -> str:
    res = model.recognize(audio, sample_rate=sample_rate)

    if isinstance(res, str):
        return res.strip()

    if hasattr(res, "text"):
        return str(res.text).strip()

    try:
        parts = []
        for x in res:
            s = str(x).strip()
            if s:
                parts.append(s)
        return " ".join(parts).strip()
    except TypeError:
        return str(res).strip()


def downmix_and_resample(block: np.ndarray, in_channels: int, in_rate: int, out_rate: int) -> np.ndarray:
    if in_channels > 1:
        block = block.reshape(-1, in_channels).mean(axis=1)

    if in_rate == out_rate:
        return block.astype(np.float32, copy=False)

    if in_rate == 48000 and out_rate == 16000:
        n = (len(block) // 3) * 3
        if n == 0:
            return np.empty(0, dtype=np.float32)
        return block[:n].reshape(-1, 3).mean(axis=1).astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=len(block), endpoint=False)
    out_len = int(len(block) * out_rate / in_rate)
    if out_len <= 0:
        return np.empty(0, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x_new, x_old, block).astype(np.float32)


def find_stream_target(browser_pattern: str) -> str | None:
    out = subprocess.check_output(["pw-dump"], text=True)
    objects = json.loads(out)

    candidates = []

    for obj in objects:
        info = obj.get("info") or {}
        props = info.get("props") or {}

        if props.get("media.class") != "Stream/Output/Audio":
            continue

        app_name = str(props.get("application.name", ""))
        proc_bin = str(props.get("application.process.binary", ""))
        node_name = str(props.get("node.name", ""))
        media_name = str(props.get("media.name", ""))
        node_desc = str(props.get("node.description", ""))

        haystack = " ".join([app_name, proc_bin, node_name, media_name, node_desc])

        if not re.search(browser_pattern, haystack, flags=re.IGNORECASE):
            continue

        target = app_name or node_name
        if not target:
            continue

        score = 0
        if re.search(browser_pattern, proc_bin, flags=re.IGNORECASE):
            score += 5
        if re.search(browser_pattern, app_name, flags=re.IGNORECASE):
            score += 5
        if str(props.get("media.role", "")).lower() == "communication":
            score += 10

        candidates.append((score, target, props))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def spawn_pw_record(
    target: str,
    rate: int,
    channels: int,
    latency: str,
    capture_sink: bool,
) -> subprocess.Popen:
    cmd = [
        "pw-record",
        "--target", target,
        "--rate", str(rate),
        "--channels", str(channels),
        "--format", "f32",
        "--raw",
        "--latency", latency,
    ]

    if channels == 1:
        cmd += ["--channel-map", "mono"]
    elif channels == 2:
        cmd += ["--channel-map", "stereo"]

    if capture_sink:
        cmd += ["-P", '{ "stream.capture.sink": true }']

    cmd += ["-"]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def maybe_emit(text: str, last_text: str | None, prefix: str = "") -> str | None:
    text = text.strip()
    if not text:
        return last_text

    norm = normalize_text(text)
    if norm and norm != last_text:
        print(f"{prefix}{text}", flush=True)
        return norm
    return last_text


def run_capture_loop(args, model) -> None:
    input_block_frames = int(args.input_rate * args.block_sec)
    input_block_bytes = input_block_frames * args.input_channels * 4

    preroll_blocks = max(1, int(args.preroll_sec / args.block_sec))
    rms_log_every_blocks = max(1, int(args.rms_log_sec / args.block_sec))

    last_target = None
    global_last_text = None

    while True:
        target = args.target or find_stream_target(args.browser)

        if not target:
            if args.debug:
                eprint("DEBUG no target yet; waiting...")
            time.sleep(1.0)
            continue

        if target != last_target:
            eprint(f"Attached to PipeWire target: {target}")
            last_target = target

        proc = spawn_pw_record(
            target=target,
            rate=args.input_rate,
            channels=args.input_channels,
            latency=args.latency,
            capture_sink=args.capture_sink,
        )

        assert proc.stdout is not None
        assert proc.stderr is not None

        preroll = deque(maxlen=preroll_blocks)
        utterance = []
        in_speech = False
        silence_sec = 0.0
        utterance_sec = 0.0
        next_emit_at = args.emit_every_sec
        loop_idx = 0

        try:
            while True:
                buf = read_exact(proc.stdout, input_block_bytes)
                if len(buf) == 0:
                    if args.debug:
                        eprint("DEBUG pw-record EOF")
                    break
                if len(buf) < input_block_bytes:
                    if args.debug:
                        eprint(f"DEBUG short read: got={len(buf)} want={input_block_bytes}")
                    break

                raw = np.frombuffer(buf, dtype=np.float32)
                mono16 = downmix_and_resample(
                    raw,
                    in_channels=args.input_channels,
                    in_rate=args.input_rate,
                    out_rate=args.asr_rate,
                )

                if mono16.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(mono16 * mono16) + 1e-12))
                voiced = rms >= args.silence_rms

                preroll.append(mono16)

                if args.debug and loop_idx % rms_log_every_blocks == 0:
                    eprint(
                        f"DEBUG rms={rms:.5f} voiced={int(voiced)} "
                        f"in_speech={int(in_speech)} utt_sec={utterance_sec:.2f}"
                    )
                loop_idx += 1

                if voiced and not in_speech:
                    utterance = list(preroll)
                    utterance_sec = sum(len(x) for x in utterance) / args.asr_rate
                    in_speech = True
                    silence_sec = 0.0
                    next_emit_at = max(args.first_emit_sec, args.emit_every_sec)
                    if args.debug:
                        eprint(f"DEBUG speech_start rms={rms:.5f} preroll_sec={utterance_sec:.2f}")
                    continue

                if not in_speech:
                    continue

                utterance.append(mono16)
                utterance_sec += len(mono16) / args.asr_rate

                if voiced:
                    silence_sec = 0.0
                else:
                    silence_sec += args.block_sec

                if utterance_sec >= next_emit_at:
                    audio = np.concatenate(utterance)
                    if len(audio) / args.asr_rate >= args.min_utt_sec:
                        text = robust_recognize(model, audio, sample_rate=args.asr_rate)
                        if args.debug:
                            eprint(f"DEBUG partial utt_sec={utterance_sec:.2f} text={text!r}")
                        global_last_text = maybe_emit(text, global_last_text)
                    next_emit_at += args.emit_every_sec

                if silence_sec >= args.tail_silence_sec or utterance_sec >= args.max_utt_sec:
                    audio = np.concatenate(utterance)
                    if len(audio) / args.asr_rate >= args.min_utt_sec:
                        text = robust_recognize(model, audio, sample_rate=args.asr_rate)
                        if args.debug:
                            eprint(
                                f"DEBUG final utt_sec={utterance_sec:.2f} "
                                f"silence_sec={silence_sec:.2f} text={text!r}"
                            )
                        global_last_text = maybe_emit(text, global_last_text)

                    preroll.clear()
                    utterance = []
                    in_speech = False
                    silence_sec = 0.0
                    utterance_sec = 0.0
                    next_emit_at = args.emit_every_sec

        finally:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                proc.kill()

        time.sleep(0.5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--browser", default=r"brave|brave-browser")
    p.add_argument("--target", default=None, help="explicit PipeWire target, e.g. Brave or bluez_output...")
    p.add_argument("--capture-sink", action="store_true", help="capture sink monitor instead of app stream")
    p.add_argument("--model", default="gigaam-v3-e2e-ctc")
    p.add_argument("--quantization", default=None)

    p.add_argument("--input-rate", type=int, default=48000)
    p.add_argument("--input-channels", type=int, default=2)
    p.add_argument("--asr-rate", type=int, default=16000)
    p.add_argument("--latency", default="50ms")

    p.add_argument("--block-sec", type=float, default=0.20)
    p.add_argument("--preroll-sec", type=float, default=0.40)

    p.add_argument("--silence-rms", type=float, default=0.0035)
    p.add_argument("--tail-silence-sec", type=float, default=0.80)
    p.add_argument("--min-utt-sec", type=float, default=0.60)
    p.add_argument("--first-emit-sec", type=float, default=2.0)
    p.add_argument("--emit-every-sec", type=float, default=2.5)
    p.add_argument("--max-utt-sec", type=float, default=6.0)

    p.add_argument("--debug", action="store_true")
    p.add_argument("--rms-log-sec", type=float, default=1.0)
    return p


def main() -> int:
    args = build_parser().parse_args()

    eprint(f"Loading model: {args.model!r} quantization={args.quantization!r}")
    model = onnx_asr.load_model(args.model, quantization=args.quantization)
    eprint("Ready. Waiting for audio stream...")

    run_capture_loop(args, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
