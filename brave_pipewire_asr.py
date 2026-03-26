#!/usr/bin/env python3
import argparse
import json
import os
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from textwrap import wrap

import numpy as np
import onnx_asr
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class RawKeyboard:
    def __init__(self, stream):
        self.stream = stream
        self.fd = stream.fileno()
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)


def read_key(fd: int, timeout: float = 0.1) -> str | None:
    r, _, _ = select.select([fd], [], [], timeout)
    if not r:
        return None

    data = os.read(fd, 1)
    if not data:
        return None

    if data == b"\x1b":
        time.sleep(0.01)
        while True:
            r, _, _ = select.select([fd], [], [], 0)
            if not r:
                break
            data += os.read(fd, 1)

    s = data.decode(errors="ignore")

    mapping = {
        "k": "up",
        "j": "down",
        "g": "home",
        "G": "end",
        "f": "follow",
        "q": "quit",
        " ": "pagedown",
        "b": "pageup",
        "\x1b[A": "up",
        "\x1b[B": "down",
        "\x1b[5~": "pageup",
        "\x1b[6~": "pagedown",
        "\x1b[H": "home",
        "\x1b[F": "end",
        "\x1bOH": "home",
        "\x1bOF": "end",
    }
    return mapping.get(s)


@dataclass
class SubtitleEntry:
    ts: str
    text: str


class TerminalUI:
    def __init__(
        self,
        target: str = "",
        history_max: int = 500,
        log_file: str | None = None,
    ):
        self.console = Console(stderr=True, soft_wrap=True)
        self.target = target
        self.state = "idle"
        self.rms = 0.0
        self.voiced = False
        self.in_speech = False
        self.partial = ""
        self.last_final = ""
        self.history = deque(maxlen=history_max)
        self.model = ""
        self.log_fp = open(log_file, "a", encoding="utf-8") if log_file else None

        self.follow_history = True
        self.history_scroll = 0
        self.quit_requested = False
        self._lock = threading.RLock()

        self.live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=8,
            redirect_stdout=False,
            redirect_stderr=False,
            transient=False,
            screen=False,
        )

    def close(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def set_model(self, model_name: str):
        self.model = model_name
        self.refresh()

    def set_target(self, target: str):
        self.target = target
        self.refresh()

    def set_state(self, state: str):
        self.state = state
        self.refresh()

    def set_levels(self, rms: float, voiced: bool, in_speech: bool):
        self.rms = rms
        self.voiced = voiced
        self.in_speech = in_speech
        self.refresh()

    def set_partial(self, text: str):
        self.partial = text.strip()
        self.refresh()

    def add_final(self, text: str):
        text = text.strip()
        if not text:
            return

        ts = datetime.now().strftime("%H:%M:%S")

        prev_count = None
        if not self.follow_history:
            prev_count = len(self._flatten_history_lines(self.console.size.width))

        self.last_final = text
        self.history.append(SubtitleEntry(ts=ts, text=text))
        self.partial = ""

        if self.log_fp:
            self.log_fp.write(f"[{ts}] {text}\n")
            self.log_fp.flush()

        if prev_count is not None:
            new_count = len(self._flatten_history_lines(self.console.size.width))
            self.history_scroll += max(0, new_count - prev_count)

        self.refresh()

    def rms_bar(self, width: int = 16) -> str:
        level = min(1.0, self.rms / 0.03)
        filled = int(level * width)
        return "█" * filled + "░" * (width - filled)

    def _iter_history_items(self):
        for item in self.history:
            if isinstance(item, str):
                yield "", item
            else:
                yield getattr(item, "ts", ""), getattr(item, "text", str(item))

    def _flatten_history_lines(self, width: int) -> list[str]:
        body_width = max(20, width - 8)
        lines: list[str] = []

        for ts, text in self._iter_history_items():
            prefix = f"[{ts}] " if ts else ""
            usable = max(10, body_width - len(prefix))
            chunks = wrap(
                text,
                width=usable,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [text]

            lines.append(prefix + chunks[0])
            for chunk in chunks[1:]:
                lines.append(" " * len(prefix) + chunk)

        return lines

    def _history_viewport(self, width: int, visible_height: int) -> list[str]:
        all_lines = self._flatten_history_lines(width)
        if not all_lines:
            return ["История пока пуста"]

        max_scroll = max(0, len(all_lines) - visible_height)
        scroll = 0 if self.follow_history else min(self.history_scroll, max_scroll)

        end = len(all_lines) - scroll
        start = max(0, end - visible_height)
        return all_lines[start:end]

    def scroll_up(self, lines: int = 1):
        self.follow_history = False
        self.history_scroll += max(1, lines)
        self.refresh()

    def scroll_down(self, lines: int = 1):
        self.history_scroll = max(0, self.history_scroll - max(1, lines))
        if self.history_scroll == 0:
            self.follow_history = True
        self.refresh()

    def page_up(self):
        self.scroll_up(max(5, self.console.size.height // 3))

    def page_down(self):
        self.scroll_down(max(5, self.console.size.height // 3))

    def scroll_home(self):
        all_lines = self._flatten_history_lines(self.console.size.width)
        visible_h = max(3, max(8, self.console.size.height - 11) - 2)
        self.follow_history = False
        self.history_scroll = max(0, len(all_lines) - visible_h)
        self.refresh()

    def scroll_end(self):
        self.history_scroll = 0
        self.follow_history = True
        self.refresh()

    def handle_key(self, key: str):
        if key == "up":
            self.scroll_up(1)
        elif key == "down":
            self.scroll_down(1)
        elif key == "pageup":
            self.page_up()
        elif key == "pagedown":
            self.page_down()
        elif key == "home":
            self.scroll_home()
        elif key == "end":
            self.scroll_end()
        elif key == "follow":
            self.scroll_end()
        elif key == "quit":
            self.quit_requested = True

    def render(self):
        state_style = {
            "idle": "yellow",
            "listening": "green",
            "waiting": "cyan",
            "error": "bold red",
        }.get(self.state, "white")

        # Компактная верхняя строка вместо большого блока
        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(justify="right")
        top.add_row(
            f"[bold]GigaAM[/bold] [dim]{self.model}[/dim]  [dim]{self.target or '-'}[/dim]",
            f"[{state_style}]{self.state.upper()}[/{state_style}]",
        )
        top.add_row(
            f"[dim]signal:[/dim] {self.rms_bar()}  [dim]rms:[/dim] {self.rms:.5f}",
            f"[dim]voiced:[/dim] {int(self.voiced)}  "
            f"[dim]speech:[/dim] {int(self.in_speech)}",
        )

        current = Table.grid(expand=True)
        current.add_column()
        current.add_row(Text(self.partial or "…", style="bold white"))
        current.add_row(Text(self.last_final or "—", style="green"))

        term_w = self.console.size.width
        term_h = self.console.size.height

        layout = Layout()
        layout.split_column(
            Layout(Panel(top, title="Status", border_style="blue"), size=5),
            Layout(Panel(current, title="Now", border_style="yellow"), size=6),
            Layout(name="history"),
        )

        history_height = max(8, term_h - 11)
        visible_h = max(3, history_height - 2)

        history_lines = self._history_viewport(term_w, visible_h)
        history_text = Text("\n".join(history_lines), style="green")

        mode = (
            "FOLLOW"
            if self.follow_history and self.history_scroll == 0
            else f"SCROLL +{self.history_scroll}"
        )
        layout["history"].update(
            Panel(
                history_text,
                title=f"Subtitles ({len(self.history)}/{self.history.maxlen}) [{mode}]",
                subtitle="\u2191/k older  \u2193/j newer  PgUp/PgDn  g/G  f follow  q quit",
                border_style="magenta",
            )
        )
        return layout

    def refresh(self):
        if hasattr(self, "live"):
            self.live.update(self.render(), refresh=True)


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


def downmix_and_resample(
    block: np.ndarray, in_channels: int, in_rate: int, out_rate: int
) -> np.ndarray:
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


@dataclass
class PWTarget:
    object_serial: str
    node_name: str
    app_name: str
    media_name: str
    state: str


def pw_dump_objects():
    out = subprocess.check_output(["pw-dump"], text=True)
    return json.loads(out)


def iter_output_streams():
    for obj in pw_dump_objects():
        info = obj.get("info") or {}
        props = info.get("props") or {}

        if props.get("media.class") != "Stream/Output/Audio":
            continue

        object_serial = props.get("object.serial")
        node_name = props.get("node.name")
        if not object_serial or not node_name:
            continue

        yield PWTarget(
            object_serial=str(object_serial),
            node_name=str(node_name),
            app_name=str(props.get("application.name", "")),
            media_name=str(props.get("media.name", "")),
            state=str(info.get("state", "")),
        )


STATE_SCORE = {
    "running": 30,
    "paused": 20,
    "idle": 10,
}


def find_stream_target(pattern: str) -> PWTarget | None:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates = []

    for s in iter_output_streams():
        haystack = " ".join([s.node_name, s.app_name, s.media_name])

        if not rx.search(haystack):
            continue

        score = STATE_SCORE.get(s.state.lower(), 0)

        if rx.search(s.node_name):
            score += 20
        if rx.search(s.app_name):
            score += 10
        if rx.search(s.media_name):
            score += 5

        candidates.append((score, s))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_exact_target(spec: str) -> PWTarget | None:
    spec = str(spec)
    for s in iter_output_streams():
        if spec == s.object_serial or spec == s.node_name:
            return s
    return None


def resolve_target(explicit: str | None, browser_pattern: str) -> PWTarget | None:
    if not explicit:
        return find_stream_target(browser_pattern)

    # first try exact target match by object.serial or node.name
    exact = find_exact_target(explicit)
    if exact:
        return exact

    # otherwise treat as human-friendly pattern
    return find_stream_target(re.escape(explicit))


def target_exists(object_serial: str) -> bool:
    for s in iter_output_streams():
        if s.object_serial == str(object_serial):
            return True
    return False


def spawn_pw_record(
    target_object: str,
    rate: int,
    channels: int,
    latency: str,
    capture_sink: bool,
) -> subprocess.Popen:
    props = {
        "node.dont-fallback": True,
        "node.dont-reconnect": True,
        "node.dont-move": True,
    }

    if capture_sink:
        props["stream.capture.sink"] = True

    cmd = [
        "pw-record",
        "--target",
        str(target_object),
        "--rate",
        str(rate),
        "--channels",
        str(channels),
        "--format",
        "f32",
        "--raw",
        "--latency",
        latency,
        "-P",
        json.dumps(props),
    ]

    if channels == 1:
        cmd += ["--channel-map", "mono"]
    elif channels == 2:
        cmd += ["--channel-map", "stereo"]

    cmd += ["-"]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def keyboard_loop(ui: TerminalUI, stop_event: threading.Event):
    fd = sys.stdin.fileno()
    while not stop_event.is_set() and not ui.quit_requested:
        key = read_key(fd, timeout=0.1)
        if key:
            ui.handle_key(key)


def run_capture_loop(args, model, ui) -> None:
    input_block_frames = int(args.input_rate * args.block_sec)
    input_block_bytes = input_block_frames * args.input_channels * 4

    preroll_blocks = max(1, int(args.preroll_sec / args.block_sec))
    rms_log_every_blocks = max(1, int(args.rms_log_sec / args.block_sec))

    last_target = None
    global_last_text = None

    while not ui.quit_requested:
        resolved = resolve_target(args.target, args.browser)

        if not resolved:
            ui.set_state("idle")
            if args.debug:
                eprint("DEBUG no target yet; waiting...")
            time.sleep(1.0)
            continue

        target_serial = resolved.object_serial

        if target_serial != last_target:
            eprint(
                f"Attached to PipeWire target: "
                f"{resolved.app_name or resolved.node_name} "
                f"(serial={resolved.object_serial}, node={resolved.node_name}, state={resolved.state})"
            )
            last_target = target_serial
            ui.set_target(
                f"{resolved.app_name or resolved.node_name} [{resolved.object_serial}]"
            )
            ui.set_state("waiting")

        proc = spawn_pw_record(
            target_object=target_serial,
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
            while not ui.quit_requested:
                buf = read_exact(proc.stdout, input_block_bytes)
                if len(buf) == 0:
                    if args.debug:
                        eprint("DEBUG pw-record EOF")
                    break
                if len(buf) < input_block_bytes:
                    if args.debug:
                        eprint(
                            f"DEBUG short read: got={len(buf)} want={input_block_bytes}"
                        )
                    break

                raw = np.frombuffer(buf, dtype=np.float32)
                mono16 = downmix_and_resample(
                    raw,
                    in_channels=args.input_channels,
                    in_rate=args.input_rate,
                    out_rate=args.asr_rate,
                )

                if mono16.size == 0:
                    loop_idx += 1
                    continue

                rms = float(np.sqrt(np.mean(mono16 * mono16) + 1e-12))
                voiced = rms >= args.silence_rms

                ui.set_levels(rms, voiced, in_speech)

                preroll.append(mono16)

                if args.debug and loop_idx % rms_log_every_blocks == 0:
                    eprint(
                        f"DEBUG rms={rms:.5f} voiced={int(voiced)} "
                        f"in_speech={int(in_speech)} utt_sec={utterance_sec:.2f}"
                    )
                loop_idx += 1

                # periodically check that pw-record is alive and target still exists
                if loop_idx % rms_log_every_blocks == 0:
                    if proc.poll() is not None:
                        if args.debug:
                            eprint("DEBUG pw-record exited")
                        break

                    if not target_exists(target_serial):
                        if args.debug:
                            eprint(f"DEBUG target disappeared: serial={target_serial}")
                        break

                if voiced and not in_speech:
                    utterance = list(preroll)
                    utterance_sec = sum(len(x) for x in utterance) / args.asr_rate
                    in_speech = True
                    silence_sec = 0.0
                    next_emit_at = max(args.first_emit_sec, args.emit_every_sec)
                    ui.set_state("listening")
                    if args.debug:
                        eprint(
                            f"DEBUG speech_start rms={rms:.5f} preroll_sec={utterance_sec:.2f}"
                        )
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
                            eprint(
                                f"DEBUG partial utt_sec={utterance_sec:.2f} text={text!r}"
                            )
                        ui.set_partial(text)
                    next_emit_at += args.emit_every_sec

                if (
                    silence_sec >= args.tail_silence_sec
                    or utterance_sec >= args.max_utt_sec
                ):
                    audio = np.concatenate(utterance)
                    if len(audio) / args.asr_rate >= args.min_utt_sec:
                        text = robust_recognize(model, audio, sample_rate=args.asr_rate)
                        if args.debug:
                            eprint(
                                f"DEBUG final utt_sec={utterance_sec:.2f} "
                                f"silence_sec={silence_sec:.2f} text={text!r}"
                            )
                        norm = normalize_text(text)
                        if norm and norm != global_last_text:
                            print(text, flush=True)
                            global_last_text = norm

                        ui.add_final(text)
                        ui.set_state("waiting")

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
    p.add_argument(
        "--target",
        default=None,
        help="exact PipeWire target: object.serial or node.name (e.g. 1234 or Brave); if not exact, treated as search pattern",
    )
    p.add_argument(
        "--capture-sink",
        action="store_true",
        help="capture sink monitor instead of app stream",
    )
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
    p.add_argument("--history-max", type=int, default=500)
    p.add_argument(
        "--log-file",
        default=None,
        help="path to append finalized subtitles, e.g. /tmp/meeting_subtitles.txt",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    eprint(f"Loading model: {args.model!r} quantization={args.quantization!r}")
    model = onnx_asr.load_model(args.model, quantization=args.quantization)
    eprint("Ready. Waiting for audio stream...")

    ui = TerminalUI(
        target="",
        history_max=args.history_max,
        log_file=args.log_file,
    )
    ui.set_model(args.model)

    stop_event = threading.Event()

    try:
        if sys.stdin.isatty():
            with RawKeyboard(sys.stdin):
                t = threading.Thread(
                    target=keyboard_loop,
                    args=(ui, stop_event),
                    daemon=True,
                )
                t.start()
                with ui.live:
                    run_capture_loop(args, model, ui)
        else:
            with ui.live:
                run_capture_loop(args, model, ui)
    finally:
        stop_event.set()
        ui.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
