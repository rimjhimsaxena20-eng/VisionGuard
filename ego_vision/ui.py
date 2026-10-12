"""VisionGuard desktop control center with an embedded, low-rate preview."""
from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent.parent


class VisionGuardUI(tk.Tk):
    BG = "#0a0f19"
    PANEL = "#111a27"
    PANEL_ALT = "#182433"
    LINE = "#223247"
    TEXT = "#eef4fa"
    MUTED = "#8fa3b8"
    ACCENT = "#54d6b0"
    AMBER = "#f0ba67"

    def __init__(self):
        super().__init__()
        self.title("VisionGuard | Control Center")
        self.geometry("1240x820")
        self.minsize(1000, 700)
        self.configure(bg=self.BG)
        self.process: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue(maxsize=2)
        self.listener: socket.socket | None = None
        self.source_kind = tk.StringVar(value="Camera")
        self.source = tk.StringVar(value="0")
        self.device = tk.StringVar(value="auto")
        self.audio = tk.BooleanVar(value=True)
        self.offline = tk.BooleanVar(value=False)
        self.record = tk.BooleanVar(value=False)
        self.event_log = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="READY")
        self.preview_photo = None
        self._style()
        self._build()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=self.PANEL_ALT, background=self.PANEL_ALT,
                        foreground=self.TEXT, arrowcolor=self.MUTED, bordercolor=self.LINE,
                        padding=8, relief="flat")
        style.map("TCombobox", fieldbackground=[("readonly", self.PANEL_ALT)],
                  foreground=[("readonly", self.TEXT)])

    def _build(self):
        shell = tk.Frame(self, bg=self.BG, padx=24, pady=20)
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, bg=self.BG)
        header.pack(fill="x", pady=(0, 18))
        brand = tk.Frame(header, bg=self.BG)
        brand.pack(side="left")
        tk.Label(brand, text="VG", bg=self.ACCENT, fg="#08130f", width=3, height=1,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 12), ipady=7)
        titles = tk.Frame(brand, bg=self.BG)
        titles.pack(side="left")
        tk.Label(titles, text="VisionGuard", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w")
        tk.Label(titles, text="LOCAL PEDESTRIAN AWARENESS  ·  CONTROL CENTER", bg=self.BG,
                 fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 0))
        self.status_badge = tk.Label(header, textvariable=self.status, bg="#17251f", fg=self.ACCENT,
                                     padx=14, pady=8, font=("Segoe UI", 9, "bold"))
        self.status_badge.pack(side="right", pady=5)

        content = tk.Frame(shell, bg=self.BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, minsize=310, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_controls(content)
        self._build_workspace(content)

    def _card(self, parent, title, subtitle=None):
        card = tk.Frame(parent, bg=self.PANEL, padx=17, pady=15,
                        highlightthickness=1, highlightbackground=self.LINE)
        tk.Label(card, text=title, bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        if subtitle:
            tk.Label(card, text=subtitle, bg=self.PANEL, fg=self.MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 11))
        return card

    def _build_controls(self, parent):
        controls = tk.Frame(parent, bg=self.BG, width=310)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        controls.grid_propagate(False)

        source_card = self._card(controls, "INPUT SOURCE", "Choose a camera or local recording.")
        source_card.pack(fill="x", pady=(0, 12))
        self._field_label(source_card, "Source type")
        source_type = ttk.Combobox(source_card, textvariable=self.source_kind,
                                   values=("Camera", "Video file"), state="readonly")
        source_type.pack(fill="x", pady=(0, 10), ipady=2)
        source_type.bind("<<ComboboxSelected>>", self._source_changed)
        self._field_label(source_card, "Camera index / file")
        source_row = tk.Frame(source_card, bg=self.PANEL)
        source_row.pack(fill="x")
        self.source_entry = tk.Entry(source_row, textvariable=self.source, bg=self.PANEL_ALT,
                                     fg=self.TEXT, insertbackground=self.TEXT, relief="flat",
                                     font=("Segoe UI", 9))
        self.source_entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(0, 8))
        self.browse_button = self._button(source_row, "Browse", self._browse, quiet=True)
        self.browse_button.pack(side="right", ipady=5)

        settings = self._card(controls, "SESSION SETTINGS", "Models stay local after first download.")
        settings.pack(fill="x", pady=(0, 12))
        self._field_label(settings, "Compute device")
        ttk.Combobox(settings, textvariable=self.device, values=("auto", "cpu", "cuda:0"),
                     state="readonly").pack(fill="x", pady=(0, 8), ipady=2)
        self._check(settings, "Enable spoken hazard warnings", self.audio).pack(anchor="w", pady=3)
        self._check(settings, "Offline mode (models already downloaded)", self.offline).pack(anchor="w", pady=3)

        evidence = self._card(controls, "SESSION OUTPUTS", "Saved under the project outputs folder.")
        evidence.pack(fill="x", pady=(0, 12))
        self._check(evidence, "Annotated video", self.record).pack(anchor="w", pady=3)
        self._check(evidence, "Event measurements (CSV)", self.event_log).pack(anchor="w", pady=3)

        actions = tk.Frame(controls, bg=self.BG)
        actions.pack(fill="x", pady=(2, 12))
        self.start_button = self._button(actions, "Start session", self._start, primary=True)
        self.start_button.pack(fill="x", ipady=9, pady=(0, 8))
        self.stop_button = self._button(actions, "Stop session", self._stop, quiet=True)
        self.stop_button.pack(fill="x", ipady=8)
        self.stop_button.configure(state="disabled")

        tk.Label(controls, text="DEMONSTRATION PROTOTYPE\nApproximate depth. Keep a sighted operator present.",
                 justify="left", bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

    def _build_workspace(self, parent):
        workspace = tk.Frame(parent, bg=self.BG)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.rowconfigure(0, weight=3)
        workspace.rowconfigure(1, weight=1)
        workspace.columnconfigure(0, weight=1)

        preview_card = self._card(workspace, "LIVE PREVIEW", "Processed view · 12 fps stream · starts after model initialization")
        preview_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        preview_card.rowconfigure(2, weight=1)
        preview_card.columnconfigure(0, weight=1)
        preview_header = tk.Frame(preview_card, bg=self.PANEL)
        preview_header.pack(fill="x")
        tk.Label(preview_header, text="CAMERA / VIDEO", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self.preview_state = tk.Label(preview_header, text="WAITING FOR SESSION", bg=self.PANEL,
                                      fg=self.MUTED, font=("Segoe UI", 8, "bold"))
        self.preview_state.pack(side="right")
        self.preview = tk.Label(preview_card, text="Preview will appear here\n\nStart a session to connect the camera",
                                bg="#070b11", fg=self.MUTED, justify="center",
                                font=("Segoe UI", 12), compound="center")
        self.preview.pack(fill="both", expand=True, pady=(12, 0))
        legend = tk.Frame(preview_card, bg=self.PANEL)
        legend.pack(fill="x", pady=(12, 0))
        tk.Label(legend, text="RISK GUIDE", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 12))
        for label, color in (("LOW", "#59c789"), ("CAUTION", "#e5c45b"),
                             ("HIGH", "#f39a51"), ("CRITICAL", "#f16f70"),
                             ("UNKNOWN", "#aab6c4")):
            tk.Label(legend, text=label, bg=self.PANEL_ALT, fg=color,
                     padx=8, pady=4, font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        tk.Label(preview_card, text="LOW means no relevant detected threat. UNKNOWN means essential evidence is unavailable.",
                 bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(9, 0))

        log_card = self._card(workspace, "RUNTIME OUTPUT", "Startup, model, camera, and audio status")
        log_card.grid(row=1, column=0, sticky="nsew")
        self.log = tk.Text(log_card, bg="#080d14", fg="#b9c9d7", insertbackground=self.TEXT,
                           relief="flat", height=6, wrap="word", font=("Cascadia Mono", 9),
                           padx=11, pady=9, state="disabled")
        self.log.tag_configure("error", foreground="#ff8883")
        self.log.tag_configure("notice", foreground=self.ACCENT)
        self.log.pack(fill="both", expand=True, pady=(9, 0))
        self._append("Ready. First start may download the depth model; the preview begins after loading.\n")

    def _field_label(self, parent, text):
        tk.Label(parent, text=text.upper(), bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 6))

    def _check(self, parent, text, variable):
        return tk.Checkbutton(parent, text=text, variable=variable, bg=self.PANEL, fg=self.TEXT,
                              activebackground=self.PANEL, activeforeground=self.TEXT,
                              selectcolor=self.PANEL_ALT, font=("Segoe UI", 9), relief="flat",
                              anchor="w", padx=0)

    def _button(self, parent, text, command, primary=False, quiet=False):
        bg = self.ACCENT if primary else (self.PANEL_ALT if quiet else self.PANEL)
        fg = "#07130f" if primary else self.TEXT
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground="#79e6c8" if primary else "#25364a",
                         activeforeground=fg, disabledforeground="#647487", relief="flat", bd=0,
                         padx=12, font=("Segoe UI", 9, "bold"), cursor="hand2")

    def _source_changed(self, _event=None):
        is_file = self.source_kind.get() == "Video file"
        self.browse_button.configure(state="normal" if is_file else "disabled")
        if is_file and self.source.get() == "0":
            self.source.set("")
        elif not is_file and not self.source.get():
            self.source.set("0")

    def _browse(self):
        path = filedialog.askopenfilename(title="Choose a video", filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")])
        if path:
            self.source.set(path)

    def _start(self):
        source = self.source.get().strip()
        if not source:
            messagebox.showerror("Source required", "Enter a camera index or choose a video file.")
            return
        if self.source_kind.get() == "Video file" and not Path(source).is_file():
            messagebox.showerror("Video not found", "Choose an existing local video file.")
            return
        try:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.listener.settimeout(.4)
        except OSError as exc:
            self._close_listener()
            messagebox.showerror("Preview unavailable", f"Could not reserve the local preview channel: {exc}")
            return
        host, port = self.listener.getsockname()
        threading.Thread(target=self._receive_preview, daemon=True).start()

        output_dir = ROOT / "outputs"
        command = [sys.executable, "-m", "ego_vision.pipeline", "--source", source,
                   "--no-show", "--device", self.device.get(), "--preview-stream", f"{host}:{port}"]
        if self.offline.get():
            command.append("--offline")
        if not self.audio.get():
            command.append("--mute")
        if self.record.get():
            command += ["--output", str(output_dir / "visionguard-session.mp4")]
        if self.event_log.get():
            command += ["--events", str(output_dir / "visionguard-events.csv")]
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                            errors="replace", bufsize=1,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            self._close_listener()
            messagebox.showerror("Could not start VisionGuard", str(exc))
            return
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("STARTING")
        self.status_badge.configure(fg=self.AMBER)
        self.preview_state.configure(text="LOADING MODELS", fg=self.AMBER)
        self.preview.configure(image="", text="Loading models…\n\nPreview connects when initialization is complete")
        self.preview_photo = None
        self._append("\n$ " + subprocess.list2cmdline(command) + "\n")

    def _receive_preview(self):
        connection = None
        try:
            while self.listener:
                try:
                    connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    if self.process and self.process.poll() is not None:
                        return
            if not connection:
                return
            connection.settimeout(3.0)
            while True:
                header = self._recv_exact(connection, 4)
                if not header:
                    break
                length = struct.unpack("!I", header)[0]
                if length <= 0 or length > 8_000_000:
                    break
                data = self._recv_exact(connection, length)
                if not data:
                    break
                with Image.open(BytesIO(data)) as image:
                    frame = image.convert("RGB")
                self._queue_frame(frame)
        except (OSError, ValueError):
            pass
        finally:
            if connection:
                connection.close()

    @staticmethod
    def _recv_exact(connection, length):
        parts = bytearray()
        while len(parts) < length:
            chunk = connection.recv(length - len(parts))
            if not chunk:
                return None
            parts.extend(chunk)
        return bytes(parts)

    def _queue_frame(self, frame):
        try:
            self.events.put_nowait(("frame", frame))
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(("frame", frame))
            except queue.Full:
                pass

    def _read_output(self, process):
        if process.stdout:
            for line in process.stdout:
                self.events.put(("line", line))
        self.events.put(("exit", process.wait()))

    def _stop(self):
        if self.process and self.process.poll() is None:
            self.status.set("STOPPING")
            self.status_badge.configure(fg=self.AMBER)
            self.process.terminate()

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    line = value.rstrip()
                    self._append(value)
                    if "Loading VisionGuard models" in line:
                        self.status.set("LOADING MODELS")
                    elif "Embedded preview connected" in line:
                        self.status.set("LIVE")
                        self.status_badge.configure(fg=self.ACCENT)
                        self.preview_state.configure(text="LIVE", fg=self.ACCENT)
                    elif "Audio unavailable" in line:
                        self._append("Check that a Windows SAPI voice is installed and audio is enabled.\n")
                elif kind == "frame":
                    available_w = max(320, self.preview.winfo_width() - 24)
                    available_h = max(200, self.preview.winfo_height() - 24)
                    value.thumbnail((available_w, available_h), Image.Resampling.LANCZOS)
                    self.preview_photo = ImageTk.PhotoImage(value)
                    self.preview.configure(image=self.preview_photo, text="")
                    self.preview_state.configure(text="LIVE", fg=self.ACCENT)
                    if self.status.get() == "STARTING":
                        self.status.set("LIVE")
                        self.status_badge.configure(fg=self.ACCENT)
                else:
                    self.process = None
                    self._close_listener()
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status.set("READY" if value == 0 else f"EXIT {value}")
                    self.status_badge.configure(fg=self.ACCENT if value == 0 else self.AMBER)
                    self.preview_state.configure(text="SESSION ENDED", fg=self.MUTED)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _close_listener(self):
        if self.listener:
            try:
                self.listener.close()
            except OSError:
                pass
            self.listener = None

    def _append(self, text):
        self.log.configure(state="normal")
        normalized = text.lower()
        tag = "error" if any(word in normalized for word in ("error", "failed", "unavailable", "traceback")) else None
        if tag is None and any(word in normalized for word in ("connected", "loading visionguard models", "models loaded")):
            tag = "notice"
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self._close_listener()
        self.destroy()


def main():
    VisionGuardUI().mainloop()


if __name__ == "__main__":
    main()
