"""Tkinter GUI for 360-Video-Manager.

The GUI orchestrates operations by delegating to the unified workflow layer
(:mod:`workflows.unified_pipeline`).  No pipeline logic lives here — all
processing callbacks are offloaded to daemon threads and results are
delivered back to the main thread via ``master.after()``.
"""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

from config.logging_config import setup_logging
from config.settings import get_settings
from utils.exceptions import (
    DownloadError,
    MediaCMSError,
    WorkflowError,
    YouTubeAPIError,
)
from workflows.unified_pipeline import JobOptions, process_video_job


logger = logging.getLogger(__name__)


class VR360ManagerApp:
    """Main Tkinter application window for VR360 Media Manager."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("VR360 Media Manager")
        self.master.geometry("960x620")

        cfg = get_settings()
        cfg.ensure_runtime_dirs()

        self.search_results: List[dict] = []
        self.video_path: Optional[str] = None

        self.status_var = tk.StringVar(value="Ready")
        self.search_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.playlist_var = tk.StringVar()
        self.desc_text: Optional[tk.Text] = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        main = ttk.Frame(self.master, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Search row ---
        search_row = ttk.Frame(main)
        search_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_row, text="Search YouTube:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        search_entry.bind("<Return>", lambda _e: self._on_search())
        ttk.Button(search_row, text="Search", command=self._on_search).pack(side=tk.LEFT)

        # --- Results list ---
        list_frame = ttk.Frame(main)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.results_list = tk.Listbox(list_frame, height=12)
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.results_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_list.config(yscrollcommand=scrollbar.set)

        # --- Action buttons ---
        actions = ttk.Frame(main)
        actions.pack(fill=tk.X, pady=10)
        ttk.Button(
            actions, text="Download selected", command=self._on_download_selected
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            actions, text="Detect & convert projection", command=self._on_detect
        ).pack(side=tk.LEFT)

        # --- Upload form ---
        upload_frame = ttk.LabelFrame(main, text="Upload to MediaCMS", padding=8)
        upload_frame.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(upload_frame, text="Title:").grid(row=0, column=0, sticky="w")
        ttk.Entry(upload_frame, textvariable=self.title_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )

        ttk.Label(upload_frame, text="Playlist (optional):").grid(row=1, column=0, sticky="w")
        ttk.Entry(upload_frame, textvariable=self.playlist_var).grid(
            row=1, column=1, sticky="ew", padx=6
        )

        ttk.Label(upload_frame, text="Description:").grid(row=2, column=0, sticky="nw")
        self.desc_text = tk.Text(upload_frame, height=4)
        self.desc_text.grid(row=2, column=1, sticky="ew", padx=6, pady=(0, 4))

        ttk.Button(upload_frame, text="Upload", command=self._on_upload).grid(
            row=3, column=1, sticky="e"
        )
        upload_frame.columnconfigure(1, weight=1)

        # --- Status bar ---
        status_bar = ttk.Label(main, textvariable=self.status_var, anchor="w", relief="sunken")
        status_bar.pack(fill=tk.X, pady=(8, 0))

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _get_selected(self) -> Optional[dict]:
        sel = self.results_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        if 0 <= idx < len(self.search_results):
            return self.search_results[idx]
        return None

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_search(self) -> None:
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Enter a search query.")
            return
        self._set_status("Searching...")
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    def _search_thread(self, query: str) -> None:
        from core.youtube import search_videos
        try:
            results = search_videos(query)
            self.search_results = results
            self.master.after(0, self._update_results_list)
            self.master.after(
                0, lambda: self._set_status(f"Found {len(results)} 360° video(s).")
            )
        except Exception as exc:
            self.master.after(
                0, lambda: messagebox.showerror("Search error", str(exc))
            )
            self.master.after(0, lambda: self._set_status("Search failed."))

    def _update_results_list(self) -> None:
        self.results_list.delete(0, tk.END)
        for item in self.search_results:
            title = item.get("title") or "No title"
            channel = item.get("channel") or "Unknown channel"
            self.results_list.insert(tk.END, f"{title} | {channel}")

    def _on_download_selected(self) -> None:
        selected = self._get_selected()
        if not selected:
            messagebox.showwarning("Warning", "Select a video first.")
            return
        url = selected.get("url")
        if not url:
            messagebox.showerror("Error", "Selected result has no URL.")
            return
        self.title_var.set(selected.get("title") or "")
        self._set_status("Downloading…")
        threading.Thread(target=self._download_thread, args=(url,), daemon=True).start()

    def _download_thread(self, url: str) -> None:
        from core.downloader import download_video

        cfg = get_settings()

        def _progress(d: dict) -> None:
            if d.get("status") == "downloading":
                pct = d.get("_percent_str", "?%").strip()
                self.master.after(0, lambda: self._set_status(f"Downloading… {pct}"))

        try:
            path = download_video(url, output_dir=cfg.downloads_dir, progress_callback=_progress)
            self.video_path = path
            self.master.after(
                0, lambda: self._set_status("Download complete. Ready to detect / upload.")
            )
        except DownloadError as exc:
            self.master.after(0, lambda: messagebox.showerror("Download error", str(exc)))
            self.master.after(0, lambda: self._set_status("Download failed."))

    def _on_detect(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Warning", "No downloaded video available.")
            return
        self._set_status("Running projection detection…")
        threading.Thread(target=self._detect_thread, args=(self.video_path,), daemon=True).start()

    def _detect_thread(self, video_path: str) -> None:
        """Normalise codec → extract previews → detect projection → optionally convert."""
        try:
            # Codec normalisation
            self.master.after(0, lambda: self._set_status("Normalising codec…"))
            from detector.video_io import convert_video_codec
            norm_path = convert_video_codec(video_path)
            self.video_path = norm_path

            # Preview frames
            from core.preview_frames import extract_preview_frames
            cfg = get_settings()
            extract_preview_frames(norm_path, num_frames=5, output_dir=os.path.join(cfg.downloads_dir, "previews"))

            # Detection
            self.master.after(0, lambda: self._set_status("Detecting projection…"))
            from detector.pipeline import run_detection_pipeline
            det = run_detection_pipeline(norm_path, num_frames=10)
            proj = det.get("projection_type", "unknown")
            conf = float(det.get("confidence", 0.0))

            # Conversion
            converted: Optional[str] = None
            if proj not in ("equirectangular", "stereo_equi", "unknown") and conf >= 0.5:
                self.master.after(0, lambda: self._set_status("Converting to equirectangular…"))
                from detector.projection_conversion import convert_detected_projection_to_equirectangular
                converted = convert_detected_projection_to_equirectangular(
                    norm_path, proj, conf
                )
                if converted and converted != norm_path:
                    self.video_path = converted
                    norm_path = converted

            msg = (
                f"Detection complete\n"
                f"Projection: {proj}\n"
                f"Confidence: {conf:.1%}"
            )
            if converted:
                msg += f"\nConverted: {os.path.basename(converted)}"

            self.master.after(0, lambda: messagebox.showinfo("Detection", msg))
            self.master.after(0, lambda: self._set_status("Ready to upload."))

        except Exception as exc:
            self.master.after(0, lambda: messagebox.showerror("Detection error", str(exc)))
            self.master.after(0, lambda: self._set_status("Detection failed."))

    def _on_upload(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Warning", "No video available to upload.")
            return
        title = self.title_var.get().strip() or os.path.basename(self.video_path)
        desc = self.desc_text.get("1.0", tk.END).strip() if self.desc_text else ""
        playlist = self.playlist_var.get().strip() or None
        self._set_status("Uploading…")
        threading.Thread(
            target=self._upload_thread,
            args=(self.video_path, title, desc, playlist),
            daemon=True,
        ).start()

    def _upload_thread(
        self,
        video_path: str,
        title: str,
        description: str,
        playlist: Optional[str],
    ) -> None:
        from core.uploader import upload_video_asset
        try:
            upload_result = upload_video_asset(
                video_path=video_path,
                title=title,
                description=description,
                playlist_id=playlist,
            )
            if upload_result.success:
                self.master.after(
                    0, lambda: messagebox.showinfo("Upload", "Video uploaded successfully.")
                )
                self.master.after(0, lambda: self._set_status("Upload complete."))
            else:
                err = upload_result.error or "Unknown error"
                self.master.after(0, lambda: messagebox.showerror("Upload failed", err))
                self.master.after(0, lambda: self._set_status("Upload failed."))
        except MediaCMSError as exc:
            self.master.after(0, lambda: messagebox.showerror("Upload error", str(exc)))
            self.master.after(0, lambda: self._set_status("Upload failed."))


def run_gui() -> None:
    """Launch the VR360 Manager Tkinter GUI."""
    setup_logging(level=logging.INFO)
    root = tk.Tk()
    VR360ManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
