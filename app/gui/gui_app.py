"""CustomTkinter GUI for 360-Video-Manager.

The GUI orchestrates operations by delegating to the unified workflow layer
(:mod:`workflows.unified_pipeline`).  No pipeline logic lives here — all
processing callbacks are offloaded to daemon threads and results are
delivered back to the main thread via ``master.after()``.
"""

from __future__ import annotations

import enum
import io
import logging
import threading
import time
import tkinter
import tkinter.messagebox
import urllib.request
from typing import Any, Dict, List, Optional

import customtkinter as ctk
from PIL import Image

from app.gui.progress_utils import (
    DOWNLOAD_PROGRESS_UPDATE_MS,
    clamp_progress,
    compute_progress_update_delay_ms,
    extract_download_progress_fraction,
)
from config.logging_config import setup_logging
from config.settings import get_settings
from utils.exceptions import (
    MediaCMSError,
    NoYouTubeAPIKeyError,
    YouTubeAPIError,
)

logger = logging.getLogger(__name__)

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Layout constants ──────────────────────────────────────────────────────────
_THUMB_CARD   = (120, 68)
_THUMB_DETAIL = (320, 180)
_PAGE_SIZE    = 5
_ACCENT       = "#1a7fd4"


# ── App state ─────────────────────────────────────────────────────────────────

class AppState(enum.Enum):
    IDLE       = "idle"
    SEARCHING  = "searching"
    READY      = "ready"
    PROCESSING = "processing"
    PROCESSED  = "processed"
    UPLOADING  = "uploading"


# search_btn / dl_btn / up_btn
_STATE_BUTTONS: Dict[AppState, tuple] = {
    AppState.IDLE:       ("normal",   "disabled", "disabled"),
    AppState.SEARCHING:  ("disabled", "disabled", "disabled"),
    AppState.READY:      ("normal",   "normal",   "disabled"),
    AppState.PROCESSING: ("disabled", "disabled", "disabled"),
    AppState.PROCESSED:  ("normal",   "normal",   "normal"),
    AppState.UPLOADING:  ("disabled", "disabled", "disabled"),
}


# ── Log handler ───────────────────────────────────────────────────────────────

class GUILogHandler(logging.Handler):
    """Appends log records to a CTkTextbox via root.after() — always thread-safe."""

    def __init__(self, textbox: ctk.CTkTextbox, root: ctk.CTk) -> None:
        super().__init__()
        self._box  = textbox
        self._root = root
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            self._root.after(0, lambda m=msg: self._append(m))
        except Exception:
            self.handleError(record)

    def _append(self, msg: str) -> None:
        self._box.configure(state="normal")
        self._box.insert("end", msg)
        self._box.see("end")
        self._box.configure(state="disabled")


# ── Thumbnail helpers ─────────────────────────────────────────────────────────

def _fetch_thumbnail(video_id: str, size: tuple) -> Optional[ctk.CTkImage]:
    """Download a YouTube thumbnail and return a CTkImage, or None on failure."""
    from core.youtube import get_video_thumbnail_urls
    urls = get_video_thumbnail_urls(video_id)
    for key in ("mqdefault", "hqdefault", "sddefault", "default"):
        url = urls.get(key, "")
        if not url:
            continue
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return ctk.CTkImage(light_image=img, size=size)
        except Exception:
            continue
    return None


def _grey_image(size: tuple) -> ctk.CTkImage:
    """Return a neutral grey placeholder CTkImage."""
    img = Image.new("RGB", size, color=(210, 210, 210))
    return ctk.CTkImage(light_image=img, size=size)


# ── Main application ──────────────────────────────────────────────────────────

class VR360ManagerApp:
    """Main CustomTkinter application window for VR360 Media Manager."""

    def __init__(self, master: ctk.CTk) -> None:
        self.master = master
        self.master.title("VR360 Media Manager")
        self.master.geometry("1000x760")
        self.master.minsize(700, 560)

        get_settings().ensure_runtime_dirs()

        # ── Runtime state ──
        self._state:        AppState              = AppState.IDLE
        self._results:      List[Dict[str, Any]]  = []
        self._max_results:  int                   = _PAGE_SIZE
        self._selected:     Optional[Dict]        = None
        self._ready_path:   Optional[str]         = None
        self._cards:             List[ctk.CTkFrame]    = []
        self._res_children:      List                  = []   # all widgets in results_frame
        self._card_title_labels: List[ctk.CTkLabel]   = []   # for adaptive wraplength
        self._playlists:         List[Dict]            = []
        self._categories:        List[Dict]            = []
        self._log_visible:  bool                  = False
        self._status_var  = tkinter.StringVar(value="Ready")
        self._download_progress_lock = threading.Lock()
        self._download_progress_pending: Optional[tuple[Optional[float], str]] = None
        self._download_progress_scheduled: bool = False
        self._download_progress_last_ts: float = 0.0

        self._build_ui()
        self._attach_log_handler()

        # Pre-load playlists from CMS
        threading.Thread(target=self._bg_load_playlists, daemon=True).start()
        threading.Thread(target=self._bg_load_categories, daemon=True).start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        root_frame.pack(fill="both", expand=True, padx=16, pady=12)
        # Row 0 grows with the window; row 1 (bottom actions) stays fixed.
        root_frame.grid_rowconfigure(0, weight=1)
        root_frame.grid_rowconfigure(1, weight=0)
        root_frame.grid_columnconfigure(0, weight=1)

        # ── Scrollable content area (top, fills available height) ──
        content = ctk.CTkScrollableFrame(root_frame, fg_color="transparent")
        content.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        # ── Fixed action area (bottom, always visible) ──
        bottom = ctk.CTkFrame(root_frame, fg_color="transparent")
        bottom.grid(row=1, column=0, sticky="ew")

        # ── Search row ──
        search_row = ctk.CTkFrame(content, fg_color="transparent")
        search_row.pack(fill="x", pady=(0, 8))

        self._search_entry = ctk.CTkEntry(
            search_row,
            placeholder_text="Search YouTube for 360° videos…",
            height=38,
        )
        self._search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._search_entry.bind("<Return>", lambda _: self._on_search())

        self._search_btn = ctk.CTkButton(
            search_row, text="Search", width=100, height=38,
            command=self._on_search,
        )
        self._search_btn.pack(side="left")

        # ── Results ──
        ctk.CTkLabel(
            content, text="Results",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", pady=(4, 2))

        self._results_frame = ctk.CTkScrollableFrame(content, height=220)
        self._results_frame.pack(fill="x")

        # Container for "Show more" — always in layout; child shown/hidden
        _sm_cont = ctk.CTkFrame(content, fg_color="transparent")
        _sm_cont.pack(fill="x")
        self._show_more_btn = ctk.CTkButton(
            _sm_cont,
            text="Show more results",
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray20", "gray80"),
            command=self._on_show_more,
        )
        # Not packed yet — shown only when a full page was returned

        # ── Selected video panel ──
        ctk.CTkLabel(
            content, text="Selected video",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", pady=(10, 2))

        detail = ctk.CTkFrame(
            content, corner_radius=8,
            border_width=1, border_color=("gray80", "gray30"),
        )
        detail.pack(fill="x")

        self._detail_thumb = ctk.CTkLabel(
            detail, text="",
            image=_grey_image(_THUMB_DETAIL),
            width=_THUMB_DETAIL[0], height=_THUMB_DETAIL[1],
        )
        self._detail_thumb.pack(side="left", padx=12, pady=10)

        info = ctk.CTkFrame(detail, fg_color="transparent")
        info.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=10)

        self._detail_title_lbl = ctk.CTkLabel(
            info, text="No video selected",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w", wraplength=540,
        )
        self._detail_title_lbl.pack(fill="x")

        self._detail_channel_lbl = ctk.CTkLabel(
            info, text="", anchor="w",
            text_color=("gray45", "gray60"),
        )
        self._detail_channel_lbl.pack(fill="x", pady=(4, 0))

        self._detail_url_lbl = ctk.CTkLabel(
            info, text="", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
        )
        self._detail_url_lbl.pack(fill="x", pady=(2, 0))

        # Bind for adaptive detail-title wraplength
        info.bind("<Configure>", self._on_info_resize)

        # ── Upload options ──
        ctk.CTkLabel(
            content, text="Upload options",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", pady=(10, 2))

        upload_opts = ctk.CTkFrame(
            content, corner_radius=8,
            border_width=1, border_color=("gray80", "gray30"),
        )
        upload_opts.pack(fill="x")
        upload_opts.columnconfigure(1, weight=1)

        ctk.CTkLabel(upload_opts, text="Title:").grid(
            row=0, column=0, padx=(12, 6), pady=(10, 4), sticky="w")
        self._title_entry = ctk.CTkEntry(
            upload_opts, placeholder_text="Video title…", height=32)
        self._title_entry.grid(
            row=0, column=1, columnspan=2, padx=(0, 12), pady=(10, 4), sticky="ew")

        ctk.CTkLabel(upload_opts, text="Playlist:").grid(
            row=1, column=0, padx=(12, 6), pady=(4, 10), sticky="w")
        self._playlist_var = tkinter.StringVar(value="— no playlist —")
        self._playlist_menu = ctk.CTkOptionMenu(
            upload_opts,
            variable=self._playlist_var,
            values=["— no playlist —"],
            width=300,
            dynamic_resizing=False,
        )
        self._playlist_menu.grid(row=1, column=1, padx=(0, 6), pady=(4, 10), sticky="w")
        ctk.CTkButton(
            upload_opts, text="＋ New…", width=72,
            command=self._on_new_playlist,
        ).grid(row=1, column=2, padx=(0, 12), pady=(4, 10))

        ctk.CTkLabel(upload_opts, text="Patient (Category):").grid(
            row=2, column=0, padx=(12, 6), pady=(4, 10), sticky="w")
        self._category_var = tkinter.StringVar(value="— no category —")
        self._category_menu = ctk.CTkOptionMenu(
            upload_opts,
            variable=self._category_var,
            values=["— no category —"],
            width=300,
            dynamic_resizing=False,
        )
        self._category_menu.grid(row=2, column=1, padx=(0, 6), pady=(4, 10), sticky="w")
        ctk.CTkButton(
            upload_opts, text="＋ New…", width=72,
            command=self._on_new_category,
        ).grid(row=2, column=2, padx=(0, 12), pady=(4, 10))

        ctk.CTkLabel(upload_opts, text="Tags:").grid(
            row=3, column=0, padx=(12, 6), pady=(0, 10), sticky="w")
        self._tags_entry = ctk.CTkEntry(
            upload_opts, placeholder_text="tag1, tag2, tag3", height=32)
        self._tags_entry.grid(
            row=3, column=1, columnspan=2, padx=(0, 12), pady=(0, 10), sticky="ew")

        # ── Action buttons (in always-visible bottom frame) ──
        actions = ctk.CTkFrame(bottom, fg_color="transparent")
        actions.pack(fill="x", pady=(6, 4))

        self._dl_btn = ctk.CTkButton(
            actions,
            text="⬇  Download & Process",
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self._on_download_process,
        )
        self._dl_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._up_btn = ctk.CTkButton(
            actions,
            text="⬆  Upload to CMS",
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("gray65", "gray35"),
            hover_color=("gray55", "gray45"),
            state="disabled",
            command=self._on_upload,
        )
        self._up_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # ── Progress bar ──
        self._progress = ctk.CTkProgressBar(bottom, height=10)
        self._progress.pack(fill="x", pady=(8, 2))
        self._progress.set(0)

        # ── Status label ──
        self._status_lbl = ctk.CTkLabel(
            bottom, textvariable=self._status_var,
            anchor="w", font=ctk.CTkFont(size=12),
        )
        self._status_lbl.pack(fill="x")

        # ── Log toggle ──
        self._log_toggle_btn = ctk.CTkButton(
            bottom,
            text="▶  Show log",
            fg_color="transparent",
            text_color=("gray45", "gray65"),
            font=ctk.CTkFont(size=11),
            anchor="w",
            height=22,
            command=self._toggle_log,
        )
        self._log_toggle_btn.pack(fill="x", pady=(4, 0))

        # ── Log panel (inside bottom container; text box shown/hidden on toggle) ──
        self._log_container = ctk.CTkFrame(bottom, fg_color="transparent")
        self._log_container.pack(fill="x")
        self._log_box = ctk.CTkTextbox(
            self._log_container,
            height=120,
            state="disabled",
            font=ctk.CTkFont(family="monospace", size=11),
        )
        # Not packed until toggled on

        # Bind master resize for adaptive card-title wraplengths
        self.master.bind("<Configure>", self._on_root_resize)

    def _attach_log_handler(self) -> None:
        handler = GUILogHandler(self._log_box, self.master)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

    # ── Adaptive resize handlers ──────────────────────────────────────────────

    def _on_info_resize(self, event) -> None:
        """Recalculate detail-title wraplength when its container is resized."""
        w = event.width
        if w > 1:
            self._detail_title_lbl.configure(wraplength=max(100, w - 8))

    def _on_root_resize(self, event) -> None:
        """Recalculate card-title wraplengths when the root window is resized."""
        if event.widget is not self.master:
            return
        w = event.width
        # Subtract root padx (32), scrollbar (~16), thumb width, thumb padx (32), inner pad (~20)
        card_wrap = max(150, w - 32 - _THUMB_CARD[0] - 68)
        for lbl in self._card_title_labels:
            try:
                lbl.configure(wraplength=card_wrap)
            except Exception:
                pass

    # ── State machine ─────────────────────────────────────────────────────────

    def _set_state(self, state: AppState) -> None:
        self._state = state
        s_s, dl_s, up_s = _STATE_BUTTONS[state]
        self._search_btn.configure(state=s_s)
        self._search_entry.configure(state=s_s)
        self._dl_btn.configure(state=dl_s)
        self._up_btn.configure(state=up_s)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _progress_reset(self, status: Optional[str] = None) -> None:
        self._clear_download_progress_queue()
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        if status is not None:
            self._set_status(status)

    def _progress_set_determinate(self, value: float, status: Optional[str] = None) -> None:
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(clamp_progress(value))
        if status is not None:
            self._set_status(status)

    def _progress_start_indeterminate(self, status: Optional[str] = None) -> None:
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        if status is not None:
            self._set_status(status)

    def _clear_download_progress_queue(self) -> None:
        with self._download_progress_lock:
            self._download_progress_pending = None

    def _queue_download_progress_update(self, progress: Optional[float], status: str) -> None:
        with self._download_progress_lock:
            self._download_progress_pending = (progress, status)
            if self._download_progress_scheduled:
                return
        self._schedule_download_progress_flush()

    def _schedule_download_progress_flush(self) -> None:
        with self._download_progress_lock:
            if self._download_progress_scheduled:
                return
            delay_ms = compute_progress_update_delay_ms(
                last_update_monotonic=self._download_progress_last_ts,
                now_monotonic=time.monotonic(),
            )
            self._download_progress_scheduled = True
        self.master.after(delay_ms, self._flush_download_progress_update)

    def _flush_download_progress_update(self) -> None:
        with self._download_progress_lock:
            pending = self._download_progress_pending
            self._download_progress_pending = None
            self._download_progress_scheduled = False
            self._download_progress_last_ts = time.monotonic()

        if not pending:
            return

        progress, status = pending
        if progress is None:
            self._progress_start_indeterminate(status)
        else:
            self._progress_set_determinate(progress, status)

        with self._download_progress_lock:
            has_more = self._download_progress_pending is not None
        if has_more:
            self._schedule_download_progress_flush()

    # ── Log toggle ────────────────────────────────────────────────────────────

    def _toggle_log(self) -> None:
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_box.pack(fill="x", pady=(2, 0))
            self._log_toggle_btn.configure(text="▼  Hide log")
        else:
            self._log_box.pack_forget()
            self._log_toggle_btn.configure(text="▶  Show log")

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self) -> None:
        query = self._search_entry.get().strip()
        if not query:
            tkinter.messagebox.showwarning("Warning", "Enter a search query.")
            return
        self._max_results = _PAGE_SIZE
        self._set_state(AppState.SEARCHING)
        self._progress_start_indeterminate("Searching…")
        threading.Thread(
            target=self._bg_search, args=(query, self._max_results), daemon=True,
        ).start()

    def _on_show_more(self) -> None:
        query = self._search_entry.get().strip()
        if not query:
            return
        self._max_results += _PAGE_SIZE
        self._set_state(AppState.SEARCHING)
        self._progress_start_indeterminate("Loading more results…")
        threading.Thread(
            target=self._bg_search, args=(query, self._max_results), daemon=True,
        ).start()

    def _bg_search(self, query: str, max_results: int) -> None:
        from core.youtube import search_videos
        try:
            results = search_videos(query, max_results=max_results)
        except (NoYouTubeAPIKeyError, YouTubeAPIError, Exception) as exc:
            self.master.after(0, lambda e=str(exc): self._on_search_error(e))
            return
        self.master.after(0, lambda r=results: self._on_search_done(r))

    def _on_search_done(self, results: List[Dict]) -> None:
        self._progress_reset()
        self._results = results
        self._render_cards(results)
        n = len(results)
        self._set_status(f"Found {n} 360° video{'s' if n != 1 else ''}.")
        if n >= self._max_results:
            self._show_more_btn.pack(pady=(4, 2))
        else:
            self._show_more_btn.pack_forget()
        self._set_state(AppState.IDLE)

    def _on_search_error(self, msg: str) -> None:
        self._progress_reset()
        self._set_state(AppState.IDLE)
        self._set_status("Search failed.")
        tkinter.messagebox.showerror("Search error", msg)

    # ── Results cards ─────────────────────────────────────────────────────────

    def _render_cards(self, results: List[Dict]) -> None:
        """Destroy existing cards and render fresh ones."""
        for w in self._res_children:
            try:
                w.destroy()
            except Exception:
                pass
        self._res_children.clear()
        self._cards.clear()
        self._card_title_labels.clear()
        self._selected   = None
        self._ready_path = None

        if not results:
            lbl = ctk.CTkLabel(
                self._results_frame,
                text="No 360° videos found. Try a different query.",
                text_color=("gray50", "gray50"),
            )
            lbl.pack(pady=16)
            self._res_children.append(lbl)
            return

        for item in results:
            self._add_card(item)

    def _add_card(self, item: Dict) -> None:
        card = ctk.CTkFrame(
            self._results_frame,
            height=_THUMB_CARD[1] + 12,
            corner_radius=6,
            border_width=1,
            border_color=("gray80", "gray30"),
            cursor="hand2",
        )
        card.pack(fill="x", pady=3, padx=2)
        card.pack_propagate(False)
        self._cards.append(card)
        self._res_children.append(card)

        ph = _grey_image(_THUMB_CARD)
        thumb_lbl = ctk.CTkLabel(
            card, text="", image=ph,
            width=_THUMB_CARD[0], height=_THUMB_CARD[1],
        )
        thumb_lbl.pack(side="left", padx=(8, 8), pady=6)

        txt = ctk.CTkFrame(card, fg_color="transparent")
        txt.pack(side="left", fill="both", expand=True, pady=8, padx=(0, 8))

        title_lbl = ctk.CTkLabel(
            txt,
            text=item.get("title") or "Untitled",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w", wraplength=600,
        )
        title_lbl.pack(fill="x")
        self._card_title_labels.append(title_lbl)

        ch_lbl = ctk.CTkLabel(
            txt,
            text=f"{item.get('channel') or 'Unknown'}  ·  360°",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
        )
        ch_lbl.pack(fill="x")

        # Bind click on every widget inside the card
        for widget in (card, thumb_lbl, txt, title_lbl, ch_lbl):
            widget.bind("<Button-1>",
                        lambda _, i=item, c=card: self._on_card_click(i, c))

        vid_id = item.get("id")
        if vid_id:
            threading.Thread(
                target=self._bg_card_thumb,
                args=(vid_id, thumb_lbl),
                daemon=True,
            ).start()

    def _bg_card_thumb(self, video_id: str, lbl: ctk.CTkLabel) -> None:
        img = _fetch_thumbnail(video_id, _THUMB_CARD)
        if img:
            self.master.after(0, lambda i=img: lbl.configure(image=i))

    def _on_card_click(self, item: Dict, clicked: ctk.CTkFrame) -> None:
        # Same card clicked again — nothing to do
        if self._selected and self._selected.get("id") == item.get("id"):
            return
        for c in self._cards:
            c.configure(border_color=("gray80", "gray30"))
        clicked.configure(border_color=(_ACCENT, _ACCENT))
        self._selected   = item
        self._ready_path = None  # new selection clears any prior download
        self._update_detail(item)
        if self._state not in (AppState.PROCESSING, AppState.UPLOADING, AppState.SEARCHING):
            self._set_state(AppState.READY)

    def _update_detail(self, item: Dict) -> None:
        title = item.get("title") or "Untitled"
        self._detail_title_lbl.configure(text=title)
        self._detail_channel_lbl.configure(text=item.get("channel") or "")
        self._detail_url_lbl.configure(text=item.get("url") or "")
        self._detail_thumb.configure(image=_grey_image(_THUMB_DETAIL))
        self._title_entry.delete(0, "end")
        self._title_entry.insert(0, title)
        vid_id = item.get("id")
        if vid_id:
            threading.Thread(
                target=self._bg_detail_thumb, args=(vid_id,), daemon=True
            ).start()

    def _bg_detail_thumb(self, video_id: str) -> None:
        img = _fetch_thumbnail(video_id, _THUMB_DETAIL)
        if img:
            self.master.after(0, lambda i=img: self._detail_thumb.configure(image=i))

    # ── Playlists & Categories ───────────────────────────────────────────────

    def _bg_load_playlists(self) -> None:
        from core.uploader import get_playlists
        try:
            playlists = get_playlists()
        except Exception as exc:
            logger.debug("Playlist load failed: %s", exc)
            playlists = []
        self.master.after(0, lambda p=playlists: self._set_playlists(p))

    def _set_playlists(self, playlists: List[Dict]) -> None:
        self._playlists = playlists
        values = ["— no playlist —"] + [
            str(p.get("title") or p.get("id") or f"Playlist {i + 1}")
            for i, p in enumerate(playlists)
        ]
        self._playlist_menu.configure(values=values)
        self._playlist_var.set("— no playlist —")

    def _get_playlist_id(self) -> Optional[str]:
        chosen = self._playlist_var.get()
        if chosen == "— no playlist —":
            return None
        for p in self._playlists:
            label = str(p.get("title") or p.get("id") or "")
            if label == chosen:
                return str(p.get("id") or "")
        return None

    def _on_new_playlist(self) -> None:
        dialog = ctk.CTkInputDialog(
            text="Enter a name for the new playlist:",
            title="New Playlist",
        )
        name = dialog.get_input()
        if name and name.strip():
            threading.Thread(
                target=self._bg_create_playlist, args=(name.strip(),), daemon=True,
            ).start()

    def _bg_create_playlist(self, name: str) -> None:
        from core.uploader import create_playlist
        try:
            new_id = create_playlist(name)
            if new_id:
                logger.info("Created playlist %r (id=%s)", name, new_id)
        except Exception as exc:
            logger.warning("Could not create playlist: %s", exc)
        self._bg_load_playlists()

    def _bg_load_categories(self) -> None:
        from core.uploader import get_categories
        try:
            categories = get_categories()
        except Exception as exc:
            logger.debug("Category load failed: %s", exc)
            categories = []
        self.master.after(0, lambda c=categories: self._set_categories(c))

    def _set_categories(self, categories: List[Dict]) -> None:
        self._categories = categories
        values = ["— no category —"] + [
            str(c.get("title") or c.get("id") or f"Category {i + 1}")
            for i, c in enumerate(categories)
        ]
        self._category_menu.configure(values=values)
        self._category_var.set("— no category —")

    def _get_category_id(self) -> Optional[str]:
        chosen = self._category_var.get()
        if chosen == "— no category —":
            return None
        for c in self._categories:
            label = str(c.get("title") or c.get("id") or "")
            if label == chosen:
                return str(c.get("id") or "")
        return None

    def _on_new_category(self) -> None:
        dialog = ctk.CTkInputDialog(
            text="Enter patient/category name:",
            title="New Category",
        )
        name = dialog.get_input()
        if name and name.strip():
            threading.Thread(
                target=self._bg_create_category, args=(name.strip(),), daemon=True,
            ).start()

    def _bg_create_category(self, name: str) -> None:
        from core.uploader import create_category
        try:
            new_id = create_category(name)
            if new_id:
                logger.info("Created category %r (id=%s)", name, new_id)
        except Exception as exc:
            logger.warning("Could not create category: %s", exc)
        self._bg_load_categories()

    # ── Download & Process ────────────────────────────────────────────────────

    def _on_download_process(self) -> None:
        if not self._selected:
            tkinter.messagebox.showwarning("Warning", "Select a video first.")
            return
        url = self._selected.get("url")
        if not url:
            tkinter.messagebox.showerror("Error", "Selected video has no URL.")
            return
        self._ready_path = None
        self._set_state(AppState.PROCESSING)
        self._progress_reset("Starting download…")
        threading.Thread(target=self._bg_process, args=(url,), daemon=True).start()

    def _bg_process(self, url: str) -> None:
        from workflows.unified_pipeline import JobOptions, process_video_job

        def _prog(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                frac = extract_download_progress_fraction(d)
                pct_lbl = str(d.get("_percent_str") or "").strip()
                if not pct_lbl and frac is not None:
                    pct_lbl = f"{frac * 100:.1f}%"
                status_msg = f"Downloading… {pct_lbl}".rstrip()
                self._queue_download_progress_update(frac, status_msg)

            elif status == "finished":
                self._clear_download_progress_queue()
                def _upd_proc():
                    self._progress_start_indeterminate("Processing… (see log for details)")

                self.master.after(0, _upd_proc)

        opts = JobOptions(
            source_url=url,
            upload=False,
            convert_if_needed=True,
            progress_callback=_prog,
        )
        try:
            result = process_video_job(opts)
        except Exception as exc:
            self.master.after(0, lambda e=str(exc): self._on_process_error(e))
            return
        self.master.after(0, lambda r=result: self._on_process_done(r))

    def _on_process_done(self, result) -> None:
        self._clear_download_progress_queue()
        self._progress.stop()
        self._progress.configure(mode="determinate")
        if not result.success:
            self._progress.set(0)
            self._set_state(AppState.READY)
            self._set_status("Processing failed.")
            tkinter.messagebox.showerror(
                "Processing failed", result.error or "Unknown error")
            return

        self._ready_path = (
            result.converted_video_path
            or result.normalized_video_path
            or result.original_video_path
        )
        self._progress.set(1.0)
        proj      = result.projection_type or "unknown"
        conf      = float(result.confidence or 0)
        conv_note = " — converted to equirectangular" if result.converted_video_path else ""
        self._set_status(f"Done — {proj} ({conf:.0%}){conv_note}")
        self._set_state(AppState.PROCESSED)
        tkinter.messagebox.showinfo(
            "Processing complete",
            f"Projection: {proj}\nConfidence: {conf:.0%}{conv_note}\n\nReady to upload.",
        )

    def _on_process_error(self, msg: str) -> None:
        self._progress_reset()
        self._set_state(AppState.READY if self._selected else AppState.IDLE)
        self._set_status("Processing failed.")
        tkinter.messagebox.showerror("Processing error", msg)

    # ── Upload ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tags(raw: str) -> List[str]:
        return [t.strip() for t in (raw or "").split(",") if t.strip()]

    def _on_upload(self) -> None:
        if not self._ready_path:
            tkinter.messagebox.showwarning(
                "Warning",
                "No processed video available.\nRun Download & Process first.",
            )
            return
        title = self._title_entry.get().strip()
        if not title:
            tkinter.messagebox.showwarning("Warning", "Enter a title for the upload.")
            return
        playlist_id = self._get_playlist_id()
        category_id = self._get_category_id()
        if not category_id:
            tkinter.messagebox.showwarning(
                "Warning",
                "Select or create a patient category before uploading.",
            )
            return
        tags = self._parse_tags(self._tags_entry.get().strip())
        self._set_state(AppState.UPLOADING)
        self._progress_start_indeterminate("Uploading…")
        threading.Thread(
            target=self._bg_upload,
            args=(self._ready_path, title, playlist_id, category_id, tags),
            daemon=True,
        ).start()

    def _bg_upload(
        self,
        path: str,
        title: str,
        playlist_id: Optional[str],
        category_id: Optional[str],
        tags: List[str],
    ) -> None:
        from core.uploader import upload_video_asset
        try:
            result = upload_video_asset(
                video_path=path,
                title=title,
                description="",
                playlist_id=playlist_id,
                category_id=category_id,
                tags=tags,
            )
        except (MediaCMSError, Exception) as exc:
            self.master.after(0, lambda e=str(exc): self._on_upload_error(e))
            return
        self.master.after(0, lambda r=result: self._on_upload_done(r))

    def _on_upload_done(self, result) -> None:
        self._progress.stop()
        self._progress.configure(mode="determinate")
        if result.success:
            self._progress.set(1.0)
            self._set_status("Upload complete.")
            self._set_state(AppState.PROCESSED)
            tkinter.messagebox.showinfo(
                "Upload complete",
                f"Video uploaded successfully.\n{result.media_url or ''}",
            )
        else:
            self._progress.set(0)
            self._set_state(AppState.PROCESSED)
            self._set_status("Upload failed.")
            tkinter.messagebox.showerror("Upload failed", result.error or "Unknown error")

    def _on_upload_error(self, msg: str) -> None:
        self._progress_reset()
        self._set_state(AppState.PROCESSED)
        self._set_status("Upload failed.")
        tkinter.messagebox.showerror("Upload error", msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_gui() -> None:
    """Launch the VR360 Manager GUI."""
    setup_logging(level=logging.INFO)
    root = ctk.CTk()
    try:
        VR360ManagerApp(root)
    except Exception as exc:
        logger.exception("Fatal error during GUI initialisation")
        tkinter.messagebox.showerror("Startup error", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    run_gui()
