"""Video player widget — cv2 decoding + Qt rendering + DB-driven overlays.

Why cv2-based instead of ``QMediaPlayer``: we need precise per-frame
control (frame stepping, frame-accurate seek, click hit-testing on
specific frame data), and we want to draw arbitrary overlays on top
of each frame from the analytics DB. ``QMediaPlayer`` is great for
streaming playback but its frame-extraction API is awkward and
hardware-specific. Decoding ~30 fps with cv2 is fine on CPU even on
Iris Xe — the existing pipeline already does it.

Layout:
  ┌─────────────────────────────────┐
  │                                 │
  │      VideoSurface (QLabel)      │  ← shows the rendered frame
  │                                 │
  ├─────────────────────────────────┤
  │ ⏮  ⏯  ⏭   [████─────] 1234/8332 │  ← controls + seek + frame counter
  └─────────────────────────────────┘

Signals:
  ``player_clicked(track_id: int)``   — emitted when the operator
        clicks a bbox. The main window connects this to the track
        mapping panel / event-tag pipeline.
  ``frame_changed(frame_number: int)`` — emitted on every advance,
        so other widgets (event panel) can refresh.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QMouseEvent, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .repository import (
    FramePlayerPos,
    FrameBallPos,
    TrackLabel,
    get_frame_state,
    hit_test,
)


# Overlay colours — match the existing main.py annotator look so
# operators see consistent visuals across the pipeline output and
# the tagger.
_TEAM_FALLBACK_BGR = {
    1: (255, 80, 80),    # bluish (BGR)
    2: (80, 80, 255),    # reddish
}
_REF_BGR = (0, 255, 255)        # yellow
_GK_BGR = (255, 0, 255)         # magenta
_BALL_BGR = (0, 255, 0)         # green
_BALL_INTERP_BGR = (100, 220, 100)
_HIGHLIGHT_BGR = (0, 255, 255)  # selected track halo


# ---------------------------------------------------------------------------
# VideoSurface — the QLabel that draws the rendered frame and emits clicks.
# ---------------------------------------------------------------------------
class VideoSurface(QLabel):
    """QLabel that scales pixmaps and reports click coordinates in
    *original* video pixel space (not the displayed widget space)."""

    clicked_in_video = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._video_w = 1
        self._video_h = 1

    def set_video_dimensions(self, w: int, h: int) -> None:
        self._video_w = max(1, w)
        self._video_h = max(1, h)

    def show_frame(self, bgr: np.ndarray) -> None:
        """Display a BGR ndarray as a scaled pixmap."""
        h, w = bgr.shape[:2]
        # cv2 → QImage: convert BGR → RGB, then wrap as Format_RGB888.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Each row is rgb.shape[1] * 3 bytes; provide stride explicitly so
        # arrays with non-default memory layouts work.
        qimg = QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888,
        ).copy()  # copy() so the underlying ndarray can be GC'd safely
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Translate a widget-space click into video-space coordinates
        and re-emit. Accounts for letterboxing — when the video aspect
        ratio doesn't match the widget, there are blank bars we have to
        subtract before scaling.
        """
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return
        widget_w, widget_h = self.width(), self.height()
        pix_w, pix_h = pix.width(), pix.height()
        # Letterbox offsets — pixmap is centred in the label.
        off_x = (widget_w - pix_w) / 2.0
        off_y = (widget_h - pix_h) / 2.0
        click_x = event.position().x() - off_x
        click_y = event.position().y() - off_y
        if click_x < 0 or click_y < 0 or click_x > pix_w or click_y > pix_h:
            return  # click landed in the letterbox bars — ignore
        # Scale displayed pixmap coords back to original video coords.
        vx = click_x * (self._video_w / pix_w)
        vy = click_y * (self._video_h / pix_h)
        self.clicked_in_video.emit(vx, vy)


# ---------------------------------------------------------------------------
# Drawing — reuses the look of src.trackers.annotator.draw_one_frame but
# pulls position data from the DB rather than the stub dict.
# ---------------------------------------------------------------------------
def _draw_ellipse(img, p: FramePlayerPos, color, label: str | None) -> None:
    cx = int((p.bbox_x1 + p.bbox_x2) * 0.5)
    width = int(p.bbox_x2 - p.bbox_x1)
    cv2.ellipse(
        img,
        center=(cx, int(p.bbox_y2)),
        axes=(int(width * 0.5), int(width * 0.18)),
        angle=0.0, startAngle=-45, endAngle=235,
        color=color, thickness=2, lineType=cv2.LINE_4,
    )
    if not label:
        return
    # Width-adaptive label box — short labels (just a track id) stay
    # narrow; mapped labels like "10 Petras" need more horizontal room.
    # Estimate text width via font metrics (cv2 returns size in px).
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    rect_w = max(36, tw + 10)
    rect_h = max(18, th + 8)
    rx1 = int(cx - rect_w / 2)
    ry1 = int(p.bbox_y2 + 5)
    cv2.rectangle(img, (rx1, ry1), (rx1 + rect_w, ry1 + rect_h), color, cv2.FILLED)
    cv2.putText(
        img, label, (rx1 + 5, ry1 + rect_h - 5),
        font, scale, (0, 0, 0), thick,
    )


def _draw_ball(img, b: FrameBallPos) -> None:
    cx = int(b.cx_image)
    top = int(b.bbox_y1) - 6
    color = _BALL_INTERP_BGR if b.interpolated else _BALL_BGR
    pts = np.array([[cx, top + 18], [cx - 10, top], [cx + 10, top]], np.int32)
    cv2.drawContours(img, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(img, [pts], 0, (0, 0, 0), 2)


def _color_for(p: FramePlayerPos) -> tuple[int, int, int]:
    if p.cls == "goalkeeper":
        return _GK_BGR
    if p.cls == "referee":
        return _REF_BGR
    if p.team in (1, 2):
        return _TEAM_FALLBACK_BGR[p.team]
    return (200, 200, 200)


def render_overlays(
    bgr: np.ndarray,
    players: list[FramePlayerPos],
    ball: FrameBallPos | None,
    selected_track_id: int | None,
    track_labels: dict[int, TrackLabel] | None = None,
) -> np.ndarray:
    """Draw player ellipses + ball triangle + labels on a BGR frame.

    ``track_labels`` is the override map ``{track_id: TrackLabel}`` from
    the TrackMappingPanel. A track gets its mapped name ONLY when:

      * the operator has assigned that track_id to a roster player, AND
      * the current frame's CV-detected team matches the team_side the
        mapping was stored with (or the current frame's team is None,
        which we treat as "trust the mapping" rather than overriding).

    The team-side gate is what catches ByteTrack ID reuse: if track 47
    was Petras (team 1) early in the clip, then ByteTrack frees and
    re-assigns 47 to a player on team 2 later, the renderer falls back
    to "47" rather than mis-painting "10 Petras" on the wrong team.

    A selected track gets a yellow halo so the operator has visual
    confirmation of who they clicked.

    Returns a NEW ndarray; doesn't modify input.
    """
    canvas = bgr.copy()
    labels = track_labels or {}
    for p in players:
        mapped = labels.get(p.track_id)

        # --- Colour selection ---
        # GK / referee classes always win — they're rendered with their
        # class-specific colour regardless of mapping or CV team.
        # For 'player' class:
        #   * mapped track → lock to the mapping's stored team_side.
        #     Eliminates colour flicker on home-team players when the
        #     team_assigner briefly drops them to None / opposite team.
        #   * unmapped track → use the per-frame CV team (with a gray
        #     fallback for None).
        if p.cls == "goalkeeper":
            color = _GK_BGR
        elif p.cls == "referee":
            color = _REF_BGR
        elif mapped is not None:
            color = _TEAM_FALLBACK_BGR.get(
                mapped.expected_team_side, (200, 200, 200),
            )
        elif p.team in (1, 2):
            color = _TEAM_FALLBACK_BGR[p.team]
        else:
            color = (200, 200, 200)

        # --- Label selection ---
        # Unmapped tracks render with NO label box — keeps the opposing
        # team's overlay visually clean (the operator only cares that
        # it's the right colour, not what its raw track_id is).
        # Mapped tracks paint the mapped name unless the current frame's
        # CV team is the strict OPPOSITE of the mapping (a strong
        # indicator that ByteTrack reused this track_id for a player
        # on the other team — the cross-team leak case).
        if mapped is None:
            label = None
        elif p.team is not None and p.team != mapped.expected_team_side:
            label = str(p.track_id)
        else:
            label = mapped.text

        _draw_ellipse(canvas, p, color, label)
        if p.track_id == selected_track_id and p.cls != "ball":
            cx = int((p.bbox_x1 + p.bbox_x2) * 0.5)
            width = int(p.bbox_x2 - p.bbox_x1)
            cv2.ellipse(
                canvas,
                center=(cx, int(p.bbox_y2)),
                axes=(int(width * 0.65), int(width * 0.24)),
                angle=0.0, startAngle=-45, endAngle=235,
                color=_HIGHLIGHT_BGR, thickness=4, lineType=cv2.LINE_AA,
            )
    if ball is not None:
        _draw_ball(canvas, ball)
    return canvas


# ---------------------------------------------------------------------------
# VideoWidget — the composed widget the main window embeds.
# ---------------------------------------------------------------------------
class VideoWidget(QWidget):
    """Self-contained video player + transport controls + DB-driven
    overlay rendering.

    Usage::

        v = VideoWidget(connection, match_id, video_path, fps)
        v.player_clicked.connect(on_player_click)
        v.frame_changed.connect(on_frame_change)
    """

    player_clicked = pyqtSignal(int)        # track_id
    frame_changed = pyqtSignal(int)         # frame_number

    def __init__(
        self,
        connection: sqlite3.Connection,
        match_id: int,
        video_path: str,
        fps: float,
        frame_width: int,
        frame_height: int,
        n_frames: int,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._con = connection
        self._match_id = match_id
        self._fps = fps
        self._n_frames = n_frames
        self._video_w = frame_width
        self._video_h = frame_height

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self._current_frame_number = 0
        self._selected_track_id: int | None = None
        # Map of track_id -> TrackLabel pushed in by the main window
        # whenever the TrackMappingPanel updates. Each TrackLabel carries
        # both the display text and the team_side gate that prevents
        # ByteTrack ID reuse from misnaming opposite-team players.
        self._track_labels: dict[int, TrackLabel] = {}
        # Cache the most recently decoded BGR frame so a "repaint with
        # different selection" doesn't trigger a full cv2 seek + decode
        # cycle. Without this, every player click costs ~50-100 ms of
        # backwards-seek even though the pixels haven't changed.
        self._last_bgr: np.ndarray | None = None
        self._last_bgr_frame_number: int = -1

        # ---- Widgets ----
        self._surface = VideoSurface(self)
        self._surface.set_video_dimensions(frame_width, frame_height)
        self._surface.clicked_in_video.connect(self._on_video_click)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self.toggle_play)

        self._prev_btn = QPushButton("⏮")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(self.step_back)

        self._next_btn = QPushButton("⏭")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(self.step_forward)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, n_frames - 1))
        self._slider.sliderMoved.connect(self.seek)

        self._status = QLabel("0 / 0")
        self._status.setMinimumWidth(120)
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # ---- Layout ----
        controls = QHBoxLayout()
        controls.addWidget(self._prev_btn)
        controls.addWidget(self._play_btn)
        controls.addWidget(self._next_btn)
        controls.addWidget(self._slider, stretch=1)
        controls.addWidget(self._status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._surface, stretch=1)
        layout.addLayout(controls)

        # ---- Playback timer ----
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._playing = False

        # Space-bar play/pause toggle. WindowShortcut scope so it fires
        # regardless of which widget has focus — operator shouldn't have
        # to move the mouse back to the ▶ button between tags. We bail
        # out if a text input has focus so the operator can still type
        # spaces into the player name field.
        self._play_shortcut = QShortcut(
            QKeySequence("Space"), self,
            context=Qt.ShortcutContext.WindowShortcut,
        )
        self._play_shortcut.activated.connect(self._on_space_pressed)

        # Show the first frame immediately.
        self.show_frame(0)

    def _on_space_pressed(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            return   # let the operator type a literal space
        self.toggle_play()

    # ---------------------------------------------------------- transport
    def toggle_play(self) -> None:
        if self._playing:
            self._timer.stop()
            self._play_btn.setText("▶")
        else:
            self._timer.start(int(round(1000.0 / self._fps)) if self._fps > 0 else 33)
            self._play_btn.setText("⏸")
        self._playing = not self._playing

    def step_forward(self) -> None:
        if self._playing:
            self.toggle_play()
        if self._current_frame_number < self._n_frames - 1:
            self.show_frame(self._current_frame_number + 1)

    def step_back(self) -> None:
        if self._playing:
            self.toggle_play()
        if self._current_frame_number > 0:
            self.show_frame(self._current_frame_number - 1)

    def seek(self, frame_number: int) -> None:
        if self._playing:
            self.toggle_play()
        self.show_frame(frame_number)

    # ---------------------------------------------------------- selection
    def select_track(self, track_id: int | None) -> None:
        """Highlight a specific track on the current frame. Called by
        the main window when the operator picks a track from the side
        panel, OR after a click hit-test. Skips the cv2 decode by
        repainting from the cached frame buffer."""
        self._selected_track_id = track_id
        self._repaint_overlays()

    def set_track_labels(self, labels: dict[int, TrackLabel]) -> None:
        """Push a fresh ``track_id -> TrackLabel`` map. Triggers an
        overlay-only repaint so newly mapped names appear immediately
        without re-decoding."""
        self._track_labels = dict(labels)
        self._repaint_overlays()

    def _repaint_overlays(self) -> None:
        """Re-render the current frame using the cached BGR buffer.

        Falls back to a full ``show_frame`` if no cache is available
        (first paint after construction, etc.). Saves the cv2 seek +
        decode cycle when only the overlay changes — cuts perceived
        click-lag from ~80 ms to ~5 ms.
        """
        if (
            self._last_bgr is None
            or self._last_bgr_frame_number != self._current_frame_number
        ):
            self.show_frame(self._current_frame_number)
            return
        players, ball = get_frame_state(
            self._con, self._match_id, self._current_frame_number,
        )
        rendered = render_overlays(
            self._last_bgr, players, ball, self._selected_track_id,
            track_labels=self._track_labels,
        )
        self._surface.show_frame(rendered)

    # ---------------------------------------------------------- rendering
    def show_frame(self, frame_number: int) -> None:
        if frame_number < 0 or frame_number >= self._n_frames:
            return
        # Seek if non-sequential. cv2 set() on POS_FRAMES is slow for tiny
        # jumps but unavoidable for backwards / scrubbed seeks.
        if frame_number != self._current_frame_number + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, bgr = self._cap.read()
        if not ok:
            return
        self._current_frame_number = frame_number
        # Cache for cheap overlay-only repaints (selection change,
        # mapping label updates).
        self._last_bgr = bgr
        self._last_bgr_frame_number = frame_number

        players, ball = get_frame_state(self._con, self._match_id, frame_number)
        rendered = render_overlays(
            bgr, players, ball, self._selected_track_id,
            track_labels=self._track_labels,
        )
        self._surface.show_frame(rendered)

        # Update controls — guard against re-entrancy via blockSignals.
        self._slider.blockSignals(True)
        self._slider.setValue(frame_number)
        self._slider.blockSignals(False)
        ts_s = frame_number / self._fps if self._fps > 0 else 0
        m, s = divmod(int(ts_s), 60)
        self._status.setText(
            f"{frame_number:>5} / {self._n_frames - 1}    {m:02d}:{s:02d}"
        )
        self.frame_changed.emit(frame_number)

    # ---------------------------------------------------------- internals
    def _on_tick(self) -> None:
        if self._current_frame_number >= self._n_frames - 1:
            self.toggle_play()
            return
        self.show_frame(self._current_frame_number + 1)

    def _on_video_click(self, vx: float, vy: float) -> None:
        players, _ball = get_frame_state(
            self._con, self._match_id, self._current_frame_number,
        )
        hit = hit_test(players, vx, vy)
        if hit is None:
            return
        self.player_clicked.emit(hit.track_id)
        self.select_track(hit.track_id)

    def closeEvent(self, event) -> None:  # noqa: D401
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
