#!/usr/bin/env python3
"""Export local Codex session logs as per-session HTML transcripts."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SESSIONS_ROOT = Path("/home/ch226520/.codex/sessions")
OUTPUT_ROOT = Path("session_logs")
STYLE_PATH = OUTPUT_ROOT / "style.css"
INDEX_PATH = OUTPUT_ROOT / "index.html"


@dataclass
class Message:
    timestamp: str
    role: str
    phase: str | None
    text: str


@dataclass
class SessionTranscript:
    session_id: str
    title: str
    started_at: str | None
    cwd: str | None
    source_file: Path
    messages: list[Message]


def collect_text(content: Iterable[dict], role: str) -> str:
    """Flatten message content parts into displayable text."""
    parts: list[str] = []
    for item in content or []:
        kind = item.get("type")
        if role == "user" and kind == "input_text":
            parts.append(item.get("text", ""))
        elif role == "assistant" and kind == "output_text":
            parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


def should_skip_user_text(text: str) -> bool:
    """Filter framework boilerplate that is not part of the human conversation."""
    cleaned = text.strip()
    return (
        cleaned.startswith("# AGENTS.md instructions for ")
        or cleaned.startswith("<environment_context>")
    )


def slugify(text: str) -> str:
    """Create a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return slug or "session"


def format_session_title(started_at: str | None, fallback: str) -> str:
    """Format the session start time as a human-friendly title."""
    if not started_at:
        return fallback
    try:
        dt = datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return fallback
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_message_timestamp(timestamp: str) -> str:
    """Render ISO timestamps in local time for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return timestamp
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def render_message_body(text: str) -> str:
    """Render message text while preserving terminal-like formatting."""
    return f"<pre>{html.escape(text)}</pre>"


def parse_session(session_file: Path) -> SessionTranscript:
    """Parse one Codex session JSONL file into user/assistant messages."""
    session_id = session_file.stem
    started_at: str | None = None
    cwd: str | None = None
    messages: list[Message] = []

    with session_file.open() as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "session_meta":
                payload = obj.get("payload", {})
                session_id = payload.get("id") or session_id
                started_at = payload.get("timestamp") or started_at
                cwd = payload.get("cwd") or cwd
                continue

            if obj.get("type") != "response_item":
                continue

            payload = obj.get("payload", {})
            if payload.get("type") != "message":
                continue

            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue

            text = collect_text(payload.get("content", []), role)
            if not text:
                continue
            if role == "user" and should_skip_user_text(text):
                continue

            messages.append(
                Message(
                    timestamp=obj.get("timestamp", ""),
                    role=role,
                    phase=payload.get("phase"),
                    text=text,
                )
            )

    return SessionTranscript(
        session_id=session_id,
        title=format_session_title(started_at, session_file.stem),
        started_at=started_at,
        cwd=cwd,
        source_file=session_file,
        messages=messages,
    )


def render_session_html(session: SessionTranscript) -> str:
    """Render one transcript page."""
    body_parts = []
    for message in session.messages:
        role_label = "User" if message.role == "user" else "Assistant"
        phase_label = f" · {message.phase}" if message.phase else ""
        body_parts.append(
            "\n".join(
                [
                    f'<article class="message {message.role}">',
                    '  <div class="message-meta">',
                    f"    <span class=\"role\">{role_label}</span>",
                    f"    <span class=\"phase\">{html.escape(phase_label)}</span>",
                    f"    <time>{html.escape(format_message_timestamp(message.timestamp))}</time>",
                    "  </div>",
                    f'  <div class="message-body">{render_message_body(message.text)}</div>',
                    "</article>",
                ]
            )
        )

    cwd_html = (
        f'<div class="meta-row"><span class="meta-key">CWD</span>'
        f"<code>{html.escape(session.cwd)}</code></div>"
        if session.cwd
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(session.title)}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="page">
    <header class="session-header">
      <p class="eyebrow">Codex Session Transcript</p>
      <h1>{html.escape(session.title)}</h1>
      <div class="meta-grid">
        <div class="meta-row"><span class="meta-key">Session ID</span><code>{html.escape(session.session_id)}</code></div>
        <div class="meta-row"><span class="meta-key">Source</span><code>{html.escape(str(session.source_file))}</code></div>
        {cwd_html}
        <div class="meta-row"><span class="meta-key">Messages</span><span>{len(session.messages)}</span></div>
      </div>
      <p><a class="back-link" href="index.html">Back to Index</a></p>
    </header>
    <section class="transcript">
      {''.join(body_parts)}
    </section>
  </main>
</body>
</html>
"""


def render_index_html(sessions: list[SessionTranscript]) -> str:
    """Render the session index page."""
    items = []
    for session in sessions:
        filename = f"{slugify(session.title)}-{session.session_id[:8]}.html"
        items.append(
            "\n".join(
                [
                    '<li class="session-card">',
                    f'  <a href="{html.escape(filename)}">',
                    f'    <span class="session-title">{html.escape(session.title)}</span>',
                    f'    <span class="session-subtitle">{len(session.messages)} messages</span>',
                    f'    <span class="session-subtitle">{html.escape(session.session_id)}</span>',
                    "  </a>",
                    "</li>",
                ]
            )
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codex Session Logs</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="page">
    <header class="session-header">
      <p class="eyebrow">Codex Session Logs</p>
      <h1>Session Index</h1>
      <div class="meta-grid">
        <div class="meta-row"><span class="meta-key">Source</span><code>{html.escape(str(SESSIONS_ROOT))}</code></div>
        <div class="meta-row"><span class="meta-key">Sessions</span><span>{len(sessions)}</span></div>
        <div class="meta-row"><span class="meta-key">Generated</span><span>{html.escape(datetime.now().astimezone().isoformat(timespec="seconds"))}</span></div>
      </div>
    </header>
    <section class="index-list">
      <ul>
        {''.join(items)}
      </ul>
    </section>
  </main>
</body>
</html>
"""


def stylesheet() -> str:
    """Shared stylesheet for all exported pages."""
    return """
:root {
  color-scheme: dark;
  --bg: #0b0f14;
  --panel: #101721;
  --panel-2: #151f2c;
  --border: #223041;
  --text: #d7e1ea;
  --muted: #8da0b3;
  --user: #18324b;
  --assistant: #13261e;
  --accent: #7cc7ff;
  --accent-2: #89e3a0;
  --shadow: rgba(0, 0, 0, 0.28);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background:
    radial-gradient(circle at top, rgba(124, 199, 255, 0.08), transparent 28rem),
    linear-gradient(180deg, #0a0d12, var(--bg));
  color: var(--text);
  font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.page {
  width: min(1100px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 24px 0 48px;
}

.session-header {
  position: sticky;
  top: 0;
  z-index: 10;
  margin-bottom: 20px;
  padding: 20px;
  background: rgba(11, 15, 20, 0.88);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 10px 30px var(--shadow);
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

h1 {
  margin: 0 0 16px;
  font-size: 24px;
  line-height: 1.2;
}

.meta-grid {
  display: grid;
  gap: 8px;
}

.meta-row {
  display: flex;
  gap: 12px;
  align-items: baseline;
  flex-wrap: wrap;
  color: var(--muted);
}

.meta-key {
  min-width: 84px;
  color: var(--text);
}

code {
  padding: 1px 6px;
  background: #0a1119;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.back-link,
.session-card a {
  color: var(--text);
  text-decoration: none;
}

.back-link:hover,
.session-card a:hover {
  color: var(--accent);
}

.transcript {
  display: grid;
  gap: 14px;
}

.message {
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 24px var(--shadow);
}

.message.user {
  background: linear-gradient(180deg, rgba(24, 50, 75, 0.88), rgba(18, 38, 58, 0.92));
}

.message.assistant {
  background: linear-gradient(180deg, rgba(19, 38, 30, 0.9), rgba(15, 29, 23, 0.94));
}

.message-meta {
  display: flex;
  gap: 12px;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  background: rgba(0, 0, 0, 0.18);
}

.role {
  font-weight: 700;
}

.message.user .role {
  color: var(--accent);
}

.message.assistant .role {
  color: var(--accent-2);
}

.phase,
time {
  color: var(--muted);
}

.message-body {
  padding: 0;
}

.message-body pre {
  margin: 0;
  padding: 16px 14px;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.index-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 12px;
}

.session-card a {
  display: grid;
  gap: 4px;
  padding: 16px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 10px 24px var(--shadow);
}

.session-title {
  font-size: 16px;
  color: var(--text);
}

.session-subtitle {
  color: var(--muted);
}

@media (max-width: 720px) {
  .page {
    width: min(100vw - 16px, 1100px);
    padding-top: 12px;
  }

  .session-header {
    padding: 16px;
  }

  h1 {
    font-size: 20px;
  }

  .message-meta {
    flex-wrap: wrap;
  }
}
""".strip() + "\n"


def main() -> None:
    """Generate the index page and per-session HTML pages."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STYLE_PATH.write_text(stylesheet())

    sessions = [parse_session(path) for path in sorted(SESSIONS_ROOT.rglob("rollout-*.jsonl"))]

    for session in sessions:
        filename = f"{slugify(session.title)}-{session.session_id[:8]}.html"
        (OUTPUT_ROOT / filename).write_text(render_session_html(session))

    INDEX_PATH.write_text(render_index_html(sessions))

    print(f"wrote {len(sessions)} sessions to {OUTPUT_ROOT}")
    print(f"index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
