import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LyricLine:
    text: str
    timestamp: Optional[float] = None

    def to_lrc(self) -> str:
        if self.timestamp is None:
            return self.text
        minutes = int(self.timestamp) // 60
        seconds = self.timestamp - minutes * 60
        return f"[{minutes:02d}:{seconds:05.2f}]{self.text}"


def parse_lrc_timestamp(tag: str) -> Optional[float]:
    m = re.match(r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?", tag)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    ms_raw = m.group(3)
    if ms_raw:
        if len(ms_raw) == 1:
            ms = int(ms_raw) * 100
        elif len(ms_raw) == 2:
            ms = int(ms_raw) * 10
        else:
            ms = int(ms_raw)
    else:
        ms = 0
    return minutes * 60 + seconds + ms / 1000


def parse_lyrics(lyrics_text: str) -> List[LyricLine]:
    lines: List[LyricLine] = []
    for raw_line in lyrics_text.strip().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        timestamp = None
        text = stripped
        match = re.match(r"\[(\d{1,2}:\d{2}(?:\.\d{1,3})?)\](.*)", stripped)
        if match:
            timestamp = parse_lrc_timestamp(match.group(1))
            text = match.group(2).strip()
        lines.append(LyricLine(text=text, timestamp=timestamp))
    return lines


def align_lyrics(lyrics_text: str, audio_duration: float) -> str:
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")

    lines = parse_lyrics(lyrics_text)
    if not lines:
        return ""

    has_partial = any(line.timestamp is not None for line in lines)
    has_all = all(line.timestamp is not None for line in lines)

    if has_all:
        return "\n".join(line.to_lrc() for line in lines)

    if not has_partial:
        n = len(lines)
        interval = audio_duration / n
        for i, line in enumerate(lines):
            line.timestamp = i * interval
        return "\n".join(line.to_lrc() for line in lines)

    anchors = [(i, lines[i].timestamp) for i in range(len(lines)) if lines[i].timestamp is not None]

    for idx, (anchor_i, anchor_t) in enumerate(anchors):
        if idx == len(anchors) - 1:
            next_i = len(lines)
            next_t = audio_duration
        else:
            next_i, next_t = anchors[idx + 1]

        span_count = next_i - anchor_i
        if span_count <= 1:
            continue
        interval = (next_t - anchor_t) / span_count
        for j in range(anchor_i + 1, next_i):
            lines[j].timestamp = anchor_t + (j - anchor_i) * interval

    return "\n".join(line.to_lrc() for line in lines)


def lrc_to_text(lrc_str: str) -> str:
    return re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", lrc_str)


if __name__ == "__main__":
    lyrics = """看雪在手中融化
看夜在心中发芽
看花在风中凋零
看梦在光中飞翔"""

    print("=== 均匀分配 ===")
    result = align_lyrics(lyrics, audio_duration=16.0)
    print(result)

    print("\n=== 带锚点对齐 ===")
    anchored = """[00:02.00]看雪在手中融化
看夜在心中发芽
[00:08.00]看花在风中凋零
看梦在光中飞翔"""
    result2 = align_lyrics(anchored, audio_duration=16.0)
    print(result2)

    print("\n=== 已有完整时间戳 ===")
    full_lrc = """[00:00.50]看雪在手中融化
[00:04.50]看夜在心中发芽
[00:08.50]看花在风中凋零
[00:12.50]看梦在光中飞翔"""
    result3 = align_lyrics(full_lrc, audio_duration=16.0)
    print(result3)
