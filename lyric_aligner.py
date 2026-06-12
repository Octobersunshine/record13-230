import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class LyricLine:
    text: str
    timestamp: Optional[float] = None
    weight: float = 0.0

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


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_DIGIT_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[，。！？、；：\"'（）【】《》,.;:!?()\[\]<>]")


def estimate_syllable_count(text: str, lang_weights: Optional[Dict[str, float]] = None) -> float:
    if lang_weights is None:
        lang_weights = {"zh": 1.0, "en": 1.0, "num": 0.7}

    zh_count = len(_CJK_RE.findall(text))
    en_words = _WORD_RE.findall(text)
    num_tokens = _DIGIT_RE.findall(text)

    en_syllables = 0.0
    for word in en_words:
        wl = len(word.lower())
        if wl <= 3:
            en_syllables += 1.0
        elif wl <= 5:
            en_syllables += 1.5
        elif wl <= 8:
            en_syllables += 2.0
        else:
            en_syllables += 2.5 + (wl - 8) * 0.15
        if re.search(r"(ing|tion|sion|able|ible|ment|ness|ful|less|ly)$", word.lower()):
            en_syllables += 0.3

    num_syllables = sum(max(1.0, len(tok) * 0.5) for tok in num_tokens)

    return (
        zh_count * lang_weights["zh"]
        + en_syllables * lang_weights["en"]
        + num_syllables * lang_weights["num"]
    )


def compute_line_weight(text: str) -> float:
    base = estimate_syllable_count(text)
    if base <= 0:
        base = 1.0
    clean = _PUNCT_RE.sub("", text).strip()
    if not clean:
        return base
    if _PUNCT_RE.search(text[-1:] if text else ""):
        base += 0.4
    n_pauses = len(re.findall(r"[，、；：,;:]", text))
    base += n_pauses * 0.15
    return base


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
        lines.append(LyricLine(text=text, timestamp=timestamp, weight=compute_line_weight(text)))
    return lines


def _align_to_beat(t: float, beat_interval: float, tolerance: float = 0.08) -> float:
    if beat_interval <= 0:
        return t
    nearest_beat = round(t / beat_interval) * beat_interval
    if abs(nearest_beat - t) <= beat_interval * tolerance:
        return nearest_beat
    prev_beat = int(t / beat_interval) * beat_interval
    next_beat = prev_beat + beat_interval
    if (t - prev_beat) < (next_beat - t) * 0.6:
        return prev_beat
    return next_beat


def _weighted_distribute(
    lines: List[LyricLine],
    start_idx: int,
    end_idx: int,
    start_time: float,
    end_time: float,
    beat_interval: float,
) -> None:
    count = end_idx - start_idx
    if count <= 0:
        return
    min_gap = beat_interval * 0.25 if beat_interval > 0 else 0.05
    if count == 1:
        raw_t = start_time
        t = _align_to_beat(raw_t, beat_interval) if beat_interval > 0 else raw_t
        if t < raw_t:
            t = raw_t
        lines[start_idx].timestamp = t
        return

    total_weight = sum(lines[i].weight for i in range(start_idx, end_idx))
    if total_weight <= 0:
        interval = (end_time - start_time) / count
        for j in range(start_idx, end_idx):
            raw_t = start_time + (j - start_idx) * interval
            lines[j].timestamp = _align_to_beat(raw_t, beat_interval) if beat_interval > 0 else raw_t
        return

    span_duration = end_time - start_time
    cumulative = 0.0
    prev_t = start_time
    min_gap = beat_interval * 0.25 if beat_interval > 0 else 0.05
    for j in range(start_idx, end_idx):
        if j == start_idx:
            raw_t = start_time
        else:
            raw_t = start_time + (cumulative / total_weight) * span_duration
            if raw_t < prev_t + min_gap:
                raw_t = prev_t + min_gap
                if raw_t > end_time:
                    raw_t = end_time
        lines[j].timestamp = _align_to_beat(raw_t, beat_interval) if beat_interval > 0 else raw_t
        if lines[j].timestamp < prev_t + min_gap:
            lines[j].timestamp = prev_t + min_gap
        prev_t = lines[j].timestamp
        cumulative += lines[j].weight


def align_lyrics(
    lyrics_text: str,
    audio_duration: float,
    bpm: Optional[float] = None,
    beats_per_bar: int = 4,
    beat_alignment: bool = True,
    first_line_offset: Optional[float] = None,
    lang_weights: Optional[Dict[str, float]] = None,
) -> str:
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")

    beat_interval = 0.0
    if beat_alignment and bpm and bpm > 0:
        beat_interval = 60.0 / bpm

    lines = parse_lyrics(lyrics_text)
    if not lines:
        return ""

    for line in lines:
        line.weight = compute_line_weight(line.text) if lang_weights is None else estimate_syllable_count(line.text, lang_weights)

    has_partial = any(line.timestamp is not None for line in lines)
    has_all = all(line.timestamp is not None for line in lines)

    if has_all:
        if beat_alignment and beat_interval > 0:
            for line in lines:
                if line.timestamp is not None:
                    line.timestamp = _align_to_beat(line.timestamp, beat_interval)
        return "\n".join(line.to_lrc() for line in lines)

    if not has_partial:
        effective_start = first_line_offset if first_line_offset is not None else beat_interval if beat_interval > 0 else 0.0
        effective_start = min(effective_start, audio_duration * 0.15)
        _weighted_distribute(lines, 0, len(lines), effective_start, audio_duration, beat_interval)
        return "\n".join(line.to_lrc() for line in lines)

    anchors: List[Tuple[int, float]] = []
    for i, line in enumerate(lines):
        if line.timestamp is not None:
            t = _align_to_beat(line.timestamp, beat_interval) if beat_interval > 0 else line.timestamp
            anchors.append((i, t))
            lines[i].timestamp = t

    min_gap_global = beat_interval * 0.25 if beat_interval > 0 else 0.05
    if anchors[0][0] > 0:
        seg_start = first_line_offset if first_line_offset is not None else 0.0
        if beat_alignment and beat_interval > 0 and seg_start == 0:
            seg_start = beat_interval
        _weighted_distribute(
            lines,
            0,
            anchors[0][0],
            seg_start,
            max(seg_start, anchors[0][1] - min_gap_global),
            beat_interval,
        )

    for idx, (anchor_i, anchor_t) in enumerate(anchors):
        if idx == len(anchors) - 1:
            next_i = len(lines)
            next_t = audio_duration
        else:
            next_i, next_t = anchors[idx + 1]

        if next_i - anchor_i > 1:
            seg_start_t = anchor_t + min_gap_global
            seg_end_t = next_t
            _weighted_distribute(lines, anchor_i + 1, next_i, seg_start_t, seg_end_t, beat_interval)

    return "\n".join(line.to_lrc() for line in lines)


def lrc_to_text(lrc_str: str) -> str:
    return re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", lrc_str)


if __name__ == "__main__":
    mixed_lyrics = """看雪在手中融化
The winter whispers softly, dear
看夜在心中发芽
Every memory shining clear
看花在风中凋零
I will hold you through the night
看梦在光中飞翔
Until the morning brings the light"""

    print("=== 旧版均匀分配（对照） ===")
    n = 8
    for i, line in enumerate(mixed_lyrics.strip().splitlines()):
        t = i * (24.0 / n)
        m = int(t) // 60
        s = t - m * 60
        print(f"[{m:02d}:{s:05.2f}]{line}")

    print("\n=== 新版加权分配 + 中英文权重区分 ===")
    result = align_lyrics(mixed_lyrics, audio_duration=24.0)
    print(result)

    print("\n=== 带 BPM 节拍对齐（BPM=120, 0.5秒/拍） ===")
    result_bpm = align_lyrics(mixed_lyrics, audio_duration=24.0, bpm=120.0, beat_alignment=True)
    print(result_bpm)

    print("\n=== 带锚点 + 中英文混排 ===")
    anchored_mixed = """看雪在手中融化
[00:03.00]The winter whispers softly, dear
看夜在心中发芽
[00:09.00]Every memory shining clear
看花在风中凋零
I will hold you through the night
看梦在光中飞翔
Until the morning brings the light"""
    result_anchored = align_lyrics(anchored_mixed, audio_duration=24.0, bpm=120.0)
    print(result_anchored)

    print("\n=== 自定义语言权重（英文更快，权重0.7） ===")
    custom_weights = {"zh": 1.2, "en": 0.7, "num": 0.5}
    result_custom = align_lyrics(mixed_lyrics, audio_duration=24.0, bpm=120.0, lang_weights=custom_weights)
    print(result_custom)

    pure_chinese = """看雪在手中融化
看夜在心中发芽
看花在风中凋零
看梦在光中飞翔"""
    print("\n=== 纯中文回归测试（BPM=120） ===")
    result_zh = align_lyrics(pure_chinese, audio_duration=16.0, bpm=120.0)
    print(result_zh)
