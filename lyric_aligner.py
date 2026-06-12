import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Union


@dataclass
class KaraokeToken:
    text: str
    timestamp: float
    duration: float = 0.0
    weight: float = 0.0

    def to_lrc_tag(self) -> str:
        minutes = int(self.timestamp) // 60
        seconds = self.timestamp - minutes * 60
        return f"<{minutes:02d}:{seconds:05.2f}>"


@dataclass
class LyricLine:
    text: str
    timestamp: Optional[float] = None
    weight: float = 0.0
    karaoke_tokens: List[KaraokeToken] = field(default_factory=list)

    def to_lrc(self) -> str:
        if self.timestamp is None:
            return self.text
        minutes = int(self.timestamp) // 60
        seconds = self.timestamp - minutes * 60
        return f"[{minutes:02d}:{seconds:05.2f}]{self.text}"

    def to_karaoke_lrc(self) -> str:
        if self.timestamp is None or not self.karaoke_tokens:
            return self.to_lrc()
        minutes = int(self.timestamp) // 60
        seconds = self.timestamp - minutes * 60
        parts = [f"[{minutes:02d}:{seconds:05.2f}]"]
        for token in self.karaoke_tokens:
            parts.append(token.to_lrc_tag())
            parts.append(token.text)
        return "".join(parts)


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


def tokenize_line(text: str) -> List[KaraokeToken]:
    tokens: List[KaraokeToken] = []
    i = 0
    text_len = len(text)
    while i < text_len:
        ch = text[i]
        if _CJK_RE.match(ch):
            token_text = ch
            token_weight = 1.0
            i += 1
        elif _WORD_RE.match(text[i:]):
            m = _WORD_RE.match(text[i:])
            token_text = m.group(0)
            wl = len(token_text.lower())
            if wl <= 3:
                token_weight = 1.0
            elif wl <= 5:
                token_weight = 1.5
            elif wl <= 8:
                token_weight = 2.0
            else:
                token_weight = 2.5 + (wl - 8) * 0.15
            i += len(token_text)
        elif _DIGIT_RE.match(text[i:]):
            m = _DIGIT_RE.match(text[i:])
            token_text = m.group(0)
            token_weight = max(1.0, len(token_text) * 0.5)
            i += len(token_text)
        else:
            token_text = ch
            token_weight = 0.05
            i += 1
        tokens.append(KaraokeToken(text=token_text, timestamp=0.0, weight=token_weight))
    return tokens


def distribute_line_time(
    tokens: List[KaraokeToken],
    start_time: float,
    end_time: float,
    beat_interval: float = 0.0,
) -> None:
    if not tokens:
        return
    n = len(tokens)
    duration = end_time - start_time
    if duration <= 0:
        for token in tokens:
            token.timestamp = start_time
            token.duration = 0.0
        return
    if n == 1:
        tokens[0].timestamp = start_time
        tokens[0].duration = duration
        return

    total_weight = sum(t.weight for t in tokens)
    if total_weight <= 0:
        interval = duration / n
        for j, token in enumerate(tokens):
            raw_t = start_time + j * interval
            if beat_interval > 0:
                raw_t = _align_to_beat(raw_t, beat_interval)
            token.timestamp = max(start_time, min(end_time, raw_t))
        for j in range(n):
            if j < n - 1:
                tokens[j].duration = tokens[j + 1].timestamp - tokens[j].timestamp
            else:
                tokens[j].duration = end_time - tokens[j].timestamp
        return

    min_token_gap = beat_interval * 0.15 if beat_interval > 0 else 0.02
    cumulative = 0.0
    prev_t = start_time
    for j, token in enumerate(tokens):
        if j == 0:
            raw_t = start_time
        else:
            raw_t = start_time + (cumulative / total_weight) * duration
            if raw_t < prev_t + min_token_gap:
                raw_t = prev_t + min_token_gap
                if raw_t > end_time:
                    raw_t = end_time
        if beat_interval > 0:
            aligned = _align_to_beat(raw_t, beat_interval)
            if abs(aligned - raw_t) <= beat_interval * 0.15:
                raw_t = aligned
        token.timestamp = max(prev_t, min(end_time, raw_t))
        prev_t = token.timestamp
        cumulative += token.weight

    for j in range(n):
        if j < n - 1:
            tokens[j].duration = tokens[j + 1].timestamp - tokens[j].timestamp
        else:
            tokens[j].duration = end_time - tokens[j].timestamp
        if tokens[j].duration < 0:
            tokens[j].duration = 0.0


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


def _align_lines_internal(
    lines: List[LyricLine],
    audio_duration: float,
    beat_interval: float,
    first_line_offset: Optional[float],
    lang_weights: Optional[Dict[str, float]],
    beat_alignment: bool,
) -> None:
    for line in lines:
        line.weight = compute_line_weight(line.text) if lang_weights is None else estimate_syllable_count(line.text, lang_weights)

    has_partial = any(line.timestamp is not None for line in lines)
    has_all = all(line.timestamp is not None for line in lines)

    if has_all:
        if beat_alignment and beat_interval > 0:
            for line in lines:
                if line.timestamp is not None:
                    line.timestamp = _align_to_beat(line.timestamp, beat_interval)
        return

    if not has_partial:
        effective_start = first_line_offset if first_line_offset is not None else beat_interval if beat_interval > 0 else 0.0
        effective_start = min(effective_start, audio_duration * 0.15)
        _weighted_distribute(lines, 0, len(lines), effective_start, audio_duration, beat_interval)
        return

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
            lines, 0, anchors[0][0], seg_start,
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
            _weighted_distribute(lines, anchor_i + 1, next_i, anchor_t + min_gap_global, next_t, beat_interval)

    if beat_alignment and beat_interval > 0:
        for line in lines:
            if line.timestamp is not None:
                line.timestamp = _align_to_beat(line.timestamp, beat_interval)

    prev_t = -1.0
    for line in lines:
        if line.timestamp is not None:
            if line.timestamp <= prev_t:
                line.timestamp = prev_t + min_gap_global
            prev_t = line.timestamp


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

    _align_lines_internal(lines, audio_duration, beat_interval, first_line_offset, lang_weights, beat_alignment)
    return "\n".join(line.to_lrc() for line in lines)


def _process_lines_to_karaoke(
    lines: List[LyricLine],
    audio_duration: float,
    beat_interval: float,
) -> None:
    n = len(lines)
    for i, line in enumerate(lines):
        if line.timestamp is None:
            continue
        line_start = line.timestamp
        if i < n - 1 and lines[i + 1].timestamp is not None:
            line_end = lines[i + 1].timestamp
        else:
            line_end = audio_duration
        if line_end <= line_start:
            line_end = line_start + 0.5
        tokens = tokenize_line(line.text)
        min_line_duration = len(tokens) * 0.08 + 0.2
        if line_end - line_start < min_line_duration:
            line_end = line_start + min_line_duration
            for j in range(i + 1, n):
                if lines[j].timestamp is not None and lines[j].timestamp < line_end:
                    lines[j].timestamp = line_end + (j - i) * 0.15
        distribute_line_time(tokens, line_start, line_end, beat_interval)
        line.karaoke_tokens = tokens


def align_lyrics_karaoke(
    lyrics_text: str,
    audio_duration: float,
    bpm: Optional[float] = None,
    beats_per_bar: int = 4,
    beat_alignment: bool = True,
    first_line_offset: Optional[float] = None,
    lang_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Union[str, List[Dict]]]:
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")

    beat_interval = 0.0
    if beat_alignment and bpm and bpm > 0:
        beat_interval = 60.0 / bpm

    lines = parse_lyrics(lyrics_text)
    if not lines:
        return {
            "karaoke_lrc": "",
            "lrc": "",
            "lines": [],
            "audio_duration": round(audio_duration, 3),
            "bpm": bpm,
        }

    _align_lines_internal(lines, audio_duration, beat_interval, first_line_offset, lang_weights, beat_alignment)
    _process_lines_to_karaoke(lines, audio_duration, beat_interval)

    lrc_lines = []
    for line in lines:
        if line.karaoke_tokens:
            lrc_lines.append(line.to_karaoke_lrc())
        else:
            lrc_lines.append(line.to_lrc())
    karaoke_lrc = "\n".join(lrc_lines)

    json_data = []
    for line in lines:
        line_data = {
            "text": line.text,
            "start_time": round(line.timestamp, 3) if line.timestamp is not None else None,
            "end_time": round(line.karaoke_tokens[-1].timestamp + line.karaoke_tokens[-1].duration, 3) if line.karaoke_tokens else round(audio_duration, 3),
            "tokens": []
        }
        for token in line.karaoke_tokens:
            line_data["tokens"].append({
                "text": token.text,
                "start_time": round(token.timestamp, 3),
                "end_time": round(token.timestamp + token.duration, 3),
                "duration": round(token.duration, 3),
                "weight": round(token.weight, 3),
            })
        json_data.append(line_data)

    return {
        "karaoke_lrc": karaoke_lrc,
        "lrc": "\n".join(line.to_lrc() for line in lines),
        "lines": json_data,
        "audio_duration": round(audio_duration, 3),
        "bpm": bpm,
    }


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

    print("=== 1. 旧版均匀分配（对照） ===")
    n = 8
    for i, line in enumerate(mixed_lyrics.strip().splitlines()):
        t = i * (24.0 / n)
        m = int(t) // 60
        s = t - m * 60
        print(f"[{m:02d}:{s:05.2f}]{line}")

    print("\n=== 2. 新版加权分配 + 中英文权重区分 ===")
    result = align_lyrics(mixed_lyrics, audio_duration=24.0)
    print(result)

    print("\n=== 3. 带 BPM 节拍对齐（BPM=120, 0.5秒/拍） ===")
    result_bpm = align_lyrics(mixed_lyrics, audio_duration=24.0, bpm=120.0, beat_alignment=True)
    print(result_bpm)

    print("\n=== 4. 卡拉OK逐字时间轴 - 增强LRC格式 ===")
    karaoke_result = align_lyrics_karaoke(mixed_lyrics, audio_duration=24.0, bpm=120.0)
    print(karaoke_result["karaoke_lrc"])

    print("\n=== 5. 卡拉OK逐字时间轴 - JSON格式（前2行示例） ===")
    for line_data in karaoke_result["lines"][:2]:
        print(f"\n行: {line_data['text']}")
        print(f"  起止: {line_data['start_time']}s - {line_data['end_time']}s")
        for token in line_data["tokens"]:
            print(f"    [{token['start_time']:6.2f}s] '{token['text']}' ({token['duration']:.2f}s, w={token['weight']})")

    print("\n=== 6. 纯中文卡拉OK逐字高亮 ===")
    pure_chinese = """看雪在手中融化
看夜在心中发芽
看花在风中凋零
看梦在光中飞翔"""
    zh_karaoke = align_lyrics_karaoke(pure_chinese, audio_duration=16.0, bpm=120.0)
    print(zh_karaoke["karaoke_lrc"])
    print("\nJSON 详情:")
    for line_data in zh_karaoke["lines"]:
        tokens_str = " ".join([f"<{t['start_time']:.2f}>{t['text']}" for t in line_data["tokens"]])
        print(f"  [{line_data['start_time']:.2f}] {tokens_str}")

    print("\n=== 7. 带锚点的卡拉OK时间轴 ===")
    anchored = """[00:00.50]看雪在手中融化
The winter whispers softly, dear
[00:06.50]看夜在心中发芽
Every memory shining clear"""
    anchored_karaoke = align_lyrics_karaoke(anchored, audio_duration=12.0, bpm=120.0)
    print(anchored_karaoke["karaoke_lrc"])

    print("\n=== 8. 导出完整JSON ===")
    full_json = json.dumps(karaoke_result["lines"][:1], ensure_ascii=False, indent=2)
    print(full_json)
