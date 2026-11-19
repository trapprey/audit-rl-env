import re
from datetime import datetime, timedelta, timezone

_OFFSET = re.compile(r"^([+-])(\d{2}):(\d{2})$")


def offset_text(minutes):
    sign = "-" if minutes < 0 else "+"
    minutes = abs(minutes)
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def parse_offset(text):
    match = _OFFSET.match(text)
    if not match:
        raise ValueError(f"bad offset {text!r}")
    sign = -1 if match.group(1) == "-" else 1
    return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))


def parse_ts(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("timestamp without offset")
    return value


def fmt_ts(moment, offset_minutes, zulu=False):
    local = moment.astimezone(timezone(timedelta(minutes=offset_minutes)))
    text = local.isoformat(timespec="seconds")
    if zulu and offset_minutes == 0:
        text = text[:-6] + "Z"
    return text
