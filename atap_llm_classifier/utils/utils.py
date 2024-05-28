import traceback

__all__ = [
    "format_exception",
]


def format_exception(e: Exception) -> str:
    format_str: str = "{}:{}\t{}"
    tb = list(iter(traceback.extract_tb(e.__traceback__)))
    if len(tb) > 0:
        frame = tb[-1]
        return format_str.format(frame.filename, frame.lineno, e)
    else:
        return str(e)
