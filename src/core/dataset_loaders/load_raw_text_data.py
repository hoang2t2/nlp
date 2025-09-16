def load_raw_text_data(file_path: str) -> str:
    from pathlib import Path
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(...)
    return p.read_text(encoding="utf-8")
