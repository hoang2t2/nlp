from pathlib import Path
import os
def load_raw_text_data(file_path: str) -> str:
 
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
