def create_chunks(text: str, chunk_size: int = 120) -> list[str]:
    """Create fixed-size character chunks, matching the notebook logic."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
