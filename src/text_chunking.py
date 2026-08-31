def create_chunks(example_para: str, chunk_size: int = 120) -> list[str]:
    chunk_list = []
    for i in range(0, len(example_para), chunk_size):
        chunk_list.append(example_para[i:i + chunk_size])
    return chunk_list
