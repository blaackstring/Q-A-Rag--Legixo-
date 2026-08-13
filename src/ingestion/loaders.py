"""Document loading: turn files on disk into LangChain Document objects.

Uses LangChain's DirectoryLoader + TextLoader so any .md/.txt file in the
corpus folder is picked up automatically, with its file path kept as metadata
(needed later for citations).
"""

from pathlib import Path

from langchain_community.document_loaders.directory import DirectoryLoader, TextLoader


def build_text_loader(file_path: str) -> TextLoader:
    """Create a TextLoader for one file (UTF-8, auto-detect encoding)."""
    loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    return loader


def load_corpus(corpus_dir: str) -> list:
    """Load every text/markdown document under `corpus_dir` into Documents.

    glob="**/*" matches every file, in this folder and nested sub-folders.
    Returns Documents with `.page_content` and `.metadata["source"]`.
    """
    loader = DirectoryLoader(
        path=corpus_dir,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
        recursive=True,
    )
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No readable text files found under: {corpus_dir}")
    return docs