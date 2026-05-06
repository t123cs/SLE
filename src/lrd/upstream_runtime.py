import os
import sys


def ensure_scaling_retriever_importable():
    try:
        import scaling_retriever  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = os.environ.get("SCALING_RETRIEVER_ROOT")
    if not repo_root:
        raise ImportError(
            "Could not import scaling_retriever. Clone the upstream repository and "
            "set SCALING_RETRIEVER_ROOT=/path/to/scaling-retriever."
        )

    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import scaling_retriever  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Failed to import scaling_retriever from SCALING_RETRIEVER_ROOT={repo_root!r}."
        ) from exc
