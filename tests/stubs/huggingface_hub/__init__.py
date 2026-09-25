"""Tiny Hugging Face Hub stand-in that avoids network access in offline tests."""


def hf_hub_download(repo_id, filename):
    """Return the requested filename; the Piper test double does not read it."""
    del repo_id
    return filename
