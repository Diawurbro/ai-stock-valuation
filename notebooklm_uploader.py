"""
notebooklm_uploader.py — Adds stock analysis reports to a NotebookLM notebook.

One-time setup:
  1. Activate your venv: source venv/bin/activate
  2. Run: notebooklm login
     A browser will open → Sign in with Google → Allow → storage state saved.
  3. That's it — all future runs authenticate automatically.

Requires:
  pip install "notebooklm-py[browser]"
  playwright install chromium  (only needed once)
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def _upload_to_notebooklm(
    ticker: str,
    uploaded_files: list[dict],
    notebook_title: Optional[str] = None,
) -> str:
    """Create (or reuse) a NotebookLM notebook and add Drive files as sources.

    Args:
        ticker:         Stock ticker, e.g. "TSLA".
        uploaded_files: List of {"id": ..., "name": ..., "mime_type": ...} dicts
                        returned by drive_uploader.upload_reports_to_drive().
        notebook_title: Override the auto-generated notebook title.

    Returns:
        NotebookLM notebook URL, or empty string on failure.
    """
    from notebooklm import NotebookLMClient

    title = notebook_title or f"Stock Analysis — {ticker.upper()}"

    async with await NotebookLMClient.from_storage() as client:
        # Reuse existing notebook with the same title if it exists
        notebook = None
        notebooks = await client.notebooks.list()
        for nb in notebooks:
            if nb.title == title:
                notebook = nb
                logger.info("  ♻  Reusing existing notebook: %s", title)
                break

        if notebook is None:
            notebook = await client.notebooks.create(title)
            logger.info("  ✓  Created notebook: %s (id=%s)", title, notebook.id)

        # Add each Drive file as a source
        for f in uploaded_files:
            try:
                await client.sources.add_drive(
                    notebook_id=notebook.id,
                    file_id=f["id"],
                    title=f["name"],
                    mime_type=f["mime_type"],
                    wait=True,
                    wait_timeout=120,
                )
                logger.info("  ✓  Added source: %s", f["name"])
            except Exception as exc:
                logger.warning("  ⚠  Could not add source %s: %s", f["name"], exc)

        return f"https://notebooklm.google.com/notebook/{notebook.id}"


def upload_to_notebooklm(
    ticker: str,
    uploaded_files: list[dict],
    notebook_title: Optional[str] = None,
) -> str:
    """Synchronous wrapper around _upload_to_notebooklm."""
    return asyncio.run(_upload_to_notebooklm(ticker, uploaded_files, notebook_title))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TEST"
    print(f"Testing NotebookLM connection for ticker {ticker} …")
    url = upload_to_notebooklm(ticker=ticker, uploaded_files=[])
    print(f"✅  Notebook ready: {url}")
