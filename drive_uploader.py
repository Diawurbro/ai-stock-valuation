"""
drive_uploader.py — Uploads stock analysis reports to Google Drive via OAuth.

One-time setup:
  1. Go to https://console.cloud.google.com/ → Project: DriveAPI
  2. Google Auth Platform → Audience → Add Users → add your Gmail → Save
  3. APIs & Services → Credentials → Download OAuth 2.0 Client ID JSON
     Rename it to "credentials.json" and put it in this project folder
  4. Run: python3 drive_uploader.py
     A browser will open → Sign in with Gmail you added as Test User → Allow
     "token.json" will be saved automatically for all future runs.
"""

import logging
import time
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES               = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_FILE     = Path(__file__).parent / "credentials.json"
TOKEN_FILE           = Path(__file__).parent / "token.json"
PARENT_FOLDER_NAME   = "StockReports"


def _get_drive_service():
    """Authenticate and return a Google Drive API service object."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "\n\n❌  credentials.json not found!\n"
                    "   See drive_uploader.py for setup instructions.\n"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _get_or_create_folder(service, folder_name: str, parent_id: str | None = None) -> str:
    """Find or create a Drive folder, return its ID."""
    query = (
        f"name='{folder_name}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_reports_to_drive(ticker: str, reports_dir: str = "reports") -> tuple[str, list[dict]]:
    """Upload all report files for the given ticker to Google Drive.

    Returns:
        (folder_url, uploaded_files) where uploaded_files is a list of
        {"id": ..., "name": ..., "mime_type": ...} dicts for each file.
    """
    reports_path = Path(reports_dir)
    report_files = sorted(reports_path.glob(f"*_{ticker.upper()}.md"))

    if not report_files:
        logger.warning("No report files found for ticker %s in %s", ticker, reports_dir)
        return "", []

    logger.info("🔗 Connecting to Google Drive …")
    service = _get_drive_service()

    root_id   = _get_or_create_folder(service, PARENT_FOLDER_NAME)
    ticker_id = _get_or_create_folder(service, ticker.upper(), parent_id=root_id)

    logger.info("📤 Uploading %d report(s) → StockReports/%s …", len(report_files), ticker.upper())

    GDOC_MIME = "application/vnd.google-apps.document"
    uploaded_files: list[dict] = []

    for file_path in report_files:
        # Upload as Google Docs so NotebookLM can read them
        media    = MediaFileUpload(str(file_path), mimetype="text/plain", resumable=False)
        doc_name = file_path.stem  # filename without .md extension
        metadata = {
            "name": doc_name,
            "parents": [ticker_id],
            "mimeType": GDOC_MIME,
        }

        existing = service.files().list(
            q=f"name='{doc_name}' and '{ticker_id}' in parents and trashed=false",
            fields="files(id)"
        ).execute().get("files", [])

        file_id = None
        # Retry up to 3 times on transient errors (500, 503)
        for attempt in range(1, 4):
            try:
                if existing:
                    service.files().update(fileId=existing[0]["id"], media_body=media).execute()
                    file_id = existing[0]["id"]
                    logger.info("  ↺  Updated: %s", doc_name)
                else:
                    result = service.files().create(body=metadata, media_body=media, fields="id").execute()
                    file_id = result["id"]
                    logger.info("  ✓  Uploaded: %s", doc_name)
                break  # Success — exit retry loop
            except HttpError as e:
                if e.status_code in (500, 503) and attempt < 3:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning("  ⚠  %s — retrying in %ds (attempt %d/3)…", doc_name, wait, attempt)
                    time.sleep(wait)
                    media = MediaFileUpload(str(file_path), mimetype="text/plain", resumable=False)
                else:
                    raise

        if file_id:
            uploaded_files.append({"id": file_id, "name": doc_name, "mime_type": GDOC_MIME})

    folder_url = f"https://drive.google.com/drive/folders/{ticker_id}"
    return folder_url, uploaded_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Testing Google Drive connection …")
    service = _get_drive_service()
    about = service.about().get(fields="user").execute()
    print(f"✅  Connected as: {about['user']['displayName']} ({about['user']['emailAddress']})")
    print("Authentication successful! All future runs will upload automatically.")
