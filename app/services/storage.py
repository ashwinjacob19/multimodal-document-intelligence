import shutil
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


class StorageService:
    def __init__(self, upload_dir: str = settings.UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        # Ensure the directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, file: UploadFile, unique_filename: str) -> str:
        """Saves an uploaded file to the local directory.

        Returns:
            The absolute path of the saved file as a string.
        """
        file_path = self.upload_dir / unique_filename

        # Reset file pointer to the beginning
        file.file.seek(0)

        # Save content using shutil
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return str(file_path.absolute())
