# utils.py
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_upload(file_obj, upload_folder: str) -> str:
    ensure_dir(upload_folder)
    filename = secure_filename(file_obj.filename)
    save_path = os.path.join(upload_folder, filename)
    file_obj.save(save_path)
    return save_path
