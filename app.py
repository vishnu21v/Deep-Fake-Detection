# app.py
import os
from flask import Flask, render_template, request, url_for
from deepfake_model import DeepfakeModel
from utils import allowed_file, save_upload

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

# init model once
model = DeepfakeModel()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            result = {"error": "No file part in the request"}
            return render_template("index.html", result=result, image_url=image_url)

        file = request.files["file"]
        if file.filename == "":
            result = {"error": "No file selected"}
            return render_template("index.html", result=result, image_url=image_url)

        if file and allowed_file(file.filename):
            saved_path = save_upload(file, app.config["UPLOAD_FOLDER"])
            # run model
            pred = model.predict(saved_path)

            # build response
            image_url = url_for("static", filename=f"uploads/{os.path.basename(saved_path)}")
            result = pred
        else:
            result = {"error": "File type not allowed. Please upload JPG/PNG/GIF."}

    return render_template("index.html", result=result, image_url=image_url)

if __name__ == "__main__":
    # for local run: python app.py
    app.run(debug=True)
