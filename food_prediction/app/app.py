import os
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, current_app
from PIL import Image

# imports from local package -- these assume app/ is a package and models.py is app/models.py
from models import db, User, Prediction
from ml.inference import load_model, predict_image

# helper for calorie mapping
from ml.utils import get_calories

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # configuration
    # Use DATABASE_URL env var if present, otherwise sqlite file in project root
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///foodapp.db")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # initialize DB
    db.init_app(app)
    with app.app_context():
        db.create_all()

    # load model once at startup (or set to None if not available)
    try:
        model, classes, device = load_model()
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        model, classes, device = None, [], None
        app.logger.error("Model failed to load at startup: %s", e)

    @app.route("/")
    def index():
        """
        Render home page (upload form)
        """
        return render_template("index.html")

    @app.route("/history")
    def history():
        """
        Show recent predictions (most recent first)
        """
        preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(500).all()
        return render_template("history.html", preds=preds)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accepts a multipart/form-data request with key 'file' (or 'image').
        Returns JSON:
        {
          "label": "<top label>",
          "calories": <cal_per_100g>,
          "predictions": [ {"label":..., "confidence":...}, ... ]
        }
        """
        # accept either 'file' or 'image' keys (frontend uses 'file' or 'image')
        upload_key = "file" if "file" in request.files else ("image" if "image" in request.files else None)
        if upload_key is None:
            return jsonify({"error": "no image file provided (expected form key 'file' or 'image')"}), 400

        uploaded = request.files[upload_key]
        if uploaded.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # read image
        try:
            img_bytes = uploaded.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            current_app.logger.exception("Failed to read uploaded image: %s", e)
            return jsonify({"error": "unable to read image"}), 400

        # ensure model is loaded
        if model is None:
            return jsonify({"error": "model not loaded on server"}), 500

        # run inference (top-k)
        try:
            results = predict_image(model, classes, device, img, topk=3)
        except Exception as e:
            current_app.logger.exception("Inference error: %s", e)
            return jsonify({"error": "inference failed"}), 500

        # top result + calories lookup
        top = results[0] if results else {"label": "unknown", "confidence": 0.0}
        calories_per_100g = get_calories(top["label"])

        # persist to DB (no user auth for now; user_id left None)
        try:
            pred = Prediction(
                user_id=None,
                label=top["label"],
                confidence=float(top["confidence"]),
                calories=float(calories_per_100g),
                timestamp=datetime.now()
            )
            db.session.add(pred)
            db.session.commit()
        except Exception as e:
            # don't fail the whole request if DB write fails — log and continue
            current_app.logger.exception("Failed to save prediction to DB: %s", e)

        # return structured JSON to match frontend expectations
        response = {
            "label": top["label"],
            "confidence": top["confidence"],
            "calories": calories_per_100g,
            "predictions": results
        }
        return jsonify(response), 200

    @app.route("/health")
    def health():
        """
        Basic health check for deployments.
        """
        ok = model is not None
        return jsonify({
            "status": "ok" if ok else "model-not-loaded",
            "model_loaded": ok,
            "classes_count": len(classes)
        })

    # error handlers (optional nicer JSON errors)
    @app.errorhandler(500)
    def internal_error(error):
        current_app.logger.exception("Unhandled exception: %s", error)
        return jsonify({"error": "internal server error"}), 500

    return app


if __name__ == "__main__":
    # helpful defaults for local development
    # ensure MODEL_PATH and CLASSES_PATH env vars are set if model not in default location
    # example:
    # set MODEL_PATH=C:\path\to\food_model_best.pth
    # set CLASSES_PATH=C:\path\to\classes.json
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
