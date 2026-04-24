import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from models import db
from routes import register_routes
from services.index_store import IndexStore

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

app = Flask(
    __name__,
    static_folder=os.path.join(project_root, "frontend", "dist"),
    static_url_path="",
)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            os.environ.get("FRONTEND_URL", ""),
        ]
    }
})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "20 per minute"],
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

artifacts_dir = Path(project_root) / "data" / "artifacts"
store = IndexStore(artifacts_dir)

try:
    store.load()
except FileNotFoundError as e:
    print(f"[warn] {e}")
    print("[warn] The app will start but recommendation endpoints will return empty results.")
    print("[warn] Run: python scripts/build_indices.py")

app.config["INDEX_STORE"] = store
app.config["SQLITE_DB_PATH"] = str(Path(project_root) / "data" / "database.sqlite")

register_routes(app, limiter)


def init_db():
    with app.app_context():
        db.create_all()


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
