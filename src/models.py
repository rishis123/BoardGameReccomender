from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Feedback(db.Model):
    """User feedback (+1 / -1) on chat responses and substitution suggestions."""
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    context_type = db.Column(db.String(32), nullable=False)   # "chat" | "substitution"
    context_id = db.Column(db.String(256), nullable=True)
    score = db.Column(db.Integer, nullable=False)              # +1 or -1
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"Feedback({self.context_type}, {self.score})"


class MetricsCache(db.Model):
    """Cached evaluation metrics (precision@k, etc.)."""
    __tablename__ = "metrics_cache"
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Float, nullable=False)
    updated_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"Metric({self.key}={self.value})"
