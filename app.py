from __future__ import annotations

from flask import Flask, render_template, request

from config import config
from services import run_forecast

app = Flask(__name__)
app.config["SECRET_KEY"] = config.secret_key


@app.get("/")
def index():
    return render_template("index.html", default_horizon=config.default_horizon, default_rows=config.default_rows)


@app.post("/forecast")
def forecast():
    try:
        horizon = int(request.form.get("horizon", config.default_horizon))
        rows = int(request.form.get("rows", config.default_rows))
        source = request.form.get("source", "auto")

        horizon = max(1, min(horizon, 72))
        rows = max(500, min(rows, config.max_rows))

        result = run_forecast(horizon=horizon, rows=rows, source=source)
        return render_template("result.html", result=result)
    except Exception as exc:
        return render_template("index.html", error=str(exc), default_horizon=config.default_horizon, default_rows=config.default_rows)


@app.get("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
