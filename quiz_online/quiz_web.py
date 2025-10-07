from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

# Important filesystem locations.
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = (BASE_DIR / "outputs").resolve()
DEFAULT_DATA_FILE = OUTPUTS_DIR / "all_data.json"
RESPONSES_FILE = OUTPUTS_DIR / "responses.csv"

# Limit uploaded JSON payload size.
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16 MiB


class DataLoadError(Exception):
    """Raised when an SJT dataset cannot be parsed."""


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def normalise_dataset(raw: object) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """
    Convert a raw JSON payload into the {question_id: {situation, options}} format.

    Returns:
        (clean_items, skipped_ids)
    """
    if not isinstance(raw, dict):
        raise DataLoadError("Expected the JSON root to be an object mapping ids to items.")

    cleaned: Dict[str, Dict[str, object]] = {}
    skipped: List[str] = []

    for qid, record in raw.items():
        if not isinstance(record, dict):
            skipped.append(str(qid))
            continue

        situation = record.get("situation")
        options = record.get("options")

        if not isinstance(situation, str):
            situation = ""
        if not isinstance(options, dict):
            options = {}

        cleaned[str(qid)] = {
            "situation": situation,
            "options": {str(opt_key): str(opt_text) for opt_key, opt_text in options.items()},
        }

    return cleaned, skipped


def load_dataset_from_path(path: Path) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    if not path.exists():
        raise DataLoadError(f"JSON file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON: {exc.msg}") from exc
    return normalise_dataset(payload)


def load_dataset_from_bytes(raw_bytes: bytes) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataLoadError("Uploaded file must be UTF-8 encoded.") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Uploaded JSON is invalid: {exc.msg}") from exc
    return normalise_dataset(payload)


def resolve_media_reference(raw_value: str) -> Tuple[str | None, bool]:
    """
    Resolve an image path from the dataset to a safe relative path under outputs/.

    Returns:
        (relative_path, exists_flag)
    """
    if not raw_value:
        return None, False

    ensure_outputs_dir()

    raw_path = Path(raw_value).expanduser()
    outputs_root = OUTPUTS_DIR.resolve()

    def normalise(relative_path: Path) -> str:
        return str(relative_path).replace("\\", "/")

    def mirror_external(resolved: Path) -> Tuple[str, bool]:
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()
        mirror_dir = outputs_root / "_external_media" / digest[:2] / digest[2:10]
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirrored = mirror_dir / resolved.name
        if not mirrored.exists() or resolved.stat().st_mtime > mirrored.stat().st_mtime:
            shutil.copy2(resolved, mirrored)
        return normalise(mirrored.relative_to(outputs_root)), True

    # Mirror absolute paths that sit outside outputs/ into a safe subdirectory.
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        if resolved.exists():
            try:
                rel = resolved.relative_to(outputs_root)
            except ValueError:
                return mirror_external(resolved)
            else:
                return normalise(rel), True
        return None, False

    candidates: List[Path] = [
        OUTPUTS_DIR / raw_path,
        BASE_DIR / raw_path,
        BASE_DIR.parent / raw_path,
    ]

    # Prefer candidates that exist, mirroring them if required.
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists():
            continue
        try:
            relative = resolved.relative_to(outputs_root)
        except ValueError:
            return mirror_external(resolved)
        return normalise(relative), True

    # Fall back to the first candidate under outputs/ even if it is missing.
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(outputs_root)
        except ValueError:
            continue
        return normalise(relative), False

    return None, False


def get_data_store(app: Flask) -> Dict[str, Dict[str, Dict[str, object]]]:
    return app.config.setdefault("DATA_STORE", {})


def build_fieldnames(existing: List[str] | None, ordered_questions: List[str]) -> List[str]:
    fieldnames: List[str] = ["id"]
    seen = {"id"}

    for qid in ordered_questions:
        if qid not in seen:
            fieldnames.append(qid)
            seen.add(qid)

    if existing:
        for name in existing:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)

    return fieldnames


def save_responses_to_csv(user_id: str, question_order: List[str], responses: Dict[str, str]) -> None:
    ensure_outputs_dir()
    existing_rows: List[Dict[str, str]] = []
    existing_fields: List[str] | None = None

    if RESPONSES_FILE.exists():
        with RESPONSES_FILE.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames
            for row in reader:
                existing_rows.append(dict(row))

    fieldnames = build_fieldnames(existing_fields, question_order)

    # Prepare the new/updated row.
    new_row = {field: "" for field in fieldnames}
    new_row["id"] = user_id
    for qid in question_order:
        new_row[qid] = responses.get(qid, "")

    updated = False
    for row in existing_rows:
        for field in fieldnames:
            row.setdefault(field, "")
        if row.get("id") == user_id:
            for qid in question_order:
                row[qid] = responses.get(qid, "")
            updated = True

    if not updated:
        existing_rows.append(new_row)

    with RESPONSES_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "picsjt-dev-key")
    get_data_store(app)  # Ensure storage is available.

    @app.route("/", methods=["GET", "POST"])
    def start():
        default_available = DEFAULT_DATA_FILE.exists()

        if request.method == "POST":
            user_id = (request.form.get("user_id") or "").strip()

            if not user_id:
                flash("请输入您的ID。", "error")
                return render_template("start.html", default_available=default_available)

            upload = request.files.get("data_file")

            try:
                if upload and upload.filename:
                    dataset, skipped = load_dataset_from_bytes(upload.read())
                    data_origin = f"Uploaded file: {upload.filename}"
                else:
                    dataset, skipped = load_dataset_from_path(DEFAULT_DATA_FILE)
                    data_origin = f"Default file: {DEFAULT_DATA_FILE.relative_to(BASE_DIR)}"
            except DataLoadError as exc:
                flash(str(exc), "error")
                return render_template("start.html", default_available=default_available)

            if not dataset:
                flash("数据集为空，无法开始测验。", "warning")
                return render_template("start.html", default_available=default_available)

            state_id = secrets.token_urlsafe(16)
            data_store = get_data_store(app)

            previous_state = session.get("state_id")
            if previous_state:
                data_store.pop(previous_state, None)

            data_store[state_id] = dataset

            session.clear()
            order = sorted(dataset.keys())
            session["state_id"] = state_id
            session["user_id"] = user_id
            session["question_order"] = order
            session["current_index"] = 0
            session["responses"] = {}
            session["skipped"] = skipped
            session["data_origin"] = data_origin

            flash(f"已加载 {len(order)} 道题。", "success")
            return redirect(url_for("quiz"))

        state_id = session.get("state_id")
        data_store = get_data_store(app)
        if state_id and state_id in data_store:
            return redirect(url_for("quiz"))

        return render_template("start.html", default_available=default_available)

    @app.route("/quiz", methods=["GET", "POST"])
    def quiz():
        state_id = session.get("state_id")
        data_store = get_data_store(app)

        if not state_id or state_id not in data_store:
            session.clear()
            flash("请先开始新的测验。", "warning")
            return redirect(url_for("start"))

        dataset = data_store[state_id]
        order: List[str] = session.get("question_order", [])
        if not order:
            data_store.pop(state_id, None)
            session.clear()
            flash("当前测验没有题目。", "error")
            return redirect(url_for("start"))

        responses = dict(session.get("responses", {}))
        total = len(order)
        index = session.get("current_index", 0)

        if request.method == "POST":
            question_id = request.form.get("question_id")
            selected_option = request.form.get("selected_option")
            action = request.form.get("action")
            goto_qid = (request.form.get("goto_qid") or "").strip()

            if question_id and selected_option:
                responses[question_id] = selected_option
                session["responses"] = responses

            if action == "prev":
                index = max(0, index - 1)
            elif action == "next":
                index = min(total - 1, index + 1)
            elif action == "goto":
                if goto_qid and goto_qid in order:
                    index = order.index(goto_qid)
                elif goto_qid:
                    flash(f"未找到题号 {goto_qid}。", "warning")
            elif action == "finish":
                user_id = session.get("user_id", "")
                save_responses_to_csv(user_id, order, responses)
                data_store.pop(state_id, None)
                session.clear()
                flash(f"已保存 {user_id} 的答卷结果。", "success")
                return redirect(url_for("start"))

            index = max(0, min(index, total - 1))
            session["current_index"] = index
            return redirect(url_for("quiz", index=index))

        if "index" in request.args:
            try:
                index = int(request.args["index"])
            except (TypeError, ValueError):
                index = session.get("current_index", 0)

        index = max(0, min(index, total - 1))
        session["current_index"] = index

        question_id = order[index]
        record = dataset.get(question_id, {})
        situation = str(record.get("situation", ""))
        options = record.get("options", {})

        option_items = []
        if isinstance(options, dict):
            for opt_key, opt_text in sorted(options.items()):
                option_items.append(
                    {
                        "key": str(opt_key),
                        "text": str(opt_text),
                    }
                )

        rel_path, exists_flag = resolve_media_reference(situation)
        image_url = url_for("media", filename=rel_path) if rel_path and exists_flag else None

        return render_template(
            "question.html",
            progress=index + 1,
            total=total,
            question_id=question_id,
            options=option_items,
            current_choice=responses.get(question_id, ""),
            has_prev=index > 0,
            has_next=index < total - 1,
            image_url=image_url,
            raw_situation=situation,
            data_origin=session.get("data_origin"),
            user_id=session.get("user_id"),
            skipped=session.get("skipped", []),
        )

    @app.route("/reset")
    def reset():
        state_id = session.get("state_id")
        data_store = get_data_store(app)
        if state_id:
            data_store.pop(state_id, None)
        session.clear()
        flash("测验已重置。", "info")
        return redirect(url_for("start"))

    @app.route("/media/<path:filename>")
    def media(filename: str):
        target = (OUTPUTS_DIR / filename).resolve()
        outputs_root = OUTPUTS_DIR.resolve()

        if not target.exists():
            abort(404)
        if outputs_root not in target.parents and target != outputs_root:
            abort(404)

        return send_file(target)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
