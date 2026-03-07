from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil
import os

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
CONDA_ENV = "nifty-rtx5060"
TRAIN_SCRIPT = ROOT / "src" / "scripts" / "train.py"
PREDICT_SCRIPT = ROOT / "src" / "scripts" / "predict.py"
TRAIN_DATA = ROOT / "data" / "inputs" / "prod_train.csv"
TEST_DATA = ROOT / "data" / "inputs" / "prod_test.csv"
MODEL_FILE = ROOT / "models" / "prod_binary_lstm_best.pth"
SCALER_FILE = ROOT / "models" / "prod_scaler.joblib"
PREDICTIONS_FILE = ROOT / "data" / "predictions" / "production_predictions.csv"


st.set_page_config(page_title="NIFTY-NLP GUI", page_icon="N", layout="wide")


def resolve_runtime() -> tuple[list[str], str]:
    """Pick the most reliable runtime command prefix."""
    direct_python = Path.home() / "miniconda3" / "envs" / CONDA_ENV / "bin" / "python"
    if direct_python.exists():
        return [str(direct_python)], f"Direct env python: {direct_python}"

    conda_path = shutil.which("conda")
    if conda_path:
        return [conda_path, "run", "-n", CONDA_ENV, "python"], f"Conda env: {CONDA_ENV}"

    return [sys.executable], f"Current python: {sys.executable}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
            :root {
                --bg: #f5f6ef;
                --card: #ffffff;
                --ink: #18212d;
                --muted: #5b6572;
                --accent: #0f766e;
                --accent-2: #f59e0b;
            }
            .stApp {
                font-family: "Space Grotesk", sans-serif;
                color: var(--ink);
                background:
                    radial-gradient(1200px 500px at -10% -10%, #d9f3ef 0%, transparent 60%),
                    radial-gradient(900px 400px at 110% 0%, #ffe9c5 0%, transparent 60%),
                    var(--bg);
            }
            .block-container {
                padding-top: 2.2rem;
            }
            .metric-card {
                background: var(--card);
                border: 1px solid #dce2e8;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.7rem;
            }
            .metric-title {
                font-size: 0.85rem;
                color: var(--muted);
            }
            .metric-value {
                font-size: 1rem;
                font-weight: 600;
            }
            .stButton > button {
                width: 100%;
                color: #ffffff;
                background: linear-gradient(120deg, #0f766e 0%, #115e59 100%);
                border: 1px solid #0b5e57;
                border-radius: 10px;
                font-weight: 700;
            }
            .stButton > button:hover {
                color: #ffffff;
                background: linear-gradient(120deg, #0d6b64 0%, #0f4f4a 100%);
                border-color: #0a514b;
            }
            .stButton > button:focus {
                color: #ffffff;
                box-shadow: 0 0 0 0.2rem rgba(15, 118, 110, 0.25);
                outline: none;
            }
            .stDownloadButton > button {
                color: #ffffff;
                background: linear-gradient(120deg, #0f766e 0%, #115e59 100%);
                border: 1px solid #0b5e57;
                border-radius: 10px;
                font-weight: 700;
            }
            .stDownloadButton > button:hover {
                color: #ffffff;
                background: linear-gradient(120deg, #0d6b64 0%, #0f4f4a 100%);
                border-color: #0a514b;
            }
            .stDownloadButton > button:focus {
                color: #ffffff;
                box-shadow: 0 0 0 0.2rem rgba(15, 118, 110, 0.25);
                outline: none;
            }
            code, pre {
                font-family: "IBM Plex Mono", monospace !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_card(title: str, exists: bool, path: Path) -> None:
    label = "Available" if exists else "Missing"
    st.markdown(
        (
            "<div class='metric-card'>"
            f"<div class='metric-title'>{title}</div>"
            f"<div class='metric-value'>{label}</div>"
            f"<div class='metric-title'>{path}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def find_files(pattern: str, default_path: Path | None = None) -> list[Path]:
    files = sorted([p for p in ROOT.glob(pattern) if p.is_file()])
    if default_path is not None and default_path.exists() and default_path not in files:
        files.insert(0, default_path)
    return files


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / uploaded_file.name
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def artifact_selector(
    title: str,
    pattern: str,
    default_path: Path,
    upload_dir: Path,
    upload_types: list[str],
    key_prefix: str,
) -> Path:
    options = find_files(pattern, default_path)
    option_labels = [str(p.relative_to(ROOT)) for p in options]
    if option_labels:
        default_index = option_labels.index(str(default_path.relative_to(ROOT))) if default_path in options else 0
        selected_label = st.selectbox(
            f"{title} (select existing)",
            options=option_labels,
            index=default_index,
            key=f"{key_prefix}_select",
        )
        selected_path = ROOT / selected_label
    else:
        st.caption("No existing file found yet for this item.")
        selected_path = default_path

    uploaded = st.file_uploader(
        f"{title} (browse/upload)",
        type=upload_types,
        key=f"{key_prefix}_upload",
    )
    if uploaded is not None:
        saved_path = save_uploaded_file(uploaded, upload_dir)
        st.success(f"Uploaded: {saved_path.relative_to(ROOT)}")
        st.session_state[f"{key_prefix}_select"] = str(saved_path.relative_to(ROOT))
        st.rerun()

    return selected_path


def run_script(script_path: Path, args: list[str] | None = None) -> tuple[bool, str]:
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    runtime_prefix, runtime_label = resolve_runtime()
    command = runtime_prefix + [str(script_path)] + (args or [])
    env = dict(os.environ)
    # Avoid MKL/OpenMP conflict in subprocesses launched from Streamlit.
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    output = f"Runtime: {runtime_label}\nCommand: {' '.join(command)}\n\n{output}".strip()
    return result.returncode == 0, output.strip()


def render_results(conf_threshold: float, predictions_path: Path) -> None:
    if not predictions_path.exists():
        st.info("Predictions file does not exist yet. Run prediction first.")
        return

    df = pd.read_csv(predictions_path)
    if df.empty:
        st.warning("Predictions file is empty.")
        return

    if "Confidence" in df.columns:
        conf_series = pd.to_numeric(
            df["Confidence"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        df = df[conf_series >= conf_threshold].copy()

    if df.empty:
        st.warning("No rows left after confidence filter.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_styles()

    st.title("NIFTY-NLP Streamlit GUI")
    st.caption("Simple controls for training and prediction on the current project artifacts.")

    with st.sidebar:
        st.subheader("Controls")
        _, runtime_label = resolve_runtime()
        st.caption(f"Script runtime: `{runtime_label}`")
        if runtime_label.startswith("Current python:"):
            st.warning("Using current Python runtime. Activate `nifty-rtx5060` before launching Streamlit.")
        confidence_filter = st.slider(
            "Min confidence (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
        )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Project Status")
        selected_train = artifact_selector(
            title="Train data CSV",
            pattern="data/**/*.csv",
            default_path=TRAIN_DATA,
            upload_dir=ROOT / "data" / "inputs",
            upload_types=["csv"],
            key_prefix="train_data",
        )
        status_card("Train data", selected_train.exists(), selected_train)

        selected_test = artifact_selector(
            title="Test data CSV",
            pattern="data/**/*.csv",
            default_path=TEST_DATA,
            upload_dir=ROOT / "data" / "inputs",
            upload_types=["csv"],
            key_prefix="test_data",
        )
        status_card("Test data", selected_test.exists(), selected_test)

        selected_model = artifact_selector(
            title="Model file",
            pattern="models/**/*.pth",
            default_path=MODEL_FILE,
            upload_dir=ROOT / "models",
            upload_types=["pth", "pt"],
            key_prefix="model_file",
        )
        status_card("Model file", selected_model.exists(), selected_model)

        selected_scaler = artifact_selector(
            title="Scaler file",
            pattern="models/**/*.joblib",
            default_path=SCALER_FILE,
            upload_dir=ROOT / "models",
            upload_types=["joblib", "pkl"],
            key_prefix="scaler_file",
        )
        status_card("Scaler file", selected_scaler.exists(), selected_scaler)

        selected_predictions = artifact_selector(
            title="Predictions CSV",
            pattern="data/predictions/**/*.csv",
            default_path=PREDICTIONS_FILE,
            upload_dir=ROOT / "data" / "predictions",
            upload_types=["csv"],
            key_prefix="predictions_file",
        )
        status_card("Predictions file", selected_predictions.exists(), selected_predictions)

    with col_b:
        st.subheader("Actions")
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Running training script..."):
                ok, log_text = run_script(
                    TRAIN_SCRIPT,
                    args=[
                        "--train-path", str(selected_train),
                        "--test-path", str(selected_test),
                        "--model-out", str(selected_model),
                        "--scaler-out", str(selected_scaler),
                    ],
                )
            if ok:
                st.success("Training completed.")
            else:
                st.error("Training failed.")
            if log_text:
                st.code(log_text, language="text")

        if st.button("Run Prediction", use_container_width=True):
            with st.spinner("Running prediction script..."):
                ok, log_text = run_script(
                    PREDICT_SCRIPT,
                    args=[
                        "--model-path", str(selected_model),
                        "--scaler-path", str(selected_scaler),
                        "--test-path", str(selected_test),
                        "--output-path", str(selected_predictions),
                    ],
                )
            if ok:
                st.success("Prediction completed.")
            else:
                st.error("Prediction failed.")
            if log_text:
                st.code(log_text, language="text")

    st.subheader("Prediction Results")
    render_results(confidence_filter, selected_predictions)


if __name__ == "__main__":
    main()
