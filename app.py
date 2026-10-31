import io
import json
import hashlib
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


st.set_page_config(
    page_title="Automated Data Label Quality Auditor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


CLASSES = ["cat", "dog", "bird", "fish", "horse"]
VALID_CLASSES = set(CLASSES)
ANNOTATORS = [f"annotator_{i:02d}" for i in range(1, 9)]
GUIDELINES = ["GL-2026.09", "GL-2026.10"]
CURRENT_GUIDELINE = GUIDELINES[-1]
DEMO_NOW = datetime(2026, 10, 31, 18, 0, 0)
DEFAULT_MAX_ROWS = 3000
SUSPICION_THRESHOLD_FOR_METRICS = 0.60


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().lower()


def _corrupt_label(label: str, rng: np.random.Generator) -> str:
    candidates = [
        label.upper(),
        label.capitalize(),
        f" {label} ",
        f"{label}s",
        label.replace("a", "@") if "a" in label else f"{label}?",
        "unknown",
        "",
        "object",
        "other",
    ]
    return str(rng.choice(candidates))


def _make_row(
    annotation_id: int,
    item_id: str,
    annotator_id: str,
    label: str,
    timestamp: datetime,
    guideline_version: str,
    confidence: float,
    source_format: str,
) -> dict[str, Any]:
    return {
        "annotation_id": f"ann_{annotation_id:07d}",
        "item_id": item_id,
        "annotator_id": annotator_id,
        "label": label,
        "timestamp": timestamp,
        "guideline_version": guideline_version,
        "confidence": float(np.clip(confidence, 0.01, 0.99)),
        "source_format": source_format,
    }


def generate_initial_data(seed: int, target_items: int = 180) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(seed)
    annotator_quality = {
        annotator: max(0.72, 0.99 - index * 0.025)
        for index, annotator in enumerate(ANNOTATORS)
    }

    rows: list[dict[str, Any]] = []
    annotation_counter = 1

    for item_index in range(target_items):
        item_id = f"asset_{item_index:06d}"
        true_label = str(rng.choice(CLASSES))

        n_annotations = int(
            rng.choice(
                [0, 1, 2, 3, 4, 5],
                p=[0.05, 0.12, 0.25, 0.30, 0.18, 0.10],
            )
        )

        if n_annotations > 0:
            annotators_for_item = rng.choice(ANNOTATORS, size=n_annotations, replace=False)
        else:
            annotators_for_item = []

        base_timestamp = DEMO_NOW - timedelta(minutes=int(rng.integers(15, 540)))
        guideline_version = (
            GUIDELINES[0]
            if base_timestamp < DEMO_NOW - timedelta(hours=3)
            else GUIDELINES[1]
        )

        for raw_annotator in annotators_for_item:
            annotator_id = str(raw_annotator)
            quality = annotator_quality[annotator_id]

            if rng.random() > quality:
                label = _corrupt_label(true_label, rng)
            else:
                label = true_label

            if rng.random() < 0.015:
                label = ""

            confidence = float(
                np.clip(
                    rng.normal(0.55 + 0.40 * quality, 0.10),
                    0.01,
                    0.99,
                )
            )

            timestamp = base_timestamp + timedelta(seconds=int(rng.integers(0, 180)))

            rows.append(
                _make_row(
                    annotation_id=annotation_counter,
                    item_id=item_id,
                    annotator_id=annotator_id,
                    label=label,
                    timestamp=timestamp,
                    guideline_version=guideline_version,
                    confidence=confidence,
                    source_format="synthetic_coco_metadata",
                )
            )
            annotation_counter += 1

            if rng.random() < 0.02:
                rows.append(
                    _make_row(
                        annotation_id=annotation_counter,
                        item_id=item_id,
                        annotator_id=annotator_id,
                        label=label,
                        timestamp=timestamp + timedelta(seconds=7),
                        guideline_version=guideline_version,
                        confidence=confidence,
                        source_format="synthetic_label_studio_export",
                    )
                )
                annotation_counter += 1

    df = pd.DataFrame(rows)
    return df, annotation_counter, target_items


def generate_live_rows(
    existing_df: pd.DataFrame,
    tick: int,
    rng: np.random.Generator,
    next_annotation_id: int,
    next_item_number: int,
    max_new: int = 7,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    n_new = int(rng.integers(2, max_new + 1))

    existing_items = (
        existing_df.item_id.dropna().unique().tolist()
        if not existing_df.empty
        else []
    )

    simulated_timestamp = DEMO_NOW + timedelta(seconds=tick)
    anomaly_burst = tick % 9 == 0

    for _ in range(n_new):
        if existing_items and rng.random() < 0.78:
            item_id = str(rng.choice(existing_items))
            raw_labels = (
                existing_df.loc[existing_df.item_id == item_id, "label"]
                .dropna()
                .astype(str)
                .map(normalize_label)
            )
            valid_labels = [label for label in raw_labels if label in VALID_CLASSES]
            base_label = (
                Counter(valid_labels).most_common(1)[0][0]
                if valid_labels
                else str(rng.choice(CLASSES))
            )
        else:
            item_id = f"asset_{next_item_number:06d}"
            next_item_number += 1
            base_label = str(rng.choice(CLASSES))

        annotator_id = str(rng.choice(ANNOTATORS))

        noise_probability = 0.22 if anomaly_burst and rng.random() < 0.45 else 0.08

        if rng.random() < noise_probability:
            label = _corrupt_label(base_label, rng)
        else:
            label = base_label

        if rng.random() < 0.02:
            label = ""

        confidence = float(
            np.clip(
                rng.normal(0.78 if label == base_label else 0.48, 0.12),
                0.01,
                0.99,
            )
        )

        timestamp = simulated_timestamp + timedelta(milliseconds=int(rng.integers(0, 950)))

        rows.append(
            _make_row(
                annotation_id=next_annotation_id,
                item_id=item_id,
                annotator_id=annotator_id,
                label=label,
                timestamp=timestamp,
                guideline_version=CURRENT_GUIDELINE,
                confidence=confidence,
                source_format="synthetic_jsonl_stream",
            )
        )
        next_annotation_id += 1

        if rng.random() < 0.03:
            rows.append(
                _make_row(
                    annotation_id=next_annotation_id,
                    item_id=item_id,
                    annotator_id=annotator_id,
                    label=label,
                    timestamp=timestamp + timedelta(seconds=3),
                    guideline_version=CURRENT_GUIDELINE,
                    confidence=confidence,
                    source_format="synthetic_duplicate_replay",
                )
            )
            next_annotation_id += 1

    return rows, next_annotation_id, next_item_number


def cohen_kappa(x: list[str], y: list[str]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")

    ax = np.array(x)
    ay = np.array(y)

    observed_agreement = float(np.mean(ax == ay))

    px = pd.Series(ax).value_counts(normalize=True)
    py = pd.Series(ay).value_counts(normalize=True)

    categories = set(ax.tolist()) | set(ay.tolist())
    expected_agreement = float(
        sum(
            float(px.get(category, 0.0)) * float(py.get(category, 0.0))
            for category in categories
        )
    )

    if np.isclose(expected_agreement, 1.0):
        return 1.0 if np.isclose(observed_agreement, 1.0) else 0.0

    return float((observed_agreement - expected_agreement) / (1.0 - expected_agreement))


def compute_quality(raw_df: pd.DataFrame, sim_now: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw_df.copy()

    if df.empty:
        df["norm_label"] = pd.Series(dtype="string")
        df["is_valid"] = pd.Series(dtype="bool")
        df["is_duplicate"] = pd.Series(dtype="bool")
        df["suspicion_score"] = pd.Series(dtype="float")

        metrics = {
            "total_annotations": 0,
            "total_items": 0,
            "completeness": 0.0,
            "validity": 0.0,
            "uniqueness": 1.0,
            "consistency": float("nan"),
            "kappa": float("nan"),
            "suspicious_rate": 0.0,
            "flagged_count": 0,
            "median_age_hours": float("nan"),
        }
        return df, metrics

    df["norm_label"] = df["label"].map(normalize_label)
    df["is_valid"] = (
        df["norm_label"].isin(VALID_CLASSES)
        & df["label"].map(lambda value: not pd.isna(value) and str(value).strip() != "")
    )
    df["is_duplicate"] = df.duplicated(subset=["item_id", "annotator_id"], keep=False)

    total_annotations = int(len(df))
    total_items = int(df.item_id.nunique())
    annotated_items = int(df.loc[df.is_valid, "item_id"].nunique())

    completeness = float(annotated_items / total_items) if total_items else 0.0
    validity = float(df.is_valid.mean())
    uniqueness = float(1.0 - df.is_duplicate.mean())

    valid_df = df[df.is_valid]

    if valid_df.empty:
        consistency = float("nan")
        kappa = float("nan")
        majority_ratio: dict[str, float] = {}
        class_freq: dict[str, float] = {}
        ann_valid = pd.Series(dtype="float")
        ann_disagree: dict[str, float] = {}
    else:
        grouped_valid = valid_df.groupby("item_id")["norm_label"]
        counts = grouped_valid.size()
        majority_ratio = grouped_valid.apply(
            lambda series: float(series.value_counts(normalize=True).iloc[0])
        ).to_dict()

        multi_label_items = counts[counts >= 2].index
        consistency = (
            float(np.mean([majority_ratio[item_id] for item_id in multi_label_items]))
            if len(multi_label_items)
            else float("nan")
        )

        class_freq = valid_df["norm_label"].value_counts(normalize=True).to_dict()
        ann_valid = df.groupby("annotator_id")["is_valid"].mean()

        ann_disagree = {}
        for annotator_id, group in valid_df.groupby("annotator_id"):
            ratios = group["item_id"].map(majority_ratio)
            ann_disagree[str(annotator_id)] = (
                float((1.0 - ratios).mean()) if len(ratios) else 0.0
            )

        kappa = float("nan")
        if valid_df.annotator_id.nunique() >= 2:
            pivot = valid_df.pivot_table(
                index="item_id",
                columns="annotator_id",
                values="norm_label",
                aggfunc="first",
            )
            pair_scores: list[float] = []

            for annotator_a, annotator_b in combinations(pivot.columns.tolist(), 2):
                subset = pivot[[annotator_a, annotator_b]].dropna()
                if len(subset) >= 12:
                    score = cohen_kappa(
                        subset[annotator_a].astype(str).tolist(),
                        subset[annotator_b].astype(str).tolist(),
                    )
                    if not np.isnan(score):
                        pair_scores.append(score)

            if pair_scores:
                kappa = float(np.mean(pair_scores))

    scores: list[float] = []

    for row in df.itertuples(index=False):
        score = 0.0

        if not row.is_valid:
            score += 0.45

        if row.is_duplicate:
            score += 0.20

        item_majority_ratio = majority_ratio.get(row.item_id, float("nan"))
        if row.is_valid and not pd.isna(item_majority_ratio):
            score += 0.35 * (1.0 - float(item_majority_ratio))

        score += 0.15 * (1.0 - float(row.confidence))

        if row.is_valid and class_freq.get(row.norm_label, 1.0) < 0.06:
            score += 0.10

        if float(ann_valid.get(row.annotator_id, 1.0)) < 0.85:
            score += 0.10

        if float(ann_disagree.get(row.annotator_id, 0.0)) > 0.25:
            score += 0.10

        scores.append(min(max(score, 0.0), 1.0))

    df["suspicion_score"] = scores

    suspicious_rate = float((df.suspicion_score >= SUSPICION_THRESHOLD_FOR_METRICS).mean())
    flagged_count = int((df.suspicion_score >= SUSPICION_THRESHOLD_FOR_METRICS).sum())

    ages_hours = (
        pd.to_datetime(sim_now) - pd.to_datetime(df.timestamp)
    ).dt.total_seconds() / 3600.0
    median_age_hours = float(np.nanmedian(ages_hours)) if len(ages_hours) else float("nan")

    metrics = {
        "total_annotations": total_annotations,
        "total_items": total_items,
        "completeness": completeness,
        "validity": validity,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "kappa": kappa,
        "suspicious_rate": suspicious_rate,
        "flagged_count": flagged_count,
        "median_age_hours": median_age_hours,
    }

    return df, metrics


def _history_row(tick: int, sim_now: datetime, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "tick": tick,
        "sim_time": sim_now.strftime("%H:%M:%S"),
        **metrics,
    }


def _fmt_fraction(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pp_delta(current: Any, previous: Any) -> str | None:
    if current is None or previous is None:
        return None
    if pd.isna(current) or pd.isna(previous):
        return None
    delta = (float(current) - float(previous)) * 100.0
    return f"{delta:+.1f} pp"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def build_demo_gif(
    history_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    review_threshold: float = 0.60,
) -> bytes | None:
    if not PIL_AVAILABLE:
        return None

    if history_df.empty:
        history_df = pd.DataFrame(
            [
                {
                    "tick": 0,
                    "sim_time": "18:00:00",
                    "suspicious_rate": 0.0,
                    "validity": 0.0,
                    "consistency": 0.0,
                }
            ]
        )

    width, height = 780, 440
    frame_count = int(np.clip(len(history_df) * 2, 12, 36))
    indices = np.linspace(0, len(history_df) - 1, frame_count).astype(int)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    frames: list[Image.Image] = []

    for frame_position, history_index in enumerate(indices):
        img = Image.new("RGB", (width, height), (17, 24, 39))
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, width, 58], fill=(30, 41, 59))
        draw.text(
            (18, 18),
            "Automated Data Label Quality Auditor — live demo",
            fill=(226, 232, 240),
            font=font,
        )

        row = history_df.iloc[history_index]
        tick_value = int(row.get("tick", history_index))
        sim_time = str(row.get("sim_time", ""))

        left_lines = [
            f"Tick {tick_value}  {sim_time}",
            f"Validity: {_fmt_fraction(row.get('validity', 0.0))}",
            f"Consistency: {_fmt_fraction(row.get('consistency', 0.0))}",
            f"Suspicion: {_fmt_fraction(row.get('suspicious_rate', 0.0))}",
            f"Kappa: {_fmt_float(row.get('kappa', float('nan')))}",
        ]

        y = 78
        for line in left_lines:
            draw.text((22, y), line, fill=(203, 213, 225), font=font)
            y += 20

        chart_box = (310, 78, width - 42, height - 118)
        draw.rectangle(chart_box, outline=(71, 85, 105), width=1)
        draw.text(
            (chart_box[0] + 8, chart_box[1] - 20),
            "Suspicion rate over time (%)",
            fill=(148, 163, 184),
            font=font,
        )

        subset = history_df.iloc[: history_index + 1]
        if len(subset) > 1 and "suspicious_rate" in subset.columns:
            values = (
                pd.to_numeric(subset["suspicious_rate"], errors="coerce")
                .fillna(0.0)
                .to_numpy()
                * 100.0
            )
            xs = np.linspace(chart_box[0] + 12, chart_box[2] - 12, len(values))
            ys = (
                chart_box[3]
                - 12
                - (np.clip(values, 0.0, 100.0) / 100.0)
                * (chart_box[3] - chart_box[1] - 24)
            )
            points = [(int(x), int(y_point)) for x, y_point in zip(xs, ys)]
            draw.line(points, fill=(248, 113, 113), width=3)

            if points:
                last_x, last_y = points[-1]
                draw.ellipse(
                    [last_x - 4, last_y - 4, last_x + 4, last_y + 4],
                    fill=(251, 191, 36),
                )

        draw.text(
            (22, height - 100),
            "Top flagged records",
            fill=(226, 232, 240),
            font=font,
        )

        if not enriched_df.empty and "suspicion_score" in enriched_df.columns:
            top = enriched_df.sort_values("suspicion_score", ascending=False).head(3)
        else:
            top = pd.DataFrame()

        yy = height - 78
        if top.empty:
            draw.text(
                (22, yy),
                "No flagged records in current window",
                fill=(148, 163, 184),
                font=font,
            )
        else:
            for _, record in top.iterrows():
                score = float(record.get("suspicion_score", 0.0) or 0.0)
                if np.isnan(score):
                    score = 0.0

                label_text = normalize_label(record.get("label", "")) or "missing"
                line = (
                    f"{record.get('item_id', '')} | "
                    f"{record.get('annotator_id', '')} | "
                    f"{label_text[:18]} | "
                    f"score {score:.2f}"
                )

                color = (251, 191, 36) if score >= review_threshold else (148, 163, 184)
                draw.text((22, yy), line[:96], fill=color, font=font)
                yy += 18

        draw.text(
            (width - 160, height - 26),
            f"Frame {frame_position + 1}/{frame_count}",
            fill=(100, 116, 139),
            font=font,
        )

        frames.append(img)

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=260,
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()


def live_panel_impl() -> None:
    if not hasattr(st, "fragment"):
        st.caption(
            "Automatic refresh is unavailable in this Streamlit runtime. "
            "Use the sidebar controls or review-queue sync button to rerun."
        )

    st.subheader("Live annotation stream")

    if not st.session_state.get("sim_running", False):
        st.info("Simulation paused. Toggle 'Run live simulation' in the sidebar.")
    else:
        tick = int(st.session_state.get("tick", 0)) + 1
        sim_now = DEMO_NOW + timedelta(seconds=tick)
        rng = np.random.default_rng(int(st.session_state.get("seed", 42)) + 10000 + tick)

        new_rows, next_annotation_id, next_item_number = generate_live_rows(
            existing_df=st.session_state.df,
            tick=tick,
            rng=rng,
            next_annotation_id=int(st.session_state.next_id),
            next_item_number=int(st.session_state.next_item),
        )

        df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)

        max_rows = int(st.session_state.get("max_rows", DEFAULT_MAX_ROWS))
        if len(df) > max_rows:
            df = df.iloc[-max_rows:].reset_index(drop=True)

        enriched, metrics = compute_quality(df, sim_now)

        new_history_row = _history_row(tick, sim_now, metrics)
        history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_history_row])],
            ignore_index=True,
        )

        if len(history) > 240:
            history = history.tail(240).reset_index(drop=True)

        st.session_state.update(
            {
                "df": df,
                "enriched": enriched,
                "metrics": metrics,
                "history": history,
                "tick": tick,
                "next_id": next_annotation_id,
                "next_item": next_item_number,
                "sim_now": sim_now,
            }
        )

    metrics = st.session_state.metrics
    history = st.session_state.history
    enriched = st.session_state.enriched
    sim_now = st.session_state.sim_now

    previous_row = history.iloc[-2] if len(history) > 1 else None

    annotation_delta = None
    if previous_row is not None:
        previous_total = previous_row.get("total_annotations")
        if not pd.isna(previous_total):
            annotation_delta = f"{int(metrics['total_annotations'] - int(previous_total)):+d}"

    cols = st.columns(5)
    cols[0].metric("Annotations", f"{metrics['total_annotations']:,}", delta=annotation_delta)
    cols[1].metric(
        "Completeness",
        _fmt_fraction(metrics["completeness"]),
        delta=_pp_delta(metrics["completeness"], previous_row.get("completeness") if previous_row is not None else None),
    )
    cols[2].metric(
        "Validity",
        _fmt_fraction(metrics["validity"]),
        delta=_pp_delta(metrics["validity"], previous_row.get("validity") if previous_row is not None else None),
    )
    cols[3].metric(
        "Consistency",
        _fmt_fraction(metrics["consistency"]),
        delta=_pp_delta(metrics["consistency"], previous_row.get("consistency") if previous_row is not None else None),
    )
    cols[4].metric(
        "Suspicious",
        _fmt_fraction(metrics["suspicious_rate"]),
        delta=_pp_delta(metrics["suspicious_rate"], previous_row.get("suspicious_rate") if previous_row is not None else None),
    )

    kappa_text = _fmt_float(metrics["kappa"])
    median_age_text = _fmt_float(metrics["median_age_hours"], digits=2)

    st.caption(
        f"Simulated clock: {sim_now:%H:%M:%S} | "
        f"Tick: {st.session_state.tick} | "
        f"Items: {metrics['total_items']:,} | "
        f"Cohen kappa proxy: {kappa_text} | "
        f"Median age: {median_age_text} h"
    )

    chart_df = history.copy()
    for column in ["suspicious_rate", "validity", "consistency"]:
        if column in chart_df.columns:
            chart_df[column] = (
                pd.to_numeric(chart_df[column], errors="coerce").fillna(0.0) * 100.0
            )

    if "tick" in chart_df.columns:
        chart_df = chart_df.set_index("tick")
        visible_columns = [
            column
            for column in ["suspicious_rate", "validity", "consistency"]
            if column in chart_df.columns
        ]
        if visible_columns:
            st.line_chart(chart_df[visible_columns], height=250)

    st.markdown("**Recent annotation events**")
    if not enriched.empty:
        event_df = enriched.tail(10).copy()
        event_df["timestamp"] = pd.to_datetime(event_df["timestamp"]).dt.strftime("%H:%M:%S")
        event_df["label"] = event_df["label"].astype(str).replace({"nan": ""})
        event_df["confidence"] = event_df["confidence"].round(3)
        event_df["suspicion_score"] = event_df["suspicion_score"].round(3)

        st.dataframe(
            event_df[
                [
                    "timestamp",
                    "item_id",
                    "annotator_id",
                    "label",
                    "confidence",
                    "suspicion_score",
                    "source_format",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No events yet.")


if hasattr(st, "fragment"):
    live_panel = st.fragment(live_panel_impl, run_every=1.0)
else:
    live_panel = live_panel_impl


st.sidebar.title("Audit controls")

seed = int(
    st.sidebar.number_input(
        "Deterministic seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )
)

max_rows = int(
    st.sidebar.number_input(
        "Max live rows",
        min_value=500,
        max_value=10000,
        value=DEFAULT_MAX_ROWS,
        step=100,
    )
)

review_threshold = float(
    st.sidebar.slider(
        "Review queue threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
    )
)

st.sidebar.checkbox("Run live simulation", value=True, key="sim_running")

if st.sidebar.button("Reset demo"):
    st.session_state.clear()
    st.rerun()

needs_init = (
    "initialized" not in st.session_state
    or st.session_state.get("seed") != seed
    or st.session_state.get("max_rows") != max_rows
)

if needs_init:
    with st.spinner("Building deterministic synthetic annotation corpus..."):
        base_df, next_annotation_id, next_item_number = generate_initial_data(seed)

        df = base_df.copy()
        enriched, metrics = compute_quality(df, DEMO_NOW)
        history = pd.DataFrame([_history_row(0, DEMO_NOW, metrics)])

        for tick in range(1, 16):
            rng = np.random.default_rng(seed + 1000 + tick)
            rows, next_annotation_id, next_item_number = generate_live_rows(
                existing_df=df,
                tick=tick,
                rng=rng,
                next_annotation_id=next_annotation_id,
                next_item_number=next_item_number,
            )

            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

            if len(df) > max_rows:
                df = df.iloc[-max_rows:].reset_index(drop=True)

            sim_now = DEMO_NOW + timedelta(seconds=tick)
            enriched, metrics = compute_quality(df, sim_now)
            history = pd.concat(
                [history, pd.DataFrame([_history_row(tick, sim_now, metrics)])],
                ignore_index=True,
            )

        initial_fingerprint = hashlib.sha256(
            base_df.to_csv(index=False).encode("utf-8")
        ).hexdigest()

        try:
            demo_gif = build_demo_gif(history, enriched, 0.60)
        except Exception:
            demo_gif = None

        st.session_state.update(
            {
                "initialized": True,
                "seed": seed,
                "max_rows": max_rows,
                "df": df,
                "enriched": enriched,
                "metrics": metrics,
                "history": history,
                "tick": 15,
                "next_id": next_annotation_id,
                "next_item": next_item_number,
                "sim_now": DEMO_NOW + timedelta(seconds=15),
                "initial_fingerprint": initial_fingerprint,
                "decisions": {},
                "demo_gif": demo_gif,
            }
        )

st.title("🧪 Automated Data Label Quality Auditor")
st.caption(
    "CPU-only Streamlit prototype. Synthetic demo data is loaded automatically. "
    "Metrics shown here are illustrative and generated from a deterministic demo corpus, "
    "not claims about a real production dataset."
)

live_panel()

st.divider()
st.subheader("Human review queue")

sync_col, note_col = st.columns([1, 4])
with note_col:
    st.caption(
        "The live panel updates automatically. Interacting with this queue, or pressing sync, "
        "refreshes the review state with the latest simulation snapshot."
    )
with sync_col:
    if st.button("Sync review queue"):
        st.rerun()

if "decisions" not in st.session_state:
    st.session_state.decisions = {}

enriched = st.session_state.enriched
flagged = (
    enriched[enriched.suspicion_score >= review_threshold]
    .sort_values("suspicion_score", ascending=False)
    .copy()
)

if flagged.empty:
    st.success("No records exceed the current review threshold. Lower the threshold or continue the simulation.")
else:
    options = flagged.annotation_id.tolist()

    if st.session_state.get("selected_flag") not in options:
        st.session_state["selected_flag"] = options[0]

    selected_annotation_id = st.selectbox(
        "Select flagged annotation",
        options,
        key="selected_flag",
    )

    row = flagged.loc[flagged.annotation_id == selected_annotation_id].iloc[0]
    row_dict = {key: _to_jsonable(value) for key, value in row.to_dict().items()}
    row_dict["timestamp"] = str(pd.to_datetime(row_dict["timestamp"]))

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("**Evidence record**")
        st.json(row_dict)

    with right:
        st.markdown("**Review action**")

        existing_decision = st.session_state.decisions.get(selected_annotation_id)
        if existing_decision:
            st.success(f"Current decision: {existing_decision.get('action', 'unknown')}")

        action_col_1, action_col_2, action_col_3 = st.columns(3)

        if action_col_1.button("Accept original", key=f"accept_{selected_annotation_id}"):
            st.session_state.decisions[selected_annotation_id] = {
                "annotation_id": selected_annotation_id,
                "action": "accept_original",
                "original_label": str(row.label),
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "evidence_score": float(row.suspicion_score),
                "guideline_version": str(row.guideline_version),
            }
            st.rerun()

        if action_col_2.button("Defer", key=f"defer_{selected_annotation_id}"):
            st.session_state.decisions[selected_annotation_id] = {
                "annotation_id": selected_annotation_id,
                "action": "defer",
                "original_label": str(row.label),
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "evidence_score": float(row.suspicion_score),
                "guideline_version": str(row.guideline_version),
            }
            st.rerun()

        if action_col_3.button("Clear decision", key=f"clear_{selected_annotation_id}"):
            st.session_state.decisions.pop(selected_annotation_id, None)
            st.rerun()

        proposed_label = st.text_input(
            "Proposed canonical label",
            value=str(row.norm_label),
            key=f"proposed_{selected_annotation_id}",
        )

        if st.button("Save adjudication", key=f"save_adjudication_{selected_annotation_id}"):
            st.session_state.decisions[selected_annotation_id] = {
                "annotation_id": selected_annotation_id,
                "action": "adjudicate",
                "original_label": str(row.label),
                "proposed_label": proposed_label,
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "evidence_score": float(row.suspicion_score),
                "guideline_version": str(row.guideline_version),
            }
            st.rerun()

    st.markdown("#### Change history")
    st.caption("Original labels are never overwritten. Decisions are stored as separate review derivatives.")

    if st.session_state.decisions:
        trail = pd.DataFrame(list(st.session_state.decisions.values()))
        st.dataframe(trail, use_container_width=True, hide_index=True)

        st.download_button(
            "Export audit trail JSON",
            data=json.dumps(st.session_state.decisions, indent=2, default=str).encode("utf-8"),
            file_name="label_audit_decisions.json",
            mime="application/json",
        )
    else:
        st.info("No review decisions recorded yet.")

st.divider()
st.subheader("Working demo reel")

gif_col, preview_col = st.columns([1, 2])

with gif_col:
    if st.button("Regenerate downloadable demo GIF"):
        with st.spinner("Rendering frames from the live simulation..."):
            generated_gif = build_demo_gif(
                st.session_state.history,
                st.session_state.enriched,
                review_threshold,
            )
        st.session_state.demo_gif = generated_gif

    if not PIL_AVAILABLE:
        st.warning(
            "Pillow is unavailable, so GIF generation is disabled. "
            "The interactive dashboard still works."
        )

with preview_col:
    if st.session_state.get("demo_gif"):
        st.image(
            st.session_state.demo_gif,
            caption="Auto-generated demo reel (animated GIF)",
            use_container_width=True,
        )
        st.download_button(
            "Download demo GIF",
            data=st.session_state.demo_gif,
            file_name="label_quality_live_demo.gif",
            mime="image/gif",
            use_container_width=True,
        )
    else:
        st.info("A demo reel will appear here after rendering.")

with st.expander("Lineage, immutability, and fallback behavior"):
    st.markdown(
        f"""
- Initial synthetic corpus SHA-256: `{st.session_state.get('initial_fingerprint', 'n/a')}`
- Raw demo inputs are treated as immutable evidence.
- Review decisions are stored separately and exported as derivatives.
- Deterministic seed: `{seed}`
- Simulated clock starts at: `{DEMO_NOW:%Y-%m-%d %H:%M:%S}`
- No paid API keys or external services are required.
- Optional video/GIF rendering fallback: if Pillow is unavailable, GIF export is disabled but the dashboard remains usable.
        """
    )