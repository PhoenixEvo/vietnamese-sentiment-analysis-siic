import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

warnings.filterwarnings("ignore")

# Set Vietnamese font
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]


def load_pickle_results(file_path):
    """Load results from pickle file"""
    try:
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_comparison_plots(baseline_results, finetuned_results):
    """Create comprehensive comparison plots"""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Accuracy Comparison",
            "F1-Score Comparison",
            "Confusion Matrix - Baseline",
            "Confusion Matrix - Fine-tuned",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # Extract metrics
    baseline_accuracy = baseline_results.get("accuracy", 0)
    finetuned_accuracy = finetuned_results.get("accuracy", 0)

    baseline_f1 = baseline_results.get("f1_score", 0)
    finetuned_f1 = finetuned_results.get("f1_score", 0)

    # Accuracy comparison
    fig.add_trace(
        go.Bar(
            x=["Baseline", "Fine-tuned"],
            y=[baseline_accuracy, finetuned_accuracy],
            name="Accuracy",
            marker_color=["#ff7f0e", "#2ca02c"],
            text=[f"{baseline_accuracy:.3f}", f"{finetuned_accuracy:.3f}"],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # F1-Score comparison
    fig.add_trace(
        go.Bar(
            x=["Baseline", "Fine-tuned"],
            y=[baseline_f1, finetuned_f1],
            name="F1-Score",
            marker_color=["#ff7f0e", "#2ca02c"],
            text=[f"{baseline_f1:.3f}", f"{finetuned_f1:.3f}"],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # Confusion matrices
    emotions = ["negative", "neutral", "positive"]

    # Baseline confusion matrix
    baseline_cm = baseline_results.get("confusion_matrix", np.zeros((3, 3)))
    fig.add_trace(
        go.Heatmap(
            z=baseline_cm,
            x=emotions,
            y=emotions,
            colorscale="Blues",
            text=baseline_cm.astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
            name="Baseline CM",
        ),
        row=2,
        col=1,
    )

    # Fine-tuned confusion matrix
    finetuned_cm = finetuned_results.get("confusion_matrix", np.zeros((3, 3)))
    fig.add_trace(
        go.Heatmap(
            z=finetuned_cm,
            x=emotions,
            y=emotions,
            colorscale="Greens",
            text=finetuned_cm.astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
            name="Fine-tuned CM",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "PhoBERT Baseline vs Fine-tuned Performance Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        height=800,
        showlegend=False,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Model Version", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Model Version", row=1, col=2)
    fig.update_yaxes(title_text="F1-Score", row=1, col=2)
    fig.update_xaxes(title_text="Predicted", row=2, col=1)
    fig.update_yaxes(title_text="Actual", row=2, col=1)
    fig.update_xaxes(title_text="Predicted", row=2, col=2)
    fig.update_yaxes(title_text="Actual", row=2, col=2)

    return fig


def create_detailed_metrics_comparison(baseline_results, finetuned_results):
    """Create detailed metrics comparison"""

    # Extract classification reports
    baseline_report = baseline_results.get("classification_report", {})
    finetuned_report = finetuned_results.get("classification_report", {})

    # Create metrics dataframe
    metrics_data = []
    emotions = ["negative", "neutral", "positive"]

    for emotion in emotions:
        # Baseline metrics
        if emotion in baseline_report:
            baseline_precision = baseline_report[emotion]["precision"]
            baseline_recall = baseline_report[emotion]["recall"]
            baseline_f1 = baseline_report[emotion]["f1-score"]
        else:
            baseline_precision = baseline_recall = baseline_f1 = 0

        # Fine-tuned metrics
        if emotion in finetuned_report:
            finetuned_precision = finetuned_report[emotion]["precision"]
            finetuned_recall = finetuned_report[emotion]["recall"]
            finetuned_f1 = finetuned_report[emotion]["f1-score"]
        else:
            finetuned_precision = finetuned_recall = finetuned_f1 = 0

        metrics_data.extend(
            [
                {
                    "Emotion": emotion,
                    "Model": "Baseline",
                    "Metric": "Precision",
                    "Value": baseline_precision,
                },
                {
                    "Emotion": emotion,
                    "Model": "Baseline",
                    "Metric": "Recall",
                    "Value": baseline_recall,
                },
                {
                    "Emotion": emotion,
                    "Model": "Baseline",
                    "Metric": "F1-Score",
                    "Value": baseline_f1,
                },
                {
                    "Emotion": emotion,
                    "Model": "Fine-tuned",
                    "Metric": "Precision",
                    "Value": finetuned_precision,
                },
                {
                    "Emotion": emotion,
                    "Model": "Fine-tuned",
                    "Metric": "Recall",
                    "Value": finetuned_recall,
                },
                {
                    "Emotion": emotion,
                    "Model": "Fine-tuned",
                    "Metric": "F1-Score",
                    "Value": finetuned_f1,
                },
            ]
        )

    df_metrics = pd.DataFrame(metrics_data)

    # Create detailed comparison plot
    fig = px.bar(
        df_metrics,
        x="Emotion",
        y="Value",
        color="Model",
        facet_col="Metric",
        barmode="group",
        color_discrete_map={"Baseline": "#ff7f0e", "Fine-tuned": "#2ca02c"},
        title="Detailed Metrics Comparison by Emotion",
    )

    fig.update_layout(height=600, title_x=0.5)

    return fig, df_metrics


def main():
    print("🔍 Loading PhoBERT results...")

    # Load results
    baseline_results = load_pickle_results("results/phobert_baseline_results.pkl")
    finetuned_results = load_pickle_results("results/phobert_results_20250716_172440.pkl")

    if baseline_results is None or finetuned_results is None:
        print("❌ Error: Could not load one or both result files")
        return

    print("✅ Results loaded successfully!")
    baseline_acc = baseline_results.get("accuracy", "N/A")
    finetuned_acc = finetuned_results.get("accuracy", "N/A")
    print(
        f"Baseline accuracy: {baseline_acc:.4f}"
        if isinstance(baseline_acc, (int, float))
        else f"Baseline accuracy: {baseline_acc}"
    )
    print(
        f"Fine-tuned accuracy: {finetuned_acc:.4f}"
        if isinstance(finetuned_acc, (int, float))
        else f"Fine-tuned accuracy: {finetuned_acc}"
    )

    # Create comparison plots
    print("📊 Creating comparison plots...")

    # Main comparison plot
    fig_main = create_comparison_plots(baseline_results, finetuned_results)
    fig_main.write_html("results/phobert_baseline_vs_finetuned_comparison.html")
    fig_main.write_image(
        "results/phobert_baseline_vs_finetuned_comparison.png", width=1200, height=800
    )

    # Detailed metrics comparison
    fig_detailed, df_metrics = create_detailed_metrics_comparison(
        baseline_results, finetuned_results
    )
    fig_detailed.write_html("results/phobert_detailed_metrics_comparison.html")
    fig_detailed.write_image(
        "results/phobert_detailed_metrics_comparison.png", width=1200, height=600
    )

    # Save metrics to CSV
    df_metrics.to_csv("results/phobert_metrics_comparison.csv", index=False)

    # Print summary
    print("\n📈 Performance Summary:")
    print("=" * 50)

    baseline_acc = baseline_results.get("accuracy", 0)
    finetuned_acc = finetuned_results.get("accuracy", 0)
    baseline_f1 = baseline_results.get("f1_score", 0)
    finetuned_f1 = finetuned_results.get("f1_score", 0)

    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Fine-tuned Accuracy: {finetuned_acc:.4f}")
    if baseline_acc > 0:
        improvement_acc = (finetuned_acc - baseline_acc) / baseline_acc * 100
        print(f"Improvement: {improvement_acc:.2f}%")
    else:
        print("Improvement: N/A (baseline accuracy is 0)")

    print(f"\nBaseline F1-Score: {baseline_f1:.4f}")
    print(f"Fine-tuned F1-Score: {finetuned_f1:.4f}")
    if baseline_f1 > 0:
        improvement_f1 = (finetuned_f1 - baseline_f1) / baseline_f1 * 100
        print(f"Improvement: {improvement_f1:.2f}%")
    else:
        print("Improvement: N/A (baseline F1-score is 0)")

    print("\n✅ Files created:")
    print("- results/phobert_baseline_vs_finetuned_comparison.html")
    print("- results/phobert_baseline_vs_finetuned_comparison.png")
    print("- results/phobert_detailed_metrics_comparison.html")
    print("- results/phobert_detailed_metrics_comparison.png")
    print("- results/phobert_metrics_comparison.csv")


if __name__ == "__main__":
    main()
