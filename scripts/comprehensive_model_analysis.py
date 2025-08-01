import json
import pickle
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_all_model_results():
    """Load results from all models"""
    results = {}

    # Load baseline results from CSV
    print(" Loading baseline model results...")
    try:
        baseline_df = pd.read_csv("results/baseline_results_uit-vsfc.csv")
        for _, row in baseline_df.iterrows():
            model_name = row["model_name"]
            if model_name not in ["optimized_lstm", "phobert"]:  # Skip duplicates
                results[model_name] = {
                    "accuracy": row["accuracy"],
                    "f1_score": row["f1_score"],
                    "model_type": "baseline",
                    "classification_report": literal_eval(row["classification_report"]),
                }
                print(
                    f" {model_name}: Accuracy = {row['accuracy']:.4f}, F1 = {row['f1_score']:.4f}"
                )
    except Exception as e:
        print(f"Error loading baseline results: {e}")

    # Load LSTM results
    print("\n🔥 Loading LSTM results...")
    try:
        with open("results/lstm_results.pkl", "rb") as f:
            lstm_results = pickle.load(f)

        if "test_results" in lstm_results:
            test_results = lstm_results["test_results"]
            results["LSTM"] = {
                "accuracy": test_results["accuracy"],
                "f1_score": test_results["f1_score"],
                "model_type": "deep_learning",
                "training_history": lstm_results.get("training_history", {}),
                "classification_report": test_results["classification_report"],
            }
            print(
                f" LSTM: Accuracy = {test_results['accuracy']:.4f}, F1 = {test_results['f1_score']:.4f}"
            )
    except Exception as e:
        print(f"Error loading LSTM results: {e}")

    # Load PhoBERT results
    print("\n🚀 Loading PhoBERT results...")
    try:
        with open("results/phobert_results_20250716_172440.pkl", "rb") as f:
            phobert_results = pickle.load(f)

        results["PhoBERT"] = {
            "accuracy": float(phobert_results["accuracy"]),
            "f1_score": float(phobert_results["f1_score"]),
            "model_type": "transformer",
            "training_history": phobert_results.get("training_history", {}),
            "classification_report": phobert_results["classification_report"],
        }
        print(
            f" PhoBERT: Accuracy = {phobert_results['accuracy']:.4f}, F1 = {phobert_results['f1_score']:.4f}"
        )
    except Exception as e:
        print(f"Error loading PhoBERT results: {e}")

    return results


def plot_model_performance_comparison(results):
    """Plot comprehensive model performance comparison"""
    plt.figure(figsize=(16, 12))

    # Prepare data
    models = list(results.keys())
    accuracies = [results[model]["accuracy"] for model in models]
    f1_scores = [results[model]["f1_score"] for model in models]

    # Color mapping by model type
    color_map = {
        "baseline": "#3498db",  # Blue
        "deep_learning": "#e74c3c",  # Red
        "transformer": "#2ecc71",  # Green
    }
    colors = [color_map[results[model]["model_type"]] for model in models]

    # 1. Accuracy Comparison
    plt.subplot(2, 3, 1)
    bars1 = plt.bar(models, accuracies, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.75, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. F1-Score Comparison
    plt.subplot(2, 3, 2)
    bars2 = plt.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    plt.title("Model F1-Score Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.75, 1.0)

    # Add value labels on bars
    for bar, f1 in zip(bars2, f1_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{f1:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Accuracy vs F1-Score Scatter Plot
    plt.subplot(2, 3, 3)
    for i, model in enumerate(models):
        plt.scatter(
            accuracies[i],
            f1_scores[i],
            color=colors[i],
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )
        plt.annotate(
            model,
            (accuracies[i], f1_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel("Accuracy")
    plt.ylabel("F1-Score")
    plt.title("Accuracy vs F1-Score", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # 4. Model Type Performance
    plt.subplot(2, 3, 4)
    model_types = ["baseline", "deep_learning", "transformer"]
    type_labels = ["Traditional ML", "Deep Learning", "Transformer"]

    type_accuracies = []
    type_f1s = []

    for model_type in model_types:
        type_models = [m for m in models if results[m]["model_type"] == model_type]
        if type_models:
            avg_acc = np.mean([results[m]["accuracy"] for m in type_models])
            avg_f1 = np.mean([results[m]["f1_score"] for m in type_models])
            type_accuracies.append(avg_acc)
            type_f1s.append(avg_f1)
        else:
            type_accuracies.append(0)
            type_f1s.append(0)

    x = np.arange(len(type_labels))
    width = 0.35

    bars3 = plt.bar(
        x - width / 2,
        type_accuracies,
        width,
        label="Accuracy",
        color="lightblue",
        alpha=0.7,
        edgecolor="black",
    )
    bars4 = plt.bar(
        x + width / 2,
        type_f1s,
        width,
        label="F1-Score",
        color="lightcoral",
        alpha=0.7,
        edgecolor="black",
    )

    plt.xlabel("Model Type")
    plt.ylabel("Score")
    plt.title("Average Performance by Model Type", fontsize=14, fontweight="bold")
    plt.xticks(x, type_labels)
    plt.legend()
    plt.ylim(0.75, 1.0)

    # Add value labels
    for bar, val in zip(bars3, type_accuracies):
        if val > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    for bar, val in zip(bars4, type_f1s):
        if val > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 5. Performance Ranking
    plt.subplot(2, 3, 5)

    # Sort models by accuracy
    sorted_models = sorted(models, key=lambda x: results[x]["accuracy"], reverse=True)
    sorted_accuracies = [results[model]["accuracy"] for model in sorted_models]
    sorted_colors = [color_map[results[model]["model_type"]] for model in sorted_models]

    bars5 = plt.barh(
        sorted_models, sorted_accuracies, color=sorted_colors, alpha=0.7, edgecolor="black"
    )
    plt.title("Model Ranking by Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Accuracy")

    # Add value labels
    for bar, acc in zip(bars5, sorted_accuracies):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # 6. Legend and Model Type Distribution
    plt.subplot(2, 3, 6)
    type_counts = {}
    for model in models:
        model_type = results[model]["model_type"]
        type_counts[model_type] = type_counts.get(model_type, 0) + 1

    labels = [f"{k.replace('_', ' ').title()}\n({v} models)" for k, v in type_counts.items()]
    colors_pie = [color_map[k] for k in type_counts.keys()]

    plt.pie(
        type_counts.values(), labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90
    )
    plt.title("Model Distribution by Type", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/comprehensive_model_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_detailed_performance_table(results):
    """Create detailed performance table"""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Sort models by accuracy
    sorted_models = sorted(results.keys(), key=lambda x: results[x]["accuracy"], reverse=True)

    print(
        f"{'Rank':<4} {'Model':<20} {'Type':<15} {'Accuracy':<10} {'F1-Score':<10} {'Performance'}"
    )
    print("-" * 80)

    for i, model in enumerate(sorted_models, 1):
        result = results[model]
        performance = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""

        print(
            f"{i:<4} {model:<20} {result['model_type']:<15} "
            f"{result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {performance}"
        )

    # Best performing model analysis
    best_model = sorted_models[0]
    best_result = results[best_model]

    print(f"\nBEST PERFORMING MODEL: {best_model}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   F1-Score: {best_result['f1_score']:.4f}")
    print(f"    Type: {best_result['model_type'].replace('_', ' ').title()}")

    # Model type analysis
    print(f"\n ANALYSIS BY MODEL TYPE:")
    for model_type in ["baseline", "deep_learning", "transformer"]:
        type_models = [m for m in sorted_models if results[m]["model_type"] == model_type]
        if type_models:
            avg_acc = np.mean([results[m]["accuracy"] for m in type_models])
            avg_f1 = np.mean([results[m]["f1_score"] for m in type_models])
            print(
                f"   {model_type.replace('_', ' ').title():<15}: "
                f"Avg Accuracy = {avg_acc:.4f}, Avg F1 = {avg_f1:.4f}"
            )


def plot_training_progress_comparison(results):
    """Plot training progress for models that have training history"""
    models_with_history = {
        k: v for k, v in results.items() if "training_history" in v and v["training_history"]
    }

    if not models_with_history:
        print("No training history found for comparison")
        return

    plt.figure(figsize=(15, 8))

    # Plot validation accuracy over epochs
    plt.subplot(1, 2, 1)
    for model_name, model_data in models_with_history.items():
        history = model_data["training_history"]
        if "val_accuracies" in history or "val_accuracy" in history:
            val_acc = history.get("val_accuracies", history.get("val_accuracy", []))
            if val_acc:
                epochs = range(1, len(val_acc) + 1)
                color = "#e74c3c" if "LSTM" in model_name else "#2ecc71"
                marker = "o" if "LSTM" in model_name else "s"
                plt.plot(epochs, val_acc, label=model_name, linewidth=2, marker=marker, color=color)

    plt.title("Validation Accuracy Training Progress", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot validation loss over epochs
    plt.subplot(1, 2, 2)
    for model_name, model_data in models_with_history.items():
        history = model_data["training_history"]
        if "val_losses" in history or "val_loss" in history:
            val_loss = history.get("val_losses", history.get("val_loss", []))
            if val_loss:
                epochs = range(1, len(val_loss) + 1)
                color = "#e74c3c" if "LSTM" in model_name else "#2ecc71"
                marker = "o" if "LSTM" in model_name else "s"
                plt.plot(
                    epochs, val_loss, label=model_name, linewidth=2, marker=marker, color=color
                )

    plt.title("Validation Loss Training Progress", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_progress_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_class_performance(results):
    """Analyze per-class performance across all models"""
    plt.figure(figsize=(15, 10))

    classes = ["negative", "neutral", "positive"]
    metrics = ["precision", "recall", "f1-score"]

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        model_names = []
        class_scores = {cls: [] for cls in classes}

        for model_name, model_data in results.items():
            if "classification_report" in model_data:
                report = model_data["classification_report"]
                if isinstance(report, str):
                    try:
                        report = json.loads(report.replace("'", '"'))
                    except:
                        continue

                model_names.append(model_name)
                for cls in classes:
                    if cls in report:
                        class_scores[cls].append(report[cls][metric])
                    else:
                        class_scores[cls].append(0)

        x = np.arange(len(model_names))
        width = 0.25

        colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # Red, Orange, Green

        for j, cls in enumerate(classes):
            plt.bar(
                x + j * width,
                class_scores[cls],
                width,
                label=cls.title(),
                color=colors[j],
                alpha=0.7,
            )

        plt.xlabel("Models")
        plt.ylabel(metric.title())
        plt.title(f"{metric.title()} by Class", fontweight="bold")
        plt.xticks(x + width, model_names, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Overall F1-score comparison
    plt.subplot(2, 2, 4)
    model_f1s = [results[model]["f1_score"] for model in model_names]
    colors_f1 = ["#3498db", "#3498db", "#3498db", "#e74c3c", "#2ecc71"][: len(model_names)]

    bars = plt.bar(model_names, model_f1s, color=colors_f1, alpha=0.7, edgecolor="black")
    plt.xlabel("Models")
    plt.ylabel("Overall F1-Score")
    plt.title("Overall F1-Score Comparison", fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for bar, f1 in zip(bars, model_f1s):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{f1:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/class_performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main analysis function"""
    print(" COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 50)

    # Load all model results
    results = load_all_model_results()

    if not results:
        print(" No model results found!")
        return

    print(f"\n Loaded results from {len(results)} models")

    # Create detailed performance table
    create_detailed_performance_table(results)

    print(f"\n Creating comprehensive visualizations...")

    # Plot comprehensive performance comparison
    plot_model_performance_comparison(results)

    # Plot training progress comparison (for models with training history)
    plot_training_progress_comparison(results)

    # Analyze class-wise performance
    analyze_class_performance(results)

    print(f"\n Analysis complete! Generated files:")
    print("   - results/comprehensive_model_analysis.png")
    print("   - results/training_progress_comparison.png")
    print("   - results/class_performance_analysis.png")


if __name__ == "__main__":
    main()
