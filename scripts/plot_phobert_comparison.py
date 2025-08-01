import pickle
import os
import matplotlib.pyplot as plt

# Đường dẫn tới các file kết quả
baseline_path = 'results/phobert_baseline_results.pkl'
finetuned_path = 'results/phobert_results_20250716_172440.pkl'
output_path = 'results/comparison_phobert.png'

# Hàm load kết quả từ file pkl
def load_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load kết quả
baseline_data = load_results(baseline_path)
finetuned_data = load_results(finetuned_path)

# Lấy metric từ đúng key
baseline_metrics_dict = baseline_data['baseline_results']
finetuned_metrics_dict = finetuned_data

# Các metric cần so sánh
metrics = ['accuracy', 'f1_score']
baseline_metrics = []
finetuned_metrics = []

for metric in metrics:
    if metric not in baseline_metrics_dict or metric not in finetuned_metrics_dict:
        raise KeyError(f"Metric '{metric}' không có trong kết quả. Hãy kiểm tra lại file pkl.")
    baseline_metrics.append(baseline_metrics_dict[metric])
    finetuned_metrics.append(finetuned_metrics_dict[metric])

# Plot
plt.figure(figsize=(6, 5))
x = range(len(metrics))
width = 0.35
bars1 = plt.bar([i - width/2 for i in x], baseline_metrics, width, label='PhoBERT Baseline')
bars2 = plt.bar([i + width/2 for i in x], finetuned_metrics, width, label='PhoBERT Fine-tuned')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('So sánh kết quả PhoBERT Baseline vs Fine-tuned')
plt.legend(loc='lower right')
plt.tight_layout()

# Thêm số liệu lên trên mỗi cột
for bar in bars1:
    height = bar.get_height()
    plt.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    plt.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

plt.savefig(output_path)
print(f"Đã lưu biểu đồ so sánh vào {output_path}") 