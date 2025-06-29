"""
AKI预测模型训练脚本
"""

import os

import numpy as np
import pandas as pd

# 导入自定义模块
from src.data_processor_dl import prepare_time_series_data
from src.dl_models import DeepLearningModels

# 设置随机种子
np.random.seed(42)

# 创建输出目录
output_dir = r"d:\Programfile\AIIntellligence\models\aki_prediction"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data_path = r"d:\Programfile\AIIntellligence\data\raw\base_stay_dl.csv"
df = pd.read_csv(data_path)

# 数据探索
print(f"数据集形状: {df.shape}")
print(f"唯一患者数量: {df['subject_id'].nunique()}")
print(f"AKI患者比例: {df[df['aki_label']==1]['subject_id'].nunique() / df['subject_id'].nunique():.2f}")

# 准备时序数据
# 准备时序数据时增加数据预处理选项
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_time_series_data(
    df, 
    time_col='hours_since_admission',
    measurement_col='measurement',
    value_col='valuenum',
    patient_id_col='subject_id',
    target_col='aki_label',
    # 添加以下参数
    normalize_method='robust',  # 使用稳健缩放，对异常值不敏感
    handle_missing='forward_fill_mean',  # 先前向填充再均值填充
    time_window=True,  # 添加时间窗口特征
    diff_features=True  # 添加差分特征
)


# 创建深度学习模型
dl_models = DeepLearningModels()

# 构建LSTM+Attention模型
input_shape = (X_train.shape[1], X_train.shape[2])
# 构建改进的LSTM+Attention模型
model = dl_models.build_lstm_attention(
    input_shape=input_shape,
    lstm_units=128,  # 增加单元数
    attention_heads=8,  # 增加注意力头数
    dense_units=64,  # 增加全连接层单元数
    dropout_rate=0.4,  # 增加dropout比例
    learning_rate=0.0005,  # 降低学习率
    l2_reg=0.001  # 添加L2正则化
)

# 训练模型
model_path = os.path.join(output_dir, "lstm_attention_model.keras")
history = dl_models.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=16,  # 减小批量大小
    epochs=50,
    patience=10,
    model_path=model_path,
    class_weight={0: 1, 1: 3},  # 添加类别权重处理不平衡
    use_focal_loss=True,  # 使用Focal Loss
    use_lr_scheduler=True  # 使用学习率调度器
)

# 评估模型
metrics = dl_models.evaluate(X_test, y_test)

# 绘制训练历史
history_path = os.path.join(output_dir, "training_history.png")
dl_models.plot_training_history(save_path=history_path)

# 尝试其他模型
print("\n尝试Transformer模型...")
transformer_model = dl_models.build_transformer(
    input_shape=input_shape,
    num_layers=2,
    d_model=64,
    num_heads=4
)

transformer_path = os.path.join(output_dir, "transformer_model.keras")
dl_models.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50,
    patience=10,
    model_path=transformer_path
)

transformer_metrics = dl_models.evaluate(X_test, y_test)

# 尝试CNN+LSTM模型
print("\n尝试CNN+LSTM模型...")
cnn_lstm_model = dl_models.build_cnn_lstm(
    input_shape=input_shape,
    filters=64,
    kernel_size=3,
    lstm_units=64
)

cnn_lstm_path = os.path.join(output_dir, "cnn_lstm_model.keras")
dl_models.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50,
    patience=10,
    model_path=cnn_lstm_path
)

cnn_lstm_metrics = dl_models.evaluate(X_test, y_test)

# 比较模型性能
print("\n模型性能比较:")
models = ["LSTM+Attention", "Transformer", "CNN+LSTM"]
metrics_list = [metrics, transformer_metrics, cnn_lstm_metrics]

for model_name, model_metrics in zip(models, metrics_list):
    print(f"{model_name}:")
    for metric_name, value in model_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

print("\n训练完成！模型已保存到:", output_dir)

# 在训练完所有模型后添加集成学习
print("\n构建集成模型...")

# 获取各模型的预测概率
y_pred_lstm = dl_models.predict(X_test).flatten()
dl_models.model = transformer_model
y_pred_transformer = dl_models.predict(X_test).flatten()
dl_models.model = cnn_lstm_model
y_pred_cnn_lstm = dl_models.predict(X_test).flatten()

# 保存预测结果
ensemble_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred_lstm': y_pred_lstm,
    'y_pred_transformer': y_pred_transformer,
    'y_pred_cnn_lstm': y_pred_cnn_lstm,
})
ensemble_results.to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)

# 添加可视化功能
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. 模型性能对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Deep Learning Models Performance Comparison', fontsize=16, fontweight='bold')

# 提取指标数据
metric_names = ['auc', 'precision', 'recall', 'f1']
metric_labels = ['AUC', 'Precision', 'Recall', 'F1 Score']
model_names = ['LSTM+Attention', 'Transformer', 'CNN+LSTM']

for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    row = idx // 2
    col = idx % 2
    
    values = [metrics_list[i][metric] for i in range(len(models))]
    
    bars = axes[row, col].bar(model_names, values, alpha=0.8)
    axes[row, col].set_title(f'{label} Comparison', fontweight='bold')
    axes[row, col].set_ylabel(label)
    axes[row, col].set_ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 旋转x轴标签
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. ROC曲线对比
plt.figure(figsize=(10, 8))

# 计算每个模型的ROC曲线
colors = ['blue', 'red', 'green']
predictions = [y_pred_lstm, y_pred_transformer, y_pred_cnn_lstm]

for i, (pred, model_name, color) in enumerate(zip(predictions, model_names, colors)):
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Precision-Recall曲线对比
plt.figure(figsize=(10, 8))

for i, (pred, model_name, color) in enumerate(zip(predictions, model_names, colors)):
    precision, recall, _ = precision_recall_curve(y_test, pred)
    avg_precision = metrics_list[i]['precision']
    plt.plot(recall, precision, color=color, lw=2,
             label=f'{model_name} (AP = {avg_precision:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'pr_curves_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. 模型预测分布对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Prediction Distribution Comparison', fontsize=16, fontweight='bold')

for i, (pred, model_name) in enumerate(zip(predictions, model_names)):
    axes[i].hist(pred[y_test == 0], bins=30, alpha=0.7, label='Non-AKI', color='lightblue', density=True)
    axes[i].hist(pred[y_test == 1], bins=30, alpha=0.7, label='AKI', color='lightcoral', density=True)
    axes[i].set_title(f'{model_name}', fontweight='bold')
    axes[i].set_xlabel('Prediction Probability', fontweight='bold')
    axes[i].set_ylabel('Density', fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. 创建综合性能雷达图
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 设置雷达图的角度
angles = [n / float(len(metric_labels)) * 2 * pi for n in range(len(metric_labels))]
angles += angles[:1]  # 闭合图形

# 为每个模型绘制雷达图
for i, (model_name, color) in enumerate(zip(model_names, colors)):
    values = [metrics_list[i][metric] for metric in metric_names]
    values += values[:1]  # 闭合图形
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.savefig(os.path.join(output_dir, 'performance_radar_chart.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n可视化图表已保存到: {output_dir}")
print("生成的图表包括:")
print("1. model_performance_comparison.png - 模型性能对比柱状图")
print("2. roc_curves_comparison.png - ROC曲线对比")
print("3. pr_curves_comparison.png - Precision-Recall曲线对比")
print("4. prediction_distribution.png - 预测分布对比")
print("5. performance_radar_chart.png - 性能雷达图")