"""
AKI预测模型训练脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

# 导入自定义模块
from src.data_processor_dl import prepare_time_series_data
from src.dl_models import DeepLearningModels

# 设置随机种子
np.random.seed(42)

# 创建输出目录
output_dir = r"d:\Programfile\AIIntellligence\models\aki_prediction"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
print("加载数据...")
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
from sklearn.ensemble import VotingClassifier
import joblib

# 获取各模型的预测概率
y_pred_lstm = dl_models.predict(X_test).flatten()
dl_models.model = transformer_model
y_pred_transformer = dl_models.predict(X_test).flatten()
dl_models.model = cnn_lstm_model
y_pred_cnn_lstm = dl_models.predict(X_test).flatten()

# 简单平均集成
y_pred_ensemble = (y_pred_lstm + y_pred_transformer + y_pred_cnn_lstm) / 3
ensemble_auc = roc_auc_score(y_test, y_pred_ensemble)
print(f"集成模型AUC: {ensemble_auc:.4f}")

# 保存预测结果
ensemble_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred_lstm': y_pred_lstm,
    'y_pred_transformer': y_pred_transformer,
    'y_pred_cnn_lstm': y_pred_cnn_lstm,
    'y_pred_ensemble': y_pred_ensemble
})
ensemble_results.to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)