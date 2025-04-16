"""
ICU患者急性肾损伤(AKI)预测主程序

该程序整合了数据处理、模型训练和模型评估模块，实现完整的AKI预测流程：
1. 数据加载和预处理
2. 特征工程
3. 训练基线模型 (逻辑回归、随机森林、梯度提升树、XGBoost)
4. 训练深度学习模型 (LSTM+Attention)
5. 模型评估和比较
6. 结果可视化

使用方法: python main.py
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入自定义模块
from src.data_processor import process_data
from src.model_trainer import BaselineModels, LSTMWithAttention, prepare_time_series_data
from src.model_evaluator import ModelEvaluator


# 配置
class Config:
    # 数据路径
    RAW_DATA_PATH = "data/raw/base_stay.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
    
    # 输出路径
    MODEL_OUTPUT_DIR = "output/models"
    REPORT_OUTPUT_DIR = "output/reports"
    PLOT_OUTPUT_DIR = "output/plots"
    
    # 模型参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15  # 验证集大小 (占训练集的比例)
    
    # 深度学习模型参数
    DL_EPOCHS = 10  # 减少epoch数以加快训练
    DL_BATCH_SIZE = 32
    DL_PATIENCE = 3  # 减少早停patience以加快训练
    DL_LEARNING_RATE = 0.001
    LSTM_UNITS = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.3
    
    # 时序数据参数 (若使用时序模型)
    TIME_STEPS = 48  # 48小时的数据
    
    # 是否使用时序模型
    USE_TIME_SERIES = False  # 设置为True启用时序模型
    
    # 是否训练基线模型
    TRAIN_BASELINE = True
    
    # 是否训练深度学习模型
    TRAIN_DL = True  # 关闭深度学习模型


def setup_environment():
    """
    设置环境，创建必要的目录
    """
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.PLOT_OUTPUT_DIR, exist_ok=True)
    
    # 确保processed目录存在
    processed_dir = os.path.dirname(Config.PROCESSED_DATA_PATH)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("环境设置完成，所有必要的目录已创建")


def run_baseline_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    训练和评估基线模型
    
    参数:
        X_train, X_test, y_train, y_test: 训练和测试数据
        preprocessor: 预处理流水线
        
    返回:
        best_model_name: 最佳模型名称
        best_model: 最佳模型
        all_model_probas: 所有模型的预测概率
    """
    print("\n" + "="*50)
    print("训练和评估基线模型")
    print("="*50)
    
    # 初始化基线模型
    baseline = BaselineModels(random_state=Config.RANDOM_STATE)
    
    # 训练模型
    baseline.train(X_train, y_train, preprocessor)
    
    # 获取原始特征名称（用于特征重要性分析）
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = None
        print("警告: 无法获取特征名称，特征重要性分析可能使用索引作为特征名称")
    
    # 评估模型，并计算特征重要性
    results = baseline.evaluate(
        X_test, y_test, 
        feature_names=feature_names,
        output_dir=os.path.join(Config.REPORT_OUTPUT_DIR, "feature_importance")
    )
    
    # 保存模型
    baseline.save_models(Config.MODEL_OUTPUT_DIR)
    
    # 获取所有模型的预测概率
    all_model_probas = {}
    model_names = []
    model_probas = []
    
    for name, model in baseline.trained_models.items():
        all_model_probas[name] = model.predict_proba(X_test)[:, 1]
        model_names.append(name)
        model_probas.append(all_model_probas[name])
    
    # 生成每个模型的评估报告
    for name, proba in all_model_probas.items():
        print(f"\n生成 {name} 模型的评估报告")
        evaluator = ModelEvaluator(y_test, proba)
        evaluator.generate_report(
            os.path.join(Config.REPORT_OUTPUT_DIR, name),
            prefix=f"{name}_"
        )
    
    return baseline.best_model_name, baseline.best_model, all_model_probas


def run_deep_learning_model(X_train, X_test, y_train, y_test, preprocessor=None, is_time_series=False):
    """
    训练和评估深度学习模型
    
    参数:
        X_train, X_test, y_train, y_test: 训练和测试数据
        preprocessor: 预处理流水线
        is_time_series: 是否使用时序数据
        
    返回:
        dl_model: 训练好的深度学习模型
        y_pred_proba: 预测概率
    """
    print("\n" + "="*50)
    print("训练和评估深度学习模型")
    print("="*50)
    
    # 分割验证集
    X_train_dl, X_val, y_train_dl, y_val = train_test_split(
        X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # 如果需要，应用预处理器
    if preprocessor and not is_time_series:
        X_train_processed = preprocessor.transform(X_train_dl)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # 获取处理后的特征数量
        n_features = X_train_processed.shape[1]
        
        # 初始化深度学习模型
        dl_model = LSTMWithAttention(input_shape=n_features)
    elif is_time_series:
        # 使用时序数据
        X_train_processed = X_train_dl
        X_val_processed = X_val
        X_test_processed = X_test
        
        time_steps = X_train_processed.shape[1]
        n_features = X_train_processed.shape[2]
        
        # 初始化时序深度学习模型
        dl_model = LSTMWithAttention()
        dl_model.build_model(
            lstm_units=Config.LSTM_UNITS,
            dense_units=Config.DENSE_UNITS,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.DL_LEARNING_RATE,
            time_steps=time_steps,
            features=n_features
        )
    else:
        # 直接使用原始特征
        X_train_processed = X_train_dl
        X_val_processed = X_val
        X_test_processed = X_test
        
        # 初始化深度学习模型
        dl_model = LSTMWithAttention(input_shape=X_train_processed.shape[1])
        dl_model.build_model(
            dense_units=Config.DENSE_UNITS,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.DL_LEARNING_RATE
        )
    
    # 检查模型是否成功构建
    if dl_model.model is None:
        print("深度学习模型构建失败，跳过训练")
        return None, None
    
    # 训练模型
    model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "dl_model.h5")
    dl_model.train(
        X_train_processed, y_train_dl,
        X_val_processed, y_val,
        batch_size=Config.DL_BATCH_SIZE,
        epochs=Config.DL_EPOCHS,
        patience=Config.DL_PATIENCE,
        model_path=model_path
    )
    
    # 评估模型
    dl_results = dl_model.evaluate(X_test_processed, y_test)
    
    # 保存训练历史图表
    history_plot_path = os.path.join(Config.PLOT_OUTPUT_DIR, "dl_training_history.png")
    dl_model.plot_training_history(save_path=history_plot_path)
    
    # 预测概率
    y_pred_proba = dl_model.model.predict(X_test_processed).flatten()
    
    # 生成评估报告
    print("\n生成深度学习模型的评估报告")
    dl_evaluator = ModelEvaluator(y_test, y_pred_proba)
    dl_evaluator.generate_report(
        os.path.join(Config.REPORT_OUTPUT_DIR, "dl_model"),
        prefix="dl_"
    )
    
    return dl_model, y_pred_proba


def compare_all_models(y_test, model_probas, model_names):
    """
    比较所有模型的性能
    
    参数:
        y_test: 测试标签
        model_probas: 不同模型的预测概率列表
        model_names: 模型名称列表
    """
    print("\n" + "="*50)
    print("比较所有模型性能")
    print("="*50)
    
    # 选择第一个模型作为参考
    reference_proba = model_probas[0]
    reference_name = model_names[0]
    
    # 初始化评估器
    evaluator = ModelEvaluator(y_test, reference_proba)
    
    # 比较模型
    other_probas = model_probas[1:]
    other_names = model_names[1:]
    
    comparison_path = os.path.join(Config.PLOT_OUTPUT_DIR, "model_comparison.png")
    evaluator.compare_models(other_probas, other_names, save_path=comparison_path)
    
    # 计算每个模型的指标
    metrics = []
    
    for name, proba in zip(model_names, model_probas):
        model_evaluator = ModelEvaluator(y_test, proba)
        metrics.append({
            'model_name': name,
            'auroc': model_evaluator.auroc,
            'auprc': model_evaluator.auprc,
            'f1': model_evaluator.f1,
            'optimal_threshold_f1': model_evaluator.find_optimal_threshold('f1')
        })
    
    # 创建比较表格
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values('auroc', ascending=False)
    
    # 保存比较表格
    comparison_csv_path = os.path.join(Config.REPORT_OUTPUT_DIR, "model_comparison.csv")
    metrics_df.to_csv(comparison_csv_path, index=False)
    print(f"模型比较结果已保存至 {comparison_csv_path}")
    
    # 打印比较结果
    print("\n模型性能比较:")
    pd.set_option('display.precision', 4)
    print(metrics_df)
    
    # 找出最佳模型
    best_model_idx = metrics_df['auroc'].idxmax()
    best_model_name = metrics_df.loc[best_model_idx, 'model_name']
    best_auroc = metrics_df.loc[best_model_idx, 'auroc']
    
    print(f"\n最佳模型是 {best_model_name}，AUROC = {best_auroc:.4f}")


def main():
    """主函数，执行完整工作流程"""
    print("="*50)
    print("ICU患者急性肾损伤(AKI)预测系统")
    print("="*50)
    
    # 记录开始时间
    start_time = time.time()
    
    # 设置环境
    setup_environment()
    
    # 处理数据
    print("\n1. 数据处理")
    X_train, X_test, y_train, y_test, preprocessor = process_data(
        Config.RAW_DATA_PATH,
        Config.PROCESSED_DATA_PATH
    )
    
    # 保存特征重要性的可视化
    model_probas = []
    model_names = []
    
    # 训练基线模型
    if Config.TRAIN_BASELINE:
        print("\n2. 训练基线模型")
        best_model_name, best_model, all_model_probas = run_baseline_models(
            X_train, X_test, y_train, y_test, preprocessor
        )
        
        # 收集所有模型的预测结果用于后续比较
        for name, proba in all_model_probas.items():
            model_probas.append(proba)
            model_names.append(name)
    
    # 训练深度学习模型
    if Config.TRAIN_DL:
        print("\n3. 训练深度学习模型")
        
        if Config.USE_TIME_SERIES:
            # 准备时序数据（示例中使用模拟数据）
            print("准备时序数据...")
            # 选择用于时序模型的特征
            processed_df = pd.read_csv(Config.PROCESSED_DATA_PATH)
            numerical_features = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # 排除一些不适合时序的特征
            exclude_cols = ['subject_id', 'hadm_id', 'icustay_id', 'aki_label']
            ts_features = [col for col in numerical_features if col not in exclude_cols]
            
            # 生成时序数据
            X_ts, y_ts = prepare_time_series_data(
                processed_df, ts_features, 
                time_steps=Config.TIME_STEPS
            )
            
            # 分割训练集和测试集
            X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
                X_ts, y_ts, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            
            # 训练时序深度学习模型
            dl_model, dl_proba = run_deep_learning_model(
                X_train_ts, X_test_ts, y_train_ts, y_test_ts, 
                is_time_series=True
            )
        else:
            # 使用非时序数据
            dl_model, dl_proba = run_deep_learning_model(
                X_train, X_test, y_train, y_test, preprocessor
            )
        
        # 添加深度学习模型结果到比较列表
        if dl_proba is not None:
            model_probas.append(dl_proba)
            model_names.append('Deep Learning')
    
    # 比较所有模型
    if len(model_probas) > 1:
        print("\n4. 比较所有模型")
        compare_all_models(y_test, model_probas, model_names)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    
    print("\n=" * 50)
    print("ICU患者急性肾损伤预测系统执行完毕")
    print("=" * 50)


if __name__ == "__main__":
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
