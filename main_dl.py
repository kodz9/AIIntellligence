"""
ICU患者急性肾损伤(AKI)预测 - 深度学习方法

该程序使用深度学习模型预测ICU患者的AKI风险：
1. 数据加载和预处理
2. 特征工程
3. 训练深度学习模型 (LSTM+Attention, Transformer, CNN+LSTM)
4. 模型评估和比较
5. 结果可视化

使用方法: python main_dl.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 导入自定义模块
from src.data_processor import load_data, verify_aki_label, handle_missing_values, feature_engineering
from src.data_processor_dl import prepare_dl_data, prepare_time_series_data, handle_class_imbalance
from src.dl_models import DeepLearningModels
from src.model_evaluator import ModelEvaluator
from src.model_explainer import ModelExplainer
from src.model_ensemble import ModelEnsemble


# 配置
# 在Config类中添加MODEL_COMPARISON_PATH属性
class Config:
    # 数据路径
    RAW_DATA_PATH = "data/raw/base_stay.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
    
    # 输出路径
    MODEL_OUTPUT_DIR = "output/models/dl"
    REPORT_OUTPUT_DIR = "output/reports/dl"
    PLOT_OUTPUT_DIR = "output/plots/dl"
    MODEL_COMPARISON_PATH = "output/reports/dl/model_comparison.csv"  # 添加这一行
    
    # 模型参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15  # 验证集大小 (占训练集的比例)
    
    # 深度学习模型参数
    DL_EPOCHS = 50
    DL_BATCH_SIZE = 32
    DL_PATIENCE = 10
    DL_LEARNING_RATE = 0.001
    
    # LSTM模型参数
    LSTM_UNITS = 64
    ATTENTION_HEADS = 4
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.3
    
    # Transformer模型参数
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_DIM = 64
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DFF = 128
    
    # CNN+LSTM模型参数
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    
    # 时序数据参数
    TIME_STEPS = 48  # 48小时


# 类别不平衡处理
    HANDLE_IMBALANCE = True
    IMBALANCE_METHOD = 'smote'  # 'smote', 'adasyn', 'random_over', 'random_under'
    
    # 是否使用时序数据
    USE_TIME_SERIES = False
    
    # 要训练的模型
    TRAIN_LSTM_ATTENTION = True
    TRAIN_TRANSFORMER = True
    TRAIN_CNN_LSTM = True


def setup_environment():
    """设置环境，创建必要的目录"""
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.PLOT_OUTPUT_DIR, exist_ok=True)
    
    # 确保processed目录存在
    processed_dir = os.path.dirname(Config.PROCESSED_DATA_PATH)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 设置TensorFlow内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个GPU，已设置内存增长")
        except RuntimeError as e:
            print(f"设置GPU内存增长时出错: {e}")
    else:
        print("未找到GPU，将使用CPU进行训练")
    
    print("环境设置完成，所有必要的目录已创建")


def load_and_process_data():
    """加载和处理数据"""
    print("\n" + "="*50)
    print("加载和处理数据")
    print("="*50)
    
    # 检查是否已有处理好的数据
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"加载已处理的数据: {Config.PROCESSED_DATA_PATH}")
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    else:
        print(f"从原始数据开始处理: {Config.RAW_DATA_PATH}")
        # 加载原始数据
        df = load_data(Config.RAW_DATA_PATH)
        
        # 验证AKI标签
        df = verify_aki_label(df)
        
        # 处理缺失值
        df = handle_missing_values(df)
        
        # 特征工程
        df = feature_engineering(df)
        
        # 保存处理后的数据
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
        print(f"处理后的数据已保存至: {Config.PROCESSED_DATA_PATH}")
    
    print(f"数据形状: {df.shape}")
    print(f"AKI发生率: {df['aki_label'].mean():.4f}")
    
    return df


def train_and_evaluate_models(df):
    """训练和评估深度学习模型"""
    print("\n" + "="*50)
    print("训练和评估深度学习模型")
    print("="*50)
    
    # 准备数据
    if Config.USE_TIME_SERIES:
        print("准备时序数据...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_time_series_data(
            df, 
            time_steps=Config.TIME_STEPS,
            target_col='aki_label',
            test_size=Config.TEST_SIZE,
            val_size=Config.VAL_SIZE,
            random_state=Config.RANDOM_STATE
        )
        input_shape = (Config.TIME_STEPS, X_train.shape[2])
    else:
        print("准备非时序数据...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_dl_data(
            df,
            target_col='aki_label',
            test_size=Config.TEST_SIZE,
            val_size=Config.VAL_SIZE,
            random_state=Config.RANDOM_STATE
        )
        input_shape = (X_train.shape[1],)
    
    # 处理类别不平衡
    if Config.HANDLE_IMBALANCE:
        print(f"处理类别不平衡 (方法: {Config.IMBALANCE_METHOD})...")
        X_train, y_train = handle_class_imbalance(
            X_train, y_train, 
            method=Config.IMBALANCE_METHOD,
            random_state=Config.RANDOM_STATE
        )
    
    # 创建模型评估结果字典
    model_results = {}
    
    # 训练LSTM+Attention模型
    if Config.TRAIN_LSTM_ATTENTION:
        print("\n" + "-"*40)
        print("训练LSTM+Attention模型")
        print("-"*40)
        
        dl_model = DeepLearningModels()
        dl_model.build_lstm_attention(
            input_shape=input_shape,
            lstm_units=Config.LSTM_UNITS,
            attention_heads=Config.ATTENTION_HEADS,
            dense_units=Config.DENSE_UNITS,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.DL_LEARNING_RATE
        )
        
        # 训练模型
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "lstm_attention_model")
        history = dl_model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=Config.DL_BATCH_SIZE,
            epochs=Config.DL_EPOCHS,
            patience=Config.DL_PATIENCE,
            model_path=model_path
        )
        
        # 评估模型
        metrics = dl_model.evaluate(X_test, y_test)
        model_results['lstm_attention'] = metrics
        
        # 绘制训练历史
        history_plot_path = os.path.join(Config.PLOT_OUTPUT_DIR, "lstm_attention_history.png")
        dl_model.plot_training_history(save_path=history_plot_path)
        
        # 获取预测概率
        y_pred_proba = dl_model.predict(X_test).flatten()
        
        # 生成评估报告
        evaluator = ModelEvaluator(y_test, y_pred_proba)
        evaluator.print_metrics()
        
        # 绘制ROC曲线
        roc_path = os.path.join(Config.PLOT_OUTPUT_DIR, "lstm_attention_roc.png")
        evaluator.plot_roc_curve(save_path=roc_path)
        
        # 绘制PR曲线
        pr_path = os.path.join(Config.PLOT_OUTPUT_DIR, "lstm_attention_pr.png")
        evaluator.plot_pr_curve(save_path=pr_path)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(Config.PLOT_OUTPUT_DIR, "lstm_attention_cm.png")
        evaluator.plot_confusion_matrix(save_path=cm_path)
        
        # 绘制阈值指标
        threshold_path = os.path.join(Config.PLOT_OUTPUT_DIR, "lstm_attention_threshold.png")
        evaluator.plot_threshold_metrics(save_path=threshold_path)
    
    # 训练Transformer模型
    if Config.TRAIN_TRANSFORMER:
        print("\n" + "-"*40)
        print("训练Transformer模型")
        print("-"*40)
        
        dl_model = DeepLearningModels()
        dl_model.build_transformer(
            input_shape=input_shape,
            num_layers=Config.TRANSFORMER_LAYERS,
            d_model=Config.TRANSFORMER_DIM,
            num_heads=Config.TRANSFORMER_HEADS,
            dff=Config.TRANSFORMER_DFF,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.DL_LEARNING_RATE
        )
        
        # 训练模型
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "transformer_model")
        history = dl_model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=Config.DL_BATCH_SIZE,
            epochs=Config.DL_EPOCHS,
            patience=Config.DL_PATIENCE,
            model_path=model_path
        )
        
        # 评估模型
        metrics = dl_model.evaluate(X_test, y_test)
        model_results['transformer'] = metrics
        
        # 绘制训练历史
        history_plot_path = os.path.join(Config.PLOT_OUTPUT_DIR, "transformer_history.png")
        dl_model.plot_training_history(save_path=history_plot_path)
        
        # 获取预测概率
        y_pred_proba = dl_model.predict(X_test).flatten()
        
        # 生成评估报告
        evaluator = ModelEvaluator(y_test, y_pred_proba)
        evaluator.print_metrics()
        
        # 绘制ROC曲线
        roc_path = os.path.join(Config.PLOT_OUTPUT_DIR, "transformer_roc.png")
        evaluator.plot_roc_curve(save_path=roc_path)
        
        # 绘制PR曲线
        pr_path = os.path.join(Config.PLOT_OUTPUT_DIR, "transformer_pr.png")
        evaluator.plot_pr_curve(save_path=pr_path)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(Config.PLOT_OUTPUT_DIR, "transformer_cm.png")
        evaluator.plot_confusion_matrix(save_path=cm_path)
        
        # 绘制阈值指标
        threshold_path = os.path.join(Config.PLOT_OUTPUT_DIR, "transformer_threshold.png")
        evaluator.plot_threshold_metrics(save_path=threshold_path)
    
    # 训练CNN+LSTM模型
    if Config.TRAIN_CNN_LSTM:
        print("\n" + "-"*40)
        print("训练CNN+LSTM模型")
        print("-"*40)
        
        dl_model = DeepLearningModels()
        dl_model.build_cnn_lstm(
            input_shape=input_shape,
            filters=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            lstm_units=Config.LSTM_UNITS,
            dense_units=Config.DENSE_UNITS,
            dropout_rate=Config.DROPOUT_RATE,
            learning_rate=Config.DL_LEARNING_RATE
        )
        
        # 训练模型
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "cnn_lstm_model")
        history = dl_model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=Config.DL_BATCH_SIZE,
            epochs=Config.DL_EPOCHS,
            patience=Config.DL_PATIENCE,
            model_path=model_path
        )
        
        # 评估模型
        metrics = dl_model.evaluate(X_test, y_test)
        model_results['cnn_lstm'] = metrics
        
        # 绘制训练历史
        history_plot_path = os.path.join(Config.PLOT_OUTPUT_DIR, "cnn_lstm_history.png")
        dl_model.plot_training_history(save_path=history_plot_path)
        
        # 获取预测概率
        y_pred_proba = dl_model.predict(X_test).flatten()
        
        # 生成评估报告
        evaluator = ModelEvaluator(y_test, y_pred_proba)
        evaluator.print_metrics()
        
        # 绘制ROC曲线
        roc_path = os.path.join(Config.PLOT_OUTPUT_DIR, "cnn_lstm_roc.png")
        evaluator.plot_roc_curve(save_path=roc_path)
        
        # 绘制PR曲线
        pr_path = os.path.join(Config.PLOT_OUTPUT_DIR, "cnn_lstm_pr.png")
        evaluator.plot_pr_curve(save_path=pr_path)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(Config.PLOT_OUTPUT_DIR, "cnn_lstm_cm.png")
        evaluator.plot_confusion_matrix(save_path=cm_path)
        
        # 绘制阈值指标
        threshold_path = os.path.join(Config.PLOT_OUTPUT_DIR, "cnn_lstm_threshold.png")
        evaluator.plot_threshold_metrics(save_path=threshold_path)
    
    # 比较所有模型
    if len(model_results) > 1:
        compare_models(model_results)
    
    return model_results


def compare_models(model_results):
    """
    比较不同模型的性能
    
    参数:
        model_results: 包含各模型评估结果的字典
    """
    print("\n" + "="*50)
    print("模型性能比较")
    print("="*50)
    
    # 创建比较表格
    results_df = pd.DataFrame(columns=['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1'])
    
    for model_name, metrics in model_results.items():
        if metrics is not None:
            # 从metrics字典中提取各项指标
            results_df.loc[model_name] = [
                metrics.get('loss', np.nan),
                metrics.get('accuracy', np.nan),
                metrics.get('auc', np.nan),
                metrics.get('precision', np.nan),
                metrics.get('recall', np.nan),
                metrics.get('f1', np.nan)
            ]
    
    # 打印比较结果
    print(results_df)
    
    # 保存比较结果
    os.makedirs(os.path.dirname(Config.MODEL_COMPARISON_PATH), exist_ok=True)
    results_df.to_csv(Config.MODEL_COMPARISON_PATH)
    print(f"模型比较结果已保存至: {Config.MODEL_COMPARISON_PATH}")
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    # 绘制多个指标比较
    metrics_to_plot = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1']
    valid_metrics = [m for m in metrics_to_plot if not results_df[m].isna().all()]
    
    for i, metric in enumerate(valid_metrics):
        plt.subplot(2, 3, i+1)
        results_df[metric].plot(kind='bar')
        plt.title(f'模型{metric}比较')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    comparison_plot_path = os.path.join(Config.PLOT_OUTPUT_DIR, "model_comparison.png")
    plt.savefig(comparison_plot_path)
    print(f"模型比较图已保存至: {comparison_plot_path}")
    
    plt.show()
    
    return results_df


def main():
    """主函数"""
    print("="*50)
    print("ICU患者急性肾损伤(AKI)预测 - 深度学习方法")
    print("="*50)
    
    start_time = time.time()
    
    # 设置环境
    setup_environment()
    
    # 加载和处理数据
    df = load_and_process_data()
    
    # 训练和评估模型
    model_results = train_and_evaluate_models(df)
    
    # 计算总运行时间
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time/60:.2f} 分钟")
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()