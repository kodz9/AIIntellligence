"""
超参数调优脚本：用于寻找深度学习模型的最佳超参数
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 添加 seaborn 导入
from sklearn.model_selection import ParameterGrid
import tensorflow as tf

# 导入自定义模块
from src.data_processor import load_data, verify_aki_label, handle_missing_values, feature_engineering
from src.data_processor_dl import prepare_dl_data, handle_class_imbalance
from src.dl_models import DeepLearningModels


# 配置
class Config:
    # 数据路径
    RAW_DATA_PATH = "data/raw/base_stay.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
    
    # 输出路径
    TUNING_OUTPUT_DIR = "output/hyperparameter_tuning"
    
    # 模型参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    
    # 超参数调优
    MAX_TRIALS = 20  # 最大试验次数
    EPOCHS = 30
    BATCH_SIZE = 32
    PATIENCE = 5
    
    # 类别不平衡处理
    HANDLE_IMBALANCE = True
    IMBALANCE_METHOD = 'smote'


def setup_environment():
    """设置环境，创建必要的目录"""
    os.makedirs(Config.TUNING_OUTPUT_DIR, exist_ok=True)
    
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
    
    print("环境设置完成")


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


def tune_lstm_attention():
    """调优LSTM+Attention模型"""
    print("\n" + "="*50)
    print("调优LSTM+Attention模型")
    print("="*50)
    
    # 加载数据
    df = load_and_process_data()
    
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_dl_data(
        df,
        target_col='aki_label',
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    # 处理类别不平衡
    if Config.HANDLE_IMBALANCE:
        X_train, y_train = handle_class_imbalance(
            X_train, y_train, 
            method=Config.IMBALANCE_METHOD,
            random_state=Config.RANDOM_STATE
        )
    
    # 定义超参数网格
    param_grid = {
        'lstm_units': [32, 64, 128],
        'attention_heads': [2, 4, 8],
        'dense_units': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }
    
    # 创建参数组合
    param_combinations = list(ParameterGrid(param_grid))
    print(f"总共 {len(param_combinations)} 种参数组合")
    
    # 限制试验次数
    if len(param_combinations) > Config.MAX_TRIALS:
        print(f"限制为 {Config.MAX_TRIALS} 次试验")
        np.random.seed(Config.RANDOM_STATE)
        param_combinations = np.random.choice(param_combinations, Config.MAX_TRIALS, replace=False)
    
    # 记录结果
    results = []
    
    # 遍历参数组合
    for i, params in enumerate(param_combinations):
        print(f"\n试验 {i+1}/{len(param_combinations)}")
        print(f"参数: {params}")
        
        # 创建模型
        dl_model = DeepLearningModels()
        dl_model.build_lstm_attention(
            input_shape=(X_train.shape[1],),
            lstm_units=params['lstm_units'],
            attention_heads=params['attention_heads'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
        
        # 训练模型
        history = dl_model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            patience=Config.PATIENCE
        )
        
        # 评估模型
        metrics = dl_model.evaluate(X_test, y_test)
        
        # 记录结果
        result = {
            'trial': i+1,
            **params,
            **metrics
        }
        results.append(result)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_path = os.path.join(Config.TUNING_OUTPUT_DIR, "lstm_attention_tuning_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"结果已保存至: {results_path}")
    
    # 分析结果
    results_df = pd.DataFrame(results)
    best_idx = results_df['auc'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print("\n" + "="*50)
    print("最佳参数:")
    print(f"LSTM单元数: {best_params['lstm_units']}")
    print(f"注意力头数: {best_params['attention_heads']}")
    print(f"全连接层单元数: {best_params['dense_units']}")
    print(f"Dropout比率: {best_params['dropout_rate']}")
    print(f"学习率: {best_params['learning_rate']}")
    print(f"AUC: {best_params['auc']:.4f}")
    
    # 绘制参数重要性图
    plt.figure(figsize=(12, 8))
    
    for i, param in enumerate(['lstm_units', 'attention_heads', 'dense_units', 'dropout_rate', 'learning_rate']):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=param, y='auc', data=results_df)
        plt.title(f'{param}对AUC的影响')
    
    plt.tight_layout()
    plot_path = os.path.join(Config.TUNING_OUTPUT_DIR, "lstm_attention_parameter_importance.png")
    plt.savefig(plot_path)
    print(f"参数重要性图已保存至: {plot_path}")
    
    return best_params


def tune_transformer():
    """调优Transformer模型"""
    # 类似于tune_lstm_attention函数，但针对Transformer模型
    # 这里省略实现，可以参考tune_lstm_attention函数
    pass


def tune_cnn_lstm():
    """调优CNN+LSTM模型"""
    # 类似于tune_lstm_attention函数，但针对CNN+LSTM模型
    # 这里省略实现，可以参考tune_lstm_attention函数
    pass


def main():
    """主函数"""
    print("="*50)
    print("ICU患者急性肾损伤(AKI)预测 - 超参数调优")
    print("="*50)
    
    start_time = time.time()
    
    # 设置环境
    setup_environment()
    
    # 调优LSTM+Attention模型
    best_lstm_attention_params = tune_lstm_attention()
    
    # 调优Transformer模型
    # best_transformer_params = tune_transformer()
    
    # 调优CNN+LSTM模型
    # best_cnn_lstm_params = tune_cnn_lstm()
    
    # 计算总运行时间
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time/60:.2f} 分钟")
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()