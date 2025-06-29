"""
模型集成示例：使用训练好的深度学习模型进行集成预测
"""

import os
import pandas as pd
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_processor_dl import prepare_time_series_data
# 导入自定义模块
from src.model_ensemble import ModelEnsemble


def load_trained_models():
    """
    加载训练好的深度学习模型
    """
    model_paths = {
        'LSTM+Attention': 'd:/Programfile/AIIntellligence/output/models/dl/lstm_attention_model.keras',
        'Transformer': 'd:/Programfile/AIIntellligence/output/models/dl/transformer_model.keras',
        'CNN+LSTM': 'd:/Programfile/AIIntellligence/output/models/dl/cnn_lstm_model.keras'
    }
    
    models = []
    model_names = []
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"加载模型: {name} from {path}")
            model = load_model(path)
            models.append(model)
            model_names.append(name)
        else:
            print(f"警告: 模型文件不存在 - {path}")
    
    return models, model_names

def load_and_prepare_data():
    """
    加载和准备数据
    """
    # 加载处理后的数据
    data_path = 'data/processed/processed_data_dl.csv'
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 - {data_path}")
        return None, None, None, None, None, None
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 准备时间序列数据 - 直接返回分割好的数据
    X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_time_series_data(df)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {len(features)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """
    主函数：演示模型集成的完整流程
    """
    print("深度学习模型集成示例")
    
    # 1. 加载训练好的模型
    print("\n1. 加载训练好的模型...")
    models, model_names = load_trained_models()
    
    if len(models) == 0:
        print("错误: 没有找到可用的模型文件")
        return
    
    print(f"成功加载 {len(models)} 个模型: {model_names}")
    
    # 2. 加载和准备数据
    print("\n2. 加载和准备数据...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    if X_test is None:
        print("错误: 数据加载失败")
        return
    
    # 3. 创建模型集成
    print("\n3. 创建模型集成...")
    ensemble = ModelEnsemble(models, model_names)
    
    # 4. 评估单个模型性能
    print("\n4. 评估单个模型性能...")
    individual_predictions = ensemble.predict_proba(X_test)
    
    for i, (name, pred) in enumerate(zip(model_names, individual_predictions)):
        auc = roc_auc_score(y_test, pred)
        print(f"{name}: AUC = {auc:.4f}")
    
    # 5. 简单平均集成
    print("\n5. 简单平均集成...")
    simple_ensemble_pred = ensemble.simple_average(X_test)
    simple_auc = roc_auc_score(y_test, simple_ensemble_pred)
    print(f"简单平均集成 AUC: {simple_auc:.4f}")
    
    # 6. 优化权重集成
    print("\n6. 优化权重集成...")
    if X_val is not None and y_val is not None:
        # 在验证集上优化权重
        best_weights = ensemble.optimize_weights(X_val, y_val, method='grid_search')
        
        # 使用优化权重进行预测
        weighted_ensemble_pred = ensemble.weighted_average(X_test)
        weighted_auc = roc_auc_score(y_test, weighted_ensemble_pred)
        print(f"加权集成 AUC: {weighted_auc:.4f}")
        print(f"最优权重: {best_weights}")
    else:
        print("跳过权重优化（验证集不可用）")
    
    # 7. 生成详细评估报告
    print("\n7. 生成详细评估报告...")
    output_dir = 'output/ensemble_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ensemble.evaluate(X_test, y_test, output_dir=output_dir)
    
    print("\n集成模型最终评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 8. 保存集成预测结果
    print("\n8. 保存集成预测结果...")
    if 'weighted_ensemble_pred' in locals():
        final_pred = weighted_ensemble_pred
        ensemble_type = "加权集成"
    else:
        final_pred = simple_ensemble_pred
        ensemble_type = "简单集成"
    
    # 创建预测结果DataFrame
    results_df = pd.DataFrame({
        'y_true': y_test,
        'ensemble_prediction': final_pred,
        'ensemble_type': ensemble_type
    })
    
    # 添加单个模型的预测
    for i, name in enumerate(model_names):
        results_df[f'{name}_prediction'] = individual_predictions[i]
    
    # 保存结果
    results_path = os.path.join(output_dir, 'ensemble_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存至: {results_path}")

    print("模型集成完成！")

def quick_ensemble_prediction(X_new):
    """
    快速集成预测函数：用于新数据的预测
    
    参数:
        X_new: 新的输入数据
        
    返回:
        ensemble_pred: 集成预测结果
    """
    # 加载模型
    models, model_names = load_trained_models()
    
    if len(models) == 0:
        print("错误: 没有找到可用的模型")
        return None
    
    # 创建集成
    ensemble = ModelEnsemble(models, model_names)
    
    # 使用简单平均进行预测
    ensemble_pred = ensemble.simple_average(X_new)
    
    return ensemble_pred

if __name__ == "__main__":
    main()