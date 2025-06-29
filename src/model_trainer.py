"""
模型训练模块：用于训练ICU患者AKI预测模型，包括基线模型和深度学习模型
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 深度学习相关导入
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Concatenate, Add, LayerNormalization, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("警告: TensorFlow未安装，深度学习模型将不可用")


class BaselineModels:
    """基线模型类"""
    
    def __init__(self, random_state=42):
        """初始化基线模型"""
        self.models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'xgboost': XGBClassifier(random_state=random_state)
        }
        
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
    
    def train(self, X_train, y_train, preprocessor=None):
        """
        训练所有基线模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            preprocessor: 预处理流水线
        """
        print("开始训练基线模型...")
        
        for name, model in self.models.items():
            print(f"训练 {name} 模型...")
            start_time = time.time()
            
            if preprocessor:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipeline.fit(X_train, y_train)
                self.trained_models[name] = pipeline
            else:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
            
            elapsed = time.time() - start_time
            print(f"训练 {name} 完成，耗时: {elapsed:.2f}秒")
    
    def evaluate(self, X_test, y_test, feature_names=None, output_dir=None):
        """
        评估所有训练好的模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称列表，用于特征重要性分析
            output_dir: 输出目录，用于保存特征重要性图
        
        返回:
            dict: 包含各模型评估结果的字典
        """
        if not self.trained_models:
            print("错误: 模型尚未训练")
            return {}
        
        print("\n基线模型评估结果:")
        results = {}
        
        # 如果没有提供特征名称，尝试从X_test获取
        if feature_names is None and hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        
        for name, model in self.trained_models.items():
            # 预测概率
            y_prob = model.predict_proba(X_test)[:, 1]
            # 使用0.5作为阈值的预测类别
            y_pred = (y_prob >= 0.5).astype(int)
            
            # 计算指标
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            model_results = {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            results[name] = model_results
            
            print(f"{name}:")
            print(f"  AUROC: {auc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            
            # 计算并可视化特征重要性（如果提供了输出目录）
            if output_dir and feature_names:
                try:
                    from src.model_evaluator import plot_feature_importance
                    
                    # 创建模型特定的目录
                    model_dir = os.path.join(output_dir, name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # 计算特征重要性
                    importance_path = os.path.join(model_dir, f"{name}_feature_importance.png")
                    feature_importance = plot_feature_importance(
                        model, feature_names, output_path=importance_path
                    )
                    
                    # 保存特征重要性CSV
                    if feature_importance is not None:
                        importance_csv = os.path.join(model_dir, f"{name}_feature_importance.csv")
                        feature_importance.to_csv(importance_csv, index=False)
                        print(f"特征重要性已保存至 {importance_csv}")
                except Exception as e:
                    print(f"计算特征重要性时出错: {str(e)}")
            
            # 更新最佳模型 (根据AUROC)
            if auc > self.best_score:
                self.best_score = auc
                self.best_model = model
                self.best_model_name = name
        
        if self.best_model:
            print(f"\n最佳基线模型: {self.best_model_name} (AUROC: {self.best_score:.4f})")
        
        return results
    
    def save_models(self, output_dir):
        """
        保存训练好的模型
        
        参数:
            output_dir: 模型保存路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f"{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"模型 {name} 已保存至 {model_path}")
    
    def load_model(self, model_path, model_name=None):
        """
        加载训练好的模型
        
        参数:
            model_path: 模型文件路径
            model_name: 模型名称
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if model_name:
            self.trained_models[model_name] = model
            print(f"模型 {model_name} 已加载")
        else:
            model_name = os.path.basename(model_path).replace('_model.pkl', '')
            self.trained_models[model_name] = model
            print(f"模型 {model_name} 已加载")

def prepare_time_series_data(df, features, time_steps=48, step=1):
    """
    准备时序数据
    
    参数:
        df: 包含患者数据的DataFrame
        features: 用于时序预测的特征列表
        time_steps: 时间窗口大小
        step: 滑动窗口步长
    
    返回:
        X: 特征序列 - 形状为 (样本数, 时间步, 特征数)
        y: 标签
    """
    print("准备时序数据...")
    
    # 此函数仅为示例，用于说明如何构建时序数据
    # 实际项目中应该从 chartevents.csv 等时间序列数据中提取患者的时序特征
    
    # 创建示例时序数据 (这只是一个示例，实际项目需要真实的时序数据)
    X, y = [], []
    
    # 示例数据生成 (实际项目中应替换为真实数据)
    n_samples = len(df)
    n_features = len(features)
    
    # 随机生成时序特征 (仅用于演示!)
    print("注意: 使用随机生成的时序数据用于演示")
    np.random.seed(42)
    
    for i in range(n_samples):
        # 为每个样本创建随机时间序列
        ts = []
        baseline_values = df.iloc[i][features].fillna(0).values
        
        for t in range(time_steps):
            # 基于基线值添加一些随机变化来模拟时间序列
            time_values = baseline_values + np.random.normal(0, 0.1, n_features) * t
            ts.append(time_values)
        
        X.append(ts)
        y.append(df.iloc[i]['aki_label'])
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # 测试模型训练
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练基线模型
    baseline = BaselineModels()
    baseline.train(X_train, y_train)
    baseline.evaluate(X_test, y_test)
    
    # 分割验证集
    X_train_dl, X_val, y_train_dl, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
