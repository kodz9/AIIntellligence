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
    
    def evaluate(self, X_test, y_test):
        """
        评估所有训练好的模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            dict: 包含各模型评估结果的字典
        """
        if not self.trained_models:
            print("错误: 模型尚未训练")
            return {}
        
        print("\n基线模型评估结果:")
        results = {}
        
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


class LSTMWithAttention:
    """LSTM + Attention深度学习模型"""
    
    def __init__(self, input_shape=None):
        """
        初始化LSTM+Attention模型
        
        参数:
            input_shape: 输入特征形状，用于非时序数据
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self, lstm_units=64, dense_units=32, dropout_rate=0.3, 
                    learning_rate=0.001, time_steps=None, features=None):
        """
        构建LSTM+Attention模型
        
        参数:
            lstm_units: LSTM层的单元数
            dense_units: 全连接层的单元数
            dropout_rate: Dropout比率
            learning_rate: 学习率
            time_steps: 时间步 (用于时序数据)
            features: 每个时间步的特征数 (用于时序数据)
        """
        try:
            # 检查是否有TensorFlow
            if not TENSORFLOW_AVAILABLE:
                print("错误: 无法构建深度学习模型，TensorFlow未安装")
                return None
            
            if time_steps and features:
                # 时序数据模型 (3D输入 [样本数, 时间步, 特征数])
                inputs = Input(shape=(time_steps, features))
                
                # LSTM层
                lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
                
                # Self-Attention层
                attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=4, key_dim=lstm_units//4
                )(lstm_out, lstm_out)
                
                # 残差连接
                attention_with_residual = tf.keras.layers.Add()([lstm_out, attention])
                
                # Layer Normalization
                normalized = tf.keras.layers.LayerNormalization()(attention_with_residual)
                
                # 全局平均池化
                pooled = tf.keras.layers.GlobalAveragePooling1D()(normalized)
                
                # 全连接层
                x = Dense(dense_units, activation='relu')(pooled)
                x = Dropout(dropout_rate)(x)
                
                # 输出层
                outputs = Dense(1, activation='sigmoid')(x)
                
                # 创建模型
                model = Model(inputs=inputs, outputs=outputs)
                
            else:
                # 非时序数据模型 (2D输入 [样本数, 特征数])
                if not self.input_shape:
                    print("错误: 非时序模型需要提供input_shape")
                    return None
                
                model = Sequential([
                    Dense(dense_units*2, activation='relu', input_shape=(self.input_shape,)),
                    Dropout(dropout_rate),
                    Dense(dense_units, activation='relu'),
                    Dropout(dropout_rate),
                    Dense(1, activation='sigmoid')
                ])
            
            # 编译模型
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            self.model = model
            print("模型构建完成:")
            model.summary()
            
            return model
            
        except Exception as e:
            print(f"构建模型时出错: {str(e)}")
            return None
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, 
              patience=10, model_path=None):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批量大小
            epochs: 训练轮数
            patience: 早停耐心值
            model_path: 模型保存路径
        """
        if self.model is None:
            print("错误: 模型尚未构建")
            return None
        
        # 定义回调函数
        callbacks = []
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 模型检查点
        if model_path:
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            checkpoint = ModelCheckpoint(
                model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # 训练模型
        print("开始训练深度学习模型...")
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        elapsed = time.time() - start_time
        print(f"深度学习模型训练完成，耗时: {elapsed:.2f}秒")
        
        self.history = history
        return history
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            dict: 包含评估结果的字典
        """
        if self.model is None:
            print("错误: 模型尚未训练")
            return {}
        
        # 预测概率
        y_prob = self.model.predict(X_test)
        # 使用0.5作为阈值的预测类别
        y_pred = (y_prob >= 0.5).astype(int)
        
        # 计算指标
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results = {
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        print("\n深度学习模型评估结果:")
        print(f"  AUROC: {auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史
        
        参数:
            save_path: 图表保存路径
        """
        if self.history is None:
            print("错误: 模型尚未训练")
            return
        
        plt.figure(figsize=(12, 5))
        
        # 绘制AUC
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['auc'])
        plt.plot(self.history.history['val_auc'])
        plt.title('模型 AUC')
        plt.ylabel('AUC')
        plt.xlabel('轮数')
        plt.legend(['训练集', '验证集'], loc='lower right')
        
        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('模型损失')
        plt.ylabel('损失')
        plt.xlabel('轮数')
        plt.legend(['训练集', '验证集'], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"训练历史图表已保存至 {save_path}")
        
        plt.show()
    
    def save_model(self, model_path):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        if self.model is None:
            print("错误: 模型尚未训练")
            return
        
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model.save(model_path)
        print(f"深度学习模型已保存至 {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型文件路径
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"深度学习模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")


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
    
    # 训练深度学习模型
    dl_model = LSTMWithAttention(input_shape=X_train.shape[1])
    dl_model.build_model()
    
    # 分割验证集
    X_train_dl, X_val, y_train_dl, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    dl_model.train(X_train_dl, y_train_dl, X_val, y_val, epochs=10)
    dl_model.evaluate(X_test, y_test) 