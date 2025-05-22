"""
深度学习模型模块：包含用于AKI预测的各种深度学习模型实现
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 直接从Keras导入所需模块
import keras
from keras.models import Model, Sequential
from keras.layers import (
    Dense, LSTM, GRU, Bidirectional, Input, Dropout, 
    Concatenate, Add, LayerNormalization, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, BatchNormalization
)
# 单独导入MultiHeadAttention
from keras.layers import MultiHeadAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


class DeepLearningModels:
    """深度学习模型类"""
    
    def __init__(self):
        """初始化深度学习模型"""
        self.model = None
        self.history = None
        
    def build_lstm_attention(self, input_shape, lstm_units=64, attention_heads=4, 
                            dense_units=32, dropout_rate=0.3, learning_rate=0.001):
        """
        构建改进的LSTM+Attention模型
        
        参数:
            input_shape: 输入形状 (time_steps, features) 或 (features,)
            lstm_units: LSTM层的单元数
            attention_heads: 注意力头数量
            dense_units: 全连接层的单元数
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        # 检查输入维度，确定是时序还是非时序数据
        if len(input_shape) == 2:  # 时序数据 (time_steps, features)
            time_steps, features = input_shape
            
            # 构建模型
            inputs = Input(shape=(time_steps, features))
            
            # 双向LSTM层
            lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
            
            # 多头注意力层
            attention = MultiHeadAttention(
                num_heads=attention_heads, key_dim=lstm_units//attention_heads
            )(lstm_out, lstm_out)
            
            # 残差连接
            attention_with_residual = Add()([lstm_out, attention])
            
            # Layer Normalization
            normalized = LayerNormalization()(attention_with_residual)
            
            # 全局平均池化
            pooled = GlobalAveragePooling1D()(normalized)
            
        else:  # 非时序数据 (features,)
            features = input_shape[0]
            
            # 构建模型
            inputs = Input(shape=(features,))
            
            # 全连接层
            pooled = Dense(dense_units*2, activation='relu')(inputs)
            pooled = BatchNormalization()(pooled)
            pooled = Dropout(dropout_rate)(pooled)
        
        # 共享的全连接层
        x = Dense(dense_units, activation='relu')(pooled)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # 输出层
        outputs = Dense(1, activation='sigmoid')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        print("LSTM+Attention模型构建完成:")
        model.summary()
        
        return model
    
    def build_transformer(self, input_shape, num_layers=2, d_model=64, num_heads=4,
                         dff=128, dropout_rate=0.1, learning_rate=0.001):
        """
        构建Transformer模型
        
        参数:
            input_shape: 输入形状 (time_steps, features) 或 (features,)
            num_layers: Transformer层数
            d_model: 模型维度
            num_heads: 注意力头数量
            dff: 前馈网络维度
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        # 检查输入维度，确定是时序还是非时序数据
        if len(input_shape) == 2:  # 时序数据 (time_steps, features)
            time_steps, features = input_shape
            
            # 构建模型
            inputs = Input(shape=(time_steps, features))
            
            # 线性投影层
            x = Dense(d_model)(inputs)
            
            # Transformer编码器层
            for _ in range(num_layers):
                # 多头注意力
                attn_output = MultiHeadAttention(
                    num_heads=num_heads, key_dim=d_model//num_heads
                )(x, x)
                
                # 残差连接和层归一化
                x = Add()([x, attn_output])
                x = LayerNormalization(epsilon=1e-6)(x)
                
                # 前馈网络
                ffn_output = Sequential([
                    Dense(dff, activation='relu'),
                    Dense(d_model)
                ])(x)
                
                # 残差连接和层归一化
                x = Add()([x, ffn_output])
                x = LayerNormalization(epsilon=1e-6)(x)
                
                # Dropout
                x = Dropout(dropout_rate)(x)
            
            # 全局平均池化
            pooled = GlobalAveragePooling1D()(x)
            
        else:  # 非时序数据 (features,)
            features = input_shape[0]
            
            # 构建模型
            inputs = Input(shape=(features,))
            
            # 全连接层
            pooled = Dense(d_model, activation='relu')(inputs)
            pooled = BatchNormalization()(pooled)
            pooled = Dropout(dropout_rate)(pooled)
        
        # 输出层
        x = Dense(d_model//2, activation='relu')(pooled)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        print("Transformer模型构建完成:")
        model.summary()
        
        return model
    
    def build_cnn_lstm(self, input_shape, filters=64, kernel_size=3, lstm_units=64,
                      dense_units=32, dropout_rate=0.3, learning_rate=0.001):
        """
        构建CNN+LSTM混合模型
        
        参数:
            input_shape: 输入形状 (time_steps, features) 或 (features,)
            filters: CNN过滤器数量
            kernel_size: CNN核大小
            lstm_units: LSTM单元数量
            dense_units: 全连接层单元数量
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        # 检查输入维度，确定是时序还是非时序数据
        if len(input_shape) == 2:  # 时序数据 (time_steps, features)
            time_steps, features = input_shape
            
            # 构建模型
            inputs = Input(shape=(time_steps, features))
            
            # CNN层
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            # LSTM层
            x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
            
            # 全局平均池化
            pooled = GlobalAveragePooling1D()(x)
            
        else:  # 非时序数据 (features,)
            features = input_shape[0]
            
            # 构建模型
            inputs = Input(shape=(features,))
            
            # 全连接层
            pooled = Dense(dense_units*2, activation='relu')(inputs)
            pooled = BatchNormalization()(pooled)
            pooled = Dropout(dropout_rate)(pooled)
        
        # 共享的全连接层
        x = Dense(dense_units, activation='relu')(pooled)
        x = Dropout(dropout_rate)(x)
        
        # 输出层
        outputs = Dense(1, activation='sigmoid')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        print("CNN+LSTM模型构建完成:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, patience=10, model_path=None):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批次大小
            epochs: 训练轮数
            patience: 早停耐心值
            model_path: 模型保存路径
        
        返回:
            history: 训练历史
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_*方法")
        
        # 确保模型路径以.keras结尾
        if model_path is not None:
            if not model_path.endswith('.keras'):
                model_path = model_path + '.keras'
            
            # 确保目录存在
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        
        # 定义回调函数
        callbacks = []
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 模型检查点
        if model_path:
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # 学习率调度器
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # 训练模型
        print("开始训练模型...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        print("模型训练完成")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        """
        if self.model is None:
            print("错误: 模型尚未构建")
            return None
        
        print("评估模型性能...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # 创建指标字典
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
            print(f"{metric_name}: {results[i]:.4f}")
        
        # 计算其他指标
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        # 确保添加这些指标到字典中
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
        
        print(f"额外计算的指标:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        使用模型进行预测
        
        参数:
            X: 输入特征
        """
        if self.model is None:
            print("错误: 模型尚未构建")
            return None
        
        return self.model.predict(X)
    
    def plot_training_history(self, figsize=(12, 5), save_path=None):
        """
        绘制训练历史
        
        参数:
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.history is None:
            print("错误: 模型尚未训练")
            return
        
        plt.figure(figsize=figsize)
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制AUC
        plt.subplot(1, 2, 2)
        plt.plot(self.history['auc'], label='Training AUC')
        if 'val_auc' in self.history:
            plt.plot(self.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图已保存至 {save_path}")
        
        plt.show()
    
    def save_model(self, model_path):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        if self.model is None:
            print("错误: 模型尚未构建")
            return
        
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.model.save(model_path)
        print(f"模型已保存至 {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型加载路径
        """
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"模型已从 {model_path} 加载")