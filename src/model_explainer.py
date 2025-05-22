"""
模型解释模块：用于解释深度学习模型的预测结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class ModelExplainer:
    """模型解释类"""
    
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        初始化模型解释器
        
        参数:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称列表
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names if feature_names is not None else [f"特征_{i}" for i in range(X_test.shape[1])]
    
    def explain_with_shap(self, sample_size=100, output_dir=None):
        """
        使用SHAP解释模型
        
        参数:
            sample_size: 用于解释的样本数量
            output_dir: 输出目录
        """
        try:
            import shap
        except ImportError:
            print("错误: SHAP库未安装，请使用 'pip install shap' 安装")
            return
        
        print("使用SHAP解释模型...")
        
        # 随机选择样本
        if sample_size > len(self.X_test):
            sample_size = len(self.X_test)
            print(f"警告: 样本大小超过测试集大小，使用全部测试集 ({sample_size} 个样本)")
        
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        X_sample = self.X_test[indices]
        
        # 创建SHAP解释器
        if hasattr(self.model, 'predict'):
            # 对于Keras模型
            explainer = shap.KernelExplainer(self.model.predict, X_sample[:10])
        else:
            # 对于其他模型
            explainer = shap.KernelExplainer(self.model, X_sample[:10])
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        
        # 如果是多分类，取第一个类别的SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 绘制摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("SHAP特征重要性")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(summary_path)
            print(f"SHAP摘要图已保存至: {summary_path}")
        
        plt.show()
        plt.close()
        
        # 绘制依赖图 (对于前5个最重要的特征)
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(feature_importance)[-5:]
        
        for i in top_indices:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(i, shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.title(f"SHAP依赖图: {self.feature_names[i]}")
            
            if output_dir:
                dep_path = os.path.join(output_dir, f"shap_dependence_{self.feature_names[i]}.png")
                plt.savefig(dep_path)
                print(f"SHAP依赖图已保存至: {dep_path}")
            
            plt.show()
            plt.close()
        
        return shap_values, explainer
    
    def explain_with_lime(self, num_samples=5, output_dir=None):
        """
        使用LIME解释模型
        
        参数:
            num_samples: 要解释的样本数量
            output_dir: 输出目录
        """
        try:
            import lime
            from lime import lime_tabular
        except ImportError:
            print("错误: LIME库未安装，请使用 'pip install lime' 安装")
            return
        
        print("使用LIME解释模型...")
        
        # 创建LIME解释器
        explainer = lime_tabular.LimeTabularExplainer(
            self.X_test,
            feature_names=self.feature_names,
            class_names=["无AKI", "AKI"],
            mode="classification"
        )
        
        # 随机选择样本
        if num_samples > len(self.X_test):
            num_samples = len(self.X_test)
            print(f"警告: 样本数量超过测试集大小，使用全部测试集 ({num_samples} 个样本)")
        
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        # 为每个样本生成解释
        for i, idx in enumerate(indices):
            # 获取样本
            instance = self.X_test[idx]
            true_label = self.y_test[idx]
            
            # 生成解释
            if hasattr(self.model, 'predict_proba'):
                exp = explainer.explain_instance(
                    instance, self.model.predict_proba, num_features=10
                )
            else:
                # 对于Keras模型，创建一个包装函数
                def predict_fn(x):
                    return self.model.predict(x)
                
                exp = explainer.explain_instance(
                    instance, predict_fn, num_features=10
                )
            
            # 绘制解释
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f"样本 {i+1} 的LIME解释 (真实标签: {'AKI' if true_label == 1 else '无AKI'})")
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                lime_path = os.path.join(output_dir, f"lime_explanation_{i+1}.png")
                plt.savefig(lime_path)
                print(f"LIME解释图已保存至: {lime_path}")
            
            plt.show()
            plt.close()
    
    def explain_with_gradcam(self, num_samples=5, output_dir=None):
        """
        使用Grad-CAM解释模型 (仅适用于CNN或具有卷积层的模型)
        
        参数:
            num_samples: 要解释的样本数量
            output_dir: 输出目录
        """
        # 检查是否为Keras模型
        if not isinstance(self.model, tf.keras.Model):
            print("错误: Grad-CAM仅适用于Keras模型")
            return
        
        # 检查模型是否包含卷积层
        has_conv = False
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                has_conv = True
                break
        
        if not has_conv:
            print("警告: 模型不包含卷积层，Grad-CAM可能不适用")
        
        print("使用Grad-CAM解释模型...")
        
        # 随机选择样本
        if num_samples > len(self.X_test):
            num_samples = len(self.X_test)
            print(f"警告: 样本数量超过测试集大小，使用全部测试集 ({num_samples} 个样本)")
        
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        # 为每个样本生成Grad-CAM
        for i, idx in enumerate(indices):
            # 获取样本
            instance = self.X_test[idx:idx+1]  # 保持批次维度
            true_label = self.y_test[idx]
            
            # 找到最后一个卷积层
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                print("错误: 未找到卷积层")
                return
            
            # 创建Grad-CAM模型
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # 计算梯度
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(instance)
                loss = predictions[:, 0]
            
            # 提取特征图和梯度
            output = conv_output[0]
            grads = tape.gradient(loss, conv_output)[0]
            
            # 计算权重
            weights = tf.reduce_mean(grads, axis=(0, 1))
            
            # 生成类激活图
            cam = tf.reduce_sum(tf.multiply(output, weights), axis=-1)
            
            # 归一化
            cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))
            
            # 调整大小以匹配输入
            if len(instance.shape) == 3:  # 时序数据
                cam = tf.image.resize(cam[..., tf.newaxis], (instance.shape[1], 1))
                cam = tf.squeeze(cam)
            
            # 绘制热力图
            plt.figure(figsize=(12, 6))
            
            if len(instance.shape) == 3:  # 时序数据
                plt.subplot(1, 2, 1)
                plt.plot(instance[0, :, 0])  # 绘制第一个特征
                plt.title(f"样本 {i+1} 的输入 (第一个特征)")
                
                plt.subplot(1, 2, 2)
                plt.imshow(cam[..., tf.newaxis], cmap='jet', alpha=0.5)
                plt.title(f"样本 {i+1} 的Grad-CAM (真实标签: {'AKI' if true_label == 1 else '无AKI'})")
            else:
                plt.bar(range(len(cam)), cam)
                plt.title(f"样本 {i+1} 的Grad-CAM (真实标签: {'AKI' if true_label == 1 else '无AKI'})")
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                gradcam_path = os.path.join(output_dir, f"gradcam_{i+1}.png")
                plt.savefig(gradcam_path)
                print(f"Grad-CAM图已保存至: {gradcam_path}")
            
            plt.show()
            plt.close()