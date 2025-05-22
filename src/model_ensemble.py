"""
模型集成模块：用于组合多个深度学习模型的预测结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from src.model_evaluator import ModelEvaluator


class ModelEnsemble:
    """模型集成类"""
    
    def __init__(self, models, model_names):
        """
        初始化模型集成
        
        参数:
            models: 模型列表
            model_names: 模型名称列表
        """
        self.models = models
        self.model_names = model_names
        self.weights = None
    
    def predict_proba(self, X):
        """
        使用所有模型进行预测并返回概率
        
        参数:
            X: 输入特征
            
        返回:
            predictions: 每个模型的预测概率
        """
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X).flatten()
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def simple_average(self, X):
        """
        简单平均集成
        
        参数:
            X: 输入特征
            
        返回:
            ensemble_pred: 集成预测概率
        """
        predictions = self.predict_proba(X)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def weighted_average(self, X, weights=None):
        """
        加权平均集成
        
        参数:
            X: 输入特征
            weights: 权重列表
            
        返回:
            ensemble_pred: 集成预测概率
        """
        predictions = self.predict_proba(X)
        
        if weights is None:
            if self.weights is None:
                # 使用均等权重
                weights = np.ones(len(self.models)) / len(self.models)
            else:
                weights = self.weights
        else:
            self.weights = weights
        
        # 确保权重和为1
        weights = np.array(weights) / np.sum(weights)
        
        # 加权平均
        ensemble_pred = np.sum(predictions * weights[:, np.newaxis], axis=0)
        return ensemble_pred
    
    def optimize_weights(self, X_val, y_val, method='grid_search'):
        """
        优化集成权重
        
        参数:
            X_val: 验证特征
            y_val: 验证标签
            method: 优化方法 ('grid_search', 'random_search')
            
        返回:
            best_weights: 最优权重
        """
        print(f"使用{method}优化集成权重...")
        
        # 获取每个模型在验证集上的预测
        predictions = self.predict_proba(X_val)
        
        if method == 'grid_search':
            # 对于两个模型的情况，使用网格搜索
            if len(self.models) == 2:
                best_auc = 0
                best_weights = [0.5, 0.5]
                
                for w1 in np.linspace(0, 1, 21):  # 0到1之间的21个权重
                    w2 = 1 - w1
                    weights = [w1, w2]
                    
                    # 计算加权预测
                    ensemble_pred = np.sum(predictions * np.array(weights)[:, np.newaxis], axis=0)
                    
                    # 计算AUC
                    auc = roc_auc_score(y_val, ensemble_pred)
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = weights
                
                print(f"最优权重: {best_weights}, AUC: {best_auc:.4f}")
                self.weights = best_weights
                return best_weights
            
            # 对于三个或更多模型，使用随机搜索
            else:
                return self.optimize_weights(X_val, y_val, method='random_search')
        
        elif method == 'random_search':
            # 随机搜索
            best_auc = 0
            best_weights = np.ones(len(self.models)) / len(self.models)
            
            for _ in range(1000):  # 尝试1000次随机权重
                # 生成随机权重
                weights = np.random.rand(len(self.models))
                weights = weights / np.sum(weights)  # 归一化
                
                # 计算加权预测
                ensemble_pred = np.sum(predictions * weights[:, np.newaxis], axis=0)
                
                # 计算AUC
                auc = roc_auc_score(y_val, ensemble_pred)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights
            
            print(f"最优权重: {best_weights}, AUC: {best_auc:.4f}")
            self.weights = best_weights
            return best_weights
        
        else:
            print(f"未知的优化方法: {method}，使用均等权重")
            weights = np.ones(len(self.models)) / len(self.models)
            self.weights = weights
            return weights
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        评估集成模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            output_dir: 输出目录
            
        返回:
            metrics: 评估指标
        """
        # 获取每个单独模型的预测
        individual_preds = self.predict_proba(X_test)
        
        # 获取集成预测
        if self.weights is not None:
            ensemble_pred = self.weighted_average(X_test)
            ensemble_name = "加权集成"
        else:
            ensemble_pred = self.simple_average(X_test)
            ensemble_name = "简单集成"
        
        # 创建评估器
        evaluator = ModelEvaluator(y_test, ensemble_pred)
        
        # 打印指标
        print(f"\n{ensemble_name}评估结果:")
        evaluator.print_metrics()
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        
        # 绘制每个单独模型的ROC曲线
        for i, (name, pred) in enumerate(zip(self.model_names, individual_preds)):
            fpr, tpr, _ = evaluator.calculate_roc_curve(y_test, pred)
            auc_score = roc_auc_score(y_test, pred)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')
        
        # 绘制集成模型的ROC曲线
        fpr, tpr, _ = evaluator.calculate_roc_curve(y_test, ensemble_pred)
        auc_score = roc_auc_score(y_test, ensemble_pred)
        plt.plot(fpr, tpr, 'k--', linewidth=3, label=f'{ensemble_name} (AUC = {auc_score:.4f})')
        
        # 添加随机基线
        plt.plot([0, 1], [0, 1], 'r--', label='随机')
        
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线比较')
        plt.legend(loc='lower right')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            roc_path = os.path.join(output_dir, "ensemble_roc.png")
            plt.savefig(roc_path)
            print(f"ROC曲线已保存至: {roc_path}")
        
        plt.show()
        plt.close()
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        
        # 绘制每个单独模型的PR曲线
        for i, (name, pred) in enumerate(zip(self.model_names, individual_preds)):
            precision, recall, _ = precision_recall_curve(y_test, pred)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f})')
        
        # 绘制集成模型的PR曲线
        precision, recall, _ = precision_recall_curve(y_test, ensemble_pred)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, 'k--', linewidth=3, label=f'{ensemble_name} (AUC = {pr_auc:.4f})')
        
        # 添加基线
        baseline = np.sum(y_test) / len(y_test)
        plt.axhline(baseline, linestyle='--', color='r', label=f'基线 = {baseline:.4f}')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('PR曲线比较')
        plt.legend(loc='lower left')
        
        if output_dir:
            pr_path = os.path.join(output_dir, "ensemble_pr.png")
            plt.savefig(pr_path)
            print(f"PR曲线已保存至: {pr_path}")
        
        plt.show()
        plt.close()
        
        return evaluator.get_metrics()