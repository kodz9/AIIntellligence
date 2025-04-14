"""
模型评估模块：用于评估AKI预测模型的性能，生成评估指标和可视化结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, confusion_matrix, classification_report
)


class ModelEvaluator:
    """模型评估类"""
    
    def __init__(self, y_true, y_pred_proba, threshold=0.5):
        """
        初始化评估器
        
        参数:
            y_true: 真实标签
            y_pred_proba: 预测概率
            threshold: 分类阈值
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        self.y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 计算评估指标
        self.auroc = roc_auc_score(y_true, y_pred_proba)
        self.auprc = average_precision_score(y_true, y_pred_proba)
        self.f1 = f1_score(y_true, self.y_pred)
        
        # 计算ROC和PR曲线
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_pred_proba)
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"AUROC: {self.auroc:.4f}")
        print(f"AUPRC: {self.auprc:.4f}")
        print(f"F1 Score: {self.f1:.4f}")
        print("\n分类报告:")
        print(classification_report(self.y_true, self.y_pred))
    
    def plot_roc_curve(self, figsize=(10, 6), save_path=None):
        """
        绘制ROC曲线
        
        参数:
            figsize: 图表大小
            save_path: 图表保存路径
        """
        plt.figure(figsize=figsize)
        
        # 绘制ROC曲线
        plt.plot(self.fpr, self.tpr, label=f'AUROC = {self.auroc:.4f}')
        
        # 添加随机猜测基线
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # 添加图表标签和图例
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲线')
        plt.legend(loc='lower right')
        
        # 保存图表
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"ROC曲线已保存至 {save_path}")
        
        plt.show()
    
    def plot_pr_curve(self, figsize=(10, 6), save_path=None):
        """
        绘制精确率-召回率曲线
        
        参数:
            figsize: 图表大小
            save_path: 图表保存路径
        """
        plt.figure(figsize=figsize)
        
        # 绘制PR曲线
        plt.plot(self.recall, self.precision, label=f'AUPRC = {self.auprc:.4f}')
        
        # 添加随机猜测基线 (正例比例)
        baseline = np.sum(self.y_true) / len(self.y_true)
        plt.axhline(baseline, linestyle='--', color='gray', label=f'Baseline = {baseline:.2f}')
        
        # 添加图表标签和图例
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall曲线')
        plt.legend(loc='lower left')
        
        # 保存图表
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"PR曲线已保存至 {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """
        绘制混淆矩阵
        
        参数:
            figsize: 图表大小
            save_path: 图表保存路径
        """
        plt.figure(figsize=figsize)
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # 绘制混淆矩阵热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['无AKI', '有AKI'],
                    yticklabels=['无AKI', '有AKI'])
        
        # 添加图表标签
        plt.xlabel('预测')
        plt.ylabel('真实')
        plt.title(f'混淆矩阵 (阈值 = {self.threshold:.2f})')
        
        # 保存图表
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"混淆矩阵已保存至 {save_path}")
        
        plt.show()
    
    def plot_threshold_metrics(self, figsize=(12, 8), save_path=None):
        """
        绘制不同阈值下的指标变化
        
        参数:
            figsize: 图表大小
            save_path: 图表保存路径
        """
        plt.figure(figsize=figsize)
        
        # 计算不同阈值下的指标
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (self.y_pred_proba >= thresh).astype(int)
            
            # 处理极端情况
            with np.errstate(divide='ignore', invalid='ignore'):
                prec = np.sum((y_pred_thresh == 1) & (self.y_true == 1)) / np.sum(y_pred_thresh == 1)
                if np.isnan(prec):
                    prec = 1.0
                
                rec = np.sum((y_pred_thresh == 1) & (self.y_true == 1)) / np.sum(self.y_true == 1)
                
                # 计算F1
                if prec + rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                else:
                    f1 = 0.0
            
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
        
        # 绘制指标随阈值变化曲线
        plt.plot(thresholds, precision_scores, label='Precision')
        plt.plot(thresholds, recall_scores, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score')
        
        # 标记当前使用的阈值
        plt.axvline(self.threshold, color='gray', linestyle='--', 
                   label=f'Current Threshold = {self.threshold:.2f}')
        
        # 找到F1最大值
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal Threshold = {optimal_threshold:.2f}')
        
        # 添加图表标签和图例
        plt.xlabel('分类阈值')
        plt.ylabel('指标值')
        plt.title('不同阈值下的模型指标')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"阈值指标图已保存至 {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def find_optimal_threshold(self, metric='f1'):
        """
        寻找最优阈值
        
        参数:
            metric: 优化指标，'f1'或'youden'(约登指数)
            
        返回:
            optimal_threshold: 最优阈值
        """
        if metric == 'f1':
            # 计算不同阈值的F1
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            
            for thresh in thresholds:
                y_pred_thresh = (self.y_pred_proba >= thresh).astype(int)
                f1 = f1_score(self.y_true, y_pred_thresh)
                f1_scores.append(f1)
            
            # 找到F1最大的阈值
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
        elif metric == 'youden':
            # 使用约登指数(敏感性+特异性-1)
            optimal_idx = np.argmax(self.tpr - self.fpr)
            optimal_threshold = self.roc_thresholds[optimal_idx]
        
        else:
            raise ValueError("不支持的指标类型，请使用'f1'或'youden'")
        
        return optimal_threshold
    
    def compare_models(self, model_probas, model_names, figsize=(12, 10), save_path=None):
        """
        比较多个模型的ROC和PR曲线
        
        参数:
            model_probas: 多个模型的预测概率列表
            model_names: 模型名称列表
            figsize: 图表大小
            save_path: 图表保存路径
        """
        plt.figure(figsize=figsize)
        
        # 创建ROC曲线子图
        plt.subplot(1, 2, 1)
        
        # 添加当前模型的ROC曲线
        plt.plot(self.fpr, self.tpr, label=f'当前模型 (AUROC={self.auroc:.4f})')
        
        # 添加其他模型的ROC曲线
        for i, (proba, name) in enumerate(zip(model_probas, model_names)):
            fpr, tpr, _ = roc_curve(self.y_true, proba)
            auroc = roc_auc_score(self.y_true, proba)
            plt.plot(fpr, tpr, label=f'{name} (AUROC={auroc:.4f})')
        
        # 添加随机猜测基线
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # 添加图表标签和图例
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲线比较')
        plt.legend(loc='lower right')
        
        # 创建PR曲线子图
        plt.subplot(1, 2, 2)
        
        # 添加当前模型的PR曲线
        plt.plot(self.recall, self.precision, label=f'当前模型 (AUPRC={self.auprc:.4f})')
        
        # 添加其他模型的PR曲线
        for i, (proba, name) in enumerate(zip(model_probas, model_names)):
            precision, recall, _ = precision_recall_curve(self.y_true, proba)
            auprc = average_precision_score(self.y_true, proba)
            plt.plot(recall, precision, label=f'{name} (AUPRC={auprc:.4f})')
        
        # 添加随机猜测基线
        baseline = np.sum(self.y_true) / len(self.y_true)
        plt.axhline(baseline, linestyle='--', color='gray', label=f'Baseline = {baseline:.2f}')
        
        # 添加图表标签和图例
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall曲线比较')
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"模型比较图已保存至 {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir, prefix=''):
        """
        生成完整评估报告，包括所有指标和图表
        
        参数:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存评估指标
        metrics = {
            'AUROC': self.auroc,
            'AUPRC': self.auprc,
            'F1': self.f1,
            'Threshold': self.threshold,
            'Optimal_Threshold_F1': self.find_optimal_threshold('f1'),
            'Optimal_Threshold_Youden': self.find_optimal_threshold('youden')
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(output_dir, f'{prefix}metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"评估指标已保存至 {metrics_path}")
        
        # 保存分类报告
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, f'{prefix}classification_report.csv')
        report_df.to_csv(report_path)
        print(f"分类报告已保存至 {report_path}")
        
        # 保存ROC曲线
        roc_path = os.path.join(output_dir, f'{prefix}roc_curve.png')
        self.plot_roc_curve(save_path=roc_path)
        
        # 保存PR曲线
        pr_path = os.path.join(output_dir, f'{prefix}pr_curve.png')
        self.plot_pr_curve(save_path=pr_path)
        
        # 保存混淆矩阵
        cm_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
        self.plot_confusion_matrix(save_path=cm_path)
        
        # 保存阈值指标图
        threshold_path = os.path.join(output_dir, f'{prefix}threshold_metrics.png')
        self.plot_threshold_metrics(save_path=threshold_path)
        
        print(f"评估报告已生成，所有文件已保存至 {output_dir}")


if __name__ == "__main__":
    # 测试评估模块
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练示例模型
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # 预测概率
    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    # 评估逻辑回归模型
    evaluator = ModelEvaluator(y_test, lr_proba)
    evaluator.print_metrics()
    
    # 绘制ROC和PR曲线
    evaluator.plot_roc_curve()
    evaluator.plot_pr_curve()
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix()
    
    # 比较模型
    evaluator.compare_models([rf_proba], ['RandomForest']) 