"""
Model Evaluation Module: For evaluating the performance of AKI prediction models, generating evaluation metrics and visualizations
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
    """Model Evaluation Class"""
    
    def __init__(self, y_true, y_pred_proba, threshold=0.5):
        """
        Initialize the evaluator
        
        Parameters:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        self.y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate evaluation metrics
        self.auroc = roc_auc_score(y_true, y_pred_proba)
        self.auprc = average_precision_score(y_true, y_pred_proba)
        self.f1 = f1_score(y_true, self.y_pred)
        
        # Calculate ROC and PR curves
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_pred_proba)
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    def print_metrics(self):
        """Print evaluation metrics"""
        print("\nModel Evaluation Metrics:")
        print(f"AUROC: {self.auroc:.4f}")
        print(f"AUPRC: {self.auprc:.4f}")
        print(f"F1 Score: {self.f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred))
    
    def plot_roc_curve(self, figsize=(10, 6), save_path=None):
        """
        Plot ROC curve
        
        Parameters:
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(self.fpr, self.tpr, label=f'AUROC = {self.auroc:.4f}')
        
        # Add random baseline
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        # Save figure
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_pr_curve(self, figsize=(10, 6), save_path=None):
        """
        Plot Precision-Recall curve
        
        Parameters:
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Plot PR curve
        plt.plot(self.recall, self.precision, label=f'AUPRC = {self.auprc:.4f}')
        
        # Add random baseline (positive class ratio)
        baseline = np.sum(self.y_true) / len(self.y_true)
        plt.axhline(baseline, linestyle='--', color='gray', label=f'Baseline = {baseline:.2f}')
        
        # Add labels and legend
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        # Save figure
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"PR curve saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Plot confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No AKI', 'AKI'],
                    yticklabels=['No AKI', 'AKI'])
        
        # Add labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Threshold = {self.threshold:.2f})')
        
        # Save figure
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_metrics(self, figsize=(12, 8), save_path=None):
        """
        Plot metrics for different thresholds
        
        Parameters:
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Calculate metrics for different thresholds
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (self.y_pred_proba >= thresh).astype(int)
            
            # Handle extreme cases
            with np.errstate(divide='ignore', invalid='ignore'):
                prec = np.sum((y_pred_thresh == 1) & (self.y_true == 1)) / np.sum(y_pred_thresh == 1)
                if np.isnan(prec):
                    prec = 1.0
                
                rec = np.sum((y_pred_thresh == 1) & (self.y_true == 1)) / np.sum(self.y_true == 1)
                
                # Calculate F1
                if prec + rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                else:
                    f1 = 0.0
            
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
        
        # Plot metrics across thresholds
        plt.plot(thresholds, precision_scores, label='Precision')
        plt.plot(thresholds, recall_scores, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score')
        
        # Mark current threshold
        plt.axvline(self.threshold, color='gray', linestyle='--', 
                   label=f'Current Threshold = {self.threshold:.2f}')
        
        # Find optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal Threshold = {optimal_threshold:.2f}')
        
        # Add labels and legend
        plt.xlabel('Classification Threshold')
        plt.ylabel('Metric Value')
        plt.title('Model Metrics for Different Thresholds')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"Threshold metrics plot saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def find_optimal_threshold(self, metric='f1'):
        """
        Find the optimal threshold
        
        Parameters:
            metric: Optimization metric, 'f1' or 'youden' (Youden's index)
            
        Returns:
            optimal_threshold: The optimal threshold
        """
        if metric == 'f1':
            # Calculate F1 for different thresholds
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            
            for thresh in thresholds:
                y_pred_thresh = (self.y_pred_proba >= thresh).astype(int)
                f1 = f1_score(self.y_true, y_pred_thresh)
                f1_scores.append(f1)
            
            # Find threshold with max F1
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
        elif metric == 'youden':
            # Use Youden's index (sensitivity + specificity - 1)
            optimal_idx = np.argmax(self.tpr - self.fpr)
            optimal_threshold = self.roc_thresholds[optimal_idx]
        
        else:
            raise ValueError("Unsupported metric type, please use 'f1' or 'youden'")
        
        return optimal_threshold
    
    def compare_models(self, model_probas, model_names, figsize=(12, 10), save_path=None):
        """
        Compare ROC and PR curves of multiple models
        
        Parameters:
            model_probas: List of predicted probabilities for multiple models
            model_names: List of model names
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Create ROC curve subplot
        plt.subplot(1, 2, 1)
        
        # Add current model's ROC curve
        plt.plot(self.fpr, self.tpr, label=f'Current Model (AUROC={self.auroc:.4f})')
        
        # Add other models' ROC curves
        for i, (proba, name) in enumerate(zip(model_probas, model_names)):
            fpr, tpr, _ = roc_curve(self.y_true, proba)
            auroc = roc_auc_score(self.y_true, proba)
            plt.plot(fpr, tpr, label=f'{name} (AUROC={auroc:.4f})')
        
        # Add random baseline
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        
        # Create PR curve subplot
        plt.subplot(1, 2, 2)
        
        # Add current model's PR curve
        plt.plot(self.recall, self.precision, label=f'Current Model (AUPRC={self.auprc:.4f})')
        
        # Add other models' PR curves
        for i, (proba, name) in enumerate(zip(model_probas, model_names)):
            precision, recall, _ = precision_recall_curve(self.y_true, proba)
            auprc = average_precision_score(self.y_true, proba)
            plt.plot(recall, precision, label=f'{name} (AUPRC={auprc:.4f})')
        
        # Add random baseline
        baseline = np.sum(self.y_true) / len(self.y_true)
        plt.axhline(baseline, linestyle='--', color='gray', label=f'Baseline = {baseline:.2f}')
        
        # Add labels and legend
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir, prefix=''):
        """
        Generate complete evaluation report, including all metrics and plots
        
        Parameters:
            output_dir: Output directory
            prefix: File name prefix
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save evaluation metrics
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
        print(f"Evaluation metrics saved to {metrics_path}")
        
        # Save classification report
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, f'{prefix}classification_report.csv')
        report_df.to_csv(report_path)
        print(f"Classification report saved to {report_path}")
        
        # Save ROC curve
        roc_path = os.path.join(output_dir, f'{prefix}roc_curve.png')
        self.plot_roc_curve(save_path=roc_path)
        
        # Save PR curve
        pr_path = os.path.join(output_dir, f'{prefix}pr_curve.png')
        self.plot_pr_curve(save_path=pr_path)
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
        self.plot_confusion_matrix(save_path=cm_path)
        
        # Save threshold metrics plot
        threshold_path = os.path.join(output_dir, f'{prefix}threshold_metrics.png')
        self.plot_threshold_metrics(save_path=threshold_path)
        
        print(f"Evaluation report generated, all files saved to {output_dir}")


def plot_feature_importance(model, feature_names, output_path=None, top_n=20):
    """
    计算并可视化模型的特征重要性
    
    参数:
        model: 训练好的模型
        feature_names: 特征名称列表
        output_path: 输出图像路径
        top_n: 显示前N个最重要的特征
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 获取模型类型
    model_type = type(model).__name__
    
    # 处理模型可能是pipeline的情况
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        clf = model.named_steps['classifier']
        model_type = type(clf).__name__
    else:
        clf = model
    
    # 获取特征重要性
    if hasattr(clf, 'feature_importances_'):
        # 随机森林、梯度提升树、XGBoost等基于树的模型
        importances = clf.feature_importances_
        
        # 检查特征名称长度是否匹配
        if len(importances) != len(feature_names):
            print(f"警告: 特征重要性维度 ({len(importances)}) 与特征名称数量 ({len(feature_names)}) 不匹配")
            # 如果不匹配，可能是因为预处理转换了特征（如独热编码）
            print("使用索引作为特征名称")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 排序并获取前N个特征
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        top_features = feature_importance.head(top_n)
        
        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_type}')
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存至 {output_path}")
            plt.close()
        else:
            plt.show()
            
        return feature_importance
        
    elif hasattr(clf, 'coef_'):
        # 逻辑回归等线性模型
        if len(clf.coef_.shape) > 1:
            # 多类分类问题
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            # 二分类问题
            importances = np.abs(clf.coef_[0])
        
        # 检查特征名称长度是否匹配
        if len(importances) != len(feature_names):
            print(f"警告: 系数维度 ({len(importances)}) 与特征名称数量 ({len(feature_names)}) 不匹配")
            # 如果不匹配，可能是因为预处理转换了特征（如独热编码）
            print("使用索引作为特征名称")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'coefficient': clf.coef_[0] if len(clf.coef_.shape) == 2 else clf.coef_
        })
        
        # 排序并获取前N个特征
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        top_features = feature_importance.head(top_n)
        
        # 绘制系数图，保留符号信息
        plt.figure(figsize=(12, 8))
        colors = ['red' if c < 0 else 'blue' for c in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], align='center', color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Coefficients - {model_type}')
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"系数图已保存至 {output_path}")
            plt.close()
        else:
            plt.show()
            
        return feature_importance
    
    else:
        print(f"警告: 模型 {model_type} 不支持特征重要性计算")
        return None


if __name__ == "__main__":
    # Test evaluation module
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate example data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train example models
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Predict probabilities
    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    # Evaluate logistic regression model
    evaluator = ModelEvaluator(y_test, lr_proba)
    evaluator.print_metrics()
    
    # Plot ROC and PR curves
    evaluator.plot_roc_curve()
    evaluator.plot_pr_curve()
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix()
    
    # Compare models
    evaluator.compare_models([rf_proba], ['RandomForest']) 