"""
数据处理模块：负责加载、清洗和预处理ICU患者数据，提取急性肾损伤(AKI)预测所需的特征
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path):
    """
    加载原始数据
    
    参数:
        data_path: 数据文件路径
        
    返回:
        pandas.DataFrame: 加载的数据
    """
    print(f"加载数据: {data_path}")
    return pd.read_csv(data_path)


def verify_aki_label(df):
    """
    验证并确认AKI标签 - 基于KDIGO标准
    
    KDIGO标准定义的AKI:
    - 血清肌酐(SCr)在48小时内升高≥0.3mg/dL(≥26.5μmol/L); 或
    - SCr升高至基线的1.5倍以上(已知或推测在7天内发生); 或
    - 尿量<0.5mL/kg/h，持续6小时以上
    
    我们的数据中已包含:
    - aki_by_creatinine: 基于肌酐变化的AKI判断
    - aki_by_urine: 基于尿量的AKI判断
    - aki_label: 综合判断的AKI标签
    
    参数:
        df: 包含患者数据和AKI标签的DataFrame
        
    返回:
        确认了AKI标签的DataFrame
    """
    print("验证AKI标签...")
    
    # 检查是否已有AKI标签
    if 'aki_label' not in df.columns:
        print("警告: 数据中没有找到aki_label字段，请确认SQL查询是否正确执行")
        return df
    
    # 填充可能的缺失值
    if df['aki_label'].isna().any():
        print(f"注意: 有 {df['aki_label'].isna().sum()} 个样本的AKI标签缺失，填充为0(无AKI)")
        df['aki_label'] = df['aki_label'].fillna(0)
    
    # 确保AKI标签是整数类型
    df['aki_label'] = df['aki_label'].astype(int)
    
    # 计算AKI发生率
    aki_rate = df['aki_label'].mean()
    print(f"AKI发生率: {aki_rate:.2f}")
    
    # 检查肌酐相关指标与标签的一致性
    if 'creatinine_baseline' in df.columns and 'creatinine_subsequent' in df.columns:
        has_both_cr = (~df['creatinine_baseline'].isna()) & (~df['creatinine_subsequent'].isna())
        valid_samples = df[has_both_cr].shape[0]
        
        print(f"有效肌酐样本数: {valid_samples} ({valid_samples/df.shape[0]:.2%})")
        
        if 'aki_by_creatinine' in df.columns:
            consistency = (df[has_both_cr]['aki_by_creatinine'] == df[has_both_cr]['aki_label']).mean()
            print(f"肌酐判断与最终标签一致性: {consistency:.2%}")
    
    return df


def handle_missing_values(df):
    """
    处理缺失值
    
    参数:
        df: 包含患者数据的DataFrame
        
    返回:
        处理后的DataFrame
    """
    print("处理缺失值...")
    # 显示缺失率
    missing_rates = df.isna().mean().sort_values(ascending=False)
    print("缺失率:")
    print(missing_rates[missing_rates > 0])
    
    # 对于缺失率超过80%的特征，可以考虑直接删除
    high_missing_cols = missing_rates[missing_rates > 0.8].index.tolist()
    if high_missing_cols:
        print(f"删除高缺失率特征: {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)
    
    return df


def feature_engineering(df):
    """
    特征工程 - 创建新特征，避免数据泄露
    
    参数:
        df: 包含患者数据的DataFrame
        
    返回:
        增加新特征后的DataFrame
    """
    print("执行特征工程...")
    
    # 计算BUN/肌酐比值 (只使用基线肌酐值，避免数据泄露)
    if 'bun' in df.columns and 'creatinine_baseline' in df.columns:
        mask = (~df['bun'].isna()) & (~df['creatinine_baseline'].isna()) & (df['creatinine_baseline'] > 0)
        df.loc[mask, 'bun_cr_baseline_ratio'] = df.loc[mask, 'bun'] / df.loc[mask, 'creatinine_baseline']
    
    # 年龄分组
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 30, 50, 70, 200], 
                                labels=['young', 'middle', 'senior', 'elderly'])
    
    # 检查尿量是否异常(低尿量可能是AKI的指标)
    if 'urineoutput' in df.columns and 'urineoutput_24h' not in df.columns:
        # 假设体重平均70kg，正常尿量应该≥0.5mL/kg/h，即≥840mL/24h
        df['low_urine_output'] = (df['urineoutput'] < 840).astype(int)
    
    # 注意：不再计算肌酐变化相关的特征（creatinine_delta, creatinine_ratio）
    # 因为这些直接反映了AKI的定义，会导致严重的数据泄露
    
    print("完成特征工程，避免数据泄露")
    
    return df


def prepare_features_and_target(df, target_col='aki_label'):
    """
    准备特征和目标变量
    
    参数:
        df: 包含患者数据的DataFrame
        target_col: 目标变量列名
        
    返回:
        X: 特征矩阵
        y: 目标变量
        preprocessor: 预处理流水线
    """
    print("准备特征和目标变量...")
    
    # 移除无用特征和导致数据泄露的特征
    cols_to_drop = [
        # ID和时间列
        'subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 
        # 目标变量
        target_col,
        # 防止数据泄露的特征（直接相关于AKI诊断的指标）
        'aki_by_creatinine', 'aki_by_urine',
        # 肌酐相关特征（这些是AKI诊断的直接依据，不应作为预测特征）
        'creatinine_subsequent', 'creatinine_delta', 'creatinine_ratio',
        # 如果有其他可能导致数据泄露的特征，也应添加到此处
    ]
    
    # 确保所有列名存在于DataFrame中
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    print("移除以下特征以防止数据泄露:")
    print(cols_to_drop)
    
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 识别数值和分类特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"数值特征 ({len(numeric_features)}): {numeric_features}")
    print(f"分类特征 ({len(categorical_features)}): {categorical_features}")
    
    # 创建预处理流水线
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor


def calculate_feature_correlation(df, target_col='aki_label', output_dir=None):
    """
    计算特征与目标变量的相关性，并生成相关性热图
    
    参数:
        df: 包含特征和目标变量的DataFrame
        target_col: 目标变量列名
        output_dir: 输出目录，如果提供则保存相关性矩阵和热图
        
    返回:
        corr_with_target: 特征与目标变量的相关性Series，按绝对值排序
    """
    print("计算特征相关性...")
    
    # 选择数值型特征
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # 确保目标变量在数据集中
    if target_col not in numeric_df.columns:
        print(f"警告: 目标变量 {target_col} 不在数值型特征中，将其添加到相关性计算中")
        if target_col in df.columns:
            numeric_df[target_col] = df[target_col]
        else:
            print(f"错误: 目标变量 {target_col} 不在数据集中")
            return None
    
    # 计算相关性矩阵
    corr_matrix = numeric_df.corr()
    
    # 获取与目标变量的相关性，并按绝对值排序
    corr_with_target = corr_matrix[target_col].drop(target_col)
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)
    
    # 输出前10个相关性最强的特征
    print(f"\n与{target_col}相关性最强的10个特征:")
    for feature, corr in corr_with_target.head(10).items():
        # 获取原始相关系数（有正负）
        orig_corr = corr_matrix[target_col][feature]
        print(f"{feature}: {orig_corr:.4f}")
    
    # 如果提供了输出目录，保存相关性矩阵和热图
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存与目标变量的相关性
        target_corr_path = os.path.join(output_dir, f"corr_with_{target_col}.csv")
        corr_with_target_df = pd.DataFrame({
            'feature': corr_with_target.index,
            'correlation': [corr_matrix[target_col][feature] for feature in corr_with_target.index],
            'abs_correlation': corr_with_target.values
        })
        corr_with_target_df.to_csv(target_corr_path, index=False)
        print(f"特征与{target_col}的相关性已保存至 {target_corr_path}")
        
        # 生成热图 - 目标变量与前15个最相关特征
        plt.figure(figsize=(12, 10))
        top_features = corr_with_target.head(15).index.tolist()
        top_features.append(target_col)
        
        # 提取子矩阵并创建热图
        corr_subset = corr_matrix.loc[top_features, top_features]
        mask = np.triu(np.ones_like(corr_subset, dtype=bool))
        
        sns.heatmap(corr_subset, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title(f"Top 15 Features Correlation with {target_col}", fontsize=14)
        plt.tight_layout()
        
        # 保存热图
        heatmap_path = os.path.join(output_dir, f"correlation_heatmap_{target_col}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"相关性热图已保存至 {heatmap_path}")
        plt.close()
        
        # 生成与目标变量相关性的条形图
        plt.figure(figsize=(12, 8))
        
        # 获取与目标的相关系数（保留符号）
        target_corrs = [corr_matrix[target_col][feature] for feature in corr_with_target.head(15).index]
        features = corr_with_target.head(15).index
        
        # 创建条形图
        bars = plt.barh(range(len(features)), target_corrs, color=['red' if c < 0 else 'blue' for c in target_corrs])
        plt.yticks(range(len(features)), features)
        plt.xlabel(f'Correlation with {target_col}')
        plt.title(f'Top 15 Features Correlation with {target_col}')
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 添加相关系数标签
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() * (1.01 if bar.get_width() > 0 else 0.99),
                bar.get_y() + bar.get_height()/2,
                f'{target_corrs[i]:.3f}',
                va='center'
            )
        
        # 保存条形图
        barplot_path = os.path.join(output_dir, f"correlation_barplot_{target_col}.png")
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        print(f"相关性条形图已保存至 {barplot_path}")
        plt.close()
    
    return corr_with_target


def process_data(raw_data_path, output_path=None):
    """
    完整的数据处理流程
    
    参数:
        raw_data_path: 原始数据路径
        output_path: 处理后数据保存路径
        
    返回:
        X_train: 训练特征
        X_test: 测试特征  
        y_train: 训练标签
        y_test: 测试标签
        preprocessor: 预处理流水线
    """
    # 加载数据
    df = load_data(raw_data_path)
    
    # 验证AKI标签 (现在直接使用SQL查询生成的标签)
    df = verify_aki_label(df)
    
    # 处理缺失值
    df = handle_missing_values(df)
    
    # 特征工程
    df = feature_engineering(df)
    
    # 计算特征相关性（如果输出路径存在，则保存相关性分析结果）
    if output_path:
        output_dir = os.path.dirname(output_path)
        calculate_feature_correlation(df, target_col='aki_label', output_dir=output_dir)
    
    # 准备特征和目标变量
    X, y, preprocessor = prepare_features_and_target(df)
    
    # 保存处理后的数据
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False)
        print(f"处理后的数据已保存至: {output_path}")
    
    # 使用时间序列分割
    tscv = TimeSeriesSplit(n_splits=5)
    # 获取最后一次分割作为训练集和测试集
    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, preprocessor


def extract_features(df, prediction_time):
    features = {}
    # 只使用prediction_time之前的数据
    past_data = df[df['time'] < prediction_time]
    # 计算特征
    features['mean_creatinine'] = past_data['creatinine'].mean()
    # ... 其他特征
    return features


def select_features(df):
    # 移除与AKI直接相关的特征
    features_to_exclude = ['aki_label', 'aki_stage', 'aki_time']
    features = [col for col in df.columns if col not in features_to_exclude]
    return features


if __name__ == "__main__":
    # 测试数据处理流程
    raw_data_path = "../data/raw/base_stay.csv"
    processed_data_path = "../data/processed/processed_data.csv"
    
    X_train, X_test, y_train, y_test, preprocessor = process_data(
        raw_data_path, processed_data_path
    )
    
    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}") 