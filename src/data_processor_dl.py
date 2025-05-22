"""
深度学习数据处理模块：为深度学习模型准备数据
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def prepare_dl_data(df, target_col='aki_label', test_size=0.2, val_size=0.15, random_state=42):
    """
    准备深度学习模型的数据
    
    参数:
        df: 包含特征和目标的DataFrame
        target_col: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例(占训练集的比例)
        random_state: 随机种子
        
    返回:
        X_train, X_val, X_test: 训练、验证和测试特征
        y_train, y_val, y_test: 训练、验证和测试标签
        feature_names: 特征名称列表
    """
    # 移除无用特征和导致数据泄露的特征
    cols_to_drop = [
        # ID和时间列
        'subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 
        # 目标变量
        target_col,
        # 防止数据泄露的特征
        'aki_by_creatinine', 'aki_by_urine',
        # 肌酐相关特征
        'creatinine_subsequent', 'creatinine_delta', 'creatinine_ratio'
    ]
    
    # 确保所有列名存在于DataFrame中
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    feature_names = feature_cols.copy()
    
    # 提取特征和目标
    X = df[feature_cols]
    y = df[target_col]
    
    # 分割数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    # 识别分类特征和数值特征
    categorical_cols = []
    numeric_cols = []
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"分类特征: {categorical_cols}")
    print(f"数值特征: {numeric_cols}")
    
    # 创建处理管道
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # 数值特征处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 分类特征处理管道
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 组合处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # 拟合和转换数据
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 获取转换后的特征名称
    if len(categorical_cols) > 0:
        # 获取OneHotEncoder生成的特征名称
        cat_feature_names = []
        for i, col in enumerate(categorical_cols):
            encoder = preprocessor.transformers_[1][1].named_steps['onehot']
            categories = encoder.categories_[i]
            for cat in categories:
                cat_feature_names.append(f"{col}_{cat}")
        
        # 组合数值特征和分类特征名称
        transformed_feature_names = numeric_cols + cat_feature_names
    else:
        transformed_feature_names = numeric_cols
    
    print(f"数据准备完成:")
    print(f"训练集: {X_train_scaled.shape}, {y_train.shape}")
    print(f"验证集: {X_val_scaled.shape}, {y_val.shape}")
    print(f"测试集: {X_test_scaled.shape}, {y_test.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, transformed_feature_names


def prepare_time_series_data(df, time_steps=48, target_col='aki_label', test_size=0.2, val_size=0.15, random_state=42):
    """
    准备时序数据用于深度学习模型
    
    参数:
        df: 包含特征和目标的DataFrame
        time_steps: 时间步数量
        target_col: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例(占训练集的比例)
        random_state: 随机种子
        
    返回:
        X_train, X_val, X_test: 训练、验证和测试特征 (3D形状: [样本数, 时间步, 特征数])
        y_train, y_val, y_test: 训练、验证和测试标签
        feature_names: 特征名称列表
    """
    # 这里需要根据您的数据结构进行调整
    # 假设df中已经包含了时序信息，例如每个患者在不同时间点的测量值
    
    # 示例：将数据重塑为3D形状 [样本数, 时间步, 特征数]
    # 实际实现需要根据您的数据结构进行调整
    
    print("注意: 时序数据准备函数需要根据实际数据结构进行调整")
    
    # 这里是一个简化的示例，假设我们有48小时的数据，每小时一个测量值
    # 实际上，您需要根据数据的实际结构来实现这个函数
    
    # 移除无用特征和导致数据泄露的特征
    cols_to_drop = [
        # ID和时间列
        'subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 
        # 目标变量
        target_col,
        # 防止数据泄露的特征
        'aki_by_creatinine', 'aki_by_urine',
        # 肌酐相关特征
        'creatinine_subsequent', 'creatinine_delta', 'creatinine_ratio'
    ]
    
    # 确保所有列名存在于DataFrame中
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    feature_names = feature_cols.copy()
    
    # 提取特征和目标
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 假设我们可以将数据重塑为时序形式
    # 这里只是一个示例，实际上您需要根据数据结构进行调整
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # 创建一个简单的时序数据示例
    # 在实际应用中，您需要根据实际数据结构来构建时序数据
    X_time_series = np.zeros((n_samples, time_steps, n_features))
    
    # 填充时序数据 - 这里只是一个示例
    # 在实际应用中，您需要使用真实的时序数据
    for i in range(n_samples):
        for t in range(time_steps):
            # 这里我们简单地复制相同的特征值，并添加一些随机噪声
            # 在实际应用中，您应该使用真实的时序数据
            X_time_series[i, t, :] = X[i, :] + np.random.normal(0, 0.01, n_features)
    
    # 分割数据集
    indices = np.arange(n_samples)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=random_state, stratify=y[train_val_indices]
    )
    
    X_train = X_time_series[train_indices]
    X_val = X_time_series[val_indices]
    X_test = X_time_series[test_indices]
    
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]
    
    print(f"时序数据准备完成:")
    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"验证集: {X_val.shape}, {y_val.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def handle_class_imbalance(X_train, y_train, method='smote', random_state=42):
    """
    处理类别不平衡问题
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        method: 处理方法 ('smote', 'adasyn', 'random_over', 'random_under')
        random_state: 随机种子
        
    返回:
        X_resampled: 重采样后的特征
        y_resampled: 重采样后的标签
    """
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            resampler = ADASYN(random_state=random_state)
        elif method == 'random_over':
            from imblearn.over_sampling import RandomOverSampler
            resampler = RandomOverSampler(random_state=random_state)
        elif method == 'random_under':
            from imblearn.under_sampling import RandomUnderSampler
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            print(f"未知的重采样方法: {method}，返回原始数据")
            return X_train, y_train
        
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        print(f"类别不平衡处理完成 (方法: {method}):")
        print(f"原始类别分布: {np.bincount(y_train)}")
        print(f"重采样后类别分布: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
        
    except ImportError:
        print("警告: imblearn库未安装，无法进行类别不平衡处理")
        return X_train, y_train