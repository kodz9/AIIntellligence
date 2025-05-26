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


def prepare_time_series_data(df, time_col='hours_since_admission', measurement_col='measurement', 
                           value_col='valuenum', patient_id_col='subject_id', target_col='aki_label', 
                           test_size=0.2, val_size=0.15, random_state=42,
                           normalize_method='standard', handle_missing='forward_fill_mean',
                           time_window=False, diff_features=False):
    """
    准备时序数据用于深度学习模型
    
    参数:
        df: 包含特征和目标的DataFrame (base_stay_dl.csv格式)
        time_col: 时间列名
        measurement_col: 测量项目列名
        value_col: 测量值列名
        patient_id_col: 患者ID列名
        target_col: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例(占训练集的比例)
        random_state: 随机种子
        normalize_method: 标准化方法，'standard'或'robust'或'minmax'
        handle_missing: 缺失值处理方法，'forward_fill_mean'或'mean'或'median'
        time_window: 是否添加时间窗口特征
        diff_features: 是否添加差分特征
        
    返回:
        X_train, X_val, X_test: 训练、验证和测试特征 (3D形状: [样本数, 时间步, 特征数])
        y_train, y_val, y_test: 训练、验证和测试标签
        feature_names: 特征名称列表
    """
    print("开始处理时序数据...")
    
    # 获取唯一的患者ID
    patient_ids = df[patient_id_col].unique()
    print(f"数据集中共有 {len(patient_ids)} 名患者")
    
    # 获取唯一的时间点
    time_points = sorted(df[time_col].unique())
    print(f"数据集中共有 {len(time_points)} 个时间点")
    
    # 获取唯一的测量项目
    features = sorted(df[measurement_col].unique())
    print(f"数据集中共有 {len(features)} 个测量项目: {features}")
    
    # 分割患者ID为训练、验证和测试集
    # 首先获取每个患者的AKI标签
    patient_labels = {}
    for patient_id in patient_ids:
        patient_data = df[df[patient_id_col] == patient_id]
        if len(patient_data) > 0:
            patient_labels[patient_id] = patient_data[target_col].iloc[0]
    
    # 将患者ID和标签转换为数组
    patient_ids_array = np.array(list(patient_labels.keys()))
    patient_labels_array = np.array(list(patient_labels.values()))
    
    # 分层分割
    from sklearn.model_selection import train_test_split
    
    # 分割为训练+验证集和测试集
    train_val_ids, test_ids, _, _ = train_test_split(
        patient_ids_array, patient_labels_array, 
        test_size=test_size, random_state=random_state, 
        stratify=patient_labels_array
    )
    
    # 进一步分割为训练集和验证集
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids, [patient_labels[pid] for pid in train_val_ids], 
        test_size=val_size, random_state=random_state, 
        stratify=[patient_labels[pid] for pid in train_val_ids]
    )
    
    print(f"训练集: {len(train_ids)} 名患者")
    print(f"验证集: {len(val_ids)} 名患者")
    print(f"测试集: {len(test_ids)} 名患者")
    
    # 创建3D数据结构 [样本数, 时间步, 特征数]
    def create_patient_matrix(patient_id):
        patient_data = df[df[patient_id_col] == patient_id]
        
        # 创建一个空矩阵
        patient_matrix = np.zeros((len(time_points), len(features)))
        patient_matrix.fill(np.nan)  # 填充为NaN
        
        # 填充矩阵
        for _, row in patient_data.iterrows():
            time_idx = time_points.index(row[time_col])
            feature_idx = features.index(row[measurement_col])
            patient_matrix[time_idx, feature_idx] = row[value_col]
        
        return patient_matrix
    
    # 创建训练、验证和测试数据
    X_train = np.array([create_patient_matrix(pid) for pid in train_ids])
    y_train = np.array([patient_labels[pid] for pid in train_ids])
    
    X_val = np.array([create_patient_matrix(pid) for pid in val_ids])
    y_val = np.array([patient_labels[pid] for pid in val_ids])
    
    X_test = np.array([create_patient_matrix(pid) for pid in test_ids])
    y_test = np.array([patient_labels[pid] for pid in test_ids])
    
    print(f"X_train 形状: {X_train.shape}")
    print(f"X_val 形状: {X_val.shape}")
    print(f"X_test 形状: {X_test.shape}")
    
    # 处理缺失值
    print("处理缺失值...")
    
    # 对每个患者的序列进行填充
    def fill_missing_values(X, method=handle_missing):
        for i in range(X.shape[0]):
            # 对每个特征进行填充
            for j in range(X.shape[2]):
                # 提取该患者该特征的时间序列
                feature_series = pd.Series(X[i, :, j])
                
                if method == 'forward_fill_mean':
                    # 前向填充
                    feature_series = feature_series.fillna(method='ffill')
                    # 后向填充
                    feature_series = feature_series.fillna(method='bfill')
                    # 如果仍有缺失值，用该特征的平均值填充
                    if feature_series.isnull().any():
                        feature_mean = np.nanmean(X[:, :, j])
                        feature_series = feature_series.fillna(feature_mean)
                elif method == 'mean':
                    # 直接用平均值填充
                    feature_mean = np.nanmean(X[:, :, j])
                    feature_series = feature_series.fillna(feature_mean)
                elif method == 'median':
                    # 用中位数填充
                    feature_median = np.nanmedian(X[:, :, j])
                    feature_series = feature_series.fillna(feature_median)
                
                # 更新数据
                X[i, :, j] = feature_series.values
        return X
    
    X_train = fill_missing_values(X_train)
    X_val = fill_missing_values(X_val)
    X_test = fill_missing_values(X_test)
    
    # 添加时间窗口特征
    if time_window:
        print("添加时间窗口特征...")
        
        def add_window_features(X):
            # 原始特征数
            original_features = X.shape[2]
            
            # 创建新的特征矩阵，包含原始特征和窗口特征
            # 窗口特征：过去12小时的均值、最大值、最小值
            window_size = 3  # 假设每4小时一个时间点，12小时就是3个时间点
            new_features_per_original = 3  # 均值、最大值、最小值
            
            # 新特征数 = 原始特征数 * (1 + 新特征数/原始特征)
            new_feature_count = original_features * (1 + new_features_per_original)
            
            # 创建新的特征矩阵
            X_new = np.zeros((X.shape[0], X.shape[1], int(new_feature_count)))
            
            # 复制原始特征
            X_new[:, :, :original_features] = X
            
            # 对每个样本
            for i in range(X.shape[0]):
                # 对每个时间点（从窗口大小开始）
                for t in range(window_size, X.shape[1]):
                    # 获取窗口数据
                    window_data = X[i, t-window_size:t, :]
                    
                    # 计算窗口特征
                    window_mean = np.nanmean(window_data, axis=0)
                    window_max = np.nanmax(window_data, axis=0)
                    window_min = np.nanmin(window_data, axis=0)
                    
                    # 添加窗口特征
                    feature_idx = original_features
                    
                    # 添加均值特征
                    X_new[i, t, feature_idx:feature_idx+original_features] = window_mean
                    feature_idx += original_features
                    
                    # 添加最大值特征
                    X_new[i, t, feature_idx:feature_idx+original_features] = window_max
                    feature_idx += original_features
                    
                    # 添加最小值特征
                    X_new[i, t, feature_idx:feature_idx+original_features] = window_min
            
            return X_new
        
        X_train = add_window_features(X_train)
        X_val = add_window_features(X_val)
        X_test = add_window_features(X_test)
        
        # 更新特征名称
        new_features = []
        for f in features:
            new_features.append(f)
            new_features.append(f + "_mean_12h")
            new_features.append(f + "_max_12h")
            new_features.append(f + "_min_12h")
        
        features = new_features
        
        print(f"添加时间窗口特征后的形状: {X_train.shape}")
    
    # 添加差分特征
    if diff_features:
        print("添加差分特征...")
        
        def add_diff_features(X):
            # 原始特征数
            original_features = X.shape[2]
            
            # 创建新的特征矩阵，包含原始特征和差分特征
            X_new = np.zeros((X.shape[0], X.shape[1], original_features * 2))
            
            # 复制原始特征
            X_new[:, :, :original_features] = X
            
            # 对每个样本
            for i in range(X.shape[0]):
                # 对每个时间点（从第二个开始）
                for t in range(1, X.shape[1]):
                    # 计算差分
                    diff = X[i, t, :] - X[i, t-1, :]
                    
                    # 添加差分特征
                    X_new[i, t, original_features:] = diff
            
            return X_new
        
        X_train = add_diff_features(X_train)
        X_val = add_diff_features(X_val)
        X_test = add_diff_features(X_test)
        
        # 更新特征名称
        new_features = []
        for f in features:
            new_features.append(f)
            new_features.append(f + "_diff")
        
        features = new_features
        
        print(f"添加差分特征后的形状: {X_train.shape}")
    
    # 特征缩放
    print("进行特征缩放...")
    
    # 重塑数据以便于缩放
    train_samples, train_timesteps, train_features = X_train.shape
    val_samples, val_timesteps, val_features = X_val.shape
    test_samples, test_timesteps, test_features = X_test.shape
    
    X_train_reshaped = X_train.reshape(-1, train_features)
    X_val_reshaped = X_val.reshape(-1, val_features)
    X_test_reshaped = X_test.reshape(-1, test_features)
    
    # 根据选择的方法进行缩放
    if normalize_method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif normalize_method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif normalize_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    # 拟合缩放器并转换数据
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # 将数据重塑回原始形状
    X_train = X_train_scaled.reshape(train_samples, train_timesteps, train_features)
    X_val = X_val_scaled.reshape(val_samples, val_timesteps, val_features)
    X_test = X_test_scaled.reshape(test_samples, test_timesteps, test_features)
    
    print("时序数据准备完成")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, features


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