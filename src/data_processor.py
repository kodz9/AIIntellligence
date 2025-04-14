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


def extract_aki_label(df, creatinine_threshold=1.5):
    """
    提取AKI标签 - 根据KDIGO标准
    
    KDIGO标准定义的AKI:
    - 血清肌酐(SCr)在48小时内升高≥0.3mg/dL(≥26.5μmol/L); 或
    - SCr升高至基线的1.5倍以上(已知或推测在7天内发生); 或
    - 尿量<0.5mL/kg/h，持续6小时以上
    
    由于我们的数据限制，这里使用简化版本:
    - 检查肌酐是否升高超过基线的1.5倍
    
    参数:
        df: 包含患者数据的DataFrame
        creatinine_threshold: 肌酐升高倍数阈值，默认为1.5倍
        
    返回:
        带有AKI标签的DataFrame
    """
    print("提取AKI标签...")
    # 这里假设数据中已有肌酐值，如果没有，需要从其他数据源中提取
    
    # 在实际实现中，需要比较入院48小时内的基线肌酐值与后续的肌酐值
    # 由于数据示例中只有单一肌酐值，我们临时使用随机生成的标签用于示范
    
    # 实际项目中应该:
    # 1. 从labevents.csv中获取患者48小时内的肌酐值作为基线
    # 2. 从labevents.csv中获取48-72小时内的肌酐值
    # 3. 比较是否达到AKI标准
    
    # 临时模拟标签 (0=无AKI，1=有AKI)
    np.random.seed(42)  # 固定随机数种子以便复现
    
    # 标记肌酐值超过1.0的患者更可能出现AKI (只是为了演示)
    has_creatinine = ~df['creatinine'].isna()
    prob_aki = np.where(has_creatinine & (df['creatinine'] > 1.0), 0.6, 0.2)
    
    df['aki_label'] = np.random.binomial(1, prob_aki)
    
    print(f"AKI发生率: {df['aki_label'].mean():.2f}")
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
    特征工程 - 创建新特征
    
    参数:
        df: 包含患者数据的DataFrame
        
    返回:
        增加新特征后的DataFrame
    """
    print("执行特征工程...")
    
    # 计算BUN/肌酐比值 (肾功能指标)
    if 'bun' in df.columns and 'creatinine' in df.columns:
        df['bun_creatinine_ratio'] = df['bun'] / df['creatinine']
    
    # 年龄分组
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 30, 50, 70, 200], 
                                labels=['young', 'middle', 'senior', 'elderly'])
    
    # 检查尿量是否异常(低尿量可能是AKI的指标)
    if 'urineoutput' in df.columns:
        # 假设体重平均70kg，正常尿量应该≥0.5mL/kg/h，即≥840mL/24h
        df['low_urine_output'] = (df['urineoutput'] < 840).astype(int)
    
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
    
    # 移除无用特征
    cols_to_drop = ['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', target_col]
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
    
    # 提取AKI标签
    df = extract_aki_label(df)
    
    # 处理缺失值
    df = handle_missing_values(df)
    
    # 特征工程
    df = feature_engineering(df)
    
    # 准备特征和目标变量
    X, y, preprocessor = prepare_features_and_target(df)
    
    # 保存处理后的数据
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False)
        print(f"处理后的数据已保存至: {output_path}")
    
    # 分割训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # 测试数据处理流程
    raw_data_path = "../data/raw/base_stay.csv"
    processed_data_path = "../data/processed/processed_data.csv"
    
    X_train, X_test, y_train, y_test, preprocessor = process_data(
        raw_data_path, processed_data_path
    )
    
    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}") 