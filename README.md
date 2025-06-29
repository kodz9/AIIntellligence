# ICU患者急性肾损伤(AKI)预测系统

本项目旨在基于ICU患者入院最初48小时的数据，预测患者在接下来的24-72小时内是否会发展为急性肾损伤(AKI)。该预测系统可以帮助医生提前干预，防止患者病情进一步恶化。
项目代码在deeperlearning分支下

## 📋 项目简介

急性肾损伤(AKI)是ICU患者的常见并发症，可能导致严重的健康问题甚至死亡。通过机器学习和深度学习技术，本系统利用患者的生理参数、实验室检查结果、人口统计学信息等多维度数据，构建预测模型，实现对AKI的早期预警。

### 🏆 主要特点

- 支持多种机器学习模型(逻辑回归、随机森林、梯度提升树、XGBoost)
- 实现LSTM+Attention深度学习模型处理时序数据
- 完整的数据预处理和特征工程流程
- 全面的模型评估和可视化
- 模型比较和自动选择最优模型

## 🔧 项目架构

```
ICU_AKI_Prediction/
│
├── data/               # 数据目录
│   ├── raw/            # 原始数据
│   └── processed/      # 处理后的数据
│
├── src/                # 源代码
│   ├── data_processor.py     # 数据处理模块
│   ├── model_trainer.py      # 模型训练模块
│   └── model_evaluator.py    # 模型评估模块
│
├── output/             # 输出目录
│   ├── models/         # 保存训练好的模型
│   ├── reports/        # 评估报告和指标
│   └── plots/          # 可视化图表
│
├── main.py             # 主程序
└── README.md           # 项目说明文档
```

## 📝 数据说明

本项目使用的主要数据包括：

1. **生理参数**：尿量、心率、收缩压、呼吸频率、体温等
2. **实验室检查**：肌酐(creatinine)、尿素氮(BUN)、电解质(钠、钾、钙)、白细胞计数(WBC)、红细胞压积(HCT)等
3. **人口统计学**：年龄、性别等
4. **治疗措施**：是否使用利尿剂、是否接受静脉补液等

数据来源主要为MIMIC-III数据库相关表格(chartevents.csv、labevents.csv和admissions.csv)。

## 💡 AKI定义

本项目采用KDIGO(肾脏疾病改善全球组织)标准定义AKI，主要标准包括：

- 血清肌酐(SCr)在48小时内升高≥0.3mg/dL(≥26.5μmol/L)
- SCr升高至基线的1.5倍以上(已知或推测在7天内发生)
- 尿量<0.5mL/kg/h，持续6小时以上

## 🔍 未来改进

- 整合更多医疗数据源
- 增加模型可解释性分析
- 开发临床决策支持系统界面
- 增加更多深度学习模型（如Transformer）
- 实现在线学习功能

## 📚 参考文献

1. KDIGO Clinical Practice Guideline for Acute Kidney Injury. Kidney Int. 2012;2:1–138.
2. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Scientific Data. 2016;3:160035.
3. Tomašev N, Glorot X, Rae JW, et al. A clinically applicable approach to continuous prediction of future acute kidney injury. Nature. 2019;572:116-119.

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 👥 贡献

欢迎提交Pull Request或Issue来改进这个项目！ 
