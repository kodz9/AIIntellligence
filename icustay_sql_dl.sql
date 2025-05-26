-- 从MIMIC-III数据库直接提取AKI患者的时序数据（优化数据量）
WITH base_stays AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.icustay_id,
        i.intime,
        i.outtime,
        ROUND(EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0, 2) AS icu_los_hours,
        CASE
            WHEN p.dob > '2100-01-01' THEN 91
            ELSE DATE_PART('year', i.intime) - DATE_PART('year', p.dob)
        END AS age,
        p.gender
    FROM mimiciii.icustays i
    JOIN mimiciii.patients p ON i.subject_id = p.subject_id
    JOIN mimiciii.admissions a ON i.hadm_id = a.hadm_id
    WHERE
        EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 >= 72
        AND (
            CASE
                WHEN p.dob > '2100-01-01' THEN 91
                ELSE DATE_PART('year', i.intime) - DATE_PART('year', p.dob)
            END
        ) >= 18
    ORDER BY RANDOM()
    LIMIT 1000  -- 减少患者数量从5000到1000
),

-- 提取前48小时 lab 特征（基线特征）
lab_features AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum ELSE NULL END) AS creatinine,
        AVG(CASE WHEN le.itemid = 51006 THEN le.valuenum ELSE NULL END) AS bun,
        AVG(CASE WHEN le.itemid = 50983 THEN le.valuenum ELSE NULL END) AS sodium,
        AVG(CASE WHEN le.itemid = 50971 THEN le.valuenum ELSE NULL END) AS potassium,
        AVG(CASE WHEN le.itemid = 51300 THEN le.valuenum ELSE NULL END) AS wbc,
        AVG(CASE WHEN le.itemid = 51221 THEN le.valuenum ELSE NULL END) AS hematocrit
    FROM mimiciii.labevents le
    JOIN base_stays b ON le.hadm_id = b.hadm_id
    WHERE le.charttime BETWEEN b.intime - INTERVAL '48 hours' AND b.intime
    AND le.valuenum IS NOT NULL
    GROUP BY le.subject_id, le.hadm_id
),

-- 提取入院后48小时内的肌酐基线值
creatinine_baseline AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        MIN(le.valuenum) AS creatinine_baseline  -- 使用最小值作为基线
    FROM mimiciii.labevents le
    JOIN base_stays b ON le.hadm_id = b.hadm_id
    WHERE le.itemid = 50912  -- 肌酐的itemid
    AND le.charttime BETWEEN b.intime AND b.intime + INTERVAL '48 hours'
    AND le.valuenum IS NOT NULL
    GROUP BY le.subject_id, le.hadm_id
),

-- 提取入院后48-72小时的肌酐值，用于判断AKI
creatinine_subsequent AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        MAX(le.valuenum) AS creatinine_subsequent  -- 使用最大值作为后续值以检测升高
    FROM mimiciii.labevents le
    JOIN base_stays b ON le.hadm_id = b.hadm_id
    WHERE le.itemid = 50912  -- 肌酐的itemid
    AND le.charttime BETWEEN b.intime + INTERVAL '48 hours' AND b.intime + INTERVAL '72 hours'
    AND le.valuenum IS NOT NULL
    GROUP BY le.subject_id, le.hadm_id
),

-- 提取前48小时生命体征（chart）特征
chart_features AS (
    SELECT
        ce.subject_id,
        ce.icustay_id,
        AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum ELSE NULL END) AS heartrate,
        AVG(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum ELSE NULL END) AS sbp,
        AVG(CASE WHEN ce.itemid IN (615, 220210) THEN ce.valuenum ELSE NULL END) AS resp_rate,
        AVG(CASE WHEN ce.itemid IN (223761, 678) THEN ce.valuenum ELSE NULL END) AS temperature
    FROM mimiciii.chartevents ce
    JOIN base_stays b ON ce.icustay_id = b.icustay_id
    WHERE ce.charttime BETWEEN b.intime - INTERVAL '48 hours' AND b.intime
    AND ce.valuenum IS NOT NULL
    GROUP BY ce.subject_id, ce.icustay_id
),

-- 尿量（前48小时总量）
urine_features AS (
    SELECT
        oe.subject_id,
        oe.icustay_id,
        SUM(oe.value) AS urineoutput
    FROM mimiciii.outputevents oe
    JOIN base_stays b ON oe.icustay_id = b.icustay_id
    WHERE
        oe.charttime BETWEEN b.intime - INTERVAL '48 hours' AND b.intime
    GROUP BY oe.subject_id, oe.icustay_id
),

-- 计算24小时尿量用于AKI判断
urine_24h AS (
    SELECT
        oe.subject_id,
        oe.icustay_id,
        SUM(oe.value) AS urineoutput_24h
    FROM mimiciii.outputevents oe
    JOIN base_stays b ON oe.icustay_id = b.icustay_id
    WHERE
        oe.charttime BETWEEN b.intime AND b.intime + INTERVAL '24 hours'
    GROUP BY oe.subject_id, oe.icustay_id
),

-- 计算患者体重
weight_data AS (
    SELECT
        ce.subject_id,
        ce.icustay_id,
        AVG(ce.valuenum) AS weight_kg
    FROM mimiciii.chartevents ce
    JOIN base_stays b ON ce.icustay_id = b.icustay_id
    WHERE ce.itemid IN (762, 763, 3723, 224639)  -- 体重相关的itemid
    AND ce.valuenum > 0 AND ce.valuenum < 300  -- 合理值范围检查
    GROUP BY ce.subject_id, ce.icustay_id
),

-- 整合所有特征并计算AKI标签
integrated_data AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.icustay_id,
        b.intime,
        b.outtime,
        b.icu_los_hours,
        b.age,
        b.gender,
        lf.creatinine,
        lf.bun,
        lf.sodium,
        lf.potassium,
        lf.wbc,
        lf.hematocrit,
        cf.heartrate,
        cf.sbp,
        cf.resp_rate,
        cf.temperature,
        uf.urineoutput,
        cb.creatinine_baseline,
        cs.creatinine_subsequent,
        CASE
            WHEN cs.creatinine_subsequent IS NULL OR cb.creatinine_baseline IS NULL THEN NULL
            WHEN cs.creatinine_subsequent >= cb.creatinine_baseline * 1.5 THEN 1
            WHEN cs.creatinine_subsequent - cb.creatinine_baseline >= 0.3 THEN 1  -- 0.3 mg/dL增长标准
            ELSE 0
        END AS aki_by_creatinine,
        u24.urineoutput_24h,
        w.weight_kg,
        CASE
            WHEN u24.urineoutput_24h IS NULL OR w.weight_kg IS NULL THEN NULL
            WHEN (u24.urineoutput_24h / (w.weight_kg * 24)) < 0.5 THEN 1  -- 尿量<0.5ml/kg/h
            ELSE 0
        END AS aki_by_urine,
        -- 综合判断AKI
        CASE
            WHEN (CASE
                    WHEN cs.creatinine_subsequent IS NULL OR cb.creatinine_baseline IS NULL THEN NULL
                    WHEN cs.creatinine_subsequent >= cb.creatinine_baseline * 1.5 THEN 1
                    WHEN cs.creatinine_subsequent - cb.creatinine_baseline >= 0.3 THEN 1
                    ELSE 0
                 END) = 1 OR 
                 (CASE
                    WHEN u24.urineoutput_24h IS NULL OR w.weight_kg IS NULL THEN NULL
                    WHEN (u24.urineoutput_24h / (w.weight_kg * 24)) < 0.5 THEN 1
                    ELSE 0
                 END) = 1 THEN 1
            ELSE 0
        END AS aki_label
    FROM base_stays b
    LEFT JOIN lab_features lf ON b.subject_id = lf.subject_id AND b.hadm_id = lf.hadm_id
    LEFT JOIN chart_features cf ON b.subject_id = cf.subject_id AND b.icustay_id = cf.icustay_id
    LEFT JOIN urine_features uf ON b.subject_id = uf.subject_id AND b.icustay_id = uf.icustay_id
    LEFT JOIN creatinine_baseline cb ON b.subject_id = cb.subject_id AND b.hadm_id = cb.hadm_id
    LEFT JOIN creatinine_subsequent cs ON b.subject_id = cs.subject_id AND b.hadm_id = cs.hadm_id
    LEFT JOIN urine_24h u24 ON b.subject_id = u24.subject_id AND b.icustay_id = u24.icustay_id
    LEFT JOIN weight_data w ON b.subject_id = w.subject_id AND b.icustay_id = w.icustay_id
),

-- 获取所有患者（AKI和非AKI）
all_patients AS (
    SELECT 
        subject_id, hadm_id, icustay_id, aki_label,
        intime, outtime,
        -- 限制时间范围为前72小时（而不是整个住院期间）
        intime + INTERVAL '72 hours' AS max_time
    FROM integrated_data
),

-- 获取实验室检查结果（限制时间范围和采样频率）
lab_events AS (
    SELECT le.subject_id, le.hadm_id, le.charttime,
           CASE 
               WHEN le.itemid = 50912 THEN 'creatinine'
               WHEN le.itemid = 51006 THEN 'bun'
               WHEN le.itemid = 50971 THEN 'potassium'
               WHEN le.itemid = 50983 THEN 'sodium'
               WHEN le.itemid = 50902 THEN 'chloride'
               WHEN le.itemid = 50882 THEN 'bicarbonate'
               WHEN le.itemid = 50931 THEN 'glucose'
               WHEN le.itemid = 51221 THEN 'hematocrit'
               WHEN le.itemid = 51222 THEN 'hemoglobin'
               WHEN le.itemid = 51301 THEN 'wbc'
               WHEN le.itemid = 51265 THEN 'platelet'
           END AS measurement,
           le.valuenum
    FROM mimiciii.labevents le
    JOIN all_patients p ON le.subject_id = p.subject_id AND le.hadm_id = p.hadm_id
    WHERE le.itemid IN (
        50912, -- 肌酐
        51006, -- 尿素氮
        50971, -- 钾
        50983, -- 钠
        50902, -- 氯
        50882, -- 碳酸氢盐
        50931, -- 葡萄糖
        51221, -- 血细胞比容
        51222, -- 血红蛋白
        51301, -- 白细胞
        51265  -- 血小板
    )
    AND le.valuenum IS NOT NULL
    AND le.charttime BETWEEN p.intime AND p.max_time  -- 限制为前72小时
),

-- 获取生命体征（限制时间范围和采样频率）
chart_events AS (
    SELECT ce.subject_id, ce.hadm_id, ce.charttime,
           CASE 
               WHEN ce.itemid IN (211, 220045) THEN 'heart_rate'
               WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN 'systolic_bp'
               WHEN ce.itemid IN (8368, 8440, 8441, 8555, 220180, 220051) THEN 'diastolic_bp'
               WHEN ce.itemid IN (456, 52, 6702, 443, 220052, 220181, 225312) THEN 'mean_bp'
               WHEN ce.itemid IN (615, 618, 220210, 224690) THEN 'respiratory_rate'
               WHEN ce.itemid IN (223761, 678) THEN 'temperature'
               WHEN ce.itemid IN (646, 220277) THEN 'spo2'
               WHEN ce.itemid IN (807, 811, 1529, 3745, 3744, 225664, 220621, 226537) THEN 'gcs'
           END AS measurement,
           ce.valuenum
    FROM mimiciii.chartevents ce
    JOIN all_patients p ON ce.subject_id = p.subject_id AND ce.hadm_id = p.hadm_id
    WHERE ce.itemid IN (
        -- 心率
        211, 220045,
        -- 收缩压
        51, 442, 455, 6701, 220179, 220050,
        -- 舒张压
        8368, 8440, 8441, 8555, 220180, 220051,
        -- 平均动脉压
        456, 52, 6702, 443, 220052, 220181, 225312,
        -- 呼吸频率
        615, 618, 220210, 224690,
        -- 体温
        223761, 678,
        -- 血氧饱和度
        646, 220277,
        -- GCS评分
        807, 811, 1529, 3745, 3744, 225664, 220621, 226537
    )
    AND ce.valuenum IS NOT NULL
    AND ce.charttime BETWEEN p.intime AND p.max_time  -- 限制为前72小时
),

-- 获取尿量（限制时间范围和采样频率）
urine_output AS (
    SELECT oe.subject_id, oe.hadm_id, oe.charttime,
           'urine_output' AS measurement,
           oe.value AS valuenum
    FROM mimiciii.outputevents oe
    JOIN all_patients p ON oe.subject_id = p.subject_id AND oe.hadm_id = p.hadm_id
    WHERE oe.itemid IN (
        -- 尿量相关的itemid
        40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086,
        40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557,
        226558, 227488, 227489
    )
    AND oe.value > 0
    AND oe.charttime BETWEEN p.intime AND p.max_time  -- 限制为前72小时
),

-- 合并所有测量值
all_measurements AS (
    SELECT subject_id, hadm_id, charttime, measurement, valuenum
    FROM lab_events
    UNION ALL
    SELECT subject_id, hadm_id, charttime, measurement, valuenum
    FROM chart_events
    UNION ALL
    SELECT subject_id, hadm_id, charttime, measurement, valuenum
    FROM urine_output
),

-- 对时间进行分桶，每4小时一个时间点
time_buckets AS (
    SELECT 
        m.subject_id, 
        m.hadm_id, 
        m.measurement,
        -- 将时间分成4小时一个桶
        FLOOR(EXTRACT(EPOCH FROM (m.charttime - p.intime))/14400) AS time_bucket,
        AVG(m.valuenum) AS avg_value
    FROM all_measurements m
    JOIN all_patients p ON m.subject_id = p.subject_id AND m.hadm_id = p.hadm_id
    GROUP BY m.subject_id, m.hadm_id, m.measurement, time_bucket
)

-- 最终查询：为每个患者创建时序数据（采样后）
SELECT 
    p.subject_id, 
    p.hadm_id, 
    p.icustay_id,
    p.aki_label,
    p.intime,
    p.outtime,
    tb.time_bucket * 4 AS hours_since_admission,  -- 转换回小时
    tb.measurement,
    tb.avg_value AS valuenum
FROM all_patients p
JOIN time_buckets tb ON p.subject_id = tb.subject_id AND p.hadm_id = tb.hadm_id
WHERE tb.time_bucket <= 18  -- 限制为前72小时（18个4小时时间段）
ORDER BY p.subject_id, p.hadm_id, p.icustay_id, hours_since_admission, tb.measurement;