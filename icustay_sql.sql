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
    LIMIT 5000
),

-- 🎯 提取前48小时 lab 特征（基线特征）
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

-- 🎯 提取入院后48小时内的肌酐基线值
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

-- 🎯 提取入院后48-72小时的肌酐值，用于判断AKI
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

-- 🩺 提取前48小时生命体征（chart）特征
chart_features AS (
    SELECT
        ce.subject_id,
        ce.icustay_id,
        AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum ELSE NULL END) AS sbp,
        AVG(CASE WHEN ce.itemid IN (220074, 618) THEN ce.valuenum ELSE NULL END) AS heartrate,
        AVG(CASE WHEN ce.itemid IN (615, 220210) THEN ce.valuenum ELSE NULL END) AS resp_rate,
        AVG(CASE WHEN ce.itemid IN (223761, 678) THEN ce.valuenum ELSE NULL END) AS temperature
    FROM mimiciii.chartevents ce
    JOIN base_stays b ON ce.icustay_id = b.icustay_id
    WHERE ce.charttime BETWEEN b.intime - INTERVAL '48 hours' AND b.intime
    AND ce.valuenum IS NOT NULL
    GROUP BY ce.subject_id, ce.icustay_id
),

-- 💧 尿量（前48小时总量）
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

-- 📊 计算24小时尿量用于AKI判断
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

-- 🧮 计算患者体重
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
)

-- 🧩 整合所有特征到主表
SELECT
    b.*,
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
    -- 新增AKI判断所需的字段
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
LEFT JOIN weight_data w ON b.subject_id = w.subject_id AND b.icustay_id = w.icustay_id;
