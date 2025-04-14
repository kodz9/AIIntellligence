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

-- 🎯 提取前48小时 lab 特征
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
    uf.urineoutput

FROM base_stays b
LEFT JOIN lab_features lf ON b.subject_id = lf.subject_id AND b.hadm_id = lf.hadm_id
LEFT JOIN chart_features cf ON b.subject_id = cf.subject_id AND b.icustay_id = cf.icustay_id
LEFT JOIN urine_features uf ON b.subject_id = uf.subject_id AND b.icustay_id = uf.icustay_id;
