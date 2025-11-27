# -*- coding: utf-8 -*-
# 数据生成模块：生成合成AKI预测数据集
import numpy as np
import pandas as pd

def make_synthetic(n=5000, random_state=42):
    """
    生成合成的急性肾损伤（AKI）预测数据集
    
    返回:
        pd.DataFrame: 包含特征和目标变量的数据框
    """
    rng = np.random.default_rng(random_state)
    
    # --- 1. 基础人口学特征 ---
    age = rng.normal(65, 12, n).clip(18, 95)
    female = rng.binomial(1, 0.45, n)
    
    # --- 2. 身高 (cm) ---
    # 男性: Mean 175, Std 7; 女性: Mean 162, Std 6
    height_mean = np.where(female==0, 175, 162)
    height_std = np.where(female==0, 7, 6)
    height = rng.normal(height_mean, height_std, n).clip(140, 200)
    
    # --- 3. 体重 (kg) ---
    # 通过BMI生成体重以保证合理性 (BMI Mean ~27, Std ~5)
    bmi = rng.normal(27, 5, n).clip(15, 50)
    weight = bmi * (height / 100) ** 2
    weight = weight.clip(40, 180) # 限制合理范围

    # --- 4. 基础肾功能指标 ---
    # baseline SCr (mg/dL), 对数正态分布, 男性或CKD患者稍高
    base_scr_log_mean = np.log(1.0 + 0.2*(1-female))
    base_scr = np.exp(rng.normal(base_scr_log_mean, 0.35, n)).clip(0.4, 6)
    
    # CKD 标记: 基于肌酐生成的概率 (肌酐越高，CKD概率越大)
    ckd_prob = 1 / (1 + np.exp(-(-2 + 1.2 * (base_scr - 1.2))))
    ckd = rng.binomial(1, ckd_prob, n)
    
    # --- 5. 计算肌酐清除率 (Creatinine Clearance, Cockcroft-Gault公式) ---
    # 公式: ((140 - age) * weight) / (72 * Scr) * (0.85 if female)
    crcl = ((140 - age) * weight) / (72 * base_scr)
    crcl = np.where(female == 1, crcl * 0.85, crcl)
    crcl = crcl.clip(5, 200) # 限制在生理合理范围

    # --- 6. 基础共病 ---
    diabetes = rng.binomial(1, 0.28, n)
    hypertension = rng.binomial(1, 0.55, n)
    heart_failure = rng.binomial(1, 0.12, n)
    
    # --- 7. 临床事件与药物 ---
    icu = rng.binomial(1, 0.35, n)
    sepsis = rng.binomial(1, 0.30 + 0.15*icu, n).clip(0,1)
    hypotension = rng.binomial(1, 0.15 + 0.20*sepsis + 0.05*icu, n).clip(0,1)
    dehydration = rng.binomial(1, 0.18 + 0.08*sepsis, n).clip(0,1)
    
    vanco_use = rng.binomial(1, 0.35 + 0.10*sepsis, n).clip(0,1)
    # 谷浓度仅在使用万古霉素时存在
    vanco_trough = (vanco_use * rng.normal(12, 4, n)).clip(0, 40)
    
    piptazo = rng.binomial(1, 0.30 + 0.10*sepsis, n).clip(0,1)
    aminogly = rng.binomial(1, 0.08, n)
    nsaid = rng.binomial(1, 0.10, n)
    loop_diur = rng.binomial(1, 0.25, n)
    contrast = rng.binomial(1, 0.20, n)
    
    # --- 8. 生成 AKI 标签 (Latent Risk Model) ---
    # 调整逻辑：加入肌酐清除率的影响、交互项和非线性项
    # 系数选择是为了保持约 ~20-25% 的AKI发生率
    z = (
        -1.8
        + 0.015 * (age - 65)
        + 0.0008 * (age - 65) ** 2          # 非线性年龄效应
        + 0.35 * ckd
        - 0.015 * (crcl - 80)               # CrCl每降低1单位，风险增加 (基准80)
        + 0.55 * sepsis
        + 0.50 * hypotension
        + 0.35 * contrast
        + 0.25 * dehydration * contrast     # 交互项：脱水+造影剂风险更高
        + 0.40 * vanco_use
        + 0.035 * vanco_use * (vanco_trough - 12)  # 修复：只在使用万古霉素时计算谷浓度影响
        + 0.30 * piptazo
        + 0.50 * vanco_use * piptazo        # 交互项：万古+哌拉西林的协同肾毒性
        + 0.45 * aminogly
        + 0.20 * nsaid
        + 0.25 * nsaid * ckd                # 交互项：CKD患者用NSAID风险更高
        + 0.30 * loop_diur
        + 0.15 * dehydration
        + 0.10 * icu
        + 0.08 * diabetes
    )
    p = 1 / (1 + np.exp(-z))
    aki = rng.binomial(1, p, n)
    
    # --- 9. 构建 DataFrame ---
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),  # 从1开始的顺序ID
        "age": age.round(1),
        "female": female,
        "height_cm": height.round(1),          # 新增字段
        "weight_kg": weight.round(1),          # 新增字段
        "baseline_scr_mgdl": base_scr.round(2),
        "creatinine_clearance": crcl.round(1), # 新增字段
        "ckd": ckd, 
        "diabetes": diabetes, 
        "hypertension": hypertension, 
        "heart_failure": heart_failure,
        "icu_admit": icu, 
        "sepsis": sepsis, 
        "hypotension": hypotension, 
        "dehydration_flag": dehydration,
        "vanco_use": vanco_use, 
        "vanco_trough": vanco_trough.round(1),
        "pip_tazo_use": piptazo, 
        "aminoglycoside_use": aminogly, 
        "nsaid_use": nsaid,
        "loop_diuretic_use": loop_diur, 
        "contrast_use": contrast,
        "aki_48h": aki
    })
    
    return df

if __name__ == "__main__":
    # 测试数据生成功能
    import os
    print("正在生成包含身高和肌酐清除率的合成数据集...")
    df = make_synthetic(n=10000)
    
    print(f"\n数据集形状: {df.shape}")
    print(f"AKI患病率: {df['aki_48h'].mean():.3f}")
    
    # 显示所有列名，确认新字段存在
    print(f"\n所有字段列表 ({len(df.columns)}个):")
    for i, col in enumerate(df.columns, 1):
        marker = " [新增]" if col in ['height_cm', 'weight_kg', 'creatinine_clearance'] else ""
        print(f"  {i:2d}. {col}{marker}")
    
    print("\n新增字段统计:")
    new_fields = ['height_cm', 'weight_kg', 'creatinine_clearance']
    if all(field in df.columns for field in new_fields):
        print(df[new_fields].describe())
    else:
        missing = [f for f in new_fields if f not in df.columns]
        print(f"⚠ 警告: 以下字段缺失: {missing}")
    
    print("\n前5行数据:")
    print(df.head())
    
    # 导出数据
    output_file = "nephro_small.csv"
    try:
        # 如果文件已存在，询问是否覆盖
        if os.path.exists(output_file):
            print(f"\n⚠ 文件 {output_file} 已存在，将被覆盖")
            print(f"   旧文件大小: {os.path.getsize(output_file):,} 字节")
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 验证导出结果
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            # 读取验证
            df_check = pd.read_csv(output_file, nrows=1)
            print(f"\n✓ 数据已成功导出到: {output_file}")
            print(f"  文件大小: {file_size:,} 字节")
            print(f"  导出字段数: {len(df_check.columns)}")
            
            # 检查新字段是否在导出的文件中
            missing_in_file = [f for f in new_fields if f not in df_check.columns]
            if missing_in_file:
                print(f"  ⚠ 警告: 以下新字段未在CSV中找到: {missing_in_file}")
            else:
                print(f"  ✓ 所有新字段已成功导出: {new_fields}")
        else:
            print(f"\n✗ 错误: 文件 {output_file} 导出后未找到")
    except Exception as e:
        print(f"\n✗ 导出错误: {e}")
        import traceback
        traceback.print_exc()