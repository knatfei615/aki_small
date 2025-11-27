# -*- coding: utf-8 -*-
"""
AKI预测机器学习教学平台
面向药学专业人员的机器学习入门工具
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面配置
st.set_page_config(
    page_title="AKI预测 - 机器学习教学平台！！！",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============== 数据加载 ==============
@st.cache_data
def load_data():
    """加载训练数据"""
    df = pd.read_csv("splits/train.csv")
    return df

@st.cache_data
def load_test_data():
    """加载测试数据（无标签）"""
    test_df = pd.read_csv("splits/test.csv")
    return test_df

# ============== 特征说明 ==============
FEATURE_DESCRIPTIONS = {
    'age': '年龄 (岁)',
    'female': '性别 (1=女性, 0=男性)',
    'height_cm': '身高 (cm)',
    'weight_kg': '体重 (kg)',
    'baseline_scr_mgdl': '基线血清肌酐 (mg/dL)',
    'creatinine_clearance': '肌酐清除率 (mL/min)',
    'ckd': '慢性肾病 (0=否, 1=是)',
    'diabetes': '糖尿病 (0=否, 1=是)',
    'hypertension': '高血压 (0=否, 1=是)',
    'heart_failure': '心力衰竭 (0=否, 1=是)',
    'icu_admit': 'ICU入院 (0=否, 1=是)',
    'sepsis': '脓毒症 (0=否, 1=是)',
    'hypotension': '低血压 (0=否, 1=是)',
    'dehydration_flag': '脱水 (0=否, 1=是)',
    'vanco_use': '万古霉素使用 (0=否, 1=是)',
    'vanco_trough': '万古霉素谷浓度 (μg/mL)',
    'pip_tazo_use': '哌拉西林/他唑巴坦使用 (0=否, 1=是)',
    'aminoglycoside_use': '氨基糖苷类使用 (0=否, 1=是)',
    'nsaid_use': 'NSAIDs使用 (0=否, 1=是)',
    'loop_diuretic_use': '袢利尿剂使用 (0=否, 1=是)',
    'contrast_use': '造影剂使用 (0=否, 1=是)',
    'aki_48h': '48小时内发生AKI (目标变量)'
}

# ============== 主程序 ==============
def main():
    # 标题
    st.markdown('<h1 class="main-header">🏥 急性肾损伤(AKI)预测</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">机器学习教学平台 | 面向药学专业人员</p>', unsafe_allow_html=True)
    
    # 加载数据
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ 找不到数据文件 splits/train.csv，请确保文件存在！")
        return
    
    # 侧边栏导航
    st.sidebar.title("📚 学习模块")
    page = st.sidebar.radio(
        "选择学习内容",
        ["🏠 课程介绍", "📊 数据探索", "🔍 特征筛选", "🤖 模型训练", "📈 模型评估", "🎯 预测演示"]
    )
    
    if page == "🏠 课程介绍":
        page_intro()
    elif page == "📊 数据探索":
        page_data_exploration(df)
    elif page == "🔍 特征筛选":
        page_feature_selection(df)
    elif page == "🤖 模型训练":
        page_model_training(df)
    elif page == "📈 模型评估":
        page_model_evaluation(df)
    elif page == "🎯 预测演示":
        page_prediction_demo(df)


def page_intro():
    """课程介绍页面"""
    st.header("👋 欢迎来到机器学习教学平台")
    
    st.markdown("""
    <div class="info-box">
    <h4>📌 什么是急性肾损伤(AKI)?</h4>
    <p>急性肾损伤是指肾功能在短时间内（通常48小时内）急剧下降的临床综合征。
    早期识别高风险患者对于预防AKI的发生至关重要。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 学习目标")
        st.markdown("""
        通过本平台，您将学习：
        1. **数据探索** - 了解临床数据的基本特征
        2. **特征筛选** - 识别对预测最重要的变量
        3. **模型训练** - 使用不同算法构建预测模型
        4. **模型评估** - 理解ROC曲线、AUC等指标
        5. **临床应用** - 如何解读和使用预测结果
        """)
    
    with col2:
        st.subheader("📚 机器学习简介")
        st.markdown("""
        **机器学习**是让计算机从数据中"学习"规律的技术：
        
        - **监督学习**: 用已知结果的数据训练模型
        - **特征(Feature)**: 用于预测的输入变量（如年龄、用药情况）
        - **标签(Label)**: 我们要预测的目标（如是否发生AKI）
        - **训练集/测试集**: 分别用于学习和验证模型
        """)
    
    st.markdown("---")
    st.subheader("🔬 本数据集包含的临床特征")
    
    # 分类展示特征
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 患者基本信息**")
        st.markdown("- 年龄、性别\n- 身高、体重\n- 基线肌酐、肌酐清除率")
    
    with col2:
        st.markdown("**🏥 合并症**")
        st.markdown("- 慢性肾病(CKD)\n- 糖尿病、高血压\n- 心力衰竭、脓毒症\n- ICU入院")
    
    with col3:
        st.markdown("**💊 肾毒性药物**")
        st.markdown("- 万古霉素\n- 哌拉西林/他唑巴坦\n- 氨基糖苷类\n- NSAIDs、袢利尿剂\n- 造影剂")


def page_data_exploration(df):
    """数据探索页面"""
    st.header("📊 数据探索")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 学习要点：</b> 在建模之前，我们需要先了解数据的基本情况，包括数据量、缺失值、各特征的分布等。
    这一步骤被称为<b>探索性数据分析(EDA)</b>。
    </div>
    """, unsafe_allow_html=True)
    
    # 基本统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样本数量", f"{len(df):,}")
    with col2:
        st.metric("特征数量", f"{len(df.columns)-2}")  # 减去Id和目标变量
    with col3:
        aki_rate = df['aki_48h'].mean() * 100
        st.metric("AKI发生率", f"{aki_rate:.1f}%")
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("缺失值", f"{missing}")
    
    st.markdown("---")
    
    # 数据预览
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10), use_container_width=True)
    
    # 特征说明
    with st.expander("📖 点击查看特征说明"):
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            if feat in df.columns:
                st.markdown(f"- **{feat}**: {desc}")
    
    st.markdown("---")
    
    # 目标变量分布
    st.subheader("🎯 目标变量分布")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        aki_counts = df['aki_48h'].value_counts()
        st.markdown(f"""
        - **未发生AKI (0)**: {aki_counts[0]:,} 例 ({aki_counts[0]/len(df)*100:.1f}%)
        - **发生AKI (1)**: {aki_counts[1]:,} 例 ({aki_counts[1]/len(df)*100:.1f}%)
        """)
        
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ 注意：</b> 这是一个<b>不平衡数据集</b>，AKI病例较少。
        在模型训练时需要特别处理。
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#4CAF50', '#F44336']
        bars = ax.bar(['未发生AKI (0)', '发生AKI (1)'], aki_counts.values, color=colors)
        ax.set_ylabel('样本数量')
        ax.set_title('AKI发生情况分布')
        for bar, count in zip(bars, aki_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{count}', ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # 特征分布可视化
    st.subheader("📈 特征分布可视化")
    
    # 选择要可视化的特征
    numeric_cols = ['age', 'height_cm', 'weight_kg', 'baseline_scr_mgdl', 
                    'creatinine_clearance', 'vanco_trough']
    selected_feature = st.selectbox("选择一个连续特征查看其分布：", numeric_cols)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 整体分布
    axes[0].hist(df[selected_feature].dropna(), bins=30, color='steelblue', edgecolor='white')
    axes[0].set_xlabel(FEATURE_DESCRIPTIONS.get(selected_feature, selected_feature))
    axes[0].set_ylabel('频次')
    axes[0].set_title(f'{selected_feature} 整体分布')
    
    # 按AKI分组
    for aki_val, color, label in [(0, '#4CAF50', '未发生AKI'), (1, '#F44336', '发生AKI')]:
        data = df[df['aki_48h'] == aki_val][selected_feature].dropna()
        axes[1].hist(data, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[1].set_xlabel(FEATURE_DESCRIPTIONS.get(selected_feature, selected_feature))
    axes[1].set_ylabel('频次')
    axes[1].set_title(f'{selected_feature} 按AKI状态分布')
    axes[1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 相关性热力图
    st.markdown("---")
    st.subheader("🔥 特征相关性热力图")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 什么是相关性？</b> 相关系数衡量两个变量之间的线性关系强度，范围从-1到+1。
    接近+1表示正相关，接近-1表示负相关，接近0表示无线性相关。
    </div>
    """, unsafe_allow_html=True)
    
    # 计算相关性
    feature_cols = [c for c in df.columns if c not in ['Id']]
    corr_matrix = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('特征相关性热力图', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 与目标变量的相关性排序
    st.subheader("📊 与AKI的相关性排名")
    target_corr = corr_matrix['aki_48h'].drop('aki_48h').sort_values(key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#F44336' if x > 0 else '#2196F3' for x in target_corr.values]
    bars = ax.barh(range(len(target_corr)), target_corr.values, color=colors)
    ax.set_yticks(range(len(target_corr)))
    ax.set_yticklabels(target_corr.index)
    ax.set_xlabel('相关系数')
    ax.set_title('各特征与AKI的相关性')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def page_feature_selection(df):
    """特征筛选页面"""
    st.header("🔍 特征筛选")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 为什么要进行特征筛选？</b><br>
    1. <b>降低过拟合风险</b>：减少不相关特征可以让模型更加稳定<br>
    2. <b>提高模型可解释性</b>：更少的特征更容易理解<br>
    3. <b>减少计算成本</b>：特征越少，训练速度越快<br>
    4. <b>处理多重共线性</b>：去除高度相关的冗余特征
    </div>
    """, unsafe_allow_html=True)
    
    # 准备数据
    feature_cols = [c for c in df.columns if c not in ['Id', 'aki_48h']]
    X = df[feature_cols].copy()
    y = df['aki_48h'].astype(int)
    
    # 填充缺失值
    X = X.fillna(X.median())
    
    st.markdown("---")
    
    # 方法选择
    st.subheader("🛠️ 选择特征筛选方法")
    
    method = st.selectbox(
        "选择特征筛选方法：",
        ["手动选择特征", "单变量统计检验 (ANOVA F-test)", "互信息 (Mutual Information)", 
         "递归特征消除 (RFE)", "基于随机森林的重要性"]
    )
    
    # 手动选择特征模式
    if method == "手动选择特征":
        st.markdown("""
        <div class="info-box">
        <b>📖 手动选择特征：</b><br>
        根据您的专业知识和临床经验，手动选择您认为对AKI预测最重要的特征。
        这种方法可以结合领域专家的先验知识，选择有临床意义的变量。
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 选择特征")
        
        # 按类别分组显示特征
        col1, col2, col3 = st.columns(3)
        
        # 特征分类
        patient_features = ['age', 'female', 'height_cm', 'weight_kg', 'baseline_scr_mgdl', 'creatinine_clearance']
        comorbidity_features = ['ckd', 'diabetes', 'hypertension', 'heart_failure', 'icu_admit', 'sepsis', 'hypotension', 'dehydration_flag']
        drug_features = ['vanco_use', 'vanco_trough', 'pip_tazo_use', 'aminoglycoside_use', 'nsaid_use', 'loop_diuretic_use', 'contrast_use']
        
        with col1:
            st.markdown("**👤 患者基本信息**")
            selected_patient = []
            for feat in patient_features:
                if feat in feature_cols:
                    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                    if st.checkbox(desc, value=True, key=f"manual_{feat}"):
                        selected_patient.append(feat)
        
        with col2:
            st.markdown("**🏥 合并症**")
            selected_comorbidity = []
            for feat in comorbidity_features:
                if feat in feature_cols:
                    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                    if st.checkbox(desc, value=True, key=f"manual_{feat}"):
                        selected_comorbidity.append(feat)
        
        with col3:
            st.markdown("**💊 肾毒性药物**")
            selected_drug = []
            for feat in drug_features:
                if feat in feature_cols:
                    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                    if st.checkbox(desc, value=True, key=f"manual_{feat}"):
                        selected_drug.append(feat)
        
        # 汇总选择的特征
        manual_selected = selected_patient + selected_comorbidity + selected_drug
        
        st.markdown("---")
        
        # 显示选择统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("患者信息", f"{len(selected_patient)}/{len([f for f in patient_features if f in feature_cols])}")
        with col2:
            st.metric("合并症", f"{len(selected_comorbidity)}/{len([f for f in comorbidity_features if f in feature_cols])}")
        with col3:
            st.metric("药物特征", f"{len(selected_drug)}/{len([f for f in drug_features if f in feature_cols])}")
        with col4:
            st.metric("总计选择", f"{len(manual_selected)}/{len(feature_cols)}")
        
        # 确认按钮
        if st.button("✅ 确认选择", type="primary"):
            if len(manual_selected) < 1:
                st.error("❌ 请至少选择1个特征！")
            else:
                st.session_state['selected_features'] = manual_selected
                st.success(f"✅ 已选择 {len(manual_selected)} 个特征：")
                st.write(", ".join([f"**{f}**" for f in manual_selected]))
                st.info("💡 请前往 **🤖 模型训练** 页面使用选择的特征训练模型！")
        
        return  # 手动选择模式不执行后续的自动筛选逻辑
    
    n_features = st.slider("选择保留的特征数量：", 3, len(feature_cols), 10)
    
    if st.button("🚀 开始特征筛选", type="primary"):
        with st.spinner("正在进行特征筛选..."):
            if method == "单变量统计检验 (ANOVA F-test)":
                st.markdown("""
                <div class="info-box">
                <b>📖 ANOVA F-test 原理：</b><br>
                通过分析各特征在不同类别（AKI vs 非AKI）之间的方差差异来评估特征重要性。
                F值越高，说明该特征在两组间的差异越显著。
                </div>
                """, unsafe_allow_html=True)
                
                selector = SelectKBest(score_func=f_classif, k=n_features)
                selector.fit(X, y)
                scores = pd.DataFrame({
                    '特征': feature_cols,
                    'F分数': selector.scores_,
                    'P值': selector.pvalues_
                }).sort_values('F分数', ascending=False)
                
            elif method == "互信息 (Mutual Information)":
                st.markdown("""
                <div class="info-box">
                <b>📖 互信息原理：</b><br>
                互信息衡量两个变量之间的依赖关系，可以捕捉非线性关系。
                互信息值越高，说明该特征与目标变量的关联越强。
                </div>
                """, unsafe_allow_html=True)
                
                mi_scores = mutual_info_classif(X, y, random_state=42)
                scores = pd.DataFrame({
                    '特征': feature_cols,
                    '互信息分数': mi_scores
                }).sort_values('互信息分数', ascending=False)
                
            elif method == "递归特征消除 (RFE)":
                st.markdown("""
                <div class="info-box">
                <b>📖 RFE原理：</b><br>
                递归特征消除通过反复构建模型并移除最不重要的特征来筛选。
                这是一种包装式(Wrapper)方法，考虑特征之间的相互作用。
                </div>
                """, unsafe_allow_html=True)
                
                estimator = LogisticRegression(max_iter=200, solver='liblinear')
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                rfe = RFE(estimator, n_features_to_select=n_features, step=1)
                rfe.fit(X_scaled, y)
                
                scores = pd.DataFrame({
                    '特征': feature_cols,
                    'RFE排名': rfe.ranking_,
                    '是否选中': ['✅ 是' if s else '❌ 否' for s in rfe.support_]
                }).sort_values('RFE排名')
                
            else:  # 随机森林重要性
                st.markdown("""
                <div class="info-box">
                <b>📖 随机森林特征重要性原理：</b><br>
                基于特征在决策树分裂中的贡献度来评估重要性。
                使用的特征越频繁、带来的纯度提升越大，重要性越高。
                </div>
                """, unsafe_allow_html=True)
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                scores = pd.DataFrame({
                    '特征': feature_cols,
                    '重要性分数': rf.feature_importances_
                }).sort_values('重要性分数', ascending=False)
            
            # 显示结果
            st.markdown("---")
            st.subheader("📊 特征筛选结果")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📋 特征评分表**")
                st.dataframe(scores, use_container_width=True)
            
            with col2:
                st.markdown("**📈 特征重要性可视化**")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                if '互信息分数' in scores.columns:
                    score_col = '互信息分数'
                elif 'F分数' in scores.columns:
                    score_col = 'F分数'
                elif '重要性分数' in scores.columns:
                    score_col = '重要性分数'
                else:
                    score_col = None
                
                if score_col:
                    top_scores = scores.head(n_features)
                    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_scores)))[::-1]
                    bars = ax.barh(range(len(top_scores)), top_scores[score_col].values, color=colors)
                    ax.set_yticks(range(len(top_scores)))
                    ax.set_yticklabels(top_scores['特征'].values)
                    ax.set_xlabel(score_col)
                    ax.set_title(f'Top {n_features} 特征')
                    ax.invert_yaxis()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # 选中的特征
            st.markdown("---")
            if 'RFE排名' in scores.columns:
                selected_features = scores[scores['RFE排名'] <= n_features]['特征'].tolist()
            else:
                selected_features = scores.head(n_features)['特征'].tolist()
            
            st.success(f"✅ 选中的 {n_features} 个特征：")
            st.write(", ".join([f"**{f}**" for f in selected_features]))
            
            # 保存到session state供后续使用
            st.session_state['selected_features'] = selected_features


def page_model_training(df):
    """模型训练页面"""
    st.header("🤖 模型训练")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 机器学习模型训练流程：</b><br>
    1. <b>数据准备</b>：从训练集划分出验证集用于评估<br>
    2. <b>选择算法</b>：根据问题特点选择合适的模型<br>
    3. <b>训练模型</b>：用训练数据让模型学习规律<br>
    4. <b>模型验证</b>：用验证集评估模型效果<br>
    5. <b>预测输出</b>：对测试集(test.csv)进行预测
    </div>
    """, unsafe_allow_html=True)
    
    # 准备数据
    feature_cols = [c for c in df.columns if c not in ['Id', 'aki_48h']]
    
    # 检查是否有选择的特征
    if 'selected_features' in st.session_state:
        use_selected = st.checkbox("使用特征筛选结果", value=True)
        if use_selected:
            feature_cols = st.session_state['selected_features']
            st.info(f"📌 使用已筛选的 {len(feature_cols)} 个特征")
    
    X = df[feature_cols].copy()
    y = df['aki_48h'].astype(int)
    
    st.markdown("---")
    
    # 数据分割设置
    st.subheader("📊 数据分割")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("验证集比例", 0.1, 0.4, 0.25, 0.05)
    with col2:
        random_state = st.number_input("随机种子 (用于结果复现)", 0, 999, 42)
    
    st.markdown(f"""
    - 训练集大小: **{int(len(df) * (1-test_size)):,}** 样本
    - 验证集大小: **{int(len(df) * test_size):,}** 样本
    """)
    
    st.markdown("---")
    
    # 模型选择
    st.subheader("🧠 选择机器学习模型")
    
    model_options = {
        "逻辑回归 (Logistic Regression)": {
            "description": "经典的线性分类模型，可解释性强，适合作为基线模型",
            "model": LogisticRegression(max_iter=200, class_weight='balanced', solver='liblinear')
        },
        "随机森林 (Random Forest)": {
            "description": "集成多棵决策树，抗过拟合能力强，可以捕捉非线性关系",
            "model": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        },
        "梯度提升 (Gradient Boosting)": {
            "description": "迭代地训练决策树来纠正错误，通常能取得很好的预测效果",
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        "支持向量机 (SVM)": {
            "description": "在高维空间寻找最优分隔超平面，适合中小规模数据",
            "model": SVC(probability=True, class_weight='balanced', random_state=42)
        }
    }
    
    selected_model = st.selectbox("选择模型：", list(model_options.keys()))
    
    st.markdown(f"""
    <div class="info-box">
    <b>📖 {selected_model}</b><br>
    {model_options[selected_model]['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # 训练按钮
    if st.button("🚀 开始训练模型", type="primary"):
        with st.spinner("正在训练模型..."):
            # 数据预处理
            X_filled = X.fillna(X.median())
            
            # 分割数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_filled, y, test_size=test_size, stratify=y, random_state=random_state
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 训练模型
            model = model_options[selected_model]['model']
            model.fit(X_train_scaled, y_train)
            
            # 在验证集上预测
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # 计算指标
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, y_pred)
            auprc = average_precision_score(y_val, y_proba)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            # 保存到session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_cols'] = feature_cols
            st.session_state['X_val'] = X_val
            st.session_state['y_val'] = y_val
            st.session_state['y_proba'] = y_proba
            st.session_state['y_pred'] = y_pred
            st.session_state['train_median'] = X.median()  # 保存训练集中位数供预测使用
            
            # 显示结果
            st.markdown("---")
            st.subheader("📊 训练结果")
            
            st.markdown("""
            <div class="success-box">
            <b>✅ 模型训练完成！</b>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("准确率 (Accuracy)", f"{accuracy:.3f}")
            with col2:
                st.metric("AUC-ROC", f"{auc:.3f}")
            with col3:
                st.metric("F1分数", f"{f1:.3f}")
            with col4:
                st.metric("AUC-PR", f"{auprc:.3f}")
            
            st.markdown("---")
            st.subheader("🔄 5折交叉验证结果")
            st.markdown(f"""
            - 平均AUC: **{cv_scores.mean():.3f}** ± {cv_scores.std():.3f}
            - 各折AUC: {', '.join([f'{s:.3f}' for s in cv_scores])}
            """)
            
            st.markdown("""
            <div class="info-box">
            <b>💡 什么是交叉验证？</b><br>
            交叉验证将数据分成K份，轮流用其中一份做验证、其余做训练。
            这样可以更可靠地评估模型的泛化能力，减少因数据分割导致的偶然性。
            </div>
            """, unsafe_allow_html=True)
            
            st.success("💡 请前往 **📈 模型评估** 页面查看详细分析，或前往 **🎯 预测演示** 页面对测试集进行预测！")


def page_model_evaluation(df):
    """模型评估页面"""
    st.header("📈 模型评估")
    
    # 检查是否有已训练的模型
    if 'model' not in st.session_state:
        st.warning("⚠️ 请先在 **🤖 模型训练** 页面训练一个模型！")
        return
    
    st.markdown("""
    <div class="info-box">
    <b>💡 为什么需要评估模型？</b><br>
    模型评估帮助我们了解模型在未见过的数据上的表现，
    判断模型是否可以可靠地用于临床决策支持。
    </div>
    """, unsafe_allow_html=True)
    
    # 获取数据
    y_test = st.session_state['y_val']
    y_proba = st.session_state['y_proba']
    y_pred = st.session_state['y_pred']
    
    st.markdown("---")
    
    # ROC曲线
    st.subheader("📊 ROC曲线")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>📖 什么是ROC曲线？</b><br>
        ROC曲线展示了不同分类阈值下<b>敏感度(Sensitivity)</b>和<b>1-特异度(1-Specificity)</b>的权衡关系。<br><br>
        - <b>敏感度/召回率</b>：正确识别AKI患者的比例<br>
        - <b>特异度</b>：正确识别非AKI患者的比例<br>
        - <b>AUC</b>：曲线下面积，越接近1越好，0.5表示随机猜测
        </div>
        """, unsafe_allow_html=True)
        
        auc = roc_auc_score(y_test, y_proba)
        st.markdown(f"""
        **模型AUC = {auc:.3f}**
        
        - AUC > 0.9: 优秀
        - AUC 0.8-0.9: 良好
        - AUC 0.7-0.8: 一般
        - AUC < 0.7: 较差
        """)
    
    with col2:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', label='随机猜测')
        ax.fill_between(fpr, tpr, alpha=0.2)
        ax.set_xlabel('1 - 特异度 (假阳性率)', fontsize=11)
        ax.set_ylabel('敏感度 (真阳性率)', fontsize=11)
        ax.set_title('ROC曲线', fontsize=12)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # PR曲线
    st.subheader("📊 精确率-召回率曲线 (PR曲线)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>📖 什么是PR曲线？</b><br>
        PR曲线特别适合<b>不平衡数据集</b>，展示精确率和召回率的权衡：<br><br>
        - <b>精确率(Precision)</b>：预测为AKI的患者中，真正发生AKI的比例<br>
        - <b>召回率(Recall)</b>：实际AKI患者中，被模型正确识别的比例<br>
        - <b>AUPRC</b>：PR曲线下面积
        </div>
        """, unsafe_allow_html=True)
        
        auprc = average_precision_score(y_test, y_proba)
        prevalence = y_test.mean()
        st.markdown(f"""
        **模型AUPRC = {auprc:.3f}**
        
        基线（随机猜测）= {prevalence:.3f}（AKI发生率）
        """)
    
    with col2:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, 'g-', linewidth=2, label=f'PR曲线 (AUPRC = {auprc:.3f})')
        ax.axhline(y=prevalence, color='r', linestyle='--', label=f'基线 = {prevalence:.3f}')
        ax.fill_between(recall, precision, alpha=0.2, color='green')
        ax.set_xlabel('召回率 (Recall)', fontsize=11)
        ax.set_ylabel('精确率 (Precision)', fontsize=11)
        ax.set_title('精确率-召回率曲线', fontsize=12)
        ax.legend(loc='upper right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # 混淆矩阵
    st.subheader("📊 混淆矩阵")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>📖 什么是混淆矩阵？</b><br>
        混淆矩阵展示了模型预测结果与实际结果的对比：<br><br>
        - <b>真阴性(TN)</b>：正确预测为非AKI<br>
        - <b>假阳性(FP)</b>：错误预测为AKI（假警报）<br>
        - <b>假阴性(FN)</b>：错误预测为非AKI（漏诊）<br>
        - <b>真阳性(TP)</b>：正确预测为AKI
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['预测: 非AKI', '预测: AKI'],
                    yticklabels=['实际: 非AKI', '实际: AKI'])
        ax.set_xlabel('预测值', fontsize=11)
        ax.set_ylabel('实际值', fontsize=11)
        ax.set_title('混淆矩阵', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # 分类报告
    st.markdown("---")
    st.subheader("📋 分类报告")
    
    report = classification_report(y_test, y_pred, target_names=['非AKI (0)', 'AKI (1)'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # 阈值选择
    st.markdown("---")
    st.subheader("🎚️ 阈值选择")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 什么是分类阈值？</b><br>
    模型输出的是概率值(0-1)，我们需要选择一个阈值来决定预测结果。
    默认阈值是0.5，但可以根据临床需求调整：<br>
    - 降低阈值 → 提高敏感度（减少漏诊），但增加假阳性<br>
    - 提高阈值 → 提高特异度（减少假警报），但可能漏诊
    </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider("选择分类阈值", 0.0, 1.0, 0.5, 0.01)
    
    y_pred_custom = (y_proba >= threshold).astype(int)
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    
    tn, fp, fn, tp = cm_custom.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("敏感度", f"{sensitivity:.3f}")
    with col2:
        st.metric("特异度", f"{specificity:.3f}")
    with col3:
        st.metric("阳性预测值(PPV)", f"{ppv:.3f}")
    with col4:
        st.metric("阴性预测值(NPV)", f"{npv:.3f}")


def page_prediction_demo(df):
    """预测演示页面"""
    st.header("🎯 预测演示")
    
    # 检查是否有已训练的模型
    if 'model' not in st.session_state:
        st.warning("⚠️ 请先在 **🤖 模型训练** 页面训练一个模型！")
        return
    
    st.markdown("""
    <div class="info-box">
    <b>💡 临床应用演示</b><br>
    输入患者的临床信息，模型将预测其48小时内发生AKI的风险。
    </div>
    """, unsafe_allow_html=True)
    
    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_cols = st.session_state['feature_cols']
    
    st.markdown("---")
    st.subheader("📝 输入患者信息")
    
    # 创建输入表单
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    
    with col1:
        st.markdown("**👤 基本信息**")
        input_data['age'] = st.number_input("年龄 (岁)", 18, 100, 65)
        input_data['female'] = st.selectbox("性别", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
        input_data['height_cm'] = st.number_input("身高 (cm)", 140, 200, 170)
        input_data['weight_kg'] = st.number_input("体重 (kg)", 40, 150, 70)
        input_data['baseline_scr_mgdl'] = st.number_input("基线血清肌酐 (mg/dL)", 0.1, 5.0, 1.0, 0.1)
        input_data['creatinine_clearance'] = st.number_input("肌酐清除率 (mL/min)", 10, 150, 80)
    
    with col2:
        st.markdown("**🏥 合并症**")
        input_data['ckd'] = st.selectbox("慢性肾病", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['diabetes'] = st.selectbox("糖尿病", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['hypertension'] = st.selectbox("高血压", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['heart_failure'] = st.selectbox("心力衰竭", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['icu_admit'] = st.selectbox("ICU入院", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['sepsis'] = st.selectbox("脓毒症", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['hypotension'] = st.selectbox("低血压", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['dehydration_flag'] = st.selectbox("脱水", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
    
    with col3:
        st.markdown("**💊 肾毒性药物使用**")
        input_data['vanco_use'] = st.selectbox("万古霉素", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['vanco_trough'] = st.number_input("万古霉素谷浓度 (μg/mL)", 0.0, 30.0, 0.0, 0.5)
        input_data['pip_tazo_use'] = st.selectbox("哌拉西林/他唑巴坦", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['aminoglycoside_use'] = st.selectbox("氨基糖苷类", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['nsaid_use'] = st.selectbox("NSAIDs", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['loop_diuretic_use'] = st.selectbox("袢利尿剂", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        input_data['contrast_use'] = st.selectbox("造影剂", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
    
    st.markdown("---")
    
    if st.button("🔮 预测AKI风险", type="primary"):
        # 准备输入数据
        input_df = pd.DataFrame([input_data])
        
        # 只选择模型使用的特征
        input_df = input_df[[c for c in feature_cols if c in input_df.columns]]
        
        # 如果有缺失的特征，用0填充
        for c in feature_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        
        input_df = input_df[feature_cols]
        
        # 标准化并预测
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0, 1]
        prediction = "高风险" if probability >= 0.5 else "低风险"
        
        # 显示结果
        st.markdown("---")
        st.subheader("🔮 预测结果")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 风险仪表盘
            fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
            
            # 设置为半圆
            theta = np.linspace(0, np.pi, 100)
            r = np.ones(100)
            
            # 背景颜色区域
            colors_bg = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 100))
            for i in range(len(theta)-1):
                ax.fill_between([theta[i], theta[i+1]], 0, 1, color=colors_bg[i], alpha=0.3)
            
            # 指针
            pointer_angle = np.pi * (1 - probability)
            ax.arrow(pointer_angle, 0, 0, 0.7, head_width=0.15, head_length=0.1, 
                    fc='black', ec='black', linewidth=2)
            
            ax.set_ylim(0, 1.2)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            ax.set_thetagrids([])
            ax.set_rgrids([])
            ax.spines['polar'].set_visible(False)
            
            ax.text(np.pi, 1.1, '低风险', ha='center', fontsize=10, color='green')
            ax.text(0, 1.1, '高风险', ha='center', fontsize=10, color='red')
            ax.set_title(f'AKI风险概率: {probability*100:.1f}%', fontsize=14, pad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            if probability >= 0.5:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #c44536 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; color: white;">
                <h2>⚠️ 高风险</h2>
                <h1>{probability*100:.1f}%</h1>
                <p>该患者48小时内发生AKI的风险较高</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; color: white;">
                <h2>✅ 低风险</h2>
                <h1>{probability*100:.1f}%</h1>
                <p>该患者48小时内发生AKI的风险较低</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ 免责声明：</b><br>
        此预测仅供教学演示和辅助参考，不能替代临床判断。
        实际诊疗决策应由专业医务人员根据患者具体情况做出。
        </div>
        """, unsafe_allow_html=True)
    
    # 测试集批量预测
    st.markdown("---")
    st.subheader("📊 测试集批量预测")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 批量预测功能</b><br>
    使用训练好的模型对测试集(test.csv)中的所有样本进行AKI风险预测，并下载预测结果。
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 对测试集进行预测", type="primary"):
        try:
            # 加载测试数据
            test_df = load_test_data()
            
            # 准备特征
            test_features = test_df[feature_cols].copy()
            
            # 使用训练集中位数填充缺失值
            if 'train_median' in st.session_state:
                for col in feature_cols:
                    if col in test_features.columns:
                        test_features[col] = test_features[col].fillna(st.session_state['train_median'].get(col, 0))
            else:
                test_features = test_features.fillna(0)
            
            # 标准化并预测
            test_scaled = scaler.transform(test_features)
            test_proba = model.predict_proba(test_scaled)[:, 1]
            test_pred = (test_proba >= 0.5).astype(int)
            
            # 创建结果DataFrame
            result_df = pd.DataFrame({
                'Id': test_df['Id'],
                'aki_48h_probability': test_proba,
                'aki_48h_prediction': test_pred
            })
            
            # 显示预测统计
            st.markdown("---")
            st.subheader("📈 预测结果统计")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("测试集样本数", f"{len(test_df):,}")
            with col2:
                high_risk_count = (test_pred == 1).sum()
                st.metric("预测高风险数", f"{high_risk_count:,} ({high_risk_count/len(test_df)*100:.1f}%)")
            with col3:
                st.metric("平均预测概率", f"{test_proba.mean()*100:.1f}%")
            
            # 预测概率分布
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(test_proba, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='分类阈值 (0.5)')
            ax.set_xlabel('预测概率')
            ax.set_ylabel('样本数量')
            ax.set_title('测试集AKI风险预测概率分布')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 显示预测结果预览
            st.markdown("**预测结果预览：**")
            st.dataframe(result_df.head(20), use_container_width=True)
            
            # 下载按钮
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果 (CSV)",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )
            
            st.success("✅ 测试集预测完成！点击上方按钮下载预测结果。")
            
        except FileNotFoundError:
            st.error("❌ 找不到测试数据文件 splits/test.csv！")
        except Exception as e:
            st.error(f"❌ 预测过程中出错：{str(e)}")


# 运行主程序
if __name__ == "__main__":
    main()
