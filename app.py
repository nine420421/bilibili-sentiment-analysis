import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import base64
import io

# 设置页面
st.set_page_config(
    page_title="B站评论情感分析系统",
    page_icon="🎯",
    layout="wide"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def main():
    # 标题
    st.title("🎯 B站视频评论情感分析系统")
    st.markdown("---")

    # 演示数据（如果用户没有上传文件）
    st.info("📊 上传你的B站评论数据CSV文件，或使用下面的示例数据体验功能")

    # 示例数据
    sample_data = {
        'user_name': ['用户A', '用户B', '用户C', '用户D'],
        'content_cleaned': ['这个视频很棒，学到了很多', '内容一般，没有新意', '非常喜欢，点赞支持', '不太感兴趣'],
        'sentiment_label': ['积极', '消极', '积极', '消极'],
        'sentiment_score': [0.85, 0.25, 0.92, 0.35],
        'like_count': [156, 23, 289, 12],
        'post_time': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']
    }

    # 文件上传
    st.sidebar.header("📁 数据上传")
    uploaded_file = st.sidebar.file_uploader("选择CSV文件", type=['csv'])

    if uploaded_file is not None:
        # 使用上传的文件
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功加载 {len(df)} 条评论数据")
    else:
        # 使用示例数据
        df = pd.DataFrame(sample_data)
        st.warning("⚠️ 当前使用示例数据，请上传CSV文件获得完整分析")

    # 数据概览
    st.header("📊 数据概览")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总评论数", len(df))
    with col2:
        positive_count = len(df[df['sentiment_label'] == '积极'])
        st.metric("积极评论", positive_count)
    with col3:
        negative_count = len(df[df['sentiment_label'] == '消极'])
        st.metric("消极评论", negative_count)
    with col4:
        total_likes = df['like_count'].sum() if 'like_count' in df.columns else 0
        st.metric("总点赞数", f"{total_likes:,}")

    # 情感分析
    st.header("🎭 情感分析")

    col1, col2 = st.columns(2)

    with col1:
        # 情感分布饼图
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='评论情感分布',
            color=sentiment_counts.index,
            color_discrete_map={'积极': '#2E8B57', '消极': '#DC143C', '中性': '#1E90FF'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # 情感得分分布
        if 'sentiment_score' in df.columns:
            fig_hist = px.histogram(
                df, x='sentiment_score',
                title='情感得分分布',
                nbins=20,
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)

    # 词云生成
    st.header("☁️ 词云分析")

    if st.button("生成词云图", type="primary"):
        with st.spinner("正在生成词云..."):
            # 准备文本数据
            all_text = ' '.join(df['content_cleaned'].astype(str))

            # 生成词云
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                font_path=None  # 在云端使用默认字体
            ).generate(all_text)

            # 显示词云
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('评论词云图', fontsize=16)
            st.pyplot(fig)

    # 评论浏览
    st.header("💬 评论详情")

    # 筛选选项
    col1, col2 = st.columns(2)

    with col1:
        sentiment_filter = st.multiselect(
            "情感筛选",
            options=df['sentiment_label'].unique(),
            default=df['sentiment_label'].unique()
        )

    with col2:
        sort_by = st.selectbox(
            "排序方式",
            ["默认", "点赞数", "情感得分"]
        )

    # 应用筛选
    filtered_df = df[df['sentiment_label'].isin(sentiment_filter)]

    if sort_by == "点赞数" and 'like_count' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('like_count', ascending=False)
    elif sort_by == "情感得分" and 'sentiment_score' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('sentiment_score', ascending=False)

    # 显示评论
    st.dataframe(
        filtered_df[['user_name', 'content_cleaned', 'sentiment_label', 'like_count']],
        use_container_width=True
    )

    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ## 使用指南

        1. **数据准备**
           - 使用Python爬虫获取B站评论数据
           - 进行数据清洗和情感分析
           - 导出为CSV格式

        2. **上传分析**
           - 在左侧上传CSV文件
           - 系统自动生成分析报告
           - 查看各种可视化图表

        3. **功能特性**
           - 情感分布分析
           - 词云生成
           - 评论详情浏览
           - 数据筛选排序

        **技术栈**: Python + Streamlit + 机器学习
                """)

        if __name__ == "__main__":
            main()