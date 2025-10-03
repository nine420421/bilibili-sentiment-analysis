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
import os

# 设置页面
st.set_page_config(
    page_title="B站评论情感分析系统",
    page_icon="🔍",
    layout="wide"
)

# 修正字体设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # 如果字体设置失败，使用默认字体

def main():
    st.title("🔍 B站视频评论情感分析系统")
    st.markdown("---")
    st.info("📊 上传你的B站评论数据CSV文件，系统将自动进行情感分析并生成可视化报告。")

    # 示例数据
    sample_data = {
        'user_name': ['用户A', '用户B', '用户C', '用户D'],
        'content_cleaned': ['这个视频很好看', '内容一般般', '太棒了，推荐', '不喜欢这个'],
        'sentiment_label': ['积极', '消极', '积极', '消极'],
        'sentiment_score': [0.85, 0.25, 0.90, 0.15],
        'like_count': [156, 23, 289, 12],
        'post_time': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    }

    st.sidebar.header("📤 数据上传")
    uploaded_file = st.sidebar.file_uploader("选择CSV文件", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 成功加载 {len(df)} 条评论数据")
        except Exception as e:
            st.error(f"❌ 文件读取错误: {e}")
            st.warning("⚠️ 使用示例数据进行演示")
            df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(sample_data)
        st.warning("⚠️ 当前使用示例数据，请上传CSV文件")

    # 数据概览
    st.header("📈 数据概览")
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
        total_likes = df['like_count'].sum()
        st.metric("总点赞数", f"{total_likes}")

    # 情感分析图表
    st.header("📊 情感分析")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='评论情感分布',
            color=sentiment_counts.index,
            color_discrete_map={'积极': '#2E86AB', '消极': '#A23B72'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        if 'sentiment_score' in df.columns:
            fig_hist = px.histogram(
                df, x='sentiment_score',
                title='情感得分分布',
                nbins=20,
                color_discrete_sequence=['#2E86AB']
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)

    # 词云生成 - 修正了缩进，确保在main函数内部
    st.header("☁️ 词云分析")

    if st.button("生成词云图", type="primary"):
        with st.spinner("正在生成词云..."):
            # 准备文本数据
            all_text = ' '.join(df['content_cleaned'].astype(str))
            
            # 字体路径处理 - 修正云端路径
            font_paths = [
                './fonts/SimHei.ttf',           # 项目字体文件夹
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux系统字体
                None  # 最后尝试不使用字体
            ]
            
            selected_font_path = None
            for font_path in font_paths:
                if font_path is None:
                    selected_font_path = None
                    break
                try:
                    if os.path.exists(font_path):
                        selected_font_path = font_path
                        st.success(f"使用字体: {font_path}")
                        break
                except:
                    continue
            
            if selected_font_path is None:
                st.warning("⚠️ 未找到中文字体文件，词云可能无法正确显示中文")
            
            # 生成词云
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    font_path=selected_font_path,  # 使用找到的字体路径
                    stopwords=None,  # 可以添加中文停用词
                    collocations=False  # 避免重复词语
                ).generate(all_text)
                
                # 显示词云
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('评论词云图', fontsize=16)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ 生成词云时出错: {str(e)}")
                st.info("尝试使用默认设置重新生成...")
                
                # 备用方案：不使用字体
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    font_path=None
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('评论词云图', fontsize=16)
                st.pyplot(fig)

    # 评论详情
    st.header("💬 评论详情")
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

    filtered_df = df[df['sentiment_label'].isin(sentiment_filter)]

    if sort_by == "点赞数" and 'like_count' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='like_count', ascending=False)
    elif sort_by == "情感得分" and 'sentiment_score' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='sentiment_score', ascending=False)

    st.dataframe(
        filtered_df[['user_name', 'content_cleaned', 'sentiment_label', 'sentiment_score', 'like_count']],
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

        4. **技术栈**: Python + Streamlit
        """)

if __name__ == "__main__":
    main()
