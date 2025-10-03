import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import platform
import base64
import requests
from io import BytesIO

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 调试信息开关
DEBUG = st.sidebar.checkbox("显示调试信息", value=False)

def debug_info(message):
    """显示调试信息"""
    if DEBUG:
        st.sidebar.write(f"🔍 {message}")

def get_chinese_font():
    """获取中文字体 - 多种方案"""
    # 方案1: 尝试系统字体
    system = platform.system()
    font_paths = []
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc", 
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf"
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Arial.ttf"
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/arphic/uming.ttc"
        ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            debug_info(f"找到字体: {font_path}")
            return font_path
    
    # 方案2: 如果系统字体都找不到，使用默认字体（可能不支持中文）
    debug_info("未找到系统字体文件")
    return None

def display_word_frequency(word_freq):
    """显示词频的备用方案"""
    top_words = word_freq.most_common(15)
    
    if top_words:
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, counts)
        ax.set_xlabel('出现次数')
        ax.set_title('高频词汇（备用显示）')
        
        # 设置中文字体
        font_path = get_chinese_font()
        if font_path:
            plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
        
        st.pyplot(fig)

def get_words_from_segmented(segmented_str):
    """从分词字符串中提取词汇"""
    if pd.isna(segmented_str) or not isinstance(segmented_str, str):
        return []
    
    # 多种格式处理
    try:
        # 格式1: ['word1', 'word2', 'word3']
        if segmented_str.startswith('[') and segmented_str.endswith(']'):
            words = segmented_str.strip("[]").replace("'", "").replace('"', '').split(", ")
        # 格式2: "word1" "word2" "word3"
        elif '"' in segmented_str:
            words = segmented_str.replace('"', '').split()
        # 格式3: word1 word2 word3
        else:
            words = segmented_str.split()
        
        # 过滤短词和空词
        filtered_words = [word.strip() for word in words if len(word.strip()) > 1]
        debug_info(f"从字符串解析出 {len(filtered_words)} 个词汇")
        return filtered_words
    except Exception as e:
        debug_info(f"分词解析错误: {e}")
        return []

# 设置页面
st.set_page_config(
    page_title="B站评论情感分析系统",
    page_icon="📊",
    layout="wide"
)

def main():
    # 标题和介绍
    st.title("🎯 B站视频评论情感分析系统")
    st.markdown("---")

    # 侧边栏 - 文件上传和设置
    st.sidebar.header("📁 数据上传")
    uploaded_file = st.sidebar.file_uploader("上传评论数据CSV文件", type=['csv'])

    if uploaded_file is not None:
        # 读取数据
        df = pd.read_csv(uploaded_file)
        
        # 数据预处理
        if 'post_time' in df.columns:
            df['post_time'] = pd.to_datetime(df['post_time'])

        # 数据检查
        st.sidebar.subheader("📋 数据检查")
        st.sidebar.write(f"数据形状: {df.shape}")
        st.sidebar.write("列名:", list(df.columns))
        
        # 检查必要列是否存在
        required_columns = ['segmented_words', 'sentiment_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ 缺少必要列: {missing_columns}")
            st.stop()
        
        # 显示数据样例
        if st.sidebar.checkbox("显示数据样例"):
            st.sidebar.dataframe(df.head(3))
        
        # 检查分词列数据
        if st.sidebar.checkbox("检查分词数据"):
            st.sidebar.write("分词列样例:")
            for i, seg_text in enumerate(df['segmented_words'].head(3)):
                st.sidebar.write(f"{i+1}: {seg_text}")

        # 显示基本信息
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
            neutral_count = len(df[df['sentiment_label'] == '中性'])
            st.metric("中性评论", neutral_count)

        # 情感分布饼图
        st.header("🎭 情感分布分析")
        col1, col2 = st.columns(2)

        with col1:
            # 使用plotly创建交互式饼图
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

        # 时间趋势分析
        st.header("📈 评论时间趋势")
        if 'post_time' in df.columns:
            daily_stats = df.groupby(df['post_time'].dt.date).agg({
                'sentiment_score': 'mean',
                'comment_id': 'count'
            }).reset_index()

            col1, col2 = st.columns(2)

            with col1:
                fig_trend = px.line(
                    daily_stats, x='post_time', y='sentiment_score',
                    title='每日平均情感得分趋势',
                    labels={'sentiment_score': '平均情感得分', 'post_time': '日期'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with col2:
                fig_count = px.bar(
                    daily_stats, x='post_time', y='comment_id',
                    title='每日评论数量',
                    labels={'comment_id': '评论数量', 'post_time': '日期'}
                )
                st.plotly_chart(fig_count, use_container_width=True)

        # 词云分析
        st.header("☁️ 词云分析")

        # 情感选择
        sentiment_option = st.selectbox(
            "选择情感类型查看词云:",
            ["全部评论", "积极评论", "消极评论", "中性评论"]
        )

        # 词云设置选项
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("最大词汇数量", 50, 200, 100)
        with col2:
            background_color = st.selectbox("背景颜色", ["white", "black", "gray"])

        # 生成词云
        if st.button("生成词云"):
            with st.spinner("正在生成词云..."):
                # 根据选择过滤数据
                if sentiment_option == "全部评论":
                    target_df = df
                    color_map = 'viridis'
                elif sentiment_option == "积极评论":
                    target_df = df[df['sentiment_label'] == '积极']
                    color_map = 'spring'
                elif sentiment_option == "消极评论":
                    target_df = df[df['sentiment_label'] == '消极']
                    color_map = 'autumn'
                else:
                    target_df = df[df['sentiment_label'] == '中性']
                    color_map = 'winter'

                debug_info(f"目标数据行数: {len(target_df)}")
                if len(target_df) > 0:
                    debug_info(f"分词样例: {target_df['segmented_words'].iloc[0]}")

                # 准备文本数据
                all_words = []
                for seg_text in target_df['segmented_words']:
                    words = get_words_from_segmented(seg_text)
                    all_words.extend(words)

                if all_words:
                    # 统计词频
                    word_freq = Counter(all_words)
                    
                    # 调试信息
                    st.write(f"✅ 成功提取 {len(all_words)} 个词汇，{len(word_freq)} 个不同词汇")
                    st.write(f"📊 前5个高频词: {word_freq.most_common(5)}")
                    
                    # 获取字体
                    font_path = get_chinese_font()
                    
                    # 创建词云配置
                    wc_config = {
                        'width': 800,
                        'height': 400,
                        'background_color': background_color,
                        'max_words': max_words,
                        'colormap': color_map,
                        'relative_scaling': 0.5,
                        'random_state': 42
                    }
                    
                    # 如果有字体就添加
                    if font_path:
                        wc_config['font_path'] = font_path
                        st.success(f"🎨 使用字体: {os.path.basename(font_path)}")
                    else:
                        st.warning("🔤 使用默认字体（可能不支持中文）")

                    try:
                        # 生成词云
                        wordcloud = WordCloud(**wc_config).generate_from_frequencies(word_freq)
                        
                        # 显示词云
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{sentiment_option} - 词云图', fontsize=16, pad=20)
                        
                        # 确保使用支持中文的字体
                        if font_path:
                            plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ 生成词云时出错: {str(e)}")
                        
                        # 备用方案：直接显示词频
                        st.info("🔄 尝试备用方案：显示词频条形图")
                        display_word_frequency(word_freq)

                    # 显示高频词表格
                    st.subheader("📋 高频词汇TOP20")
                    top_words = word_freq.most_common(20)
                    
                    # 创建数据框显示
                    word_df = pd.DataFrame(top_words, columns=['词汇', '出现次数'])
                    st.dataframe(word_df, use_container_width=True)
                    
                    # 同时显示条形图
                    fig_bar = px.bar(
                        x=[count for _, count in top_words],
                        y=[word for word, _ in top_words],
                        orientation='h',
                        title='高频词汇排行榜',
                        labels={'x': '出现次数', 'y': '词汇'},
                        color=[count for _, count in top_words],
                        color_continuous_scale='blues'
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.warning("⚠️ 没有找到足够的词汇数据来生成词云")
                    st.info("💡 提示：请检查数据中的 'segmented_words' 列格式是否正确")

        # 评论详情查看
        st.header("💬 评论详情浏览")

        # 情感筛选
        sentiment_filter = st.multiselect(
            "筛选情感类型:",
            options=['积极', '消极', '中性'],
            default=['积极', '消极', '中性']
        )

        # 点赞数范围筛选
        min_likes = 0
        max_likes = 100
        if 'like_count' in df.columns:
            min_likes = int(df['like_count'].min())
            max_likes = int(df['like_count'].max())
            
            min_likes, max_likes = st.slider(
                "点赞数范围:",
                min_value=min_likes,
                max_value=max_likes,
                value=(0, max_likes)
            )

        # 应用筛选
        filtered_df = df[df['sentiment_label'].isin(sentiment_filter)]
        
        if 'like_count' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['like_count'] >= min_likes) & 
                (filtered_df['like_count'] <= max_likes)
            ]

        # 显示筛选后的评论
        st.subheader(f"筛选结果: {len(filtered_df)} 条评论")

        # 排序选项
        sort_options = ["按点赞数降序"]
        if 'sentiment_score' in df.columns:
            sort_options.append("按情感得分降序")
        if 'post_time' in df.columns:
            sort_options.append("按时间降序")
            
        sort_option = st.selectbox("排序方式:", sort_options)

        if sort_option == "按点赞数降序" and 'like_count' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('like_count', ascending=False)
        elif sort_option == "按情感得分降序" and 'sentiment_score' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('sentiment_score', ascending=False)
        elif sort_option == "按时间降序" and 'post_time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('post_time', ascending=False)

        # 分页显示
        page_size = 10
        total_pages = max(1, (len(filtered_df) // page_size) + 1)

        page_number = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        # 显示评论
        for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            # 根据情感设置颜色
            if row['sentiment_label'] == '积极':
                color = "🟢"
                border_color = "#2E8B57"
            elif row['sentiment_label'] == '消极':
                color = "🔴"
                border_color = "#DC143C"
            else:
                color = "🔵"
                border_color = "#1E90FF"

            # 显示评论卡片
            st.markdown(f"""
            <div style="border-left: 4px solid {border_color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{color} {row['user_name'] if 'user_name' in row else '匿名用户'}</strong>
                    <span>👍 {row['like_count'] if 'like_count' in row else 0} | 情感: {row['sentiment_score'] if 'sentiment_score' in row else 'N/A'}</span>
                </div>
                <p style="margin: 5px 0;">{row['content_cleaned'] if 'content_cleaned' in row else '无内容'}</p>
                <small>时间: {row['post_time'] if 'post_time' in row else '未知'}</small>
            </div>
            """, unsafe_allow_html=True)

    else:
        # 没有上传文件时的展示
        st.info("👆 请在左侧上传B站评论数据CSV文件开始分析")

        # 显示使用说明
        st.header("📖 使用说明")
        st.markdown("""
        1. **准备数据**: 确保CSV文件包含以下字段：
           - `content_cleaned`: 清洗后的评论内容
           - `sentiment_label`: 情感标签（积极/消极/中性）
           - `sentiment_score`: 情感得分（0-1）
           - `like_count`: 点赞数
           - `user_name`: 用户名
           - `post_time`: 发布时间
           - `segmented_words`: 分词结果

        2. **上传文件**: 在左侧边栏上传CSV文件

        3. **探索分析**: 查看各种可视化图表和统计信息

        4. **筛选浏览**: 根据需要筛选和查看具体评论
        """)

        # 显示示例图片（如果有的话）
        st.header("🎯 功能预览")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("情感分析")
            st.markdown("""
            - 情感分布饼图
            - 情感得分直方图  
            - 时间趋势分析
            - 互动关系分析
            """)

        with col2:
            st.subheader("文本分析")
            st.markdown("""
            - 动态词云生成
            - 高频词汇统计
            - 情感词汇对比
            - 评论详情浏览
            """)

if __name__ == "__main__":
    main()
