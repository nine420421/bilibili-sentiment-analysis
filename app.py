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
import numpy as np
from PIL import Image
import matplotlib.font_manager as fm
import tempfile

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 调试信息开关
DEBUG = st.sidebar.checkbox("显示调试信息", value=True)

def debug_info(message):
    """显示调试信息"""
    if DEBUG:
        st.sidebar.write(f"🔍 {message}")

def get_simple_font():
    """获取简单可用的字体"""
    try:
        # 直接使用matplotlib默认字体
        return None
    except Exception as e:
        debug_info(f"字体获取失败: {e}")
        return None

def create_simple_wordcloud_direct(word_freq, max_words=100, colormap='viridis', background_color='white'):
    """直接创建词云 - 简化版本"""
    try:
        # 使用最基本的配置
        wc = WordCloud(
            width=1200,
            height=600,
            background_color=background_color,
            max_words=max_words,
            colormap=colormap,
            random_state=42,
            relative_scaling=0.3,
            min_font_size=10,
            max_font_size=120,
            collocations=False
        )
        
        # 生成词云
        wordcloud = wc.generate_from_frequencies(word_freq)
        return wordcloud
    except Exception as e:
        debug_info(f"简单词云失败: {e}")
        return None

def create_text_cloud_manual(word_freq, max_words=50, colormap='viridis', background_color='white'):
    """创建文本云 - 确保显示文字"""
    try:
        # 限制词汇数量
        top_words = word_freq.most_common(max_words)
        if not top_words:
            return None
            
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        
        # 设置背景
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # 计算字体大小
        max_count = max(counts)
        min_count = min(counts)
        
        # 简单的网格布局
        cols = 6  # 每行6个词
        rows = (len(words) + cols - 1) // cols
        
        # 颜色映射
        cmap = plt.cm.get_cmap(colormap)
        
        for i, (word, count) in enumerate(zip(words, counts)):
            # 计算位置
            row = i // cols
            col = i % cols
            
            # 计算字体大小 (20-60之间)
            if max_count == min_count:
                fontsize = 40
            else:
                fontsize = 20 + 40 * (count - min_count) / (max_count - min_count)
            
            # 计算位置
            x = (col + 0.5) * (1.0 / cols)
            y = 1.0 - (row + 0.5) * (1.0 / rows)
            
            # 计算颜色
            color = cmap(i / len(words))
            
            # 绘制文字
            ax.text(x, y, word, 
                   fontsize=fontsize,
                   ha='center', va='center',
                   color=color,
                   transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('词云图 - 手动布局', fontsize=20, pad=20)
        
        return fig
        
    except Exception as e:
        debug_info(f"文本云创建失败: {e}")
        return None

def create_fallback_chart(word_freq, title="高频词汇"):
    """创建备用图表"""
    try:
        top_words = word_freq.most_common(20)
        
        if not top_words:
            return None
            
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        # 使用plotly创建水平条形图
        fig = px.bar(
            x=counts,
            y=words,
            orientation='h',
            title=title,
            labels={'x': '出现次数', 'y': '词汇'},
            color=counts,
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            showlegend=False,
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        debug_info(f"备用图表创建失败: {e}")
        return None

def get_words_from_segmented(segmented_str):
    """从分词字符串中提取词汇"""
    if pd.isna(segmented_str) or not isinstance(segmented_str, str):
        return []
    
    try:
        clean_str = segmented_str.strip()
        
        # 处理列表格式
        if clean_str.startswith('[') and clean_str.endswith(']'):
            content = clean_str[1:-1].replace("'", "").replace('"', '')
            words = [word.strip() for word in content.split(',')]
        else:
            words = clean_str.split()
        
        # 过滤
        filtered_words = []
        for word in words:
            word_clean = word.strip()
            if (len(word_clean) > 0 and 
                not word_clean.isspace() and
                word_clean not in [' ', '', '\\n', '\\t']):
                filtered_words.append(word_clean)
        
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
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 成功读取数据，共 {len(df)} 行")
        except Exception as e:
            st.error(f"❌ 读取数据失败: {e}")
            return
        
        # 数据预处理
        if 'post_time' in df.columns:
            df['post_time'] = pd.to_datetime(df['post_time'], errors='coerce')

        # 数据检查
        st.sidebar.subheader("📋 数据检查")
        st.sidebar.write(f"数据形状: {df.shape}")
        
        # 检查必要列是否存在
        required_columns = ['segmented_words', 'sentiment_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ 缺少必要列: {missing_columns}")
            st.info("请确保CSV文件包含 'segmented_words' 和 'sentiment_label' 列")
            return

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
        else:
            st.info("数据中没有时间信息，无法显示时间趋势")

        # 词云分析
        st.header("☁️ 词云分析")

        # 情感选择
        sentiment_option = st.selectbox(
            "选择情感类型查看词云:",
            ["全部评论", "积极评论", "消极评论", "中性评论"],
            key="sentiment_selector"
        )

        # 词云设置选项
        col1, col2, col3 = st.columns(3)
        with col1:
            max_words = st.slider("最大词汇数量", 20, 100, 50, key="max_words_slider")
        with col2:
            background_color = st.selectbox("背景颜色", 
                ["white", "black", "#f0f0f0"], key="bg_color_selector")
        with col3:
            colormap_option = st.selectbox("颜色方案", 
                ["viridis", "plasma", "spring", "summer", "autumn", "winter"], 
                key="colormap_selector")

        # 生成词云
        if st.button("生成词云", type="primary", key="generate_wordcloud"):
            with st.spinner("正在生成词云..."):
                # 根据选择过滤数据
                if sentiment_option == "全部评论":
                    target_df = df
                elif sentiment_option == "积极评论":
                    target_df = df[df['sentiment_label'] == '积极']
                elif sentiment_option == "消极评论":
                    target_df = df[df['sentiment_label'] == '消极']
                else:
                    target_df = df[df['sentiment_label'] == '中性']

                debug_info(f"目标数据行数: {len(target_df)}")
                
                if len(target_df) == 0:
                    st.warning(f"⚠️ 没有找到 {sentiment_option} 的数据")
                    return

                # 准备文本数据
                all_words = []
                for seg_text in target_df['segmented_words']:
                    words = get_words_from_segmented(seg_text)
                    all_words.extend(words)

                if all_words:
                    # 统计词频
                    word_freq = Counter(all_words)
                    
                    # 显示统计信息
                    st.success(f"✅ 成功提取 {len(all_words)} 个词汇，{len(word_freq)} 个不同词汇")
                    
                    # 显示前10个高频词
                    top_10 = word_freq.most_common(10)
                    top_words_str = "、".join([f"{word}({count})" for word, count in top_10])
                    st.info(f"📊 前10个高频词: {top_words_str}")

                    # 方案1: 尝试简单词云
                    st.subheader("方案1: 简单词云")
                    wordcloud = create_simple_wordcloud_direct(
                        word_freq,
                        max_words=max_words,
                        colormap=colormap_option,
                        background_color=background_color
                    )

                    if wordcloud is not None:
                        # 显示词云
                        fig, ax = plt.subplots(figsize=(16, 8))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{sentiment_option} - 词云图', fontsize=20, pad=20)
                        fig.patch.set_facecolor(background_color)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.success("🎉 简单词云生成成功！")
                    else:
                        st.warning("❌ 简单词云生成失败，尝试方案2")

                        # 方案2: 手动文本云
                        st.subheader("方案2: 文本云布局")
                        text_fig = create_text_cloud_manual(
                            word_freq,
                            max_words=min(max_words, 30),  # 手动布局限制词汇数
                            colormap=colormap_option,
                            background_color=background_color
                        )

                        if text_fig is not None:
                            st.pyplot(text_fig)
                            plt.close(text_fig)
                            st.success("🎉 文本云生成成功！")
                        else:
                            st.error("❌ 文本云生成失败")

                            # 方案3: 使用plotly备用图表
                            st.subheader("方案3: 高频词汇图表")
                            fallback_fig = create_fallback_chart(
                                word_freq, 
                                f'{sentiment_option} - 高频词汇'
                            )
                            
                            if fallback_fig is not None:
                                st.plotly_chart(fallback_fig, use_container_width=True)
                                st.info("📊 使用高频词汇图表作为词云替代")
                            else:
                                st.error("❌ 所有方案都失败了")

                    # 显示高频词表格
                    st.subheader("📋 高频词汇TOP20")
                    top_words = word_freq.most_common(20)
                    word_df = pd.DataFrame(top_words, columns=['词汇', '出现次数'])
                    st.dataframe(word_df, use_container_width=True, height=400)

                else:
                    st.warning("⚠️ 没有找到足够的词汇数据来生成词云")

        # 评论详情查看
        st.header("💬 评论详情浏览")

        # 情感筛选
        sentiment_filter = st.multiselect(
            "筛选情感类型:",
            options=['积极', '消极', '中性'],
            default=['积极', '消极', '中性'],
            key="sentiment_filter"
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
                value=(0, max_likes),
                key="like_slider"
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
        sort_options = ["默认排序"]
        if 'like_count' in df.columns:
            sort_options.append("按点赞数降序")
        if 'sentiment_score' in df.columns:
            sort_options.append("按情感得分降序")
        if 'post_time' in df.columns:
            sort_options.append("按时间降序")
            
        sort_option = st.selectbox("排序方式:", sort_options, key="sort_selector")

        if sort_option == "按点赞数降序" and 'like_count' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('like_count', ascending=False)
        elif sort_option == "按情感得分降序" and 'sentiment_score' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('sentiment_score', ascending=False)
        elif sort_option == "按时间降序" and 'post_time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('post_time', ascending=False)

        # 分页显示
        page_size = 10
        total_pages = max(1, (len(filtered_df) // page_size) + 1)

        page_number = st.number_input("页码", min_value=1, max_value=total_pages, value=1, key="page_selector")
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
           - `segmented_words`: 分词结果
           - `sentiment_label`: 情感标签
           - 其他可选字段

        2. **上传文件**: 在左侧边栏上传CSV文件

        3. **生成词云**: 系统会尝试多种方案确保词云显示

        4. **探索分析**: 查看各种可视化图表和统计信息
        """)

if __name__ == "__main__":
    main()
