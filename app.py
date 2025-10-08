import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import numpy as np
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def create_advanced_bar_chart(word_freq, title="高频词汇云图"):
    """创建高级条形图"""
    try:
        top_words = word_freq.most_common(30)
        
        if not top_words:
            return None
            
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        # 创建水平条形图，但使用圆形标记
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=words,
            x=counts,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=counts,
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>出现次数: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="出现次数",
            yaxis_title="词汇",
            showlegend=False,
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        st.error(f"高级条形图失败: {e}")
        return None

def create_word_importance_chart(word_freq, title="词汇重要性分布"):
    """创建词汇重要性图表 - 修复版本"""
    try:
        top_words = word_freq.most_common(25)
        
        if not top_words:
            return None
            
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        # 创建散点图显示词汇重要性
        fig = go.Figure()
        
        # 修复：使用列表而不是range对象
        x_values = list(range(len(words)))
        
        fig.add_trace(go.Scatter(
            x=x_values,  # 修复：使用列表
            y=counts,
            mode='markers+text',
            text=words,
            textposition="top center",
            marker=dict(
                size=[count/2 for count in counts],
                color=counts,
                colorscale='Rainbow',
                opacity=0.7,
                line=dict(width=2, color='darkgray')
            ),
            textfont=dict(size=14),
            hovertemplate='<b>%{text}</b><br>出现次数: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                title=""
            ),
            yaxis=dict(
                title="出现次数",
                gridcolor='lightgray'
            ),
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"词汇重要性图表失败: {e}")
        return None

def create_word_frequency_heatmap(word_freq, title="词汇频率热力图"):
    """创建词汇频率热力图"""
    try:
        top_words = word_freq.most_common(20)
        
        if not top_words:
            return None
            
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        # 创建热力图样式的条形图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=words,
            y=counts,
            marker=dict(
                color=counts,
                colorscale='Hot',
                line=dict(color='white', width=1)
            ),
            text=counts,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>出现次数: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="词汇",
            yaxis_title="出现次数",
            showlegend=False,
            height=500,
            xaxis={'tickangle': 45}
        )
        
        return fig
        
    except Exception as e:
        st.error(f"热力图创建失败: {e}")
        return None

def create_word_network_chart(word_freq, title="词汇网络图"):
    """创建词汇网络图"""
    try:
        top_words = word_freq.most_common(15)
        
        if not top_words:
            return None
            
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        # 创建极坐标图
        fig = go.Figure()
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(words), endpoint=False).tolist()
        
        fig.add_trace(go.Scatterpolar(
            r=counts,
            theta=words,
            fill='toself',
            line=dict(color='blue'),
            marker=dict(
                size=[count/3 for count in counts],
                color=counts,
                colorscale='Viridis'
            ),
            text=counts,
            hovertemplate='<b>%{theta}</b><br>出现次数: %{r}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(counts)]
                )
            ),
            showlegend=False,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"网络图创建失败: {e}")
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

        # 词云分析 - 使用替代方案
        st.header("☁️ 词汇可视化分析")

        # 情感选择
        sentiment_option = st.selectbox(
            "选择情感类型:",
            ["全部评论", "积极评论", "消极评论", "中性评论"],
            key="sentiment_selector"
        )

        # 可视化方案选择
        viz_option = st.selectbox(
            "选择可视化方案:",
            [ "高级条形图", "词汇重要性图", "频率热力图", "网络图"],
            help="选择不同的方式来可视化词汇分布"
        )

        # 词汇数量设置
        max_words = st.slider("显示词汇数量", 10, 50, 25, key="max_words_slider")

        # 生成图表
        if st.button("生成可视化", type="primary", key="generate_viz"):
            with st.spinner("正在生成可视化图表..."):
                # 根据选择过滤数据
                if sentiment_option == "全部评论":
                    target_df = df
                elif sentiment_option == "积极评论":
                    target_df = df[df['sentiment_label'] == '积极']
                elif sentiment_option == "消极评论":
                    target_df = df[df['sentiment_label'] == '消极']
                else:
                    target_df = df[df['sentiment_label'] == '中性']

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

                    # 根据选择的方案生成图表
                    title_suffix = f"{sentiment_option} - "
                
                        
                    if viz_option == "高级条形图":
                        fig = create_advanced_bar_chart(
                            word_freq,
                            title=title_suffix + '高频词汇图'
                        )
                        
                    elif viz_option == "词汇重要性图":
                        fig = create_word_importance_chart(
                            word_freq,
                            title=title_suffix + '词汇重要性分布'
                        )
                        
                    elif viz_option == "频率热力图":
                        fig = create_word_frequency_heatmap(
                            word_freq,
                            title=title_suffix + '词汇频率热力图'
                        )
                        
                    elif viz_option == "网络图":
                        fig = create_word_network_chart(
                            word_freq,
                            title=title_suffix + '词汇网络图'
                        )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("🎉 可视化生成成功！")
                    else:
                        st.error("❌ 可视化生成失败")

                    # 显示高频词表格
                    st.subheader("📋 高频词汇TOP20")
                    top_words = word_freq.most_common(20)
                    word_df = pd.DataFrame(top_words, columns=['词汇', '出现次数'])
                    st.dataframe(word_df, use_container_width=True, height=400)

                else:
                    st.warning("⚠️ 没有找到足够的词汇数据")

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

        3. **探索分析**: 查看各种可视化图表和统计信息
        """)

if __name__ == "__main__":
    main()







