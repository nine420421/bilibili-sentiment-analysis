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

def get_available_fonts():
    """获取所有可用字体"""
    fonts = []
    try:
        for font in fm.fontManager.ttflist:
            fonts.append({
                'name': font.name,
                'path': font.fname
            })
        debug_info(f"找到 {len(fonts)} 个系统字体")
    except Exception as e:
        debug_info(f"获取字体列表失败: {e}")
    return fonts

def upload_custom_font():
    """上传自定义字体"""
    st.sidebar.subheader("📁 上传自定义字体")
    uploaded_font = st.sidebar.file_uploader("上传TTF字体文件", type=['ttf', 'otf'], key="font_uploader")
    
    if uploaded_font is not None:
        try:
            # 保存上传的字体到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                tmp_file.write(uploaded_font.getvalue())
                font_path = tmp_file.name
            st.sidebar.success(f"✅ 字体上传成功: {uploaded_font.name}")
            return font_path
        except Exception as e:
            st.sidebar.error(f"❌ 字体上传失败: {e}")
            return None
    return None

def get_best_chinese_font():
    """获取最佳中文字体"""
    # 首先检查上传的字体
    custom_font = upload_custom_font()
    if custom_font:
        return custom_font
    
    # 查找系统字体
    chinese_keywords = ['simhei', 'microsoft', 'pingfang', 'heiti', 'stsong', 'noto', 'cjk', 'sc', 'msyh', 'simsun']
    
    fonts = get_available_fonts()
    for font in fonts:
        font_name_lower = font['name'].lower()
        if any(keyword in font_name_lower for keyword in chinese_keywords):
            debug_info(f"选中字体: {font['name']}")
            return font['path']
    
    # 如果没找到中文字体，返回第一个可用字体
    if fonts:
        debug_info(f"使用默认字体: {fonts[0]['name']}")
        return fonts[0]['path']
    
    return None

def create_wordcloud_ultimate(word_freq, font_path, width=1200, height=600, max_words=100, colormap='viridis', background_color='white'):
    """终极版词云生成"""
    try:
        # 验证字体文件
        if font_path and os.path.exists(font_path):
            try:
                # 测试字体是否可用
                test_font = fm.FontProperties(fname=font_path)
                debug_info(f"字体验证通过: {os.path.basename(font_path)}")
            except Exception as e:
                debug_info(f"字体验证失败: {e}")
                # 使用默认字体
                font_path = None
        
        # 词云配置
        wc_config = {
            'width': width,
            'height': height,
            'background_color': background_color,
            'max_words': max_words,
            'colormap': colormap,
            'relative_scaling': 0.4,
            'random_state': 42,
            'prefer_horizontal': 0.8,
            'scale': 3,  # 提高分辨率
            'min_font_size': 8,
            'max_font_size': 150,
            'collocations': False,
            'normalize_plurals': False,
            'mode': 'RGBA'
        }
        
        # 如果有可用字体就使用
        if font_path:
            wc_config['font_path'] = font_path
        
        # 生成词云
        wc = WordCloud(**wc_config)
        wordcloud = wc.generate_from_frequencies(word_freq)
        
        debug_info("词云生成成功")
        return wordcloud
        
    except Exception as e:
        debug_info(f"词云生成错误: {e}")
        return None

def create_wordcloud_image_manual(word_freq, width=1200, height=600, max_words=100, colormap='viridis', background_color='white'):
    """手动创建词云图片（备用方案）"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # 设置背景
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # 获取颜色映射
        cmap = plt.cm.get_cmap(colormap)
        
        # 计算位置和大小
        words = list(word_freq.keys())[:max_words]
        counts = list(word_freq.values())[:max_words]
        
        # 归一化计数用于字体大小
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count == min_count:
            sizes = [50] * len(words)  # 所有词相同大小
        else:
            sizes = [20 + 80 * (count - min_count) / (max_count - min_count) for count in counts]
        
        # 简单布局算法
        x_positions = []
        y_positions = []
        
        for i in range(len(words)):
            # 简单的网格布局
            row = i // 8
            col = i % 8
            x = col * (width / 8) + (width / 16)
            y = height - (row * (height / (len(words)//8 + 1)) + (height / ((len(words)//8 + 1)*2)))
            x_positions.append(x)
            y_positions.append(y)
        
        # 绘制文字
        for i, (word, x, y, size) in enumerate(zip(words, x_positions, y_positions, sizes)):
            color = cmap(i / len(words))
            ax.text(x, y, word, fontsize=size, 
                   color=color, ha='center', va='center',
                   fontproperties=fm.FontProperties(fname=get_best_chinese_font()))
        
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        
        return fig
        
    except Exception as e:
        debug_info(f"手动词云生成失败: {e}")
        return None

def display_word_frequency(word_freq, title="高频词汇"):
    """显示词频的备用方案"""
    top_words = word_freq.most_common(20)
    
    if top_words:
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(words)), counts, color='lightblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=12)
        ax.set_xlabel('出现次数', fontsize=14)
        ax.set_title(title, fontsize=16, pad=20)
        
        # 在柱子上显示数值
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(count), ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
        
        # 显示可用字体信息
        if DEBUG:
            fonts = get_available_fonts()
            chinese_fonts = [f for f in fonts if any(kw in f['name'].lower() for kw in ['simhei', 'microsoft', 'pingfang', 'heiti'])]
            st.sidebar.info(f"找到 {len(chinese_fonts)} 个中文字体")

        # 情感选择
        sentiment_option = st.selectbox(
            "选择情感类型查看词云:",
            ["全部评论", "积极评论", "消极评论", "中性评论"],
            key="sentiment_selector"
        )

        # 词云设置选项
        col1, col2, col3 = st.columns(3)
        with col1:
            max_words = st.slider("最大词汇数量", 30, 150, 80, key="max_words_slider")
        with col2:
            background_color = st.selectbox("背景颜色", 
                ["white", "black", "#f0f0f0", "#f8f9fa"], key="bg_color_selector")
        with col3:
            colormap_option = st.selectbox("颜色方案", 
                ["viridis", "plasma", "inferno", "spring", "summer", "autumn", "winter"], 
                key="colormap_selector")

        # 生成模式选择
        generate_mode = st.radio("生成模式:", 
                                ["自动词云", "手动布局"], 
                                help="自动词云使用wordcloud库，手动布局使用matplotlib直接绘制")

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
                    
                    # 获取字体
                    font_path = get_best_chinese_font()
                    
                    if font_path:
                        st.success(f"🎨 使用字体: {os.path.basename(font_path)}")
                    else:
                        st.warning("⚠️ 使用默认字体")

                    # 根据模式生成词云
                    if generate_mode == "自动词云":
                        wordcloud = create_wordcloud_ultimate(
                            word_freq, 
                            font_path, 
                            max_words=max_words,
                            colormap=colormap_option,
                            background_color=background_color
                        )

                        if wordcloud is not None:
                            # 显示词云
                            fig, ax = plt.subplots(figsize=(16, 9))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{sentiment_option} - 词云图', 
                                       fontsize=22, pad=20, fontweight='bold')
                            fig.patch.set_facecolor(background_color)
                            st.pyplot(fig)
                            plt.close(fig)
                            st.success("🎉 自动词云生成成功！")
                        else:
                            st.error("❌ 自动词云生成失败，尝试手动布局")
                            generate_mode = "手动布局"

                    if generate_mode == "手动布局":
                        st.info("🔄 使用手动布局生成词云")
                        fig = create_wordcloud_image_manual(
                            word_freq,
                            max_words=max_words,
                            colormap=colormap_option,
                            background_color=background_color
                        )
                        
                        if fig is not None:
                            st.pyplot(fig)
                            plt.close(fig)
                            st.success("🎉 手动布局词云生成成功！")
                        else:
                            st.error("❌ 所有词云生成方案都失败了")
                            st.info("🔄 显示词频条形图作为替代")
                            display_word_frequency(word_freq, f'{sentiment_option} - 高频词汇')

                    # 显示高频词表格
                    st.subheader("📋 高频词汇TOP20")
                    top_words = word_freq.most_common(20)
                    word_df = pd.DataFrame(top_words, columns=['词汇', '出现次数'])
                    st.dataframe(word_df, use_container_width=True, height=400)

                else:
                    st.warning("⚠️ 没有找到足够的词汇数据来生成词云")

        # 评论详情查看（保持不变）
        # ... 这里保持原有的评论详情代码

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

        3. **上传字体(可选)**: 如果词云不显示文字，可以上传TTF字体文件

        4. **生成词云**: 选择手动布局模式确保显示文字
        """)

if __name__ == "__main__":
    main()
