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

   # 词云生成
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def word_cloud_analysis():
    print("🚀 开始词云与高频词分析...")
    print("=" * 50)

    # 1. 读取情感分析结果
    print("1. 读取情感分析数据...")
    df = pd.read_csv('bilibili_comments_with_sentiment.csv')
    print(f"   ✅ 分析数据: {len(df)} 条评论")

    # 2. 准备不同情感的词频数据
    print("\n2. 准备词频数据...")

    def get_words_from_segmented(segmented_str):
        """从分词语句中提取词汇列表"""
        if isinstance(segmented_str, str):
            # 移除方括号和引号，然后分割
            words = segmented_str.strip("[]").replace("'", "").split(", ")
            return [word for word in words if len(word) > 1]
        return []

    # 获取所有评论的词汇
    all_words = []
    for seg_text in df['segmented_words']:
        all_words.extend(get_words_from_segmented(seg_text))

    # 按情感分类获取词汇
    positive_words = []
    negative_words = []
    neutral_words = []

    for idx, row in df.iterrows():
        words = get_words_from_segmented(row['segmented_words'])
        if row['sentiment_label'] == '积极':
            positive_words.extend(words)
        elif row['sentiment_label'] == '消极':
            negative_words.extend(words)
        else:
            neutral_words.extend(words)

    print(f"   总词汇量: {len(all_words)} 个词")
    print(f"   积极评论词汇: {len(positive_words)} 个")
    print(f"   消极评论词汇: {len(negative_words)} 个")
    print(f"   中性评论词汇: {len(neutral_words)} 个")

    # 3. 生成词云
    print("\n3. 生成词云图...")

    # 设置中文字体（重要！）
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体字体

    # 全量词云
    print("   生成全量词云...")
    all_text = ' '.join(all_words)
    wordcloud_all = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(all_text)

    # 积极评论词云
    print("   生成积极评论词云...")
    positive_text = ' '.join(positive_words)
    wordcloud_positive = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        background_color='white',
        max_words=80,
        colormap='spring'  # 暖色调
    ).generate(positive_text)

    # 消极评论词云
    print("   生成消极评论词云...")
    negative_text = ' '.join(negative_words)
    wordcloud_negative = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        background_color='white',
        max_words=80,
        colormap='autumn'  # 冷色调
    ).generate(negative_text)

    # 4. 绘制词云图
    print("\n4. 绘制词云图...")
    plt.figure(figsize=(20, 12))

    # 全量词云
    plt.subplot(2, 2, 1)
    plt.imshow(wordcloud_all, interpolation='bilinear')
    plt.title('全量评论词云图', fontsize=16, fontweight='bold')
    plt.axis('off')

    # 积极评论词云
    plt.subplot(2, 2, 2)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title('积极评论词云图', fontsize=16, fontweight='bold', color='green')
    plt.axis('off')

    # 消极评论词云
    plt.subplot(2, 2, 3)
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title('消极评论词云图', fontsize=16, fontweight='bold', color='red')
    plt.axis('off')

    # 5. 高频词分析
    print("\n5. 高频词分析...")

    def get_top_words(word_list, top_n=15):
        """获取前N个高频词"""
        word_count = Counter(word_list)
        return word_count.most_common(top_n)

    # 获取各类高频词
    top_all = get_top_words(all_words)
    top_positive = get_top_words(positive_words)
    top_negative = get_top_words(negative_words)

    # 绘制高频词条形图
    plt.subplot(2, 2, 4)

    # 准备数据 - 取前10个词
    words_all = [word for word, count in top_all[:10]]
    counts_all = [count for word, count in top_all[:10]]

    y_pos = np.arange(len(words_all))
    plt.barh(y_pos, counts_all, color='skyblue')
    plt.yticks(y_pos, words_all, fontproperties='SimHei')
    plt.xlabel('出现频次')
    plt.title('全量评论高频词TOP10', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # 倒置Y轴，让最高的在顶部

    plt.tight_layout()
    plt.savefig('word_cloud_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ 词云图已保存: word_cloud_analysis.png")

    # 6. 高频词对比分析
    print("\n6. 高频词对比分析:")

    print(f"\n   📊 全量评论高频词TOP10:")
    for i, (word, count) in enumerate(top_all[:10], 1):
        print(f"      {i:2d}. {word:8s} : {count:3d} 次")

    print(f"\n   💚 积极评论特有高频词:")
    positive_unique = [word for word, count in top_positive if word not in [w for w, c in top_negative[:10]]]
    print(f"      {', '.join(positive_unique[:8])}")

    print(f"\n   💔 消极评论特有高频词:")
    negative_unique = [word for word, count in top_negative if word not in [w for w, c in top_positive[:10]]]
    print(f"      {', '.join(negative_unique[:8])}")

    # 7. 情感词汇分析
    print(f"\n7. 情感词汇洞察:")

    # 定义一些情感词汇示例
    positive_keywords = ['支持', '厉害', '点赞', '优秀', '发展', '进步', '希望', '感谢']
    negative_keywords = ['质疑', '反对', '浪费', '问题', '困难', '担心', '失望', '批评']

    pos_count = sum(1 for word in all_words if word in positive_keywords)
    neg_count = sum(1 for word in all_words if word in negative_keywords)

    print(f"   积极情感词汇出现: {pos_count} 次")
    print(f"   消极情感词汇出现: {neg_count} 次")
    print(f"   情感词汇比例: {pos_count / (pos_count + neg_count) * 100:.1f}% 积极")

    return {
        'all_words': all_words,
        'positive_words': positive_words,
        'negative_words': negative_words,
        'top_all': top_all,
        'top_positive': top_positive,
        'top_negative': top_negative
    }


# 运行词云分析
if __name__ == "__main__":
    word_analysis_results = word_cloud_analysis()

    print("\n" + "=" * 60)
    print("✅ 词云图生成完成")
    print("✅ 高频词分析完成")
    print("✅ 情感词汇对比完成")
    print("✅ 文本洞察挖掘完成")
    print("=" * 60)

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




