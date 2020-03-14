import pandas as pd
import jieba

df_news = pd.read_table('DATA/新闻分类任务/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
df_news = df_news.dropna()
print(df_news.head())
print(df_news.shape)

# 将新闻内容转换为list方便进行分词并查看第1000条数据内容
content = df_news.content.values.tolist()
print(content[1000])

# 下面使用jieba库进行分词
content_s = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_s.append(current_segment)
print(content_s[1000])

# 转为pandas支持的DataFrame格式
df_content = pd.DataFrame({'content_s': content_s})
print(df_content.head())

# 删除停用词
stopwords = pd.read_csv('DATA/新闻分类任务/stopwords.txt', index_col=False, sep='\t', quoting=3, names=['stopword'],
                        encoding='utf-8')  # 读取停用词表
print(stopwords.head())


# 删除语料库中的停用词，这里的all_words是为了后面的词云展示
def drop_stopwords(contents, stopwords):
    contents_clean = []  # 删除后的新闻
    all_words = []  # 构造词云所用的数据
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


contents = df_content.content_s.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)  # 得到删除停用词后的新闻以及词云数据
df_content = pd.DataFrame({'contents_clean': contents_clean})
print(df_content)
