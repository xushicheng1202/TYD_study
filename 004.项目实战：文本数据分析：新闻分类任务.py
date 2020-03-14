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
