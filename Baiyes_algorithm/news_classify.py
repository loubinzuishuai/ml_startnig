import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
df_news = pd.read_table('data/val.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()






#分词----------------------------------------------------------------------------------
content = df_news['content'].values.tolist()
content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)

df_content = pd.DataFrame(data={'content_S': content_S})




#去停用词-------------------------------------------------------------------------------
stopwords = pd.read_csv('stopwords.txt', names=['stopword'], index_col=False, sep='\t', quoting=3, encoding='utf-8')
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(word)
        contents_clean.append(line_clean)
    return contents_clean, all_words

contents = df_content['content_S'].values.tolist()
stopwords = stopwords['stopword'].values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)
df_content = pd.DataFrame(data={'contents_clean': contents_clean})




#统计字的频率
df_all_words = pd.DataFrame(data={'all_words': all_words})
words_count = df_all_words['all_words'].groupby(by=df_all_words['all_words']).agg(['count'])
words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)




#worldcloud是什么东东啊
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# wordcloud = WordCloud(font_path="./data/simhei.ttf", background_color='white', max_font_size=80)
# word_frequence =  {x[0]: x[1] for x in words_count.head(100).values}
# wordcloud.fit_words(frequencies=word_frequence)
# plt.imshow(wordcloud)
# plt.show()



#TF-IDF:提取关键词
import jieba.analyse
index = 12
print(df_news['content'][index])
content_S_str = "".join(content_S[index])
print(content_S_str)
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))




#训练集和测试集分开
df_train = pd.DataFrame(data={'content_clean': contents_clean, 'label': df_news['category']})
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
print(df_train.head())
x_train, x_test, y_train, y_test = train_test_split(df_train['content_clean'].values, df_train['label'].values, random_state=1)
print(x_train[2])



#将训练集变成句向量
words = []
for line in range(len(x_train)):
    try:
        words.append(" ".join(x_train[line]))
    except:
        print(line)

vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
vec.fit(words)
print(vec.transform(words))





#朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
classify = MultinomialNB()
classify.fit(X=vec.transform(words).toarray(), y=y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index)

print(classify.score(vec.transform(test_words), y_test))


