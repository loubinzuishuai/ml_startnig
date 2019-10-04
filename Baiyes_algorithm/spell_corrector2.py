#导入模块------------------------------------------------------------------------------------------------
import re, collections
import numpy as np




#制作字典------------------------------------------------------------------------------------------------
with open("big.txt") as file:
    text = file.read()

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(word_list):
    dict = collections.defaultdict(lambda : 1)
    for word in word_list:
        dict[word] += 1
    return dict

NWORDS = train(words(text))





#纠错-------------------------------------------------------------------------------------------------
def edit_dist(key_word, word):
    len1 = len(key_word)
    len2 = len(word)
    distant = np.zeros(shape=(len1+1, len2+1))
    for i in range(len1+1):
        distant[i][0] = i
    for i in range(len2+1):
        distant[0][i] = i
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if key_word[i-1] == word[j-1]:
                temp = 0
            else:
                temp = 1
            distant[i][j] = min(distant[i-1][j-1]+temp, distant[i-1][j]+1, distant[i][j-1]+1)
    return distant[len1][len2]


def train(word_list):
    dict = collections.defaultdict(lambda : 1)
    for word in word_list:
        dict[word] += 1
    return dict

word_frequency = train(words(text))


def dist_all(key_word):
    dict = {}
    for word, freq in word_frequency.items():
        distant = edit_dist(key_word, word)
        dict[word] = distant
    iter_dict = dict.items()
    sorted_dict = sorted(iter_dict, key=lambda x: x[1])
    print(sorted_dict)
    return sorted_dict


dict = dist_all('weaogo')
value = dict[0][1]
candidate = []
for word, freq in dict:
    if freq == value:
        candidate.append(word)
    else:
        break
result = max(candidate, key=lambda x: word_frequency[x])
print(result)
