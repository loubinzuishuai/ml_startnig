#导入模块------------------------------------------------------------------------------------------------
import re, collections




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
alphabet = 'abcdefghijklmnopqrstuvwxyz'
def edits1(word):
    n = len(word)
    return set([word[0:i]+word[i+1:] for i in range(n)] +                     # deletion
               [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] + # transposition
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] + # alteration
               [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet])  # insertion


def edits2(word):
    a =  set(e2 for e1 in edits1(word) for e2 in edits1(e1))
    print(len(a))
    print(a)
    return a


def known(words): return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    print(candidates)
    return max(candidates, key=lambda w: NWORDS[w])

print(correct('weaogo'))