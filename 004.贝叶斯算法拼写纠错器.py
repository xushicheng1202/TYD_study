import re, collections


def words(text): return re.findall('[a-z]+', text.lower())  # 去掉其他除了a到z以外的字符


def train(features):
    model = collections.defaultdict(lambda: 1)  # 导入库设置默认值为1
    for f in features:
        model[f] += 1
    return model


NWORDS = train(words(open('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words):
    return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    print(max(candidates, key=lambda w: NWORDS[w]))


a = input('输入字符：')
print(type(a))
correct(a)
# correct 函数从一个候选集合中选取最大概率的. 实际上, 就是选取有最大 P(c) 值的那个. 所有的 P(c) 值都存储在 NWORDS 结构中.
