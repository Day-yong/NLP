import codecs

f = codecs.open('wiki_zh_jian_cut.txt', 'r', encoding="utf8")
line = f.readline()
print(line)
