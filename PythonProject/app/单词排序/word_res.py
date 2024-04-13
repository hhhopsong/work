word_list = []
with open('word.txt', 'r', encoding='utf-8') as f:
    word0 = f.read()
    word_list.append(word0)
word_no = []
num0 = 0
for i in range(len(word_list[0])-1):
    if not (ord('A') <= ord(word_list[0][i]) <= ord('Z') or ord('a') <= ord(word_list[0][i]) <= ord('z') or ord(word_list[0][i]) == ord('-') or ord(word_list[0][i]) == ord('(') or ord(word_list[0][i]) == ord(' ')):
        if ord('A') <= ord(word_list[0][i+1]) <= ord('Z') or ord('a') <= ord(word_list[0][i+1]) <= ord('z') or ord(word_list[0][i+1]) == ord('-') or ord(word_list[0][i]) == ord('(') or ord(word_list[0][i]) == ord(' '):
            word_no.append(word_list[0][num0:i+1])
            num0 = i+1
word_no.sort()
with open('word_no.txt', 'w', encoding='utf-8') as f:
    for i in word_no:
        f.write(i)
