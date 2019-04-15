def prepare_data():
    import deepcut
    import json

    input = open('input.txt', 'r', encoding='utf-8')
    ans = open('ans.txt', 'r', encoding='utf-8')
    input_token = []
    for i in input:
        i = i.split('::')[1]
        i = i.replace('\n', '')
        input_token.append([deepcut.tokenize(i)])

    n = 0
    for i in ans:
        i = i.split('::')[1]
        i = i.replace('\n', '')
        if i == 'H':
            i = 0
        elif i == 'P':
            i = 1
        elif i == 'M':
            i = 2

        input_token[n].insert(0, i)
        print(input_token[n])
        n += 1

    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(input_token, file, ensure_ascii=False)

def find_duplicate():
    import json

    data = json.load(open('data.json', 'r', encoding='utf-8'))
    miss = []
    n = 1
    for i in data:
        if i[1].count('เขา') > 1:
            print(''.join(i[1]))
            miss.append(n)
        n += 1
    print(miss)

find_duplicate()