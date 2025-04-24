import pypinyin
from pathlib import Path

input_path = Path('../data/Chinese_Names_Corpus_120W.txt')
output_first_path = Path('../data/cn_first_names.all.txt')
output_last_path = Path('../data/cn_last_names.all.txt')


def convert_pinyin(pinyin_list) -> str:
    return ''.join([item[0] for item in pinyin_list])

first_name_set = set()
last_name_set = set()

with open(input_path, 'r', encoding='utf-8') as f, \
        open(output_first_path, 'w', encoding='utf-8') as fo_first, \
        open(output_last_path, 'w', encoding='utf-8') as fo_last:
    for line in f:
        name = line.strip()
        if len(name) < 2:
            continue

        last_name = name[0]
        first_name = name[1:]

        try:
            last_name_pinyin = pypinyin.pinyin(last_name,style=pypinyin.NORMAL)
            first_name_pinyin = pypinyin.pinyin(first_name,style=pypinyin.NORMAL)

            first_name_set.add(convert_pinyin(first_name_pinyin))
            last_name_set.add(convert_pinyin(last_name_pinyin))

        except Exception as e:
            print(f"The name: '{name}' has error: {e}")
            continue

with open(output_first_path, 'w', encoding='utf-8') as fo_first:
    for first_name in first_name_set:
        if len(first_name) > 1:
            fo_first.write(f"{first_name}\n")

with open(output_last_path, 'w', encoding='utf-8') as fo_last:
    for last_name in last_name_set:
        if len(last_name) > 1:
            fo_last.write(f"{last_name}\n")

print("Finish！")