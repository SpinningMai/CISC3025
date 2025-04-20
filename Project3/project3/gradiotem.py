import gradio as gr
import pickle
import nltk
from nltk.tokenize import word_tokenize
import re

# 下载 punkt_tab 数据
nltk.download('punkt')

# 加载训练好的模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class MEMM:
    def __init__(self):
        self.train_path = "data/train"
        self.dev_path = "data/dev"
        self.beta = 0.5
        self.max_iter = 5
        self.classifier = None

        # 正则表达式：用于判断驼峰命名法
        self.camel_regex = re.compile(r'^([A-Z]?[a-z]+)+([A-Z][a-z]+)*$')

        # 非英语字符集
        self.latin_letters = {'é', 'ü'}
        
        # 拼音正则表达式
        self.pinyin_regex = re.compile(
            r"^("
            r"(a[io]?|ou?|e[inr]?|ang?|ng|[bmp](a[io]?|[aei]ng?|ei|ie?|ia[no]|o|u)|"
            r"pou|me|m[io]u|[fw](a|[ae]ng?|ei|o|u)|fou|wai|[dt](a[io]?|an|e|[aeio]ng|"
            r"ie?|ia[no]|ou|u[ino]?|uan)|dei|diu|[nl](a[io]?|ei?|[eio]ng|i[eu]?|i?ang?|"
            r"iao|in|ou|u[eo]?|ve?|uan)|nen|lia|lun|[ghk](a[io]?|[ae]ng?|e|ong|ou|u[aino]?|"
            r"uai|uang?)|[gh]ei|[jqx](i(ao?|ang?|e|ng?|ong|u)?|u[en]?|uan)|([csz]h?|"
            r"r)([ae]ng?|ao|e|i|ou|u[ino]?|uan)|[csz](ai?|ong)|[csz]h(ai?|uai|"
            r"uang)|zei|[sz]hua|([cz]h|r)ong|y(ao?|[ai]ng?|e|i|ong|ou|u[en]?|uan))"
            r"){1,4}$"
        )

        # 拼音混淆词汇
        self.pinyin_confusion = {"me", "ma", "bin", "fan", "long", "sun", "panda", "china"}

    def features(self, words, previous_label, position):
        """
        为当前单词提取特征。
        :param words: 句子的单词列表
        :param previous_label: 前一个单词的标签
        :param position: 当前单词的位置
        """
        features = {}
        current_word = words[position]

        # 基本特征
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label

        # 字母大小写特征
        if current_word[0].isupper(): 
            features['Titlecase'] = 1
        if current_word.isupper(): 
            features["Allcapital"] = 1
        if self.camel_regex.fullmatch(current_word): 
            features["Camelcase"] = 1

        # 标点符号特征
        if "'" in current_word: 
            features["Apostrophe"] = 1
        if "-" in current_word: 
            features["Hyphen"] = 1

        # 后缀特征
        if current_word.endswith("son"): 
            features["Suffix_son"] = 1
        if current_word.endswith("ez"): 
            features["Suffix_ez"] = 1

        # 前缀特征
        if current_word.startswith("Mc"): 
            features["Prefix_Mc"] = 1
        if current_word.startswith("O'"): 
            features["Prefix_OAp"] = 1

        # 非英语特征（拼音和拉丁字母）
        if any(current_word) in self.latin_letters:
            features["Latinletter"] = 1
        if self.pinyin_regex.fullmatch(current_word.lower()):
            features["Pinyin"] = 1
        if current_word.lower().endswith("lyu") or current_word.lower().startswith("lyu"):
            features["Pinyin_lyu"] = 1
        if current_word.lower() in self.pinyin_confusion:
            features["Pinyin_confusion"] = 1

        return features

    def predict(self, sentence):
        """
        使用 MEMM 模型进行预测。
        :param sentence: 输入的句子
        """
        words = word_tokenize(sentence)
        predictions = []
        previous_label = "O"

        for i in range(len(words)):
            single_bunch_features = self.features(words, previous_label, i)

            current_label = model.classify(single_bunch_features)
            predictions.append(current_label)

            previous_label = current_label

        return list(zip(words, predictions))

# 初始化 MEMM 对象
ner_model = MEMM()

def predict_ner(sentence):
    """
    预测命名实体并格式化输出。
    """
    result = ner_model.predict(sentence)
   
    formatted_result = [f"\"{word}\":\t{label}" for word, label in result]
    return "\n".join(formatted_result)

# 定义 Gradio 界面
iface = gr.Interface(
    fn=predict_ner,  # 绑定的预测函数
    inputs="text",  # 输入类型：文本框
    outputs="text",  # 输出类型：文本框
    title="命名实体识别",  # 页面标题
    description="输入一个句子来识别命名实体。"  # 页面描述
)

# 启动界面
iface.launch()
