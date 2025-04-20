import gradio as gr
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
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

    def features(self, words, previous_label, position):

        features = {}
        current_word = words[position]
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label
        if current_word[0].isupper():
            features['Titlecase'] = 1
        return features

    def predict(self, sentence):

        words = word_tokenize(sentence)
        previous_labels = ["O"] + ['O'] * (len(words) - 1)
        features = [self.features(words, previous_labels[i], i) for i in range(len(words))]

        predictions = model.classify_many(features)
        
        return list(zip(words, predictions))

# 初始化 MEMM 对象
ner_model = MEMM()

def predict_ner(sentence):

    result = ner_model.predict(sentence)
   
    formatted_result = [f"{word}: {label}" for word, label in result]
    return "\n".join(formatted_result)

# 定义 Gradio 界面
iface = gr.Interface(
    fn=predict_ner,  
    inputs="text",  
    outputs="text",  
    title="命名实体识别 damn!",  
    description="输入一个句子来识别命名实体。"  
)

# 启动界面
iface.launch()
