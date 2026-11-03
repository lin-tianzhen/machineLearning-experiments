import math
import jieba
import re
from collections import Counter

def tokenize(text):
    """
    使用 jieba 对输入的文字进行分词操作。
    """
    # 使用 jieba 精确模式分词，返回列表
    words = list(jieba.cut(text))
    return words

def calculate_text_entropy(text):
    """
    计算一段文字的信息熵。
    1. 分词
    2. 计算每个词的概率
    3. 汇总计算信息熵
    返回: (分词列表, 词概率字典, 信息熵值)
    """
    #  分词
    words = tokenize(text)
    
    # 剔除中英文标点符号
    punctuation_pattern = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~，。、；：？！‘’“”（）《》【】…—～·]')
    words = [word for word in words if not punctuation_pattern.search(word)]
    
    total_words = len(words)
    if total_words == 0:
        return words, {}, 0.0
        
    # 统计词频并计算每个词的概率
    word_counts = Counter(words)
    word_probs = {}
    for word, count in word_counts.items():
        word_probs[word] = count / total_words
    
    # 计算信息熵
    entropy = 0.0
    for prob in word_probs.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
            
    return words, word_probs, entropy

def test_entropy():
    """
    测试函数：对5段不同的文字分别求信息熵，
    并输出分词结果、每个词出现的概率和最终的信息熵。
    """
    # 5段测试文字
    test_texts = [
        "Hello world! This is a test.",  
        "你好，世界！这是一个测试。",  
        "信息熵是信息论中的一个重要概念，用来度量信息的不确定性。信息熵越大，不确定性越高。",  
        "Python is a great programming language. Python is easy to learn and Python is powerful.2026.10.20",  
        "2026年10月11日，央视《新闻直播间》栏目报道了信息科学与工程学部跨学科复合型人才培养实践，展示了学部在交叉学科育人、科研攻关与成果转化等方面的探索与成效。学部计算机科学与技术学院张述副教授、计算机科学与技术（中外合作办学）专业学生接受采访。报道中，学部视觉系统与机器人实验室亮相央视新闻。报道还介绍了董军宇教授团队攻克水下三维重建技术难题，推动相关装备实现国产化替代，将育人、科研与产业串成一条线的创新实践。"  
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*20} 测试文字 {i} {'='*20}")
        print(f"[原文]: {text}")
        
        words, probs, entropy = calculate_text_entropy(text)
        
        print(f"[分词结果]: {words}，共 {len(words)} 个词")
        print("[每个词的概率]:")
        for word, prob in probs.items():
            print(f"  - '{word}': {prob:.4f} ({prob*100:.2f}%)")
        print(f"[最终信息熵]: {entropy:.4f} bits")
        print("="*55)

if __name__ == "__main__":
    # 运行测试函数
    test_entropy()