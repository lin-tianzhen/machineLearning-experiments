import csv
import math
import os
from collections import Counter, defaultdict

DATA_FILE = "data.csv"
TARGET = "PlayTennis"
IGNORED_COLUMNS = ["Day"]

def load_data(file_path):
    """读取 CSV，返回 (特征列名列表, 样本字典列表)。"""
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        columns = reader.fieldnames
    features = [c for c in columns if c not in IGNORED_COLUMNS + [TARGET]]
    return features, rows

def entropy(rows, target=TARGET):
    """计算样本集合关于目标变量的信息熵。"""
    total = len(rows)
    if total == 0:
        return 0.0
    counter = Counter(row[target] for row in rows)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())

def value_groups(rows, feature):
    """按特征取值划分子集。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row[feature]].append(row)
    return groups

def conditional_entropy(rows, feature, target=TARGET):
    """计算特征 feature 的条件熵（加权平均熵）。"""
    total = len(rows)
    groups = value_groups(rows, feature)
    return sum(len(sub) / total * entropy(sub, target) for sub in groups.values())

def split_info(rows, feature):
    """C4.5 的固有值（SplitInfo），衡量特征把样本分细的程度。"""
    total = len(rows)
    counts = Counter(row[feature] for row in rows)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def information_gain(rows, feature, target=TARGET):
    """ID3 的划分标准：信息增益。"""
    return entropy(rows, target) - conditional_entropy(rows, feature, target)

def gain_ratio(rows, feature, target=TARGET):
    """C4.5 的划分标准：信息增益率 = 信息增益 / 固有值。"""
    gain = information_gain(rows, feature, target)
    si = split_info(rows, feature)
    return (gain / si) if si > 0 else 0.0

def select_feature(rows, features, criterion, indent):
    """评估所有候选特征并返回最佳特征及其各项指标。"""
    scores = {}
    for feature in features:
        gain = information_gain(rows, feature)
        si = split_info(rows, feature)
        ratio = gain / si if si > 0 else 0.0
        scores[feature] = {"gain": gain, "split_info": si, "gain_ratio": ratio}

    key = "gain" if criterion == "ID3" else "gain_ratio"
    best = max(features, key=lambda f: scores[f][key])

    print(f"{indent}各候选特征评估：")
    for feature in features:
        s = scores[feature]
        print(f"{indent}  {feature:<12} Gain={s['gain']:.4f}  "
              f"SplitInfo={s['split_info']:.4f}  GainRatio={s['gain_ratio']:.4f}")
    basis = "信息增益最大" if criterion == "ID3" else "信息增益率最大"
    print(f"{indent}--> 选择特征 [{best}]（{criterion} 依据：{basis}）")
    return best, scores

def build_tree(rows, features, criterion, depth=0, branch_desc="根节点"):
    """递归构建决策树，并在每个节点输出构建过程。"""
    indent = "  " * depth
    labels = Counter(row[TARGET] for row in rows)
    node_entropy = entropy(rows)
    days = ",".join(row["Day"] for row in rows)

    print(f"{indent}[节点] {branch_desc} | 样本数={len(rows)} | 样本={days}")
    print(f"{indent}  标签分布={dict(labels)} | 信息熵={node_entropy:.4f}")

    if len(labels) == 1:
        label = next(iter(labels))
        print(f"{indent}  => 终止：所有样本同属 '{label}'，生成叶节点 '{label}'")
        return label

    if not features:
        label = labels.most_common(1)[0][0]
        print(f"{indent}  => 终止：已无可用特征，按多数类生成叶节点 '{label}'")
        return label

    best, _ = select_feature(rows, features, criterion, indent)
    tree = {best: {}}
    groups = value_groups(rows, best)

    for value in sorted(groups):
        subset = groups[value]
        print(f"{indent}  分支 {best}={value}：样本数={len(subset)}")
        if not subset:
            label = labels.most_common(1)[0][0]
            print(f"{indent}    => 分支为空，取父节点多数类 '{label}'")
            tree[best][value] = label
        else:
            tree[best][value] = build_tree(
                subset,
                [f for f in features if f != best],
                criterion,
                depth + 1,
                f"{best}={value}",
            )
    return tree

def print_tree(tree, indent=""):
    """以缩进文本形式打印生成的决策树。"""
    if not isinstance(tree, dict):
        print(f"{indent}=> {tree}")
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            if isinstance(subtree, dict):
                print(f"{indent}{feature} = {value}:")
                print_tree(subtree, indent + "    ")
            else:
                print(f"{indent}{feature} = {value} -> {subtree}")

def predict(tree, sample):
    """沿树自顶向下匹配一条样本，返回预测标签。"""
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample.get(feature)
        if value not in tree[feature]:
            return None
        tree = tree[feature][value]
    return tree

def run(criterion, features, rows):
    """对指定算法执行一次完整的建树流程。"""
    print("=" * 72)
    print(f"使用 {criterion} 算法构建决策树")
    print("=" * 72)
    tree = build_tree(rows, features, criterion)
    print(f"\n[{criterion}] 最终决策树：")
    print_tree(tree)
    correct = sum(predict(tree, row) == row[TARGET] for row in rows)
    print(f"[{criterion}] 训练集预测准确率：{correct}/{len(rows)}\n")
    return tree

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)
    features, rows = load_data(data_path)
    print(f"数据文件：{data_path}")
    print(f"特征列：{features} | 样本数：{len(rows)}\n")

    tree_id3 = run("ID3", features, rows)
    tree_c45 = run("C4.5", features, rows)

    print("=" * 72)
    print("两种算法生成的树结构是否相同：", tree_id3 == tree_c45)

if __name__ == "__main__":
    main()