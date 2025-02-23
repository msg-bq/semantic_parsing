import re

# 定义节点类，表示一个标签块（包括标签和其内部内容）
class Node:
    def __init__(self, tag):
        self.tag = tag  # 如 "IN:GET_WEATHER" 或 "SL:LOCATION"
        self.children = []  # 子节点列表，元素可以是字符串（自由词）或 Node 对象
    def __repr__(self):
        return f"Node({self.tag}, {self.children})"

# --- 1. 解析函数 ---
# 将输入字符串按空格分词后，根据左右括号构造嵌套树
def parse_tree(tokens, index=0):
    # tokens：分词列表；index：当前处理的位置
    if index >= len(tokens):
        return None, index
    token = tokens[index]
    if not token.startswith('['):
        raise ValueError("解析错误：预期以 '[' 开头")
    # 创建当前节点（去掉开头的 '[' ）
    node = Node(token[1:])
    index += 1
    # 处理子项，直到遇到一个单独的右括号 "]"
    while index < len(tokens) and tokens[index] != "]":
        if tokens[index].startswith('['):
            child, index = parse_tree(tokens, index)
            node.children.append(child)
        else:
            # 自由词，直接加入 children
            node.children.append(tokens[index])
            index += 1
    # 跳过匹配的 "]"
    index += 1
    return node, index

# --- 2. 过滤函数 ---
# 过滤规则：只有在 SL 标签内的自由词保留；外层（非 SL）只保留 SL 子节点；
# 当某个 SL 节点内部出现自由词时，后续出现的节点认为不再属于该 SL 块，需“提升”为平级。
def filter_node(node, in_sl=False):
    # 当前是否处于 SL 环境：若父节点已是 SL 或本节点标签以 "SL:" 开头，则认为处于 SL 内
    current_sl = in_sl or node.tag.startswith("SL:")
    new_children = []   # 留在当前节点下的子节点（Node 或自由词）
    promoted = []       # 需要提升到父节点平级的节点（Node 类型）
    free_text_encountered = False  # 标记当前 SL 节点是否已遇到自由词

    for child in node.children:
        if isinstance(child, Node):
            # 对子节点递归过滤
            filtered_child, child_promoted = filter_node(child, current_sl)
            # 若当前处于 SL 环境且已经有自由词出现，则后续的节点都“提升”
            if current_sl and free_text_encountered:
                promoted.append(filtered_child)
                promoted.extend(child_promoted)
            else:
                new_children.append(filtered_child)
                new_children.extend(child_promoted)
        else:
            # child 为自由词
            if current_sl:
                if not free_text_encountered:
                    # 若当前 SL 节点还没有遇到自由词，则允许保留自由词
                    new_children.append(child)
                else:
                    # 若 SL 节点已有自由词，则后续自由词和节点提升
                    free_text_encountered = True
            # 非 SL 环境中的自由词直接忽略

    # 如果当前节点不处于 SL 环境（例如外层 [IN:GET_WEATHER]），只保留标签为 SL 的子节点
    if not current_sl:
        new_children = [c for c in new_children if isinstance(c, Node) and c.tag.startswith("SL:")]
    node.children = new_children
    return node, promoted

# --- 3. 输出函数 ---
# 将树转换回字符串，注意：[TAG 内容 … ] 之间以空格连接
def tree_to_string(node):
    s = "[" + node.tag
    if node.children:
        parts = []
        for child in node.children:
            if isinstance(child, Node):
                parts.append(tree_to_string(child))
            else:
                parts.append(child)
        s += " " + " ".join(parts)
    s += " ]"
    return s

def process_string(input_string):
    # 定义一个正则表达式来匹配所有 ] 和 [ 之间的内容
    pattern = r']\s*(.*?)\s*\['
    # 定义一个函数用于删除 @ptr_ 前的空格和 @ptr_ 后跟数字的部分
    def remove_ptr(match):
        content = match.group(1)
        # 只有在内容不包含"SL:"或"IN:"时才进行删除操作
        if "SL:" not in content and "IN:" not in content:
            # 删除 @ptr_ 后跟数字的部分以及其前的空格
            content = re.sub(r'@ptr_\d+', '', content)
        return content
    # 使用re.sub来逐个处理匹配项，删除不符合条件的部分
    input_string = re.sub(pattern, lambda m: f"] {remove_ptr(m)} [", input_string)

    return input_string.replace("  "," ")

# --- 主函数 ---
def remove_non_slot_leaf_nodes(input_string):
    # 分词（这里假定各个标记和词之间以空格分隔）
    tokens = input_string.strip().split()
    # 解析成树
    tree, _ = parse_tree(tokens)
    # 过滤：对于最外层，不处于 SL 环境（in_sl=False），过滤后获得两个部分：
    # filtered_tree 为过滤后留在 [IN:...] 内的子节点，promoted 为被提升的 SL 节点
    filtered_tree, promoted = filter_node(tree, in_sl=False)
    # 对于外层 [IN:...]，其最终子节点为原来的留存部分加上所有被提升的节点（保持顺序，提升节点后接）
    filtered_tree.children.extend(promoted)
    remove_str = tree_to_string(filtered_tree)
    # 转回字符串
    return process_string(remove_str)


# --- 测试 ---
if __name__ == '__main__':
    from datasets import load_dataset
    train_dataset = load_dataset("/home/lzx2000/T5-base-lora/TOPv2/temp")
    file = open("preprocess.txt","w",encoding="utf-8")
    for data in train_dataset["train"]:
        # input_data = "[IN:GET_WEATHER what 's the [SL:LOCATION [IN:GET_LOCATION [SL:LOCATION_MODIFIER local ] ] radar like [SL:DATE_TIME tomorrow ] ]"
        label = data["seqlogical"]
        file.write(remove_non_slot_leaf_nodes(label) + "\n")

