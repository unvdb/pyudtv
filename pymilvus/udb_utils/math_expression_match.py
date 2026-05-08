import re

# handler math expression
def is_three_part_expression_enhanced(expr: str, string_value=False) -> bool:
    """
    增强版判断，支持更复杂的表达式
    """
    expr = expr.strip()

    # 支持的运算符
    operators = r'[+\-*/%]'
    # 支持的比较符
    comparators = r'==|!=|>|<|>=|<=|in|not\s+in'
    # 标识符
    identifier = r'[a-zA-Z_][a-zA-Z0-9_]*'
    # 数值
    number = r'\d+(?:\.\d+)?'
    # 字符串
    string = r"\'[^\']*\'|\"[^\"]*\""

    param = r'(?:[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?|\'[^\']*\'|\"[^\"]*\")'
    func_call = rf'{identifier}\s*\(\s*{param}(?:\s*,\s*{param})*\s*\)'

    # 复杂表达式: 可以是简单标识符或函数调用
    complex_expression = rf'(?:{identifier}|{func_call})'

    # 值可以是标识符、数值或字符串
    if string_value:
        value = f'({complex_expression}|{number}|{string})'
    else:
        value = f'({complex_expression}|{number})'

    # 定义各种模式
    patterns = [
        rf'^{complex_expression}\s+(==|=|!=)\s+{value}$',
        rf'^{complex_expression}\s+{comparators}\s+{value}$',
        # 模式: 标识符 运算符 比较符 值
        rf'^{complex_expression}\s+{operators}\s+{comparators}\s+{value}$',

        # 基础模式: 标识符 运算符 值
        # rf'^{complex_expression}\s+{operators}\s+{value}$',

        # 比较模式: 标识符 运算符 值 比较符 值
        rf'^{complex_expression}\s+{operators}\s+{value}\s+({comparators})\s+{value}$',

        # 带括号的表达式: (标识符 运算符 值) 比较符 值
        rf'^\({complex_expression}\s+{operators}\s+{value}\)\s+({comparators})\s+{value}$',

        # 列表比较模式: 标识符 运算符 值 IN 列表
        rf'^{complex_expression}\s+{operators}\s+{value}\s+(in|not\s+in)\s+\[.*\]$',

        # 链式运算模式: 标识符 运算符 值 运算符 值
        rf'^{complex_expression}\s+{operators}\s+{value}\s+{operators}\s+{value}$',

        # 复合比较: 标识符 运算符 值 比较符 值 逻辑符 标识符 运算符 值 比较符 值
        # rf'^{complex_expression}\s+{operators}\s+{value}\s+({comparators})\s+{value}\s+(and|or)\s+{complex_expression}\s+{operators}\s+{value}\s+({comparators})\s+{value}$',
    ]

    for i, pattern in enumerate(patterns):
        # if re.match(pattern, expr, re.IGNORECASE):
        if re.fullmatch(pattern, expr, re.IGNORECASE):
            return True

    return False


# 测试函数
def test_expression_analyzer():
    """测试各种表达式"""
    test_cases = [
        # 基础表达式
        "count % 2 == 0",  # 你的例子
        "value + 5 > 10",
        "price * quantity < 1000",
        "score / 2 >= 50",
        "total - discount == 100",

        # 复杂表达式
        "count % 2 in [0, 1]",
        "value + 5 != 0",
        "price * 1.1 <= budget",
        "score / count > average",
        "(a + b) * c == result",

        # 不符合的表达式
        "count == 0",  # 没有运算符
        "a + b + c",  # 没有比较符
        "function_call(arg)",  # 函数调用
        "true and false",  # 布尔运算
        "x > y and y < z",  # 复合表达式

        # 边缘情况
        "count%2==0",  # 无空格
        "count  %  2  ==  0",  # 多个空格
        "'status' + 'code' == '200OK'",  # 字符串操作
        "price * 0.9 < 100.5",  # 浮点数
        "user_id % 1000 == 0",  # 大数字
    ]

    print("测试表达式分析")
    print("=" * 60)

    for expr in test_cases:
        enhanced_result = is_three_part_expression_enhanced(expr)

        print(f"表达式: {expr}")
        print(f"增强匹配: {'✓' if enhanced_result else '✗'}")
        print("-" * 50)


def test_mod_cases():
    test_cases = [
        ("count % 2 == 0", True, "能匹配"),
        ("value % 3 = 0", False, "单等号 不能匹配"),
        ("id % 10 != 0", True, "能匹配"),
        ("user_id % 1000 < 500", True, "能匹配"),
        ("index % 2 > 0", True, "能匹配"),
        ("num % divisor == remainder", True, "使用变量 能匹配"),
        ("total % 2.5 <= 1.0", True, "浮点数 能匹配"),
        ("count%2==0", False, "无空格 不能匹配"),
        ("a % b == c", True, "全部变量 能匹配"),
        ("x % 2", False, "只有取模，没有比较 不能匹配"),
        ("count == 0", True, "能匹配"),
        ("mod(count, 2) == 0", True, "函数形式 能匹配"),
        ('tag == "tag_1" and id > 3 and value % 2 == 1', False, "包含逻辑运算符 不能匹配"),
    ]

    for expr, expected, description in test_cases:
        result = is_three_part_expression_enhanced(expr, string_value=True)
        if expected and not result:
            print(f"表达式 '{expr}' 应该匹配，但返回了 False")
        elif not expected and result:
            print(f"表达式 '{expr}' 不应该匹配，但返回了 True")
        else:
            print(f"表达式 '{expr}' 匹配正常 {expected} -> {result}")

# 运行测试
if __name__ == "__main__":
    test_mod_cases()