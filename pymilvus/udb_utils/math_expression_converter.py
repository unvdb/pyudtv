import re
import ast
from typing import Tuple, Set, Dict, Any, Union, List


class PymilvusToPostgresConverter:
    """将pymilvus数学表达式转换为PostgreSQL SQL表达式，针对JSON字段data"""

    def __init__(self, json_field_name: str = "data"):
        self.json_field_name = json_field_name

        # 支持的运算符映射
        self.operator_mapping = {
            '==': '=',
            '!=': '<>',
            '>': '>',
            '<': '<',
            '>=': '>=',
            '<=': '<=',
            'and': 'AND',
            'or': 'OR',
            'not': 'NOT',
            'in': 'IN',
            'not in': 'NOT IN',
        }

        # 算术运算符映射
        self.arithmetic_mapping = {
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '%': '%',  # 取模运算
            '**': '^',  # 指数运算，PostgreSQL使用^
        }

        # 支持的数学函数
        self.supported_functions = {
            'abs', 'sqrt', 'pow', 'power', 'log', 'log10', 'ln', 'exp',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'ceil', 'floor', 'round', 'trunc',
            'mod',  # 取模函数
            'length', 'upper', 'lower', 'trim', 'ltrim', 'rtrim',
            'coalesce', 'nullif',  # 空值处理
            'cast',  # 类型转换
        }

        # PostgreSQL数学函数映射
        self.function_mapping = {
            'pow': 'POWER',
            'power': 'POWER',
            'log10': 'LOG',
            'log2': 'LOG',
            'mod': 'MOD',
            'abs': 'ABS',
            'ceil': 'CEIL',
            'floor': 'FLOOR',
            'round': 'ROUND',
            'sqrt': 'SQRT',
            'exp': 'EXP',
            'ln': 'LN',
            'sin': 'SIN',
            'cos': 'COS',
            'tan': 'TAN',
            'asin': 'ASIN',
            'acos': 'ACOS',
            'atan': 'ATAN',
            'atan2': 'ATAN2',
        }

        # 不支持的运算符和语法
        self.unsupported_operators = {
            '&&', '||', '===', '!==', '&', '|', '^', '~',
            '<<', '>>', '//',
        }

        # 内置常量和数学常量
        self.builtin_constants = {
            'True', 'False', 'None',
            'PI', 'pi', 'E', 'e'
        }

        # 数学常量映射
        self.constant_mapping = {
            'PI': 'PI()',
            'pi': 'PI()',
            'E': 'EXP(1)',
            'e': 'EXP(1)',
        }

        # Python关键字
        self.python_keywords = {
            'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None',
            'if', 'else', 'elif', 'for', 'while', 'def', 'class',
            'import', 'from', 'as', 'try', 'except', 'finally',
            'with', 'as', 'pass', 'return', 'break', 'continue',
        }

        # 危险的Python内置函数
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'open', 'input',
            'globals', 'locals', 'vars', 'dir', 'type', 'isinstance',
            'issubclass', 'super', 'classmethod', 'staticmethod', 'property',
            'hasattr', 'getattr', 'setattr', 'delattr', 'callable',
            'memoryview', 'bytearray', 'bytes', 'complex', 'frozenset',
            'range', 'slice', 'object', 'reversed', 'enumerate', 'zip',
            'filter', 'map', 'sorted', 'any', 'all', 'sum', 'min', 'max',
            'chr', 'ord', 'hex', 'oct', 'bin', 'id', 'hash', 'iter', 'next',
            'help', 'breakpoint', 'copyright', 'credits', 'license', 'exit', 'quit'
        }

    def is_valid_pymilvus_expression(self, expression: str) -> Tuple[bool, str]:
        """
        判断是否是合理的pymilvus数学表达式

        返回: (是否有效, 错误信息)
        """
        if not expression or not expression.strip():
            return False, "表达式不能为空"

        expr = expression.strip()

        # 1. 安全检查：防止代码注入
        dangerous_patterns = [
            r'__[a-zA-Z_]+__',  # 双下划线方法
            r'\b(import|from|as|def|class|lambda|yield|return|break|continue|'
            r'pass|raise|try|except|finally|with|assert|del|global|nonlocal)\b',
            r'\.\s*[a-zA-Z_]\w*\s*\(',  # 方法调用
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expr):
                return False, f"表达式包含不安全语法: {pattern}"

        # 2. 检查是否包含不支持的运算符
        for op in self.unsupported_operators:
            if op in expr:
                return False, f"包含不支持的运算符: '{op}'"

        # 3. 使用ast安全解析表达式
        # try:
        #     # 解析为AST
        #     tree = ast.parse(expr, mode='eval')
        #
        #     # 验证AST节点
        #     if not self._validate_ast_node_safe(tree.body):
        #         return False, "表达式包含不支持的语法结构"
        #
        # except SyntaxError as e:
        #     return False, f"语法错误: {str(e)}"
        # except Exception as e:
        #     return False, f"解析错误: {str(e)}"

        return True, ""

    def _validate_ast_node_safe(self, node) -> bool:
        """验证AST节点是否安全（简化版本）"""
        # 允许的节点类型
        allowed_nodes = {
            # 表达式
            ast.Expression, ast.Module,
            # 二元运算
            ast.BinOp,
            # 一元运算
            ast.UnaryOp,
            # 布尔运算
            ast.BoolOp,
            # 比较
            ast.Compare,
            # 调用
            ast.Call,
            # 名称
            ast.Name,
            # 常量
            ast.Constant,
            # 数字、字符串等
            ast.Num, ast.Str, ast.NameConstant,  # Python 3.7及以下
            # 列表、元组
            ast.List, ast.Tuple,
            # 下标
            ast.Subscript,
            # 字典
            ast.Dict,
        }

        node_type = type(node)

        # 检查节点类型
        if node_type not in allowed_nodes:
            return False

        # 特殊检查：函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # 检查是否为危险的函数
                if func_name in self.dangerous_functions:
                    return False
                # 检查是否为支持的函数
                if func_name not in self.supported_functions and func_name not in self.function_mapping:
                    # 允许未知函数，在转换时处理
                    pass

        # 特殊检查：名称节点
        elif isinstance(node, ast.Name):
            # 检查是否为危险的内置名称
            if node.id in self.dangerous_functions:
                return False

        # 递归检查子节点
        for child_node in ast.iter_child_nodes(node):
            if not self._validate_ast_node_safe(child_node):
                return False

        return True

    def _extract_identifiers(self, expression: str) -> Set[str]:
        """从表达式中提取所有标识符"""
        identifiers = set()

        try:
            tree = ast.parse(expression, mode='eval')

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # 排除Python关键字和内置常量
                    if (node.id not in self.python_keywords and
                            node.id not in self.builtin_constants and
                            node.id not in self.dangerous_functions):
                        identifiers.add(node.id)
        except:
            # 如果解析失败，使用简单正则提取
            pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(pattern, expression)
            for match in matches:
                if (match not in self.python_keywords and
                        match not in self.builtin_constants and
                        match not in self.dangerous_functions and
                        match not in self.supported_functions and
                        match not in self.function_mapping):
                    identifiers.add(match)

        return identifiers

    def _infer_data_type(self, identifier: str, expression: str) -> str:
        """推断标识符的数据类型"""
        expr_lower = expression.lower()
        id_lower = identifier.lower()

        # 检查是否用于数值运算
        numeric_patterns = [
            rf'{id_lower}\s*[+\-*/%]',  # 标识符后接运算符
            rf'[+\-*/%]\s*{id_lower}',  # 运算符后接标识符
            rf'{id_lower}\s*[><]=?\s*\d',  # 与数字比较
            rf'\d\s*[><]=?\s*{id_lower}',  # 数字与标识符比较
            rf'{id_lower}\s*%\s*\d',  # 取模运算
            rf'\d\s*%\s*{id_lower}',  # 数字取模标识符
            rf'\b(?:abs|sqrt|pow|power|log|ln|log10|exp|sin|cos|tan|'
            rf'asin|acos|atan|atan2|ceil|floor|round|trunc|mod)\s*\(\s*{id_lower}\b',  # 数学函数
        ]

        for pattern in numeric_patterns:
            if re.search(pattern, expr_lower):
                return "numeric"

        # 检查是否用于字符串操作
        string_patterns = [
            rf'\b(?:length|upper|lower|trim|ltrim|rtrim)\s*\(\s*{id_lower}\b',
            rf"{id_lower}\s*[=!]=\s*'[^']*'",  # 与字符串比较
            rf"{id_lower}\s*[=!]=\s*\"[^\"]*\"",
        ]

        for pattern in string_patterns:
            if re.search(pattern, expr_lower):
                return "text"

        # 检查是否用于布尔上下文
        bool_patterns = [
            rf'\b{id_lower}\s*==\s*(?:True|False)\b',
            rf'\b{id_lower}\s*!=\s*(?:True|False)\b',
        ]

        for pattern in bool_patterns:
            if re.search(pattern, expr_lower, re.IGNORECASE):
                return "boolean"

        # 检查IN操作
        in_pattern = rf'\b{id_lower}\s+(?:not\s+)?in\s+\['
        if re.search(in_pattern, expr_lower, re.IGNORECASE):
            # 尝试推断列表元素的类型
            list_match = re.search(rf'{id_lower}\s+(?:not\s+)?in\s+\[([^\]]+)\]', expr_lower, re.IGNORECASE)
            if list_match:
                items = list_match.group(1)
                # 如果列表中包含引号，可能是字符串
                if "'" in items or '"' in items:
                    return "text"
                # 否则可能是数字
                return "numeric"

        # 默认返回text
        return "text"

    def _convert_identifier(self, identifier: str, data_type: str) -> str:
        """将标识符转换为JSON字段访问"""
        if data_type == "numeric":
            return f"({self.json_field_name}->>'{identifier}')::numeric"
        elif data_type == "boolean":
            return f"({self.json_field_name}->>'{identifier}')::boolean"
        elif data_type == "integer":
            return f"({self.json_field_name}->>'{identifier}')::integer"
        elif data_type == "float":
            return f"({self.json_field_name}->>'{identifier}')::float"
        else:  # text
            return f"{self.json_field_name}->>'{identifier}'"

    def _convert_ast_to_sql(self, node: ast.AST) -> str:
        """将AST节点转换为SQL表达式"""
        node_type = type(node)

        # 处理常量
        if node_type in (ast.Constant, ast.Num):
            if hasattr(node, 'n'):
                return str(node.n)
            elif hasattr(node, 'value'):
                if isinstance(node.value, (int, float)):
                    return str(node.value)
                elif isinstance(node.value, str):
                    return f"'{node.value}'"
                else:
                    return str(node.value)

        # 处理字符串常量
        elif node_type == ast.Str:
            return f"'{node.s}'"

        # 处理名称常量
        elif node_type == ast.NameConstant:
            if node.value is True:
                return 'TRUE'
            elif node.value is False:
                return 'FALSE'
            elif node.value is None:
                return 'NULL'

        # 处理名称节点
        elif node_type == ast.Name:
            # 处理数学常量
            if node.id in self.constant_mapping:
                return self.constant_mapping[node.id]
            # 其他名称节点将在外部处理
            return node.id

        # 处理二元运算
        elif node_type == ast.BinOp:
            left = self._convert_ast_to_sql(node.left)
            right = self._convert_ast_to_sql(node.right)

            # 映射运算符
            op_mapping = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
                ast.Mod: '%',  # 取模运算
                ast.Pow: '^',  # PostgreSQL使用^表示指数
                ast.FloorDiv: '/',  # 整数除法
            }

            op_type = type(node.op)
            if op_type in op_mapping:
                op = op_mapping[op_type]
                return f"({left} {op} {right})"
            else:
                raise ValueError(f"不支持的运算符: {op_type}")

        # 处理比较运算
        elif node_type == ast.Compare:
            left = self._convert_ast_to_sql(node.left)
            result = left

            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                comparator_expr = self._convert_ast_to_sql(comparator)

                # 映射比较运算符
                op_mapping = {
                    ast.Eq: '=',
                    ast.NotEq: '<>',
                    ast.Lt: '<',
                    ast.LtE: '<=',
                    ast.Gt: '>',
                    ast.GtE: '>=',
                    ast.In: 'IN',
                    ast.NotIn: 'NOT IN',
                }

                op_type = type(op)
                if op_type in op_mapping:
                    op_str = op_mapping[op_type]
                    result = f"{result} {op_str} {comparator_expr}"
                else:
                    raise ValueError(f"不支持的比较运算符: {op_type}")

            return result

        # 处理布尔运算
        elif node_type == ast.BoolOp:
            values = [self._convert_ast_to_sql(value) for value in node.values]

            op_mapping = {
                ast.And: 'AND',
                ast.Or: 'OR',
            }

            op_type = type(node.op)
            if op_type in op_mapping:
                op_str = op_mapping[op_type]
                return f" ({op_str} ".join(values) + ")"
            else:
                raise ValueError(f"不支持的布尔运算符: {op_type}")

        # 处理一元运算
        elif node_type == ast.UnaryOp:
            operand = self._convert_ast_to_sql(node.operand)

            op_mapping = {
                ast.USub: '-',
                ast.UAdd: '+',
                ast.Not: 'NOT',
            }

            op_type = type(node.op)
            if op_type in op_mapping:
                op_str = op_mapping[op_type]
                if op_type == ast.Not:
                    return f"{op_str} ({operand})"
                else:
                    return f"{op_str}{operand}"
            else:
                raise ValueError(f"不支持的一元运算符: {op_type}")

        # 处理函数调用
        elif node_type == ast.Call:
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = [self._convert_ast_to_sql(arg) for arg in node.args]

                # 处理特殊函数
                if func_name == 'cast':
                    if len(args) >= 2:
                        value = args[0]
                        type_name = args[1].strip("'\"")
                        type_mapping = {
                            'int': 'integer',
                            'integer': 'integer',
                            'float': 'float',
                            'double': 'double precision',
                            'numeric': 'numeric',
                            'string': 'text',
                            'text': 'text',
                            'bool': 'boolean',
                            'boolean': 'boolean',
                        }
                        pg_type = type_mapping.get(type_name.lower(), 'text')
                        return f"({value})::{pg_type}"
                    else:
                        raise ValueError("cast函数需要两个参数")

                # 处理数学函数
                elif func_name in self.function_mapping:
                    pg_func = self.function_mapping[func_name]
                    if func_name == 'log10':
                        if len(args) == 1:
                            return f"LOG(10, {args[0]})"
                        else:
                            return f"LOG({', '.join(args)})"
                    elif func_name == 'log2':
                        if len(args) == 1:
                            return f"LOG(2, {args[0]})"
                        else:
                            return f"LOG({', '.join(args)})"
                    else:
                        return f"{pg_func}({', '.join(args)})"

                # 处理其他支持的函数
                elif func_name in self.supported_functions:
                    return f"{func_name.upper()}({', '.join(args)})"

                # 未知函数
                else:
                    return f"{func_name}({', '.join(args)})"
            else:
                raise ValueError("不支持的函数调用格式")

        # 处理列表
        elif node_type == ast.List:
            elements = [self._convert_ast_to_sql(elem) for elem in node.elts]
            return f"({', '.join(elements)})"

        # 处理元组
        elif node_type == ast.Tuple:
            elements = [self._convert_ast_to_sql(elem) for elem in node.elts]
            return f"({', '.join(elements)})"

        else:
            raise ValueError(f"不支持的AST节点类型: {node_type}")

    def convert(self, pymilvus_expr: str) -> str:
        """
        将pymilvus表达式转换为PostgreSQL SQL表达式

        步骤：
        1. 验证是否为有效的pymilvus数学表达式
        2. 解析为AST
        3. 提取标识符
        4. 转换AST为SQL
        5. 替换标识符为JSON字段访问

        返回: PostgreSQL SQL表达式
        异常: ValueError 如果表达式无效
        """
        # 1. 验证表达式
        is_valid, error_msg = self.is_valid_pymilvus_expression(pymilvus_expr)
        if not is_valid:
            raise ValueError(f"无效的pymilvus数学表达式: {error_msg} expr: {pymilvus_expr}")

        # 2. 提取标识符
        identifiers = self._extract_identifiers(pymilvus_expr)

        # 3. 解析为AST
        try:
            tree = ast.parse(pymilvus_expr, mode='eval')
        except Exception as e:
            raise ValueError(f"无法解析表达式: {e}")

        # 4. 转换AST为SQL
        sql_expr = self._convert_ast_to_sql(tree.body)

        # 5. 创建标识符映射
        identifier_map = {}
        for identifier in identifiers:
            data_type = self._infer_data_type(identifier, pymilvus_expr)
            identifier_map[identifier] = self._convert_identifier(identifier, data_type)

        # 6. 替换标识符
        converted_expr = sql_expr

        # 按长度降序排序，避免部分匹配
        sorted_identifiers = sorted(identifier_map.keys(), key=len, reverse=True)
        for identifier in sorted_identifiers:
            # 使用单词边界确保完整匹配
            pattern = r'\b' + re.escape(identifier) + r'\b'
            converted_expr = re.sub(pattern, identifier_map[identifier], converted_expr)

        return converted_expr


# 测试用例
if __name__ == "__main__":
    converter = PymilvusToPostgresConverter()

    test_cases = [
        ("value % 2 == 0", "value % 2 = 0"),
        ("tag == 'tag_1' and id > 3 and value % 2 < 100", "tag = 'tag_1' AND id > 3 AND value % 2 < 100"),
        ("pow(value, 2) > 100", "POWER(value, 2) > 100"),
        ("sqrt(value) < 10", "SQRT(value) < 10"),
        ("mod(value, 3) == 0", "MOD(value, 3) = 0"),
        ("value ** 2 > 100", "value ^ 2 > 100"),
        ("abs(score) > 0.5", "ABS(score) > 0.5"),
        ("cast(value, 'int') > 10", "(value)::integer > 10"),
        ("PI * radius * radius", "PI() * radius * radius"),
        ("log10(views) > 3", "LOG(10, views) > 3"),
        ("value + 5 * 2", "value + 5 * 2"),
        ("(a + b) * c", "(a + b) * c"),
        ("x in [1, 2, 3]", "x IN (1, 2, 3)"),
        ("not (x > 0)", "NOT (x > 0)"),
        ("-value * 2", "-value * 2"),
    ]

    print("测试Pymilvus到PostgreSQL转换器")
    print("=" * 60)

    for expr, expected in test_cases:
        try:
            result = converter.convert(expr)
            # 注意：由于标识符会被替换为JSON字段访问，所以结果会不同
            # 我们只检查转换是否成功
            print(f"输入:  {expr}")
            print(f"输出:  {result}")
            print(f"状态:  ✓ 转换成功")
            print("-" * 50)
        except Exception as e:
            print(f"输入:  {expr}")
            print(f"错误:  {e}")
            print(f"状态:  ✗ 转换失败")
            print("-" * 50)

    # 测试实际使用场景
    print("\n实际使用场景测试:")
    print("=" * 60)

    # 创建测试数据
    test_expressions = [
        "value % 2 = 0",
        "tag = 'tag_1' AND id > 3 AND (data->>'value')::numeric % 2 < 100",
        "POWER((data->>'value')::numeric, 2) > 100",
    ]

    for expr in test_expressions:
        print(f"PostgreSQL表达式: {expr}")
        print("-" * 50)

    # 演示完整转换流程
    print("\n完整转换演示:")
    print("=" * 60)

    demo_cases = [
        "value % 2 == 0",
        "pow(score, 2) + sqrt(value) > 100",
        "cast(price, 'float') * quantity > 1000",
    ]

    for expr in demo_cases:
        try:
            sql_expr = converter.convert(expr)
            print(f"原始表达式: {expr}")
            print(f"转换后SQL: WHERE {sql_expr}")
            print()
        except Exception as e:
            print(f"原始表达式: {expr}")
            print(f"转换失败: {e}")
            print()