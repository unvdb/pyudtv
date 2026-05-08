import re
import ast
from typing import Any, List, Tuple, Optional, Union
# import sys

from typing import Pattern

# from pandas.core.dtypes.inference import is_number


class NumberChecker:
    """数字检查器（使用预编译正则表达式）"""

    # 预编译正则表达式（类属性）
    _INT_PATTERN: Pattern = re.compile(r'^[-+]?\d+$')
    _FLOAT_PATTERN: Pattern = re.compile(r'^[-+]?(?:\d+\.\d*|\.\d+|\d+)$')
    _SCIENTIFIC_PATTERN: Pattern = re.compile(r'^[-+]?(?:\d+\.\d*|\.\d+|\d+)e[-+]?\d+$', re.IGNORECASE)

    @classmethod
    def is_number(cls, s: str) -> bool:
        """
        判断字符串是否是数字（整数、小数、科学计数法）

        Args:
            s: 要检查的字符串

        Returns:
            bool: 是否是有效的数字
        """
        s = s.strip()

        if not s:  # 空字符串
            return False

        # 检查整数
        if cls._INT_PATTERN.match(s):
            return True

        # 检查小数
        if cls._FLOAT_PATTERN.match(s):
            return True

        # 检查科学计数法
        if cls._SCIENTIFIC_PATTERN.match(s):
            return True

        return False

    @classmethod
    def get_number_type(cls, s: str) -> str:
        """
        获取数字的具体类型

        Returns:
            str: 'integer', 'float', 'scientific', 'not_a_number'
        """
        s = s.strip()

        if not s:
            return 'empty'

        if cls._INT_PATTERN.match(s):
            return 'integer'

        if cls._SCIENTIFIC_PATTERN.match(s):
            return 'scientific'

        if cls._FLOAT_PATTERN.match(s):
            return 'float'

        return 'not_a_number'

def check_simple_type(value_str: str, return_type: bool = False) -> Union[bool, tuple]:
    """
    检查字符串是否为简单类型，并可返回实际类型

    Args:
        value_str: 要检查的字符串
        return_type: 是否返回类型信息

    Returns:
        如果return_type=False: 布尔值
        如果return_type=True: (是否匹配, 类型名称, 转换后的值)
    """
    if value_str.startswith('\'') and value_str.endswith('\'') and "'" not in value_str[1:-1] and '"' not in value_str[1:-1]:
        if return_type:
            return True, "str", value_str
        else:
            return True

    # 1. 检查整数
    int_pattern = r'^[-+]?\d+$'
    if re.match(int_pattern, value_str):
        if return_type:
            return True, 'int', int(value_str)
        return True

    # 2. 检查浮点数（小数形式）
    float_pattern = r'^[-+]?\d+\.\d+$'
    if re.match(float_pattern, value_str):
        if return_type:
            return True, 'float', float(value_str)
        return True

    # 3. 检查科学计数法
    scientific_pattern = r'^[-+]?\d+(\.\d+)?[eE][-+]?\d+$'
    if re.match(scientific_pattern, value_str):
        if return_type:
            return True, 'float', float(value_str)
        return True

    # 4. 检查布尔值
    bool_pattern = r'^(true|false|True|False|TRUE|FALSE)$'
    if re.match(bool_pattern, value_str):
        if return_type:
            bool_value = value_str.lower() == 'true'
            return True, 'bool', bool_value
        return True

    if return_type:
        return False, 'unknown', value_str
    return False

PG_KEYWORDS = {
    'ALL', 'ANALYSE', 'ANALYZE', 'AND', 'ANY', 'ARRAY', 'AS', 'ASC',
    'ASYMMETRIC', 'AUTHORIZATION', 'BETWEEN', 'BINARY', 'BOTH', 'CASE',
    'CAST', 'CHECK', 'COLLATE', 'COLUMN', 'CONCURRENTLY', 'CONSTRAINT',
    'CREATE', 'CROSS', 'CURRENT_CATALOG', 'CURRENT_DATE', 'CURRENT_ROLE',
    'CURRENT_SCHEMA', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER',
    'DEFAULT', 'DEFERRABLE', 'DESC', 'DISTINCT', 'DO', 'ELSE', 'END',
    'EXCEPT', 'FALSE', 'FETCH', 'FOR', 'FOREIGN', 'FREEZE', 'FROM',
    'FULL', 'GRANT', 'GROUP', 'HAVING', 'ILIKE', 'IN', 'INITIALLY',
    'INNER', 'INTERSECT', 'INTO', 'IS', 'ISNULL', 'JOIN', 'LEADING',
    'LEFT', 'LIKE', 'LIMIT', 'LOCALTIME', 'LOCALTIMESTAMP', 'NATURAL',
    'NOT', 'NOTNULL', 'NULL', 'OFFSET', 'ON', 'ONLY', 'OR', 'ORDER',
    'OUTER', 'OVER', 'OVERLAPS', 'PLACING', 'PRIMARY', 'REFERENCES',
    'RETURNING', 'RIGHT', 'SELECT', 'SESSION_USER', 'SIMILAR', 'SOME',
    'SYMMETRIC', 'TABLE', 'THEN', 'TO', 'TRAILING', 'TRUE', 'UNION',
    'UNIQUE', 'USER', 'USING', 'VARIADIC', 'VERBOSE', 'WHEN', 'WHERE',
    'WINDOW', 'WITH'
}

from functools import wraps

def expression_cache(function_call=False, binary_expression=False):
    def outer(func):
        @wraps(func)
        def run(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            if function_call:
                func_name = ret.split("(")[0].strip()
                if func_name.upper() in ['FLOOR', 'CEIL', 'ROUND', 'ABS', 'SQRT', 'POWER', 'EXP', 'LOG',
                                          'LOG10', 'MOD']:
                    self._expression_cache[ret] = {"func_name": func_name, "tokens": args[0] if args else [],
                                                   "data_type": 'numeric'}
                elif func_name.upper() == 'POINT':
                    self._expression_cache[ret] = {"func_name": func_name, "tokens": args[0] if args else [],
                                                   "data_type": 'point'}

                else:
                    self._expression_cache[ret] = {"func_name": func_name, "tokens": args[0] if args else []}

            elif binary_expression:
                tokens = args[0] if args else []
                op = tokens[1]
                if op in ["+", "-", '*', "/", "%", "**", "^", '%%']:
                    self._expression_cache[ret] = {"tokens": tokens,
                                                   "operation": op,
                                                   "data_type": 'numeric'}

                else:
                    self._expression_cache[ret] = {"tokens": tokens}

            else:
                self._expression_cache[ret] = True
            return ret
        return run
    return outer


class MilvusFilterToSQL:
    """将 Milvus filter 语法转换为 PostgreSQL SQL WHERE 子句"""

    def __init__(self):
        # 逻辑运算符
        self.logical_ops = {'and', 'or', 'not', 'is', 'is not', 'in', 'not in'}
        self.char_ops = {'~', '~*', '!~', '!~*', 'like', 'not like'}
        # milvus运算符与 sql运算符转换
        self.milvus_operator_to_sql_map = {"**": "^", "==": "="}
        # Milvus field type to normalized type mapping
        self.milvus_type_map = {
            # Integer types
            'Int64': 'int',
            'int64': 'int',
            'Int32': 'int',
            'int32': 'int',
            'Int16': 'int',
            'int16': 'int',
            'Int8': 'int',
            'int8': 'int',
            'UInt64': 'int',
            'uint64': 'int',
            'UInt32': 'int',
            'uint32': 'int',
            'UInt16': 'int',
            'uint16': 'int',
            'UInt8': 'int',
            'uint8': 'int',
            # Floating point types
            'Float': 'float',
            'float': 'float',
            'Double': 'double',
            'double': 'double',
            'Float16': 'float',
            'float16': 'float',
            # Boolean types
            'Bool': 'boolean',
            'bool': 'boolean',
            # String types
            'String': 'string',
            'string': 'string',
            'VARCHAR': 'string',
            # Other types
            'JSON': 'json',
            'Array': 'array',
            'Vector': 'vector',
            'FloatVector': 'vector',
            'Float16Vector': 'vector',
            'BinaryVector': 'vector',
            'SparseFloatVector': 'vector'
        }
        self._expression_cache = {}
    
    def _convert_field_with_type(self, field: str, field_type: str = None) -> str:
        """Convert field reference with proper type casting based on field type
        
        Args:
            field (str): The field name
            field_type (str): The field type from Milvus schema
            
        Returns:
            str: SQL expression with proper type casting
        """
        # Convert field reference to JSON access
        field_ref = self._convert_field_reference(field)
        
        # Normalize field type if provided
        if field_type:
            if field_type.upper() == 'POINT':
                return f"CAST({field_ref} as point)"
            elif field_type.upper() == 'VECTOR':
                return f"CAST({field_ref} as vector)"

            normalized_type = self.milvus_type_map.get(field_type, field_type)
            
            # Apply appropriate type casting
            if normalized_type in ('int', 'float', 'double', 'numeric'):
                return f"CAST({field_ref} AS numeric)"
            elif normalized_type in ('bool', 'boolean'):
                return f"({field_ref})::boolean"
            elif normalized_type in ('json'):
                return field_ref
        
        # Return field reference without casting if no type provided
        return field_ref

    def _tokenize_expression(self, expr: str) -> List[str]:
        """将表达式拆分为 token，确保正确识别逻辑运算符"""
        tokens = []
        i = 0
        n = len(expr)

        while i < n:
            char = expr[i]

            # 跳过空格
            if char in (' ', '\t', '\n'):
                i += 1
                continue

            # 处理括号
            if char in ('(', ')'):
                tokens.append(char)
                i += 1
                continue

            # 处理引号字符串
            if char == "'": # 单引号处理
                # 找到匹配的结束引号
                j = i + 1
                while j < n and expr[j] != char:
                    # 双引号
                    if j + 2 < n and expr[j + 1] == char and expr[j + 2] == char:
                        j += 3
                    else:
                        j += 1

                if j < n:
                    tokens.append(expr[i:j + 1])
                    i = j + 1
                else:
                    if expr[-1] != char:
                        raise Exception(f"No matching single quote on the right side")
                    tokens.append(expr[i:])
                    break
                continue

            if char == '"': # 双引号处理
                # 找到匹配的结束引号
                j = i + 1
                while j < n and expr[j] != char:
                    # \" 双引号转义
                    if j + 2 < n and expr[j + 1] == "\\" and expr[j + 2] == char:
                        j += 3
                    else:
                        j += 1

                if j < n:
                    tokens.append(expr[i:j + 1])
                    i = j + 1
                else:
                    if expr[-1] != char:
                        raise Exception(f"No matching double quote on the right side")
                    tokens.append(expr[i:])
                    break
                continue

            # 处理数组
            if char == '[':
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if expr[j] == '[':
                        depth += 1
                    elif expr[j] == ']':
                        depth -= 1
                    j += 1
                tokens.append(expr[i:j])
                i = j
                continue

            if char == '`': # 处理反引号标识符
                j = i + 1
                while j < n and expr[j] != char:
                    j += 1

                if j < n:
                    tokens.append(expr[i:j + 1])
                    i = j + 1
                else:
                    if expr[-1] != char:
                        raise Exception(f"No matching backtick on the right side")
                    tokens.append(expr[i:])
                    break
                continue

            # 处理标识符和逻辑运算符
            if char.isalpha() or char == '_':
                j = i
                while j < n and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1

                token = expr[i:j]
                token_lower = token.lower()

                # 检查是否是逻辑运算符
                if token_lower in self.logical_ops:
                    tokens.append(token_lower)
                else:
                    tokens.append(token)
                i = j
                continue

            if char == "*": # 处理乘方 **算符
                if i + 1 < n and expr[i + 1] == "*":
                    tokens.append(expr[i: i + 2])
                    i = i + 2
                    continue

            # 处理比较运算符
            if char in ('=', '!', '>', '<'):
                j = i
                while j < n and expr[j] in ('=', '!', '>', '<'):
                    j += 1
                tokens.append(expr[i:j])
                i = j
                continue

            # 处理数字
            if char.isdigit() or (char == '-' and i + 1 < n and expr[i + 1].isdigit()):
                j = i
                if char == '-':
                    j += 1
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(expr[i:j])
                i = j
                continue

            # 处理其他字符
            tokens.append(char)
            i += 1

        if "=" in tokens:
            raise  Exception(f"Operator '=' is not supported")

        # n = len(tokens)
        # 归并token, 处理使用空格拼接的token组合
        new_tokens = []
        merge_ops = self.logical_ops | self.char_ops
        while len(tokens) >= 2:
            t = tokens.pop(0)
            t2 = tokens[0]
            if t in merge_ops and t2 in merge_ops:
                new_t = f"{t} {t2}"
                if new_t in merge_ops:
                    new_tokens.append(new_t)
                    tokens.pop(0)
                    continue

            if len(tokens) >= 2:
                t3 = tokens[1]
                new_t = f"{t}{t2}{t3}"
                if new_t in ["<->", '<#>', '<=>']: # point 或者 vector 类型数据操作符
                    new_tokens.append(new_t)
                    tokens.pop(0)
                    tokens.pop(0)
                    continue

            new_tokens.append(t)

        while tokens:
            new_tokens.append(tokens.pop(0))

        # assert len(new_tokens) <= n, "归并token异常"
        return new_tokens

    def _parse_comparison_expression(self, expr: str) -> Optional[Tuple[str, str, Any]]:
        """解析比较表达式: field operator value"""
        expr = expr.strip()
        if not expr:
            return None

        # 处理 IN/NOT IN
        in_pattern = r'^\s*(\w+)\s+(not\s+)?in\s+\[(.*?)\]\s*$'
        in_match = re.match(in_pattern, expr, re.IGNORECASE)
        if in_match:
            field = in_match.group(1)
            not_in = bool(in_match.group(2))
            values_str = in_match.group(3)
            operator = 'not in' if not_in else 'in'

            try:
                values = ast.literal_eval(f"[{values_str}]")
            except:
                values = []
                parts = [p.strip() for p in values_str.split(',') if p.strip()]
                for part in parts:
                    if (part.startswith("'") and part.endswith("'")) or \
                            (part.startswith('"') and part.endswith('"')):
                        values.append(part[1:-1])
                    else:
                        if part.lower() in ('true', 'false'):
                            values.append(part.lower() == 'true')
                        else:
                            try:
                                if '.' in part:
                                    values.append(float(part))
                                else:
                                    values.append(int(part))
                            except:
                                values.append(part)

            return field, operator, values

        # 处理 LIKE/NOT LIKE
        like_pattern = r'^\s*(\w+)\s+(not\s+)?like\s+(.+?)\s*$'
        like_match = re.match(like_pattern, expr, re.IGNORECASE)
        if like_match:
            field = like_match.group(1)
            not_like = bool(like_match.group(2))
            value_str = like_match.group(3)
            operator = 'not like' if not_like else 'like'

            value_str = value_str.strip()
            if (value_str.startswith("'") and value_str.endswith("'")) or \
                    (value_str.startswith('"') and value_str.endswith('"')):
                value = value_str[1:-1]
            else:
                value = value_str

            return field, operator, value

        return None
        # 处理比较运算符
        # comp_pattern = r'^\s*(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$'
        # comp_match = re.match(comp_pattern, expr)
        # if comp_match:
        #     field = comp_match.group(1)
        #     operator = comp_match.group(2)
        #     value_str = comp_match.group(3)
        #     value_str = value_str.strip()
        #
        #     # 检查值的类型
        #     if not check_simple_type(value_str.strip()):
        #         return None
        #
        #     if value_str.lower() in ('true', 'false'):
        #         value = value_str.lower() == 'true'
        #     elif re.match(r'^-?\d+(\.\d+)?$', value_str):
        #         # if '.' in value_str:
        #         #     value = float(value_str)
        #         # else:
        #         #     value = int(value_str)
        #         return None
        #     elif (value_str.startswith("'") and value_str.endswith("'")) or \
        #             (value_str.startswith('"') and value_str.endswith('"')):
        #         value = value_str[1:-1]
        #     else:
        #         return None
        #         # try:
        #         #     if '.' in value_str:
        #         #         value = float(value_str)
        #         #     else:
        #         #         value = int(value_str)
        #         # except:
        #         #     value = value_str
        #
        #     return field, operator, value
        #
        # return None

    def _convert_field_reference(self, field: str) -> str:
        """转换字段引用为 PostgreSQL JSON 访问"""
        return f"data->>'{field}'"

    def _convert_value(self, value: Any) -> Tuple[str, str]:
        """转换值为 SQL 表达式"""
        if isinstance(value, bool):
            sql_value = 'true' if value else 'false'
            return sql_value, 'boolean'
        elif isinstance(value, (int, float)):
            return str(value), 'numeric'
        elif isinstance(value, str):
            # escaped = value.replace("'", "''")
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            escaped = value.replace("'", "''")
            sql_value = f"'{escaped}'"
            return sql_value, 'text'
        elif isinstance(value, list):
            if not value:
                return "()", 'text'

            values_sql = []
            types = set()

            for v in value:
                v_sql, v_type = self._convert_value(v)
                values_sql.append(v_sql)
                types.add(v_type)

            if len(types) == 1:
                value_type = next(iter(types))
            elif 'numeric' in types:
                value_type = 'numeric'
            elif 'boolean' in types:
                value_type = 'boolean'
            else:
                value_type = 'text'

            sql_value = f"({', '.join(values_sql)})"
            return sql_value, value_type
        else:
            escaped = str(value).replace("'", "''")
            sql_value = f"'{escaped}'"
            return sql_value, 'text'

    def _build_sql_expression(self, field: str, operator: str, value: Any, value_type='', field_types: dict = None) -> str:
        """构建 SQL 表达式"""
        # 获取字段类型（如果提供）
        field_type = None
        if field_types and isinstance(field, str):
            # 移除字段名中的引号
            clean_field = field.strip('`')
            field_type = field_types.get(clean_field)

        if isinstance(field, str):
            field_ref = self._convert_field_with_type(field, field_type)
        else:
            field_ref = field

        if not value_type:
            sql_value, value_type = self._convert_value(value)
        else:
            sql_value = value

        # If no field type provided, use value-based type detection
        if isinstance(field, str) and not field_type:
            if value_type == 'numeric':
                field_ref = f"CAST({self._convert_field_reference(field)} AS numeric)"
            elif value_type == 'boolean':
                field_ref = f"({self._convert_field_reference(field)})::boolean"

        if operator == '==':
            sql_operator = '='
        elif operator == '!=':
            sql_operator = '!='
        else:
            sql_operator = operator.upper()

        # IN/NOT IN
        if operator in ('in', 'not in'):
            return f"{field_ref} {sql_operator} {sql_value}"

        # LIKE/NOT LIKE
        if operator in ('like', 'not like'):
            return f"{field_ref} {sql_operator} {sql_value}"

        # 比较运算符
        if operator in ('==', '!=', '>', '<', '>=', '<='):
            return f"{field_ref} {sql_operator} {sql_value}"

        if operator in ('+', '-', '*', "/", "**", "^", "%", '//'):
            if not field_ref.startswith('CAST(') and not field_ref.startswith('('):
                # Ensure numeric casting for mathematical operations
                if isinstance(field, str):
                    field_ref = f"CAST({self._convert_field_reference(field)} AS numeric)"
            return f"{field_ref} {sql_operator} {sql_value}"

        return f"{field_ref} {sql_operator} {sql_value}"

    def _parse_expression_tokens(self, tokens: List[str]) -> str:
        """递归解析 token 表达式"""
        if not tokens:
            return ""

        # 先尝试解析整个 token 序列
        expr_str = ' '.join(tokens)
        parsed = self._parse_comparison_expression(expr_str)
        if parsed:
            field, operator, value = parsed
            return self._build_sql_expression(field, operator, value)

        # 如果整个 token 序列不是简单表达式，尝试解析逻辑表达式
        return self._parse_logical_expression_tokens(tokens)

    def _parse_logical_expression_tokens(self, tokens: List[str]) -> str:
        """解析 token 的逻辑表达式"""
        # 先按 OR 拆分
        or_parts = self._split_tokens_by_operator(tokens, 'or')
        if len(or_parts) > 1:
            sql_parts = []
            for part in or_parts:
                sql_part = self._parse_expression_tokens(part)
                sql_parts.append(sql_part)
            return ' OR '.join(sql_parts)

        # 再按 AND 拆分
        and_parts = self._split_tokens_by_operator(tokens, 'and')
        if len(and_parts) > 1:
            sql_parts = []
            for part in and_parts:
                sql_part = self._parse_expression_tokens(part)
                sql_parts.append(sql_part)
            return ' AND '.join(sql_parts)

        # 如果既不是 AND 也不是 OR，尝试解析为比较表达式
        expr_str = ' '.join(tokens)
        parsed = self._parse_comparison_expression(expr_str)
        if parsed:
            field, operator, value = parsed
            return self._build_sql_expression(field, operator, value)

        # 如果无法解析，返回原样
        return expr_str

    def _split_tokens_by_operator(self, tokens: List[str], operator: str) -> List[List[str]]:
        """按逻辑运算符拆分 token 列表"""
        parts = []
        current = []
        depth = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '(':
                depth += 1
                current.append(token)
            elif token == ')':
                depth -= 1
                current.append(token)
            elif depth == 0 and token == operator.lower():
                if current:
                    parts.append(current)
                    current = []
            else:
                current.append(token)

            i += 1

        if current:
            parts.append(current)

        return parts

    def _parse_logical_expression_full(self, expr: str, tokens=None, field_types: dict = None) -> str:
        """解析完整的逻辑表达式"""
        # if "CAST" in expr and not tokens:
        #     print('=============test =====')

        if not tokens:
            expr = expr.strip()
            if not expr:
                return ""
            # 使用 tokenize 和递归解析
            tokens = self._tokenize_expression(expr)

        if "(" in tokens:
            # 递归处理括号
            return self._expression_parse_by_stack(tokens, field_types=field_types)

        # 先按 OR 拆分
        or_parts = self._split_tokens_by_operator(tokens, 'or')
        if len(or_parts) > 1:
            sql_parts = []
            for part in or_parts:
                # sql_part = self._parse_logical_expression_full(' '.join(part))
                sql_part = self._parse_logical_expression_full(' '.join(part), tokens=part, field_types=field_types)
                sql_parts.append(sql_part)
            return ' OR '.join(sql_parts)

        # 再按 AND 拆分
        and_parts = self._split_tokens_by_operator(tokens, 'and')
        if len(and_parts) > 1:
            sql_parts = []
            for part in and_parts:
                # sql_part = self._parse_logical_expression_full(' '.join(part))
                sql_part = self._parse_logical_expression_full(' '.join(part), tokens=part, field_types=field_types)
                sql_parts.append(sql_part)
            return ' AND '.join(sql_parts)

        # 如果既不是 AND 也不是 OR，尝试解析为比较表达式
        expr_str = ' '.join(tokens)
        parsed = self._parse_comparison_expression(expr_str)
        if parsed:
            field, operator, value = parsed
            return self._build_sql_expression(field, operator, value, field_types=field_types).replace('%', '%%')

        # 运算符优先级列表（从高到低）
        # ^(**)
        # *, /, %
        # +, -
        # =, !=, <>, >, <, >=, <=
        # IN, NOT IN, LIKE, ILIKE, '~', '~*', '!~', '!~*'
        # IS, IS NOT
        # NOT
        # AND
        # OR

        for op in (['not', 'and', 'or']  + ['is', 'is not'] + ['in', 'not in', 'like', 'not like', '~', '~*', '!~', '!~*'] +
                   ["==", "!=", ">=", "<=", ">", "<"] + ['<->', '<=>', '<#>'] + ["+", "-", '*', "/", "%", "**", "^"]):
            if op.lower() not in tokens:
                continue

            if len(tokens) < 2:
                raise Exception(f"{op} parse error")
            if len(tokens) == 2:
                return self._build_unary_expression(tokens, field_types=field_types)

            if len(tokens) == 3 and op.lower() == tokens[1]: # 简单二元表达式
                return self._build_binary_expression(tokens, field_types=field_types)

            op_parts = self._split_tokens_by_operator(tokens, op)
            sql_parts = []
            is_numeric = any([i in ["+", "-", "*", "**", '%', "/", "^"] for i in tokens])

            for part in op_parts:
                # sql_part = self._parse_logical_expression_full(' '.join(part), field_types=field_types)
                sql_part = self._parse_logical_expression_full(' '.join(part), tokens=part, field_types=field_types)

                # 处理单个字段的数学表达式
                if len(part) == 1 and is_numeric:
                    is_identifier = self.is_valid_sql_identifier(part[0], field_types=field_types)[0]
                    if is_identifier and 'numeric' not in part[0].lower():
                        sql_part = f"CAST({sql_part} AS numeric)"

                sql_parts.append(sql_part)

            op_ = "=" if op == "==" else op

            # 处理in 表达式
            if op in ['in', 'not in'] and tokens[-2].lower() in ['in', 'not in'] and tokens[-1].startswith('[') and tokens[-1].endswith(']'):
                sql_parts[-1] = self.handler_tuple_str(sql_parts[-1])

            return f" {op_} ".join(sql_parts)

        # if len(tokens) == 3 and tokens[1].lower() in ['*', "**", "+", "-", "%", "/", "//", "^", 'is', 'is not']:
        #     return self._build_binary_expression(tokens, field_types=field_types)

        # 处理单个标识符字段
        if len(tokens) == 1 and isinstance(tokens[0], str) and "(" not in tokens[0]:
            is_identifier = self.is_valid_sql_identifier(tokens[0], field_types=field_types)[0]
            if is_identifier:
                field = tokens[0].replace('`', '')
                return  self._convert_field_reference(field)

        if self._is_function_expression(tokens):
            return self._build_function_expression(tokens, field_types=field_types)

        # 如果无法解析，返回原样
        return expr_str

    def handler_tuple_str(self, s):
        values = [si.strip() for si in s.strip()[1:-1].split(",")]
        for i in range(len(values)):
            if values[i].lower() in ['true', 'false']:
                values[i] = values[i].lower()
            elif values[i].startswith('"') and values[i].endswith('"'):
                values[i] = "'" + values[i][1:-1] + "'"
            elif values[i].startswith("'") and values[i].endswith("'"):
                values[i] = "'" + values[i][1:-1] + "'"

        return "(" + ','.join(values) + ")"

    def _expression_parse_by_stack(self, tokens, field_types: dict = None):
        stack = []
        n = len(tokens)

        new_tokens = []
        to_index = 0
        for i in range(n):
            if i < to_index: # 跳过已经处理的token
                continue

            if tokens[i] == ')':
                if not stack:
                    raise Exception("Parenthesis error")
                # 出栈
                child_tokens = []
                while stack and stack[-1] != '(':
                    c = stack.pop()
                    child_tokens.insert(0, c)
                stack.pop() # 弹出（
                expression = self._parse_logical_expression_full(" ".join(child_tokens), tokens=child_tokens, field_types=field_types)
                expression = f"({expression})"
                self._expression_cache[expression] = True

                if not stack:
                    new_tokens.append(expression)
                    continue
                stack.append(expression)
                continue

            if tokens[i] == '(' or stack:
                stack.append(tokens[i])
                continue

            # 处理函数调用
            if self.is_valid_sql_identifier(tokens[i], allow_quoted=False)[0]:
                if i + 1 < n and tokens[i + 1] == '(':
                    end_i = i + 2
                    s = []
                    while end_i < n and (tokens[end_i] != ')' or s):
                        if tokens[end_i] == '(':  # 嵌套函数或者括号
                            s.append('(')
                        if s and tokens[end_i] == ')':
                            s.pop()

                        end_i += 1

                    if end_i >= n or tokens[end_i] != ')':
                        raise Exception(f"Parenthesis error: {' '.join(tokens)}")

                    expression = self._build_function_expression(tokens[i:end_i + 1], field_types=field_types)
                    new_tokens.append(expression)
                    to_index = end_i + 1
                    continue

            new_tokens.append(tokens[i])

        if new_tokens:
            return self._parse_logical_expression_full(" ".join(new_tokens), new_tokens, field_types=field_types)

        raise Exception("parse error")

    def _is_function_expression(self, tokens):
        if not tokens:
            return False
        is_identifier, _, _ = self.is_valid_sql_identifier(tokens[0], allow_quoted=False)
        is_sample_function = False
        if is_identifier and len(tokens) >= 3 and  tokens[1] == "(" and tokens[-1] == ")":
            stack = []
            min_index = 1
            for i, t in enumerate(tokens):
                if t == "(":
                    stack.append((i, t))
                elif t == ")":
                    c = stack.pop()
                    if i == len(tokens) - 1 and c[0] == min_index:
                        is_sample_function = True

        return  is_sample_function

    @expression_cache(function_call=True)
    def _build_function_expression(self, tokens, field_types: dict=None) -> str:
        # ['FLOOR', '(', 'test_score', ',', '2', ')']
        function_name = tokens[0].upper()
        
        # Handle DATEDIFF function - PostgreSQL uses '-' operator instead
        if function_name == 'DATEDIFF':
            if len(tokens) >= 6:
                # Extract parameters: DATEDIFF('day', create_date, '2023-12-31')
                unit = tokens[2].strip("'\"")
                field = tokens[4].replace('`', '')
                date_value = tokens[6]
                
                # Convert field reference
                field_ref = self._convert_field_reference(field)
                
                # Build PostgreSQL equivalent
                if unit.lower() == 'day':
                    return f"(CAST({date_value} AS date) - CAST({field_ref} AS date))"
                elif unit.lower() == 'hour':
                    return f"EXTRACT(EPOCH FROM (CAST({date_value} AS timestamp) - CAST({field_ref} AS timestamp))) / 3600"
                elif unit.lower() == 'minute':
                    return f"EXTRACT(EPOCH FROM (CAST({date_value} AS timestamp) - CAST({field_ref} AS timestamp))) / 60"
                elif unit.lower() == 'second':
                    return f"EXTRACT(EPOCH FROM (CAST({date_value} AS timestamp) - CAST({field_ref} AS timestamp)))"

        if function_name == 'EXTRACT':
            new_tokens = []
            n = len(tokens)
            i = 0

            while i < n:
                if tokens[i] == '(':
                    new_tokens.append(tokens[i])
                    for j in range(i + 1, n - 1):
                        token = tokens[j]
                        is_identifier = self.is_valid_sql_identifier(token, allow_quoted=True, field_types=field_types)[0]

                        # Skip function parameters like YEAR, MONTH, DAY in EXTRACT
                        if function_name == 'EXTRACT' and token.upper() in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE',
                                                                            'SECOND', 'WEEK', 'QUARTER']:
                            new_tokens.append(token)
                        elif function_name == 'EXTRACT' and token.upper() == 'FROM':
                            new_tokens.append(token)
                        elif not is_identifier:
                            new_tokens.append(token)
                        else:
                            field = token.replace('`', '')
                            # For EXTRACT function, we need to cast to timestamp
                            if function_name == 'EXTRACT':
                                field_ref = self._convert_field_reference(field)
                                new_tokens.append(f"CAST({field_ref} AS timestamp)")
                            # For mathematical functions, ensure numeric casting
                            elif function_name in ['FLOOR', 'CEIL', 'ROUND', 'ABS', 'SQRT', 'POWER', 'EXP', 'LOG',
                                                   'LOG10', 'MOD']:
                                field_ref = self._convert_field_reference(field)
                                new_tokens.append(f"CAST({field_ref} AS numeric)")
                            else:
                                # Use the unified conversion function for other cases
                                field_ref = self._convert_field_with_type(field)
                                new_tokens.append(field_ref)
                        i = j
                    i += 1
                    continue
                new_tokens.append(tokens[i])
                i += 1
            return ' '.join(new_tokens)

        # 通用函数处理
        arg_token = tokens[2:-1]
        args = []
        a = []
        i = 0
        while i < len(arg_token):
            t = arg_token[i]
            if t == ',':
                args.append(a)
                a = []
                i += 1
                continue

            elif self.is_valid_sql_identifier(t, allow_quoted=True, field_types=field_types)[0]:
                # 处理函数嵌套函数
                j = i + 1
                if j < len(arg_token) and arg_token[j] == '(': # 函数
                    n = 1
                    while j + 1 < len(arg_token) and n:
                        j += 1
                        if arg_token[j] == ')':
                            n -= 1
                        elif arg_token[j] == '(':
                            n += 1
                    if n:
                        raise Exception(f'Left parenthesis mismatch error: {" ".join(arg_token)}')

                    if j + 1 < len(arg_token) and arg_token[j + 1] == ',':
                        args.append(arg_token[i: j + 1])
                        a = []
                        i = j + 2
                        continue
                    elif not a and j >= len(arg_token) - 1:
                        args.append(arg_token[i: j + 1])
                        a = []
                        i = j + 1
                        continue

            a.append(t)
            i += 1

        if a:
            args.append(a)

        pars_args = []
        for arg in args:
            if len(arg) == 1:
                value = arg[0]
                is_identifier, _, _ = self.is_valid_sql_identifier(value, allow_quoted=True, field_types=field_types)
                if is_identifier:

                    value = value.strip('`')
                    field_type = field_types.get(value) if field_types else None
                    if not field_type:
                        if function_name.upper() in ['FLOOR', 'CEIL', 'ROUND', 'ABS', 'SQRT', 'POWER', 'EXP', 'LOG',
                                          'LOG10', 'MOD']:
                            field_type = 'numeric'

                    value = self._convert_field_with_type(value, field_type)
                    pars_args.append(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = "'" + value[1:-1] + "'"
                    pars_args.append(value)
                elif NumberChecker.is_number(value):
                    pars_args.append(value)
                else:
                    # 递归，支持嵌套函数，表达式
                    a = self._parse_logical_expression_full(" ".join(arg), tokens=arg, field_types=field_types)
                    pars_args.append(a)

            else:
                # 递归，支持嵌套函数， 表达式
                a = self._parse_logical_expression_full(" ".join(arg), tokens=arg, field_types=field_types)
                pars_args.append(a)

        return f"{function_name}({','.join(pars_args)})"


    def is_valid_sql_identifier(self,
            identifier: str,
            allow_quoted: bool = True,
            field_types: dict=None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        检查字符串是否是合法的 PostgreSQL 标识符

        Args:
            identifier: 要检查的字符串
            allow_quoted: 是否允许加引号的标识符

        Returns:
            tuple: (是否合法, 原因说明, 建议名称)
        """
        if field_types and field_types.get(identifier):
            return True, '表定义字段', identifier

        if allow_quoted and field_types and field_types.get(identifier.strip('`')):
            return True, '表定义字段', identifier

        # 1. 检查长度
        if len(identifier.encode('utf-8')) > 63:
            return False, "标识符长度超过63字节", identifier[:20]

        # 2. 判断是否加反引号
        is_quoted = identifier.startswith('`') and identifier.endswith('`')

        if is_quoted:
            if not allow_quoted:
                return False, "不允许使用引号标识符", identifier.strip('"')

            # 处理加引号标识符
            inner = identifier[1:-1]  # 去掉引号

            # 检查空字符串
            if not inner:
                return False, "引号内标识符为空", 't'

            # 检查是否包含空字符
            if '\x00' in inner:
                return False, "标识符包含空字符", inner.replace('\x00', '')

            # 检查引号内的引号
            if '`' in inner:
                return False, "引号内包含引号", inner.replace('"', '')

            return True, "有效的引号标识符", identifier

        else:
            # 处理未加引号标识符
            # 检查首字符
            if not re.match(r'^[a-zA-Z_]', identifier):
                return False, "标识符必须以字母或下划线开头", f"t_{identifier}"

            # 检查字符集
            if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
                return False, "标识符包含非法字符", re.sub(r'[^a-zA-Z0-9_]', '_', identifier)

            # 检查是否是 PostgreSQL 关键字
            if identifier.upper() in PG_KEYWORDS:
                return False, "标识符是 PostgreSQL 保留关键字", f'"{identifier}"'

            return True, "有效的未加引号标识符", identifier

    @expression_cache()
    def _build_unary_expression(self, tokens, field_types=None):
        if len(tokens) != 2:
            raise Exception(f"parse error: {' '.join(tokens)}")

        if tokens[0] not in ["not", '-', '+']:
            raise Exception(f"parse error: {tokens[0]}")

        op, value = tokens
        if self._expression_cache.get(value):
            return f"{op} {value}"

        is_identifier, _, _ = self.is_valid_sql_identifier(value, allow_quoted=True, field_types=field_types)
        if is_identifier:
            value = value.strip('`')
            field_type = field_types.get(value) if field_types else None
            if not field_type:
                field_type = "numeric" if op in ['+', '-', '*', '/', '^', '%', '%%'] else ('boolean' if op.lower() == 'not' else None)

            value = self._convert_field_with_type(value, field_type)
        else:
            if value.lower() in ['true', 'false', 'null']:
                value = value.lower()
            if value.startswith('"') and value.endswith('"'):
                value = "'" + value[1:-1] + "'"
                value = value.replace("%", "%%")
            elif value.startswith("'") and value.endswith("'"):
                value = value.replace("%", "%%")

        return f'{op} {value}'

    @expression_cache(binary_expression=True)
    def _build_binary_expression(self, tokens, field_types: dict = None) -> str:
        op = tokens[1]
        op_ = self.milvus_operator_to_sql_map.get(op, op)
        if op_ == "%":
            op_ = "%%"

        # 纯数学表达式，直接输出即可
        if tokens[0].isdigit() and tokens[2].isdigit() and op_ in ['+', '-', '*', '/', '^', '%', '%%']:
            return f'{tokens[0]} {op_} {tokens[2]}'

        e1 = e2 = ""
        if self._expression_cache.get(tokens[0]):
           e1 = tokens[0]
        if self._expression_cache.get(tokens[2]):
            e2 = tokens[2]
        if e1 and e2:
            return f'{e1} {op_} {e2}'

        if e1:
            right = tokens[2]
            is_identifier, _, _ = self.is_valid_sql_identifier(right, allow_quoted=True, field_types=field_types)
            if is_identifier:
                right = right.strip('`')
                field_type = field_types.get(right) if field_types else None
                if not field_type:
                    field_type = "numeric" if op_ in ['+', '-', '*', '/', '^', '%', '%%'] else None

                right = self._convert_field_with_type(right, field_type)
            else:
                if right.lower() in ['true', 'false', 'null']:
                    right = right.lower()
                if right.startswith('"') and right.endswith('"'):
                    right = "'" + right[1:-1] + "'"
                    right = right.replace("%", "%%")
                elif right.startswith("'") and right.endswith("'"):
                    right = right.replace("%", "%%")
                elif right.startswith("[") and right.endswith("]"):
                    right = self.handler_tuple_str(right)

            return f'{e1} {op_} {right}'

        if e2:
            left = tokens[0]
            is_identifier, _, _ = self.is_valid_sql_identifier(left, allow_quoted=True, field_types=field_types)
            if is_identifier:
                left = left.strip('`')
                field_type = field_types.get(left) if field_types else None
                if not field_type or (op_.lower() in ['<->', '<=>', '<#>']): # todo: 优化<->判断逻辑
                    exp_type = None
                    if isinstance(self._expression_cache.get(tokens[2]), dict):
                        exp_type = self._expression_cache.get(tokens[2]).get("data_type")
                    if op_ in ['+', '-', '*', '/', '^', '%', '%%']:
                        field_type = "numeric"
                    elif op_.lower() in ['<->', '<=>', '<#>']:
                        if exp_type:
                            field_type = exp_type
                        else:
                            field_type = 'vector'
                    elif exp_type:
                        field_type = exp_type
                    else:
                        field_type = None

                left = self._convert_field_with_type(left, field_type)
            else:
                if left.lower() in ['true', 'false', 'null']:
                    left = left.lower()

                if left.startswith('"') and left.endswith('"'):
                    left = "'" + left[1:-1] + "'"

                elif left.startswith("'") and left.endswith("'"):
                    left = left.replace("%", "%%")

                elif left.startswith("[") and left.endswith("]"):
                    left = self.handler_tuple_str(left)

            return f'{left} {op_} {e2}'


        left = tokens[0]
        right = tokens[2]

        is_identifier, _, _ = self.is_valid_sql_identifier(left, allow_quoted=True, field_types=field_types)
        if is_identifier:
            left = left.strip('`')
            field_type = field_types.get(left) if field_types else None
            if not field_type:
                if op_ in ['+', '-', '*', '/', '^', '%', '%%']:
                    field_type = 'numeric'
                elif NumberChecker.is_number(right):
                    field_type = 'numeric'
                elif right.lower() in ['true', 'false']:
                    field_type = 'boolean'
                elif right.startswith('"') and right.endswith('"'):
                    field_type = 'string'
                elif right.startswith("'") and right.endswith("'"):
                    field_type = 'string'

            left = self._convert_field_with_type(left, field_type)
        else:
            if left.lower() in ['true', 'false', 'null']:
                left = left.lower()
            if left.startswith('"') and left.endswith('"'):
                left = "'" + left[1:-1].replace('%', '%%') + "'"

            elif left.startswith("'") and left.endswith("'"):
                left = left.replace("%", "%%")

            elif left.startswith("[") and left.endswith("]"):
                left = self.handler_tuple_str(left)


        is_identifier, _, _ = self.is_valid_sql_identifier(right, allow_quoted=True, field_types=field_types)
        if is_identifier:
            right = right.strip('`')
            field_type = field_types.get(right) if field_types else None
            if not field_type:
                if op_ in ['+', '-', '*', '/', '^', '%', '%%']:
                    field_type = 'numeric'
                elif NumberChecker.is_number(left):
                    field_type = 'numeric'
                elif left.lower() in ['true', 'false']:
                    field_type = 'boolean'
                elif left.startswith('"') and left.endswith('"'):
                    field_type = 'string'
                elif left.startswith("'") and left.endswith("'"):
                    field_type = 'string'

            right = self._convert_field_with_type(right, field_type)
        else:
            if right.lower() in ['true', 'false', 'null']:
                right = right.lower()
            # 字符串
            if right.startswith('"') and right.endswith('"'):
                right = "'" + right[1:-1].replace('%', '%%') + "'"

            elif right.startswith("'") and right.endswith("'"):
                right = right.replace("%", "%%")

            elif right.startswith("[") and right.endswith("]"):
                right = self.handler_tuple_str(right)

        return f'{left} {op_} {right}'

    def process_filter(self, filter_str: str, query: str, field_types: dict = None) -> str:
        """处理 filter 字符串，生成 SQL WHERE 子句并拼接到查询"""
        if not filter_str or not filter_str.strip():
            return query

        filter_str = filter_str.strip()
        where_clause = self._parse_logical_expression_full(filter_str, field_types=field_types)

        if not where_clause:
            return query

        query_lower = query.lower()
        where_pos = query_lower.find(' where ')

        if where_pos != -1:
            before_where = query[:where_pos + 7]
            after_where = query[where_pos + 7:]

            if after_where.strip():
                return f"{before_where}({after_where}) AND ({where_clause})"
            else:
                return f"{before_where}{where_clause}"
        else:
            insert_pos = len(query)
            keywords = [
                (' order by ', query_lower.find(' order by ')),
                (' group by ', query_lower.find(' group by ')),
                (' limit ', query_lower.find(' limit ')),
                (' offset ', query_lower.find(' offset ')),
                (';', query_lower.find(';'))
            ]

            for keyword, pos in keywords:
                if pos != -1 and pos < insert_pos:
                    insert_pos = pos

            if insert_pos < len(query):
                where_part = f" WHERE {where_clause}"
                if insert_pos > 0 and query[insert_pos - 1] != ' ':
                    where_part = ' ' + where_part.lstrip()
                return f"{query[:insert_pos]}{where_part} {query[insert_pos:]}".strip()
            else:
                return f"{query} WHERE {where_clause}"



def test_milvus_filter_to_sql():
    """测试 MilvusFilterToSQL 类"""
    converter = MilvusFilterToSQL()

    test_cases = [
        # 简单比较
        ("id == 1", "CAST(data->>'id' AS numeric) = 1"),
        ("age > 18", "CAST(data->>'age' AS numeric) > 18"),
        ("name == 'John'", "data->>'name' = 'John'"),
        ("active == true", "(data->>'active')::boolean = true"),
        ("score != 100", "CAST(data->>'score' AS numeric) != 100"),
        ("price == 19.99", "CAST(data->>'price' AS numeric) = 19.99"),
        ("enabled == false", "(data->>'enabled')::boolean = false"),

        # IN/NOT IN
        ("id in [1, 2, 3]", "CAST(data->>'id' AS numeric) IN (1, 2, 3)"),
        ("category in ['A', 'B', 'C']", "data->>'category' IN ('A', 'B', 'C')"),
        ("id not in [4, 5, 6]", "CAST(data->>'id' AS numeric) NOT IN (4, 5, 6)"),
        ("tags in ['tag1', 'tag2']", "data->>'tags' IN ('tag1', 'tag2')"),
        ("status in [1, 2, 3]", "CAST(data->>'status' AS numeric) IN (1, 2, 3)"),
        ("flags in [true, false]", "(data->>'flags')::boolean IN (true, false)"),

        # LIKE/NOT LIKE
        # ("name like '%John%'", "data->>'name' LIKE '%John%'"),
        ("name like '%John%'", "data->>'name' LIKE '%%John%%'"),
        # ("name not like '%Doe%'", "data->>'name' NOT LIKE '%Doe%'"),
        ("name not like '%Doe%'", "data->>'name' NOT LIKE '%%Doe%%'"),
        # ("email like '%@gmail.com'", "data->>'email' LIKE '%@gmail.com'"),
        ("email like '%@gmail.com'", "data->>'email' LIKE '%%@gmail.com'"),
        ("title like 'test_'", "data->>'title' LIKE 'test_'"),

        # 复杂表达式 - 重点测试用例
        ("(age > 18) and (active == true)",
         "(CAST(data->>'age' AS numeric) > 18) AND ((data->>'active')::boolean = true)"),
        ("(id in [1, 2]) or (name == 'John')", "(CAST(data->>'id' AS numeric) IN (1, 2)) OR (data->>'name' = 'John')"),
        ("age >= 18 and age <= 60", "CAST(data->>'age' AS numeric) >= 18 AND CAST(data->>'age' AS numeric) <= 60"), # 20

        # ("(status == 1) or (category == 'VIP')", "CAST(data->>'status' AS numeric) = 1 OR data->>'category' = 'VIP'"),
        # "(CAST(data->>'status' AS numeric) = 1) OR (data->>'category' = 'VIP')"
        ("(status == 1) or (category == 'VIP')", "(CAST(data->>'status' AS numeric) = 1) OR (data->>'category' = 'VIP')"),
        ("(score > 90) and (active == true) and (name like '%John%')",
         # "CAST(data->>'score' AS numeric) > 90 AND (data->>'active')::boolean = true AND data->>'name' LIKE '%John%'"),
         "(CAST(data->>'score' AS numeric) > 90) AND ((data->>'active')::boolean = true) AND (data->>'name' LIKE '%%John%%')"),

        # 嵌套括号
        ("((age > 18) and (active == true)) or (score > 90)",
         "((CAST(data->>'age' AS numeric) > 18) AND ((data->>'active')::boolean = true)) OR (CAST(data->>'score' AS numeric) > 90)"),

        # 转义测试
        ("name == \"O''Connor\"", "data->>'name' = 'O''Connor'"),
        ("desc like 'test%data'", "data->>'desc' LIKE 'test%%data'"),
        ("value == \"quote test\"", "data->>'value' = 'quote test'"),

        # 边界情况
        ("id == -1", "CAST(data->>'id' AS numeric) = -1"),
        ("price > -10.5", "CAST(data->>'price' AS numeric) > -10.5"),
        ("empty == ''", "data->>'empty' = ''"),

        # 复杂逻辑测试
        ("a == 1 and b == 2 or c == 3",
         "CAST(data->>'a' AS numeric) = 1 AND CAST(data->>'b' AS numeric) = 2 OR CAST(data->>'c' AS numeric) = 3"),
        ("(a == 1 or b == 2) and c == 3",
         "(CAST(data->>'a' AS numeric) = 1 OR CAST(data->>'b' AS numeric) = 2) AND CAST(data->>'c' AS numeric) = 3"),

        # 布尔IN/NOT IN
        ("status in [true]", "(data->>'status')::boolean IN (true)"),
        ("active in [false, true]", "(data->>'active')::boolean IN (false, true)"),
        ("enabled not in [false]", "(data->>'enabled')::boolean NOT IN (false)"),

        # 额外测试
        ("x > 1 and y < 2 or z == 3",
         "CAST(data->>'x' AS numeric) > 1 AND CAST(data->>'y' AS numeric) < 2 OR CAST(data->>'z' AS numeric) = 3"),
        ("(p == 1 and q == 2) or (r == 3 and s == 4)",
         "(CAST(data->>'p' AS numeric) = 1 AND CAST(data->>'q' AS numeric) = 2) OR (CAST(data->>'r' AS numeric) = 3 AND CAST(data->>'s' AS numeric) = 4)"),
    ]

    print("测试 Milvus Filter 转换:")
    print("=" * 80)

    all_passed = True
    failed_cases = []

    for i, (filter_expr, expected) in enumerate(test_cases, 1):
        try:
            where_clause = converter._parse_logical_expression_full(filter_expr)
            passed = (where_clause == expected)
            all_passed = all_passed and passed

            if not passed:
                failed_cases.append((i, filter_expr, expected, where_clause))

            status = '✓ 通过' if passed else '✗ 失败'
            print(f"测试 {i:2d}: {filter_expr}")
            print(f"预期: {expected}")
            print(f"实际: {where_clause}")
            print(f"状态: {status}")
            if not passed:
                raise Exception("test error")

        except Exception as e:
            print(f"测试 {i} 出错: {filter_expr}")
            print(f"错误: {e}")
            raise
            all_passed = False
            failed_cases.append((i, filter_expr, expected, f"错误: {e}"))

    print("\n测试 process_filter 方法:")
    print("=" * 80)

    process_test_cases = [
        ("id == 1", "SELECT * FROM users", "SELECT * FROM users WHERE CAST(data->>'id' AS numeric) = 1"),
        ("age > 18", "SELECT * FROM users WHERE name = 'John'",
         "SELECT * FROM users WHERE (name = 'John') AND (CAST(data->>'age' AS numeric) > 18)"),
        ("active == true", "SELECT * FROM users ORDER BY id",
         "SELECT * FROM users WHERE (data->>'active')::boolean = true ORDER BY id"),
        ("flags in [true, false]", "SELECT * FROM config",
         "SELECT * FROM config WHERE (data->>'flags')::boolean IN (true, false)"),
        ("(age > 18) and (active == true)", "SELECT * FROM users",
         "SELECT * FROM users WHERE (CAST(data->>'age' AS numeric) > 18) AND ((data->>'active')::boolean = true)"),
        ("age >= 18 and age <= 60", "SELECT * FROM users",
         "SELECT * FROM users WHERE CAST(data->>'age' AS numeric) >= 18 AND CAST(data->>'age' AS numeric) <= 60"),
        ("a == 1 and b == 2 or c == 3", "SELECT * FROM table",
         "SELECT * FROM table WHERE CAST(data->>'a' AS numeric) = 1 AND CAST(data->>'b' AS numeric) = 2 OR CAST(data->>'c' AS numeric) = 3"),
    ]

    for i, (filter_expr, query, expected) in enumerate(process_test_cases, 1):
        try:
            result = converter.process_filter(filter_expr, query)
            result_clean = ' '.join(result.split())
            expected_clean = ' '.join(expected.split())
            passed = (result_clean == expected_clean)

            if not passed:
                failed_cases.append((i, f"process_filter: {filter_expr}", expected, result))

            status = '✓ 通过' if passed else '✗ 失败'
            print(f"测试 {i}: filter={filter_expr}")
            print(f"预期: {expected}")
            print(f"实际: {result}")
            print(f"状态: {status}")

        except Exception as e:
            print(f"测试 {i} 出错: {filter_expr}")
            print(f"错误: {e}")
            failed_cases.append((i, f"process_filter: {filter_expr}", expected, f"错误: {e}"))

    if failed_cases:
        print("\n失败的测试用例:")
        for i, expr, expected, actual in failed_cases:
            print(f"\n测试 {i}: {expr}")
            print(f"预期: {expected}")
            print(f"实际: {actual}")

    print(f"\n总体结果: {'全部通过 ✓' if all_passed and not failed_cases else f'有 {len(failed_cases)} 个失败用例'}")
    return all_passed and not failed_cases


if __name__ == "__main__":
    # test_expression_validation()
    test_milvus_filter_to_sql()
