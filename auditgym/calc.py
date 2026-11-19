import ast
from decimal import Decimal, DecimalException, localcontext

MAX_LENGTH = 300
MAX_NODES = 80

_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_UNARY = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}


class CalcError(ValueError):
    pass


def evaluate(expression):
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    if len(expression) > MAX_LENGTH:
        raise CalcError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalcError("expression is not valid arithmetic") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError("expression is too complex")
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.Emax = 60
        ctx.Emin = -60
        try:
            value = +_eval(tree.body, expression)
            return format(value.normalize(), "f") if value != 0 else "0"
        except (DecimalException, ZeroDivisionError, ArithmeticError):
            raise CalcError("arithmetic error") from None


def _eval(node, source):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        text = ast.get_source_segment(source, node)
        try:
            return Decimal(text)
        except DecimalException:
            raise CalcError("bad number") from None
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left, source), _eval(node.right, source))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, source))
    raise CalcError("only numbers, + - * / and parentheses are allowed")
