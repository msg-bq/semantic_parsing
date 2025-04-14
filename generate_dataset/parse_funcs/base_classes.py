from __future__ import annotations
from ._co_namespace import Declared_Operators


class DuplicateError(Exception):
    def __init__(self, error_info):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class BaseIndividual(object):
    def __init__(self, value: str):
        """
        :param value: 语义解析时我们认为仅有str。正常情况下，断言逻辑的individual应当归属于某个concept，自身可以是任意格式
        """
        self.value = value

    def __eq__(self, other):
        return type(self) is type(other) and self.GetHash() == other.GetHash()

    def GetHash(self):
        return self.value

    def get_des(self):
        return None

    def __hash__(self):
        return self.GetHash().__hash__()

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"BaseIndividual(value={self.value})"


class BaseOperator(object):
    def __init__(self, name: str, input_type: list[str], output_type: str, description: str):
        if name in Declared_Operators:
            raise DuplicateError("此Operator已声明")
        Declared_Operators[name] = True

        self.inputType = input_type
        self.outputType = output_type
        self.name = name
        self.description = description

    def GetHash(self):
        return self.name  # 因为名称唯一，按理来说这就够用了

    def get_des(self):
        if self.description == "":
            return self.description
        return f"{self.name}: {self.description}"

    def __hash__(self):
        return self.GetHash().__hash__()

    def __eq__(self, other):
        return type(self) is type(other) and self.GetHash() == other.GetHash()

    def __str__(self):
        return f"{self.name}({self.inputType}={self.outputType})"

    def __repr__(self):
        return (f"BaseOperator(name={self.name}, input_type={self.inputType}, "
                f"output_type={self.outputType})")


class Term(object):
    """
    我们在这里约定，仅认为op(c1,c2...)是term，而不遵循常规的term定义。因为目前没啥必要，而且影响if语句的简洁
    """
    def __init__(self, operator: BaseOperator, variables: list[BaseIndividual | Term]):
        self.operator = operator
        self.variables = variables
        assert len(variables) == len(operator.inputType), \
            f"variables {variables} do not match inputType {operator.inputType} of operator {operator.name}"

    def GetHash(self):
        var_dict = {}
        for var, var_name in zip(self.variables, self.operator.inputType):
            var_dict[var_name] = var.GetHash()
            var_dict['operator'] = self.operator.GetHash()
        return tuple(var_dict.items())

    def get_des(self):
        des_lst = []
        for variable in self.variables:
            if variable.get_des() != None:
                des_lst.append(variable.get_des())
        variables_des = '\n'.join(des_lst)

        term_des = self.operator.get_des()

        if variables_des != "":
            return f"{self.operator.get_des()}\n{variables_des}"
        else:
            return term_des

    def __getattribute__(self, item):
        return super(Term, self).__getattribute__(item)

    def __hash__(self):
        return self.GetHash().__hash__()

    def __eq__(self, other):
        return type(self) is type(other) and self.GetHash() == other.GetHash()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __str__(self):
        variables_str = ", ".join(map(str, self.variables))
        return f"{self.operator.name}({variables_str})"

    def __repr__(self):
        variables_str = ", ".join(map(str, self.variables))
        return f"Term(operator={self.operator}, variables=[{variables_str}])"


class Assertion:
    def __init__(self, lhs: Term, rhs: Term):
        self.LHS = lhs
        self.RHS = rhs

    def GetHash(self):
        var_dict = {'LHS': self.LHS.GetHash(),
                    'RHS': self.RHS.GetHash()}

        return tuple(var_dict.items())

    def get_des(self):
        return f"{self.LHS.get_des()}\n{self.RHS.get_des()})"

    def __eq__(self, other):
        return type(self) is type(other) and self.GetHash() == other.GetHash()

    def __hash__(self):
        return self.GetHash().__hash__()

    def __str__(self):
        return f"{self.LHS} = {self.RHS}"

    def __repr__(self):
        return f"Assertion(LHS={self.LHS}, RHS={self.RHS})"


class Formula:
    def __init__(self, formula_left: Assertion | Formula,
                 connective: str,
                 formula_right: Assertion | Formula | None):
        self.formula_left = formula_left
        self.connective = connective
        self.formula_right = formula_right

    def GetHash(self):
        left_hash = self.formula_left.GetHash() if isinstance(self.formula_left, Formula) else self.formula_left
        if self.formula_right != None:
            right_hash = self.formula_right.GetHash() if isinstance(self.formula_right, Formula) else self.formula_right
        else:
            right_hash = ""
        connective_hash = hash(self.connective)
        return left_hash, connective_hash, right_hash

    def get_des(self):
        des = self.formula_left.get_des()
        if self.formula_right != None:
            des += "\n"
            des += self.formula_right.get_des()
        return des

    def __hash__(self):
        return self.GetHash().__hash__()

    def __eq__(self, other):
        return type(self) is type(other) and self.GetHash() == other.GetHash()

    def __getattribute__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return super(Formula, self).__getattribute__(attr)

    def __str__(self):
        return f"{self.formula_left} {self.connective} {self.formula_right}"

    def __repr__(self):
        return (f"Formula(formula_left={self.formula_left}, connective={self.connective}, "
                f"formula_right={self.formula_right})")
