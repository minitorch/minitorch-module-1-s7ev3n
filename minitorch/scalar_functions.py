from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x
'''
为什么ScalarFunctin不采用抽象接口继承的形式？

ABC抽象类方案：
```
from abc import ABC, abstractmethod

class ScalarFunction(ABC):
    # apply 是一个实例方法
    def apply(self, *vals):
        # ... 通用逻辑 ...
        c = self.forward(ctx, *raw_vals)  # 调用实例的 forward
        # ...
        back = ScalarHistory(self.__class__, ctx, scalars)
        return Scalar(c, back)

    @abstractmethod
    def forward(self, ctx, *args):
        pass

    @abstractmethod
    def backward(self, ctx, d_output):
        pass

class Mul(ScalarFunction):
    # 实现抽象方法
    def forward(self, ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    def backward(self, ctx, d_output):
        a, b = ctx.saved_values
        return d_output * b, d_output * a

# 使用方式
x = Scalar(3)
y = Scalar(4)
z = Mul().apply(x, y)  # 关键：需要创建实例 Mul()
```

下表中的做了不错的总结，我觉得无需实例化和无状态最能体现选择现在设计，因为：
1. 性能原因。计算图不可能为每个操作创建实例，当前设计是零开销
2. 无状态。带着状态可能会产生难以调试的bug，所以状态的管理进行了解耦，全部放在Context类里面
3. 函数式编程。ScalarFunction包装的本来就是函数，更贴近函数式编程思想：
- 函数是核心：forward 和 backward 是核心。
- 数据和行为分离：数据（Scalar）和行为（ScalarFunction）是分开的。
- 无副作用：forward 和 backward 除了修改 Context 外，没有其他副作用。

| 特性 | 设计 A (ABC 继承) | 设计 B (当前设计) | 为什么当前设计更好？ |
| :--- | :--- | :--- | :--- |
| **1.| 特性 | 设计 A (ABC 继承) | 设计 B (当前设计) | 为什么当前设计更好？ |
| :--- | :--- | :--- | :--- |
| **1. 实例化** | **需要创建实例** () | **不需要创建实例** () | **性能和内存**：在大型计算图中，每一步都创建对象 (, , ...) 会产生巨大的内存开销和垃圾回收压力。当前设计完全避免了这个问题。 |
| **2. 状态** | **有状态 (Stateful)** | **无状态 (Stateless)** | **简洁和可预测性**： 实例可以有自己的属性 ()，这使得函数行为可能依赖于实例状态，难以预测。当前设计中， 只是一个函数的容器，所有临时状态都保存在  中，更清晰、更安全。 |
| **3. 概念模型** | **"是一个" (is-a)** | **"分组" (grouping)** | **更符合函数概念**： 本身不是一个"东西"，它代表的是"乘法"这个**动作**。将  和  两个相关的**函数**分组到一个类中，比创建一个"乘法对象"更符合其数学本质。 |
| **4. 使用方式** |  |  | **更像函数调用**： 看起来更像 ，符合"调用一个函数"的直觉。 则更像"调用一个对象的方法"。 |
| **5. 接口强制** | **强制** (不实现会报错) | **约定** (不实现会报错，但更晚) | ABC 方案在加载时就能发现未实现接口，更严格。但当前设计在实际调用时也会报错，对于这个项目足够了。 |


'''

class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        # _backward(下面的_forward同理)，提供了扩展点
        # 可以在这里加入额外的逻辑，例如
        # - 输入验证
        # - 日志记录
        # - 性能监控
        # - 类型转换
        # - 输出验证
        # - 异常处理
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a, = ctx.saved_values
        return -d_output / (a * a)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a, = ctx.saved_values
        sig = operators.sigmoid(a)
        return d_output * sig * (1 - sig)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a, = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a, = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0
