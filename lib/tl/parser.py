"""Defines a simplified temporal logic with the following operators:
neg (~), or (|), and (&), implies (->)
next (X), finally (F), globally (G), until (U), weak until (W), (, )
Parsimonious is used as the backbone grammar parser.
"""

import operator as op
from functools import partialmethod, reduce

from parsimonious import Grammar, NodeVisitor
from . import syntax as ast
from . import sugar

TL_GRAMMAR = Grammar(r'''
phi = (neg / paren_phi / implies_outer / iff_outer / and_outer / or_outer / next / weak_until / until / release / since / globally / finally / AP / bot / top)

neg = "~" __ phi
paren_phi = "(" __ phi __ ")"

and_outer = "(" __ and_inner __ ")"
and_inner = (phi __ "&" __ and_inner) / phi

or_outer = "(" __ or_inner __ ")"
or_inner = (phi __ "|" __ or_inner) / phi

implies_outer = "(" __ implies_inner __ ")"
implies_inner = (phi __ "->" __ implies_inner) / phi

iff_outer = "(" __ iff_inner __ ")"
iff_inner = (phi __ "<->" __ iff_inner) / phi

next = "X" __ phi
weak_until = "(" __ phi _ "W" _ phi __ ")"
until = "(" __ phi _ "U" _ phi __ ")"
release = "(" __ phi _ "R" _ phi __ ")"
since = "(" __ phi _ "S" _ phi __ ")"
globally = "G" __ phi
finally = "F" __ phi

bot = "FALSE"
top = "TRUE"

AP = ~r"[a-z_][a-z_\d]*"

_ = ~r"\s"+
__ = ~r"\s"*
EOL = "\n"
''')

class TLVisitor(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.default_interval = ast.Interval(0.0, float('inf'))

    def binop_inner(self, _, children):
        if not isinstance(children[0], list):
            return children

        ((left, _, _, _, right),) = children
        return [left] + right

    def binop_outer(self, _, children, *, binop):
        return reduce(binop, children[2])

    def visit_const_or_unbound(self, node, children):
        child = children[0]
        return ast.Param(child) if isinstance(child, str) else float(node.text)
    
    visit_and_inner = binop_inner
    visit_iff_inner = binop_inner
    visit_implies_inner = binop_inner
    visit_or_inner = binop_inner
    visit_xor_inner = binop_inner

    visit_and_outer = partialmethod(binop_outer, binop=op.and_)
    visit_iff_outer = partialmethod(binop_outer, binop=sugar.iff)
    visit_implies_outer = partialmethod(binop_outer, binop=sugar.implies)
    visit_or_outer = partialmethod(binop_outer, binop=op.or_)

    def generic_visit(self, _, children):
        return children

    def children_getter(self, _, children, i):
        return children[i]

    visit_phi = partialmethod(children_getter, i=0)
    visit_paren_phi = partialmethod(children_getter, i=2)

    def visit_bot(self, *_):
        return ast.BOT

    def visit_top(self, *_):
        return ~ast.BOT

    def get_text(self, node, _):
        return node.text

    visit_op = get_text

    def unary_temp_op_visitor(self, _, children, op):
        _, _, phi = children
        lo, hi = self.default_interval
        return op(phi, lo=lo, hi=hi)

    visit_finally = partialmethod(unary_temp_op_visitor, op=sugar.env)
    visit_globally = partialmethod(unary_temp_op_visitor, op=sugar.alw)

    def visit_weak_until(self, _, children):
        _, _, phi1, _, _, _, phi2, _, _ = children
        return phi1.weak_until(phi2)

    def visit_until(self, _, children):
        _, _, phi1, _, _, _, phi2, _, _ = children
        return phi1.until(phi2)
    
    def visit_since(self, _, children):
        _, _, phi1, _, _, _, phi2, _, _ = children
        return phi1.since(phi2)

    def visit_release(self, _, children):
        _, _, phi1, _, _, _, phi2, _, _ = children
        return phi1.release(phi2)

    def visit_id(self, name, _):
        return name.text

    def visit_AP(self, *args):
        return ast.AtomicPred(self.visit_id(*args))

    def visit_neg(self, _, children):
        return ~children[2]

    def visit_next(self, _, children):
        return ast.Next(children[2])


def parse(input_expr):
    return TLVisitor().visit(TL_GRAMMAR.parse(input_expr))
