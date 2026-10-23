"""Parameterize the COMPLETE budget audit in the supplied native COST11 adapter.

Authenticated source: UPLOAD_B124_NATIVE_DEBUG_20261002_232505_043618.zip.
Only reviewed B-specific AST expressions in reset()/step(), and the zero-budget
branch of observed_budget(), are changed in memory. The original bounded core
still owns all mask, observation, transition, reward and physical state updates.
No source snapshot is edited. Six map channels are unrelated and untouched.
"""
from __future__ import annotations
import ast
import hashlib
from copy import deepcopy
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

PATCH_VERSION = 'b124_native_complete_budget_audit_v3_20261002'
NATIVE_SHA256 = '807817035c9d5c3da4c36caf2b56a52128f68f7eb50b2787873ac0215a4960b0'
_MARKER = '_b124_budget_guard_patch_record'


def _require(ok, message):
    if not ok:
        raise RuntimeError('B124 native budget audit: ' + message)


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _digest(node):
    return hashlib.sha256(_dump(node).encode()).hexdigest()


def _node(text):
    t = ast.parse(text)
    n = t.body[0]
    return n.value if isinstance(n, ast.Expr) else n


class _ExactReplacement(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old, self.new = _dump(_node(old)), _node(new)
        self.count = 0

    def visit(self, node):
        if _dump(node) == self.old:
            self.count += 1
            return ast.copy_location(deepcopy(self.new), node)
        return super().visit(node)


def _replace(tree, old, new, expected, audit):
    change = _ExactReplacement(old, new)
    result = change.visit(tree)
    _require(change.count == expected,
             f'authenticated AST expression count changed: {old!r}; '
             f'expected={expected}, got={change.count}')
    audit.append(dict(before=old, after=new, count=change.count))
    return result


def parameterize_source(source: str, budget: int, filename='<native>'):
    """Prepare exact reviewed reset/step/helper nodes; does not execute anything."""
    _require(type(budget) is int and budget in (0, 1, 2, 4, 6), 'unsupported B')
    tree = ast.parse(source, filename=filename)
    mixins = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'C3NoClockMixin']
    _require(len(mixins) == 1, 'C3NoClockMixin missing/duplicated')
    mixin = mixins[0]
    nodes = {n.name: n for n in mixin.body if isinstance(n, ast.FunctionDef)}
    _require({'reset', 'step'} <= set(nodes), 'reset/step missing')
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observed_budget']
    _require(len(funcs) == 1, 'observed_budget missing/duplicated')
    original = {k: deepcopy(nodes[k]) for k in ('reset', 'step')}
    original['observed_budget'] = deepcopy(funcs[0])
    patched = deepcopy(original)
    changes = []
    B = str(budget)
    r = patched['reset']
    r = _replace(r, 'np.zeros(7, np.int64)', f'np.zeros({budget+1}, np.int64)', 1, changes)
    r = _replace(r,
        "require(getattr(self.config, 'detour_budget_max', None) == 6, 'B must remain 6.')",
        f"require(getattr(self.config, 'detour_budget_max', None) == {B}, 'Assigned B must remain {B}.')",
        1, changes)
    r = _replace(r,
        "require(observed_budget(obs, 6) == 6, 'B6 reset did not expose a full budget.')",
        f"require(observed_budget(obs, {B}) == {B}, 'B{B} reset did not expose its full budget.')",
        1, changes)
    patched['reset'] = r
    s = patched['step']
    s = _replace(s, 'observed_budget(pre_obs, 6)', f'observed_budget(pre_obs, {B})', 1, changes)
    s = _replace(s, '6 if dn else budget_before if is_strict else budget_before - 1',
                    f'{B} if dn else budget_before if is_strict else budget_before - 1', 1, changes)
    s = _replace(s, 'observed_budget(self.observation(), 6)',
                    f'observed_budget(self.observation(), {B})', 2, changes)
    patched['step'] = s
    # Multiplying an incorrect normalized scalar by zero would hide it. Keep the
    # original positive-B helper and explicitly validate the B=0 convention.
    helper = patched['observed_budget']
    points = [i for i,n in enumerate(helper.body)
              if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='b' for t in n.targets)]
    _require(len(points)==1, 'native observed-budget multiplication changed')
    branch = ast.parse("if bmax == 0:\n    require(float(a[0]) == 0.0, 'B=0 budget observation must be zero.')\n    return 0\n").body[0]
    helper.body.insert(points[0], branch)
    changes.append(dict(before='b = float(a[0]) * bmax',
                        after='explicit zero-scalar validation when bmax == 0; original multiplication otherwise',count=1))
    for n in patched.values():
        ast.fix_missing_locations(n)
    return original, patched, changes


def _compile(node, filename, is_method):
    if is_method:
        holder = ast.ClassDef(name='_B124NativeCompileHolder', bases=[], keywords=[],
                             body=[deepcopy(node)], decorator_list=[])
        tree = ast.Module(body=[holder], type_ignores=[])
    else:
        tree = ast.Module(body=[deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(tree)
    code = compile(tree, filename, 'exec', dont_inherit=True, optimize=0)
    if is_method:
        holders = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name==holder.name]
        _require(len(holders)==1,'method holder compilation failed')
        code = holders[0]
    found = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name==node.name]
    _require(len(found)==1, 'function compilation failed: '+node.name)
    return found[0]


def install_budget_guard(core: ModuleType, budget: int, expected_sha256: str) -> dict:
    """Specialize all B audit sites before constructing any environment.

    One B per worker, idempotent for repeated installation of that same B.
    Refuses an unreviewed source revision; no fuzzy/global numeric replacement.
    """
    _require(type(budget) is int and budget in (0, 1, 2, 4, 6), 'unsupported B')
    path = Path(core.__file__).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == expected_sha256 == NATIVE_SHA256,
             f'unreviewed native core hash: {actual}; expected uploaded {NATIVE_SHA256}')
    previous = getattr(core, _MARKER, None)
    if previous is not None:
        _require(previous['assigned_B']==budget and previous['patch_version']==PATCH_VERSION,
                 'different B/patch already installed in this process')
        return deepcopy(previous)
    original, patched, changes = parameterize_source(raw.decode('utf-8-sig'),budget,str(path))
    cls = core.C3NoClockMixin
    edits = []
    record_methods = {}
    for name in ('reset','step','observed_budget'):
        method = name != 'observed_budget'
        func = cls.__dict__.get(name) if method else getattr(core,name)
        _require(isinstance(func,FunctionType) and func.__globals__ is vars(core),
                 'function comes from unexpected owner: '+name)
        code = _compile(patched[name],str(path)+f' [B124 complete budget B={budget}]',method)
        old = func.__code__
        _require(old.co_freevars==code.co_freevars,'closure changed: '+name)
        _require((old.co_argcount,old.co_posonlyargcount,old.co_kwonlyargcount,old.co_flags & 0x0c)==
                 (code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags & 0x0c),
                 'signature changed: '+name)
        edits.append((func,code))
        record_methods[name]=dict(original_ast_sha256=_digest(original[name]),
                                  patched_ast_sha256=_digest(patched[name]))
    # Only install once all functions have passed compilation/signature checks.
    for func, code in edits:
        func.__code__=code
    record = dict(patch_version=PATCH_VERSION, assigned_B=budget, original_expected_B=6,
                  guard_helper='require', native_source_file=str(path), native_source_sha256=actual,
                  snapshot_file_modified=False, original_super_closure_preserved=True,
                  function_identity_preserved=True, parameterized_budget_sites=7,
                  zero_budget_scalar_guard=True, methods=record_methods, changes=changes,
                  scope='reset/step budget audits, B+1 histogram, zero-budget helper; native masks/reward/physics unchanged')
    setattr(core,_MARKER,deepcopy(record))
    return record
