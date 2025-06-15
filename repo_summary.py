import os
import ast
import sys
from datetime import datetime

# Directories to skip during walk
EXCLUDE_DIRS = {'.git', 'venv', '__pycache__'}

symbol_index = {}


def is_excluded(path):
    parts = path.split(os.sep)
    return any(part in EXCLUDE_DIRS for part in parts)


def get_expr(expr):
    try:
        return ast.unparse(expr)
    except Exception:
        return '...'


def format_signature(args):
    params = []
    total = len(args.args)
    default_offset = total - len(args.defaults)
    for i, arg in enumerate(args.args):
        if i >= default_offset:
            default = args.defaults[i - default_offset]
            params.append(f"{arg.arg}={get_expr(default)}")
        else:
            params.append(arg.arg)
    if args.vararg:
        params.append('*' + args.vararg.arg)
    for kw, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            params.append(kw.arg)
        else:
            params.append(f"{kw.arg}={get_expr(default)}")
    if args.kwarg:
        params.append('**' + args.kwarg.arg)
    return '(' + ', '.join(params) + ')'


def parse_file(path):
    result = {'imports': [], 'classes': [], 'functions': [], 'constants': []}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=path)
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
        return result

    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                asname = f" as {n.asname}" if n.asname else ''
                result['imports'].append(f"import {n.name}{asname}")
        elif isinstance(node, ast.ImportFrom):
            module = '.' * node.level + (node.module or '')
            names = ', '.join(a.name + (f" as {a.asname}" if a.asname else '') for a in node.names)
            result['imports'].append(f"from {module} import {names}")
        elif isinstance(node, ast.ClassDef):
            bases = [get_expr(b) for b in node.bases]
            doc = ast.get_docstring(node)
            doc = doc.splitlines()[0].strip() if doc else ''
            result['classes'].append({'name': node.name, 'bases': bases, 'doc': doc})
            symbol_index[node.name] = (path, node.lineno)
        elif isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            doc = doc.splitlines()[0].strip() if doc else ''
            sig = format_signature(node.args)
            result['functions'].append({'name': node.name, 'signature': sig, 'doc': doc})
            symbol_index[node.name] = (path, node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    val = get_expr(node.value)
                    if len(val) > 40:
                        val = val[:37] + '...'
                    result['constants'].append(f"{t.id} = {val}")
                    symbol_index[t.id] = (path, node.lineno)
    return result


def walk_repo(root):
    summary = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        if is_excluded(dirpath):
            continue
        for file in files:
            if file.endswith('.py'):
                full = os.path.join(dirpath, file)
                rel = os.path.relpath(full, root)
                summary[rel] = parse_file(full)
    return summary


def write_summary(summary, out_path):
    timestamp = datetime.now().isoformat()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Codebase Summary\n\nGenerated {timestamp}\n\n")
        for path in sorted(summary):
            data = summary[path]
            f.write(f"## {path}\n\n")
            if data['imports']:
                f.write("* **Imports:**\n")
                for imp in data['imports']:
                    f.write(f"  - {imp}\n")
            if data['classes']:
                f.write("* **Classes:**\n")
                for c in data['classes']:
                    bases = f"({', '.join(c['bases'])})" if c['bases'] else ''
                    desc = f" - {c['doc']}" if c['doc'] else ''
                    f.write(f"  - {c['name']}{bases}{desc}\n")
            if data['functions']:
                f.write("* **Functions:**\n")
                for fn in data['functions']:
                    desc = f" - {fn['doc']}" if fn['doc'] else ''
                    f.write(f"  - {fn['name']}{fn['signature']}{desc}\n")
            if data['constants']:
                f.write("* **Globals:**\n")
                for const in data['constants']:
                    f.write(f"  - {const}\n")
            f.write("\n")


def locate_symbol(name):
    """Return (filepath, line) of symbol definition if known."""
    return symbol_index.get(name)


if __name__ == '__main__':
    summary = walk_repo(os.getcwd())
    write_summary(summary, 'CODEX_SUMMARY.md')
    print('Summary written to CODEX_SUMMARY.md')

# Usage instructions:
#   Run this script with `python3 repo_summary.py` to generate CODEX_SUMMARY.md.
#   Before requesting code modifications from Codex, load or refer to this
#   summary so the assistant understands the existing project structure.
#   Use locate_symbol('MyClass') to find where a symbol is defined.
