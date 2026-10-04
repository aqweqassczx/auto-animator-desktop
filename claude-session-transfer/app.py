#!/usr/bin/env python3
"""Claude Session Transfer — перенос сессий Claude Desktop (вкладка Code) между аккаунтами.

Claude Desktop хранит список сессий отдельно для каждого аккаунта:
    %APPDATA%\\Claude\\claude-code-sessions\\<account>\\<org>\\*.json
Сама переписка лежит в ~/.claude/projects/<проект>/<uuid>.jsonl и общая для всех аккаунтов.
Утилита копирует только метаданные сессий из папки одного аккаунта в папку другого,
поэтому в сайдбаре появляются ровно те сессии, которые были видны раньше.

Только стандартная библиотека Python. Запуск: python app.py  ->  http://127.0.0.1:8765
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
SNAP_DIR = APP_DIR / "snapshots"
BACKUP_DIR = APP_DIR / "backups"
SETTINGS_FILE = APP_DIR / "settings.json"
HOME = Path(os.environ.get("CST_HOME") or Path.home())
IS_WIN = os.name == "nt"
SESSION_KINDS = ("claude-code-sessions",)
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

TITLE_KEYS = ("title", "customTitle", "name", "summary")
CWD_KEYS = ("cwd", "originCwd", "workingDirectory", "projectPath", "folder", "worktreePath")
TIME_KEYS = ("lastActivityAt", "updatedAt", "lastUpdated", "lastModified", "modifiedAt", "createdAt")
ARCH_KEYS = ("isArchived", "archived")

# Пути процессов Claude Desktop (Squirrel-установка и MSIX из Microsoft Store)
DESKTOP_MAIN_MARKERS = ("\\anthropicclaude\\", "\\windowsapps\\claude", "\\packages\\claude_")
DESKTOP_CHILD_MARKERS = ("\\roaming\\claude\\",)  # встроенный claude-code внутри Desktop
NO_WINDOW = 0x08000000 if IS_WIN else 0

_lock = threading.Lock()


# ---------------------------------------------------------------- settings

def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text("utf-8"))
    except Exception:
        return {}


def save_settings(data):
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------- discovery

def claude_roots():
    if os.environ.get("CST_CLAUDE_DIR"):
        return [Path(os.environ["CST_CLAUDE_DIR"])]
    cands = []
    if os.environ.get("APPDATA"):
        cands.append(Path(os.environ["APPDATA"]) / "Claude")
    if os.environ.get("LOCALAPPDATA"):
        pk = Path(os.environ["LOCALAPPDATA"]) / "Packages"
        if pk.is_dir():
            cands += [p / "LocalCache" / "Roaming" / "Claude" for p in pk.glob("Claude_*")]
    cands.append(HOME / "Library" / "Application Support" / "Claude")
    cands.append(HOME / ".config" / "Claude")
    out, seen = [], set()
    for p in cands:
        key = str(p).lower()
        if p.is_dir() and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def claude_config_dir():
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or HOME / ".claude")


def oauth_account():
    for p in (HOME / ".claude.json", claude_config_dir() / ".claude.json"):
        try:
            acc = json.loads(p.read_text("utf-8")).get("oauthAccount") or {}
        except Exception:
            continue
        if acc.get("accountUuid"):
            return {
                "account": acc.get("accountUuid"),
                "org": acc.get("organizationUuid"),
                "email": acc.get("emailAddress"),
                "source": str(p),
            }
    return None


def store_dirs(base):
    """Папки внутри claude-code-sessions, где лежат json сессий (обычно <account>/<org>)."""
    out = []

    def walk(d, depth):
        has_json = any(f.is_file() for f in d.glob("*.json"))
        rel_len = len(d.relative_to(base).parts)
        if has_json or rel_len == 2:
            out.append(d)
            if has_json:
                return  # внутрь папки с сессиями не идём
        if depth < 3:
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    walk(sub, depth + 1)

    walk(base, 0)
    return out


def describe_store(root, kind, d):
    base = root / kind
    parts = d.relative_to(base).parts
    files = [f for f in d.glob("*.json") if f.is_file()]
    mtime = max([f.stat().st_mtime for f in files] or [d.stat().st_mtime])
    aliases = load_settings().get("aliases", {})
    return {
        "id": str(d),
        "path": str(d),
        "root": str(root),
        "kind": kind,
        "parts": list(parts),
        "account": parts[0] if len(parts) >= 1 else "",
        "org": parts[1] if len(parts) >= 2 else "",
        "count": len(files),
        "mtime": mtime,
        "alias": aliases.get(str(d), ""),
        "exists": True,
    }


def find_stores():
    stores = []
    for root in claude_roots():
        for kind in SESSION_KINDS:
            base = root / kind
            if base.is_dir():
                stores += [describe_store(root, kind, d) for d in store_dirs(base)]
    stores.sort(key=lambda s: -s["mtime"])
    acc = oauth_account()
    if acc:
        for s in stores:
            if s["account"] == acc["account"] and not s["alias"]:
                s["alias"] = acc.get("email") or ""
    return stores


def extra_targets(stores):
    """Папка аккаунта из ~/.claude.json, если её ещё нет (новый аккаунт без сессий)."""
    acc = oauth_account()
    roots = claude_roots()
    if not acc or not acc.get("org") or not roots:
        return []
    path = roots[0] / SESSION_KINDS[0] / acc["account"] / acc["org"]
    if any(Path(s["path"]) == path for s in stores):
        return []
    return [{
        "id": str(path), "path": str(path), "root": str(roots[0]), "kind": SESSION_KINDS[0],
        "parts": [acc["account"], acc["org"]], "account": acc["account"], "org": acc["org"],
        "count": 0, "mtime": 0, "alias": (acc.get("email") or "") + " (из ~/.claude.json)", "exists": False,
    }]


def is_allowed_target(path):
    p = Path(path).resolve()
    for root in claude_roots():
        for kind in SESSION_KINDS:
            base = (root / kind).resolve()
            if base in p.parents:
                return True
    return False


# ---------------------------------------------------------------- sessions

def transcript_index():
    idx = {}
    pdir = claude_config_dir() / "projects"
    if pdir.is_dir():
        for proj in pdir.iterdir():
            if not proj.is_dir():
                continue
            for f in proj.glob("*.jsonl"):
                try:
                    st = f.stat()
                except OSError:
                    continue
                idx[f.stem.lower()] = {"path": str(f), "size": st.st_size, "mtime": st.st_mtime,
                                       "project": proj.name, "stem": f.stem}
    return idx


def pick(data, keys, max_depth=3):
    if not isinstance(data, (dict, list)):
        return None
    for key in keys:
        lk = key.lower()
        level = [data]
        for _ in range(max_depth):
            nxt = []
            for node in level:
                items = node.items() if isinstance(node, dict) else enumerate(node)
                for k, v in items:
                    if isinstance(k, str) and k.lower() == lk and v not in (None, "", [], {}) \
                            and not isinstance(v, (dict, list)):
                        return v
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
            level = nxt
    return None


def to_epoch(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return v / 1000 if v > 1e12 else (v if v > 1e8 else None)
    if isinstance(v, str):
        try:
            return to_epoch(float(v))
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def basename(p):
    p = str(p or "").rstrip("\\/")
    return re.split(r"[\\/]", p)[-1] if p else ""


def parse_session_text(name, raw, idx, fallback_time=0):
    try:
        data = json.loads(raw)
    except Exception:
        data = None
    cwd = pick(data, CWD_KEYS) or ""
    arch = pick(data, ARCH_KEYS)
    uuids = list(dict.fromkeys(u.lower() for u in UUID_RE.findall(raw)))
    tid = next((u for u in uuids if u in idx), None)
    tr = idx.get(tid) if tid else None
    return {
        "file": name,
        "title": str(pick(data, TITLE_KEYS) or Path(name).stem),
        "cwd": str(cwd),
        "project": basename(cwd) or "(без папки)",
        "time": to_epoch(pick(data, TIME_KEYS)) or fallback_time,
        "archived": bool(arch) if arch is not None else False,
        "transcript": tr,
        "transcript_id": tr["stem"] if tr else None,
        "uuids": uuids,
        "bad_json": data is None,
    }


def siblings(f):
    """Доп. файлы/папки сессии с тем же именем (например local_xxx/ рядом с local_xxx.json)."""
    return [p for p in f.parent.iterdir() if p != f and p.name.startswith(f.stem)
            and (len(p.name) == len(f.stem) or p.name[len(f.stem)] in "._-")]


def list_store_sessions(store_path):
    d = Path(store_path)
    idx = transcript_index()
    out = []
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            if f.is_file():
                raw = f.read_text("utf-8", errors="replace")
                out.append(parse_session_text(f.name, raw, idx, f.stat().st_mtime))
    out.sort(key=lambda s: -(s["time"] or 0))
    return out


# ---------------------------------------------------------------- snapshots

def safe_name(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Недопустимое имя файла: {name!r}")
    return name


def snapshot_path(name):
    p = SNAP_DIR / safe_name(name)
    if not p.is_file():
        raise ValueError(f"Снимок не найден: {name}")
    return p


def add_dir_to_zip(z, src, arc_prefix):
    for p in Path(src).rglob("*"):
        if p.is_file():
            z.write(p, f"{arc_prefix}/{p.relative_to(src).as_posix()}")


def make_snapshot(store_path, files, include_transcripts, note=""):
    store = Path(store_path)
    if not store.is_dir():
        raise ValueError("Папка источника не найдена")
    SNAP_DIR.mkdir(exist_ok=True)
    idx = transcript_index()
    info = next((s for s in find_stores() if Path(s["path"]) == store), None)
    name = f"snapshot_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"
    sessions = []
    with zipfile.ZipFile(SNAP_DIR / name, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            f = store / safe_name(fn)
            if not f.is_file():
                continue
            raw = f.read_text("utf-8", errors="replace")
            s = parse_session_text(f.name, raw, idx, f.stat().st_mtime)
            z.write(f, f"sessions/{fn}")
            for sib in siblings(f):
                if sib.is_dir():
                    add_dir_to_zip(z, sib, f"sessions/{sib.name}")
                else:
                    z.write(sib, f"sessions/{sib.name}")
            if include_transcripts and s["transcript"]:
                tp = Path(s["transcript"]["path"])
                proj = s["transcript"]["project"]
                z.write(tp, f"transcripts/{proj}/{tp.name}")
                sub = tp.with_suffix("")  # папка <uuid>/ с субагентами и tool-results
                if sub.is_dir():
                    add_dir_to_zip(z, sub, f"transcripts/{proj}/{sub.name}")
                s["in_snapshot"] = True
            sessions.append(s)
        manifest = {
            "format": 1,
            "created": time.time(),
            "note": note,
            "source": {k: info[k] for k in ("path", "root", "kind", "parts", "account", "org", "alias")} if info
            else {"path": str(store), "parts": [], "account": "", "org": ""},
            "include_transcripts": bool(include_transcripts),
            "sessions": sessions,
        }
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"name": name, "count": len(sessions)}


def read_manifest(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("manifest.json").decode("utf-8"))


def list_snapshots():
    out = []
    if SNAP_DIR.is_dir():
        for p in sorted(SNAP_DIR.glob("*.zip"), reverse=True):
            try:
                m = read_manifest(p)
            except Exception:
                continue
            out.append({"name": p.name, "created": m.get("created"), "count": len(m.get("sessions", [])),
                        "source": m.get("source", {}), "include_transcripts": m.get("include_transcripts"),
                        "size": p.stat().st_size, "note": m.get("note", "")})
    return out


def snapshot_sessions(name):
    idx = transcript_index()
    m = read_manifest(snapshot_path(name))
    out = []
    for s in m.get("sessions", []):
        tid = next((u for u in s.get("uuids", []) if u in idx), None)
        s["transcript"] = idx.get(tid) if tid else None
        s["transcript_id"] = s["transcript"]["stem"] if s["transcript"] else s.get("transcript_id")
        out.append(s)
    out.sort(key=lambda s: -(s.get("time") or 0))
    return out


# ---------------------------------------------------------------- restore

def backup_store(target):
    files = [p for p in target.rglob("*") if p.is_file()]
    if not files:
        return None
    BACKUP_DIR.mkdir(exist_ok=True)
    name = f"backup_{'_'.join(p[:8] for p in target.parts[-2:])}_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"
    with zipfile.ZipFile(BACKUP_DIR / name, "w", zipfile.ZIP_DEFLATED) as z:
        add_dir_to_zip(z, target, ".")
    return name


def id_pairs(src_parts, dst_parts):
    pairs = []
    for a, b in zip(src_parts, dst_parts):
        if a and b and a != b and UUID_RE.fullmatch(a) and UUID_RE.fullmatch(b):
            pairs.append((a, b))
    return pairs


def rewrite(data, pairs):
    if not pairs:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for a, b in pairs:
        text = text.replace(a, b)
    return text.encode("utf-8")


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".cst-tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def restore(source, files, target_path, overwrite=False, rewrite_ids=True, restore_transcripts=True, force=False):
    if not force and desktop_status()["running"]:
        raise ValueError("Claude Desktop запущен. Закрой его кнопкой «Закрыть Claude», потом переноси.")
    target = Path(target_path)
    if not is_allowed_target(target):
        raise ValueError("Цель должна быть внутри папки claude-code-sessions")
    target.mkdir(parents=True, exist_ok=True)
    dst_base = next(Path(r) / k for r in claude_roots() for k in SESSION_KINDS
                    if (Path(r) / k).resolve() in target.resolve().parents)
    dst_parts = list(target.resolve().relative_to(dst_base.resolve()).parts)

    report = {"copied": [], "skipped": [], "transcripts": [], "errors": [], "backup": backup_store(target)}
    files = [safe_name(f) for f in files]

    if source["type"] == "store":
        src = Path(source["id"])
        src_info = next((s for s in find_stores() if Path(s["path"]) == src), None)
        src_parts = src_info["parts"] if src_info else []
        if src.resolve() == target.resolve():
            raise ValueError("Источник и цель совпадают")
        pairs = id_pairs(src_parts, dst_parts) if rewrite_ids else []
        for fn in files:
            f = src / fn
            if not f.is_file():
                report["errors"].append(f"{fn}: нет файла в источнике")
                continue
            if (target / fn).exists() and not overwrite:
                report["skipped"].append(fn)
                continue
            write_atomic(target / fn, rewrite(f.read_bytes(), pairs))
            for sib in siblings(f):
                dst = target / sib.name
                if sib.is_dir():
                    shutil.copytree(sib, dst, dirs_exist_ok=True)
                else:
                    write_atomic(dst, rewrite(sib.read_bytes(), pairs))
            report["copied"].append(fn)
    else:
        zpath = snapshot_path(source["id"])
        m = read_manifest(zpath)
        pairs = id_pairs(m.get("source", {}).get("parts", []), dst_parts) if rewrite_ids else []
        wanted = set(files)
        tr_dir = claude_config_dir() / "projects"
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            for fn in files:
                arc = f"sessions/{fn}"
                if arc not in names:
                    report["errors"].append(f"{fn}: нет в снимке")
                    continue
                if (target / fn).exists() and not overwrite:
                    report["skipped"].append(fn)
                    continue
                write_atomic(target / fn, rewrite(z.read(arc), pairs))
                stem = Path(fn).stem
                for n in names:
                    rel = n[len("sessions/"):]
                    if n.startswith("sessions/") and rel != fn and rel.startswith(stem) and not n.endswith("/"):
                        write_atomic(safe_join(target, rel), rewrite(z.read(n), pairs))
                report["copied"].append(fn)
            if restore_transcripts:
                tids = {s.get("transcript_id") for s in m.get("sessions", []) if s.get("file") in wanted}
                tids.discard(None)
                for n in names:
                    if not n.startswith("transcripts/") or n.endswith("/"):
                        continue
                    rel = n[len("transcripts/"):]
                    parts = rel.split("/")
                    if len(parts) < 2 or Path(parts[1]).stem not in tids and parts[1] not in tids:
                        continue
                    dst = safe_join(tr_dir, rel)
                    if dst.exists():
                        continue
                    write_atomic(dst, z.read(n))
                    if dst.suffix == ".jsonl":
                        report["transcripts"].append(rel)
    return report


def safe_join(base, rel):
    dst = (Path(base) / rel).resolve()
    if Path(base).resolve() not in dst.parents:
        raise ValueError(f"Недопустимый путь в архиве: {rel}")
    return dst


# ---------------------------------------------------------------- Claude Desktop process

def powershell(cmd, timeout=20):
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                       capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                       encoding="utf-8", errors="replace")
    return r.stdout.strip()


def desktop_processes():
    if not IS_WIN:
        return []
    try:
        out = powershell("Get-Process -Name claude -ErrorAction SilentlyContinue | "
                         "Select-Object Id,Path | ConvertTo-Json -Compress")
    except Exception:
        return []
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    procs = data if isinstance(data, list) else [data]
    res = []
    for p in procs:
        path = (p.get("Path") or "")
        low = path.lower()
        main = any(m in low for m in DESKTOP_MAIN_MARKERS)
        child = any(m in low for m in DESKTOP_CHILD_MARKERS)
        if main or child or not path:
            res.append({"pid": p.get("Id"), "path": path, "main": main or not path})
    return res


def desktop_status():
    procs = desktop_processes()
    mains = [p for p in procs if p["main"]]
    exe = next((p["path"] for p in mains if p["path"]), None)
    if exe:
        s = load_settings()
        if s.get("desktop_exe") != exe:
            s["desktop_exe"] = exe
            save_settings(s)
    return {"running": bool(mains), "supported": IS_WIN, "pids": [p["pid"] for p in procs]}


def close_desktop():
    procs = desktop_processes()
    if not procs:
        return {"closed": 0}
    ids = ",".join(str(p["pid"]) for p in procs)
    powershell(f"Stop-Process -Id {ids} -Force -ErrorAction SilentlyContinue")
    for _ in range(20):
        if not desktop_status()["running"]:
            break
        time.sleep(0.5)
    return {"closed": len(procs), "running": desktop_status()["running"]}


def start_desktop():
    if not IS_WIN:
        raise ValueError("Запуск поддерживается только на Windows")
    stub = Path(os.environ.get("LOCALAPPDATA", "")) / "AnthropicClaude" / "claude.exe"
    exe = load_settings().get("desktop_exe")
    if stub.is_file():
        subprocess.Popen([str(stub)], creationflags=NO_WINDOW)
        return {"started": str(stub)}
    if exe and Path(exe).is_file() and "windowsapps" not in exe.lower():
        subprocess.Popen([exe], creationflags=NO_WINDOW)
        return {"started": exe}
    app_id = powershell("(Get-StartApps | Where-Object { $_.Name -eq 'Claude' } | Select-Object -First 1).AppID")
    if app_id:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        return {"started": app_id}
    raise ValueError("Не нашёл, как запустить Claude. Запусти его вручную.")


# ---------------------------------------------------------------- diagnostics

def redact(v, depth=0):
    if isinstance(v, dict):
        return {k: redact(x, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        return [redact(x, depth + 1) for x in v[:5]]
    if isinstance(v, str) and len(v) > 80:
        return v[:77] + "..."
    return v


def diagnostics():
    lines = [f"Claude Session Transfer diagnostics  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"python {sys.version.split()[0]}  os={os.name}  home={HOME}",
             f"claude config dir: {claude_config_dir()}  (projects exists: {(claude_config_dir() / 'projects').is_dir()})",
             f"transcripts indexed: {len(transcript_index())}"]
    acc = oauth_account()
    lines.append(f"~/.claude.json oauthAccount: {acc and {k: acc[k] for k in ('account', 'org', 'source')}}")
    roots = claude_roots()
    lines.append(f"roots: {[str(r) for r in roots] or 'НЕ НАЙДЕНО'}")
    for r in roots:
        try:
            lines.append(f"\n[{r}] top-level: " + ", ".join(sorted(p.name for p in r.iterdir())))
        except OSError as e:
            lines.append(f"[{r}] error {e}")
        for kind in SESSION_KINDS + ("local-agent-mode-sessions",):
            base = r / kind
            if not base.is_dir():
                continue
            lines.append(f"  {kind}/")
            for d in sorted(p for p in base.rglob("*") if p.is_dir()):
                rel = d.relative_to(base)
                if len(rel.parts) > 3:
                    continue
                n_json = len(list(d.glob("*.json")))
                other = [p.name for p in d.iterdir() if p.is_file() and p.suffix != ".json"][:5]
                lines.append(f"    {rel.as_posix()}/  json={n_json}  other={other}")
    stores = find_stores()
    lines.append(f"\nstores: {len(stores)}")
    for s in stores:
        lines.append(f"  {s['parts']} count={s['count']} mtime={datetime.fromtimestamp(s['mtime']):%Y-%m-%d %H:%M}")
    sample = next((Path(s["path"]) for s in stores if s["count"]), None)
    if sample:
        f = next(iter(sorted(sample.glob("*.json"))))
        lines.append(f"\nsample session file: {f.name}  siblings={[p.name for p in siblings(f)]}")
        try:
            lines.append(json.dumps(redact(json.loads(f.read_text('utf-8'))), ensure_ascii=False, indent=2))
        except Exception as e:
            lines.append(f"cannot parse: {e}")
        ss = list_store_sessions(sample)
        with_tr = sum(1 for x in ss if x["transcript"])
        lines.append(f"sessions with matched transcript: {with_tr}/{len(ss)}")
    lines.append(f"\ndesktop: {desktop_status()}  procs={desktop_processes()}")
    return "\n".join(lines)


# ---------------------------------------------------------------- HTTP

def state():
    stores = find_stores()
    return {
        "stores": stores,
        "extra_targets": extra_targets(stores),
        "snapshots": list_snapshots(),
        "desktop": desktop_status(),
        "roots": [str(r) for r in claude_roots()],
        "config_dir": str(claude_config_dir()),
        "snap_dir": str(SNAP_DIR),
    }


def set_alias(store_id, name):
    s = load_settings()
    s.setdefault("aliases", {})
    if name:
        s["aliases"][store_id] = name
    else:
        s["aliases"].pop(store_id, None)
    save_settings(s)
    return {"ok": True}


def open_folder(path):
    p = Path(path)
    if not p.exists():
        raise ValueError("Папка не существует")
    if IS_WIN:
        os.startfile(str(p))  # noqa
    else:
        subprocess.Popen(["xdg-open", str(p)])
    return {"ok": True}


GET_ROUTES = {
    "/api/state": lambda q: state(),
    "/api/sessions": lambda q: list_store_sessions(q["store"]),
    "/api/snapshot": lambda q: snapshot_sessions(q["name"]),
    "/api/diagnostics": lambda q: {"text": diagnostics()},
    "/api/desktop": lambda q: desktop_status(),
}
POST_ROUTES = {
    "/api/snapshot": lambda b: make_snapshot(b["store"], b["files"], b.get("include_transcripts", True), b.get("note", "")),
    "/api/restore": lambda b: restore(b["source"], b["files"], b["target"], b.get("overwrite", False),
                                      b.get("rewrite_ids", True), b.get("restore_transcripts", True), b.get("force", False)),
    "/api/alias": lambda b: set_alias(b["id"], b.get("name", "")),
    "/api/desktop/close": lambda b: close_desktop(),
    "/api/desktop/start": lambda b: start_desktop(),
    "/api/open": lambda b: open_folder(b["path"]),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_api(self, fn, arg):
        try:
            with _lock:
                self.send(200, fn(arg))
        except (ValueError, KeyError, OSError) as e:
            self.send(400, {"error": str(e)})
        except Exception as e:  # noqa
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self.send(200, (APP_DIR / "ui.html").read_bytes(), "text/html; charset=utf-8")
        fn = GET_ROUTES.get(u.path)
        if not fn:
            return self.send(404, {"error": "not found"})
        self.handle_api(fn, {k: v[0] for k, v in parse_qs(u.query).items()})

    def do_POST(self):
        u = urlparse(self.path)
        fn = POST_ROUTES.get(u.path)
        if not fn:
            return self.send(404, {"error": "not found"})
        # защита от запросов с чужих сайтов
        origin = self.headers.get("Origin", "")
        if origin and not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
            return self.send(403, {"error": "forbidden"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send(400, {"error": "bad json"})
        self.handle_api(fn, body)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    port = int(os.environ.get("CST_PORT", "8765"))
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit("No free port")
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"Claude Session Transfer: {url}  (Ctrl+C to stop)")
    if not os.environ.get("CST_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
