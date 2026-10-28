"""How a node's ``tools["x"](**kwargs)`` reaches an MCP tool.

MRAG_TOOL_TRANSPORT, read from the environment of the process that builds the tool table:

  mcp     (the default) every call is an MCP ``tools/call`` request.  An MCP client session sends it over an
          in-memory transport (same process: no stdio, no socket) to a real FastMCP server of the class the
          package's ``server.py`` uses (``mcp.server.fastmcp.FastMCP``; ``fastmcp.FastMCP`` for enrichment_mcp).
          The server's tools are registered by the very register function the node passes to
          ``capture_mcp_tools()`` (the node's executor and closure state included).
  direct  the behaviour before 2026-10-12: ``capture_mcp_tools()`` returns the Python functions.  This is the
          in-code rollback; ``nodes/_base.py`` keeps the old code path for it unchanged.
  Any other value raises ValueError (a typo must not pick a transport).

What mcp mode guarantees (each is checked on every call, not assumed):
  arguments  sent as JSON values (pathlib paths as text, tuples as lists); a value JSON cannot carry
             (NaN, infinity, numpy scalars, sets, bytes, objects, non-text dict keys) raises McpArgumentError
             before anything is sent.  ``None`` for a parameter whose schema has no null is left out when the
             parameter's default is ``None`` (the server then supplies that same ``None``), else it raises.
             The arguments the server validated and delivered are compared with what the node passed (type
             and value, recursively; parameters not passed must arrive as their Python default) BEFORE the
             tool function runs; any difference raises McpArgumentError and the tool does not run.
  call       a wrong call (unknown keyword, missing argument) raises the TypeError Python would raise
             (same message), before anything is sent.
  return     the tool's Python return value travels as MCP content / structured content; the decoded value
             must equal the returned one in type and value (recursively) or McpTransportError is raised.
  errors     an exception raised by the tool (any BaseException) is caught in-process and re-raised in the
             node unchanged (same object: type, message, traceback, cause); the MCP result says isError.
  execution  the tool function runs in the CALLING thread with no event loop running in it, as in direct
             mode (signals, thread-locals, asyncio.run inside a tool behave the same); stdout / stderr are
             untouched (no stdio transport); Python warnings reach the normal warning machinery (the MCP
             server's per-request ``warnings.catch_warnings(record=True)`` is not used -- it would hide them
             and is not thread-safe); the mcp / fastmcp libraries' own log lines are silenced (a transport
             failure raises instead); no read timeout anywhere.
  threads    one event loop thread + one server + one client session per (tool table, calling thread), kept
             for the table's lifetime: concurrent calls from a thread pool never share a loop or a session.

The module imports only the standard library at import time.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import pathlib
import queue
import threading
import typing
import weakref

ENV = "MRAG_TOOL_TRANSPORT"
TRANSPORTS = ("mcp", "direct")
DEFAULT_TRANSPORT = "mcp"

# server names of the packages' server.py (FastMCP("<name>", ...))
SERVER_NAMES = {
    "atac_mcp": "atac-mcp", "rna_mcp": "rna-mcp", "combined_mcp": "combined-mcp",
    "diffbind_mcp": "diffbind-mcp", "homer_mcp": "homer-mcp", "trn_mcp": "trn-mcp",
    "paperqa_mcp": "paperqa-mcp", "alphagenome_mcp": "alphagenome-mcp",
    "enrichment_mcp": "enrichment-mcp", "viz_report_mcp": "viz-report-mcp",
}
# packages whose server.py uses the standalone fastmcp (3.x) instead of mcp.server.fastmcp
FASTMCP3_PACKAGES = frozenset({"enrichment_mcp"})

# sha256 of inspect.getsource(mcp.server.lowlevel.server.Server._handle_message) in mcp 1.27.1 -- the one
# library method this module replaces (per server instance); a different mcp refuses to run (re-verify first)
_HANDLE_MESSAGE_SHA256 = "00e94d1b5a2d2bbe7bfc2c92178ba116268656cc1c4998ac00929aa3eae3111f"


def transport_name(env=None) -> str:
    """'mcp' (default) or 'direct'; an unknown value raises ValueError."""
    env = os.environ if env is None else env
    v = str(env.get(ENV) or "").strip().lower()
    if not v:
        return DEFAULT_TRANSPORT
    if v not in TRANSPORTS:
        raise ValueError("%s=%r is not one of mcp | direct" % (ENV, v))
    return v


class McpTransportError(RuntimeError):
    """The MCP round trip itself failed or would change a value.  Never a tool's own error."""


class McpArgumentError(TypeError):
    """An argument cannot reach the tool unchanged over MCP; the tool was not run."""


class _ToolRaised(Exception):
    """Raised inside the server so the MCP result is isError; the node gets the tool's own exception."""


# --------------------------------------------------------------------------------------------- registration
class _Recorder:
    """Stand-in passed to the package's register function: records every ``@mcp.tool(...)`` (function, decorator
    arguments) in order and returns the function itself, exactly like nodes/_base._CaptureMCP."""

    def __init__(self):
        self.entries = []

    def tool(self, *a, **kw):
        def deco(fn):
            self.entries.append((fn, a, kw))
            return fn
        return deco


def mcp_tool_table(register_fn) -> dict:
    """``{tool name: callable}`` like capture_mcp_tools() in direct mode, every call going through MCP."""
    rec = _Recorder()
    register_fn(rec)
    table = _Table(rec.entries)
    return {name: _make_proxy(table, spec) for name, spec in table.specs.items()}


def _make_proxy(table, spec):
    def proxy(*args, **kwargs):
        return table.call(spec.key, args, kwargs)
    proxy.__name__ = spec.fn.__name__
    proxy.__qualname__ = spec.fn.__qualname__
    proxy.__doc__ = spec.fn.__doc__
    proxy.__module__ = spec.fn.__module__
    proxy.__signature__ = spec.sig
    proxy.mrag_tool_transport = "mcp"
    return proxy


class _Spec:
    """One tool: the original function, its signature, evaluated annotations, a binding stub."""

    def __init__(self, fn, reg_args, reg_kwargs):
        if not inspect.isfunction(fn):
            raise McpTransportError("%r is not a plain function" % (fn,))
        if inspect.iscoroutinefunction(fn) or inspect.isgeneratorfunction(fn) or hasattr(fn, "__wrapped__"):
            raise McpTransportError("%s: async / generator / wrapped tool functions are not supported" % fn.__name__)
        self.fn = fn
        self.key = fn.__name__                       # the dict key, as in direct mode
        self.reg_args, self.reg_kwargs = reg_args, reg_kwargs
        name = reg_kwargs.get("name")
        if name is None and reg_args and isinstance(reg_args[0], str):
            name = reg_args[0]
        self.mcp_name = name or fn.__name__          # the MCP tool name
        self.sig = inspect.signature(fn)
        self.sig_eval = inspect.signature(fn, eval_str=True)
        self.hints = typing.get_type_hints(fn)
        for p in self.sig.parameters.values():
            if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                raise McpTransportError("%s: parameter kind %s of %r is not supported" % (fn.__name__, p.kind, p.name))
        self.params = self.sig_eval.parameters
        self.ret = self.sig_eval.return_annotation
        self.stub = _binding_stub(fn, self.sig)


def _binding_stub(fn, sig):
    """An empty function with fn's parameter list and code names: calling it with a wrong argument list raises
    the very TypeError (same message) the tool would raise, without running anything."""
    parts, star = [], False
    for p in sig.parameters.values():
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        parts.append(p.name + ("=None" if p.default is not p.empty else ""))
    ns = {}
    exec("def _stub(%s):\n    pass\n" % ", ".join(parts), ns)
    st = ns["_stub"]
    st.__code__ = st.__code__.replace(co_name=fn.__code__.co_name, co_qualname=fn.__code__.co_qualname)
    st.__name__, st.__qualname__ = fn.__name__, fn.__qualname__
    return st


# --------------------------------------------------------------------------------------------- values
def _to_wire(v, where):
    """A JSON value for v, or McpArgumentError.  Paths become text, tuples lists; nothing else is converted."""
    t = type(v)
    if v is None or t is str or t is bool or t is int:
        return v
    if t is float:
        if not math.isfinite(v):
            raise McpArgumentError("%s = %r: JSON cannot carry a non-finite float (MCP would deliver null)" % (where, v))
        return v
    if isinstance(v, pathlib.PurePath):
        return str(v)
    if t is list or t is tuple:
        return [_to_wire(x, "%s[%d]" % (where, i)) for i, x in enumerate(v)]
    if t is dict:
        out = {}
        for k, x in v.items():
            if type(k) is not str:
                raise McpArgumentError("%s: dict key %r is not text (JSON object keys are text)" % (where, k))
            out[k] = _to_wire(x, "%s[%r]" % (where, k))
        return out
    raise McpArgumentError("%s = %r: a %s.%s cannot be sent as JSON unchanged (pass a plain str/int/float/bool/"
                           "list/dict, or run with %s=direct)" % (where, _short(v), t.__module__, t.__qualname__, ENV))


def _same(a, b):
    """True when a and b are the same value: same type at every level, equal values, same dict key order."""
    if type(a) is not type(b):
        return False
    if type(b) is float:
        return a == b or (a != a and b != b)
    if type(b) in (list, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(b) is dict:
        return list(a.keys()) == list(b.keys()) and all(_same(a[k], b[k]) for k in b)
    try:
        return bool(a == b)
    except Exception:
        return False


def _diff(a, b, where="value"):
    """Where a and b first differ (for error messages)."""
    if type(a) is not type(b):
        return "%s: %s %s vs %s %s" % (where, type(a).__name__, _short(a), type(b).__name__, _short(b))
    if type(b) in (list, tuple):
        if len(a) != len(b):
            return "%s: length %d vs %d" % (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            if not _same(x, y):
                return _diff(x, y, "%s[%d]" % (where, i))
    if type(b) is dict:
        if list(a.keys()) != list(b.keys()):
            return "%s: keys %r vs %r" % (where, list(a.keys())[:8], list(b.keys())[:8])
        for k in b:
            if not _same(a[k], b[k]):
                return _diff(a[k], b[k], "%s[%r]" % (where, k))
    return "%s: %s vs %s" % (where, _short(a), _short(b))


def _short(v, n=120):
    r = repr(v)
    return r if len(r) <= n else r[:n] + "..."


def _admits_none(ann) -> bool:
    if ann is typing.Any or ann is None or ann is type(None) or ann is inspect.Parameter.empty:
        return True
    if isinstance(ann, typing.TypeVar):
        return True
    args = typing.get_args(ann)
    origin = typing.get_origin(ann)
    if origin is typing.Union or (origin is not None and getattr(origin, "__name__", "") == "UnionType"):
        return any(_admits_none(x) for x in args)
    if origin is typing.Annotated:
        return _admits_none(args[0])
    return False


# --------------------------------------------------------------------------------------------- the table
class _Table:
    """The tools of one register-function call.  Nothing the event-loop threads hold refers back to it: when
    the node drops the tool dict, the table is collected and its sessions are closed."""

    def __init__(self, entries):
        self.specs = {}
        for fn, a, kw in entries:
            spec = _Spec(fn, a, kw)
            if spec.key in self.specs:
                raise McpTransportError("tool %r is registered twice by one register function (direct mode keeps the "
                                        "last, a FastMCP server the first)" % spec.key)
            self.specs[spec.key] = spec
        names = [s.mcp_name for s in self.specs.values()]
        if len(set(names)) != len(names):
            raise McpTransportError("duplicate MCP tool names %r" % names)
        pkgs = sorted({s.fn.__module__.split(".")[0] for s in self.specs.values()})
        self.package = pkgs[0] if pkgs else "tools"
        if len(pkgs) > 1:
            raise McpTransportError("one register function registered tools of several packages: %r" % pkgs)
        self.kind = "fastmcp" if self.package in FASTMCP3_PACKAGES else "mcp"
        self.server_name = SERVER_NAMES.get(self.package, self.package)
        self._workers = {}
        self._lock = threading.Lock()
        weakref.finalize(self, _close_workers, self._workers, self._lock)

    def _worker(self):
        tid = threading.get_ident()
        with self._lock:
            w = self._workers.get(tid)
        if w is None or w.broken:
            w = _Worker(self.kind, self.server_name, list(self.specs.values()))
            with self._lock:
                old = self._workers.get(tid)
                self._workers[tid] = w
            if old is not None:
                old.close()
        return w

    def call(self, key, args, kwargs):
        spec = self.specs[key]
        try:
            bound = spec.sig.bind(*args, **kwargs)
        except TypeError:
            bound = None
        if bound is None:                              # outside the except block: no implicit exception chaining
            spec.stub(*args, **kwargs)                 # raises the TypeError a direct call would raise
            raise McpArgumentError("%s: the argument list does not bind to %s%s" % (key, key, spec.sig))
        passed = dict(bound.arguments)
        wire = {}
        for k, v in passed.items():
            if v is None and not _admits_none(spec.hints.get(k, inspect.Parameter.empty)):
                if spec.params[k].default is None:
                    continue                           # the server supplies the same default None
                raise McpArgumentError(
                    "%s(%s=None): the tool's schema (%s) has no null and the default is %r, so the tool could not "
                    "receive None over MCP (run with %s=direct)" % (key, k, spec.hints.get(k), spec.params[k].default, ENV))
            wire[k] = _to_wire(v, "%s(%s=...)" % (key, k))
        return self._worker().call(spec, wire, passed)


def _close_workers(workers, lock):
    with lock:
        ws = list(workers.values())
        workers.clear()
    for w in ws:
        try:
            w.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------- one session
_LOOP_GONE = object()


class _Job:
    __slots__ = ("received", "done", "value", "exc", "check_error", "ran")

    def __init__(self, received):
        self.received = received
        self.done = threading.Event()
        self.value = None
        self.exc = None
        self.check_error = None
        self.ran = False


class _Worker:
    """One calling thread's connection to one in-memory server: an event loop in a daemon thread, the server,
    one client session.  The server's tool functions are trampolines: they hand the validated arguments back
    to the calling thread, which runs the real tool function and hands its outcome back."""

    def __init__(self, kind, server_name, specs):
        self.kind = kind
        self.pid = os.getpid()
        self.jobs = queue.SimpleQueue()
        self.busy = False
        self.broken = False
        self.loop = None
        self.session = None
        self._stop = None
        self._fatal = None
        self._ready = threading.Event()
        self.server = _build_server(kind, server_name, specs, self)
        self.thread = threading.Thread(target=self._thread_main, name="mrag-mcp-%s" % server_name, daemon=True)
        self.thread.start()
        self._ready.wait()
        if self._fatal is not None or self.session is None:
            self.broken = True
            raise McpTransportError("could not open the in-memory MCP session to %s: %r" % (server_name, self._fatal))

    # ---- the loop thread
    def _thread_main(self):
        import anyio
        try:
            anyio.run(self._amain, backend="asyncio")
        except BaseException as e:                      # noqa: BLE001 -- reported to the caller
            if self._fatal is None:
                self._fatal = e
        finally:
            self.broken = True
            self._ready.set()
            self.jobs.put(_LOOP_GONE)

    async def _amain(self):
        import asyncio
        import anyio
        self.loop = asyncio.get_running_loop()
        self._stop = anyio.Event()
        if self.kind == "mcp":
            from mcp.shared.memory import create_connected_server_and_client_session
            async with create_connected_server_and_client_session(self.server, read_timeout_seconds=None) as s:
                self.session = s
                self._ready.set()
                await self._stop.wait()
        else:
            from fastmcp import Client
            async with Client(self.server, timeout=None, init_timeout=0) as c:
                self.session = c
                self._ready.set()
                await self._stop.wait()

    async def _acall(self, name, wire):
        if self.kind == "mcp":
            return await self.session.call_tool(name, wire, read_timeout_seconds=None)
        return await self.session.call_tool_mcp(name, wire, timeout=None)

    def close(self):
        if os.getpid() != self.pid:                     # a forked child: the loop thread is not ours
            return
        if self.loop is not None and self._stop is not None and not self.broken:
            try:
                self.loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        if (not self.broken and self.thread.is_alive()
                and threading.current_thread() is not self.thread):
            self.thread.join(timeout=10)
        self.broken = True

    # ---- the calling thread
    def call(self, spec, wire, passed):
        import asyncio
        if self.busy:
            raise McpTransportError("%s: re-entrant call on the same tool table from inside one of its tools" % spec.key)
        self.busy = True
        job = None
        try:
            done = object()
            fut = asyncio.run_coroutine_threadsafe(self._acall(spec.mcp_name, wire), self.loop)
            fut.add_done_callback(lambda _f, _d=done: self.jobs.put(_d))
            while True:
                item = self.jobs.get()
                if item is done:
                    break
                if item is _LOOP_GONE:
                    self.broken = True
                    raise McpTransportError("%s: the MCP event loop stopped (%r)" % (spec.key, self._fatal))
                if job is not None:                    # the server invoked the tool twice: refuse the second run
                    item.check_error = McpTransportError("%s: the MCP server invoked the tool twice" % spec.key)
                    item.done.set()
                    continue
                job = item
                self._run_job(spec, job, passed)
            try:
                res, proto_err = fut.result(), None
            except Exception as e:                      # noqa: BLE001 -- the MCP round trip failed
                res, proto_err = None, e
                self.broken = True
        except BaseException:
            if job is None or not job.done.is_set():
                self.broken = True                      # interrupted mid-call: do not reuse this session
            raise
        finally:
            self.busy = False
        if proto_err is not None:
            if job is not None and job.exc is not None:
                _reraise(job, "exc")                    # the tool's own error wins over a later transport error
            raise McpTransportError("%s: the MCP call failed: %r" % (spec.key, proto_err)) from proto_err
        if job is None:
            raise McpTransportError("%s: the MCP server did not run the tool: %s" % (spec.key, _result_text(res)))
        if job.check_error is not None:
            _reraise(job, "check_error")
        if job.exc is not None:
            if not res.isError:
                raise McpTransportError("%s: the tool raised %r but the MCP result is not an error" % (spec.key, job.exc))
            _reraise(job, "exc")
        if res.isError:
            raise McpTransportError("%s: the tool returned but the MCP result is an error: %s" % (spec.key, _result_text(res)))
        value = _decode(res, spec)
        if not _same(value, job.value):
            raise McpTransportError("%s: the return value does not survive MCP unchanged (%s)"
                                    % (spec.key, _diff(value, job.value, "returned")))
        return value

    def _run_job(self, spec, job, passed):
        try:
            try:
                _check_received(spec, job.received, passed)
            except BaseException as e:                  # noqa: BLE001
                job.check_error = e
                return
            job.ran = True
            try:
                job.value = spec.fn(**job.received)
            except BaseException as e:                  # noqa: BLE001 -- re-raised in the node unchanged
                job.exc = e
        finally:
            job.done.set()


def _reraise(job, attr):
    """Raise the stored exception object itself (type, message, traceback, cause unchanged), keeping no
    reference to it in this module's frames (no exception -> traceback -> frame -> exception cycle)."""
    exc = getattr(job, attr)
    setattr(job, attr, None)
    try:
        raise exc
    finally:
        del exc


def _check_received(spec, received, passed):
    """The arguments the server delivered must be the ones the node passed (and the defaults for the rest)."""
    extra = [k for k in received if k not in spec.params]
    if extra:
        raise McpTransportError("%s: the server delivered unknown arguments %r" % (spec.key, extra))
    for k, p in spec.params.items():
        if k in passed:
            if k not in received:
                if passed[k] is None and p.default is None:
                    continue                            # left out on purpose; the function's default None applies
                raise McpArgumentError("%s(%s=...): the argument did not arrive" % (spec.key, k))
            exp = passed[k]
        else:
            if k not in received:
                continue                                # the function's own default applies, as in direct mode
            if p.default is p.empty:
                raise McpTransportError("%s: required %s arrived without being passed" % (spec.key, k))
            exp = p.default
        got = received[k]
        if not _same(got, exp):
            raise McpArgumentError(
                "%s(%s=...): over MCP the tool would receive a different value (%s) -- the tool was not run. "
                "Pass the exact value the tool's schema declares, or run with %s=direct"
                % (spec.key, k, _diff(got, exp, "received vs passed"), ENV))


def _result_text(res):
    try:
        return " | ".join(getattr(c, "text", repr(c)) for c in res.content)[:2000]
    except Exception:
        return repr(res)[:2000]


def _decode(res, spec):
    """The Python value of a CallToolResult, given the tool's return annotation."""
    content = list(res.content or [])
    texts = [c.text for c in content if getattr(c, "type", None) == "text"]
    if len(texts) != len(content):
        raise McpTransportError("%s: non-text MCP content %r" % (spec.key, [getattr(c, "type", c) for c in content]))
    sc = res.structuredContent
    ret = spec.ret
    origin = typing.get_origin(ret) or ret
    if sc is not None:
        if origin is dict:
            return sc
        if list(sc.keys()) != ["result"]:
            raise McpTransportError("%s: unexpected structured content keys %r" % (spec.key, list(sc.keys())))
        v = sc["result"]
        if type(v) is str and texts and texts != [v]:
            raise McpTransportError("%s: the text and the structured content of the result differ" % spec.key)
        return v
    if not texts:
        return None
    if ret is str:
        if len(texts) != 1:
            raise McpTransportError("%s: %d text blocks for a str result" % (spec.key, len(texts)))
        return texts[0]
    if len(texts) != 1:
        raise McpTransportError("%s: %d text blocks for a %r result" % (spec.key, len(texts), ret))
    return json.loads(texts[0])


# --------------------------------------------------------------------------------------------- the server
_BUILD_LOCK = threading.RLock()     # one server is built at a time (construction + registration)
_quiet_done = set()


def _quiet(name):
    """The mcp / fastmcp libraries log through their own loggers (request lines, tool tracebacks); none of that
    exists in direct mode and a transport failure raises instead, so their loggers are silenced."""
    if name not in _quiet_done:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        _quiet_done.add(name)


def _make_trampoline(spec, worker):
    def trampoline(**received):
        job = _Job(received)
        worker.jobs.put(job)
        job.done.wait()
        if job.exc is not None or job.check_error is not None:
            raise _ToolRaised(spec.key)
        return job.value
    trampoline.__name__ = spec.fn.__name__
    trampoline.__qualname__ = spec.fn.__qualname__
    trampoline.__doc__ = spec.fn.__doc__
    trampoline.__module__ = spec.fn.__module__
    trampoline.__signature__ = spec.sig_eval        # the schema comes from the tool's own signature ...
    trampoline.__annotations__ = dict(spec.hints)   # ... and its evaluated annotations
    return trampoline


def _new_server_object(kind, server_name):
    root = logging.getLogger()
    with _BUILD_LOCK:
        saved = (root.level, list(root.handlers))
        try:
            if kind == "mcp":
                import mcp.server.fastmcp.server as _fs
                _quiet("mcp")
                orig = _fs.configure_logging            # FastMCP() would logging.basicConfig() the ROOT logger
                _fs.configure_logging = lambda *a, **k: None
                try:
                    srv = _fs.FastMCP(server_name)
                finally:
                    _fs.configure_logging = orig
            else:
                import fastmcp                          # configures the 'fastmcp' logger only (propagate=False)
                _quiet("mcp")
                _quiet("fastmcp")
                srv = fastmcp.FastMCP(server_name)
        finally:
            if (root.level, list(root.handlers)) != saved:
                root.setLevel(saved[0])
                root.handlers[:] = saved[1]
    _keep_warnings(srv)
    return srv


def _build_server(kind, server_name, specs, worker):
    with _BUILD_LOCK:
        srv = _new_server_object(kind, server_name)
        for spec in specs:
            srv.tool(*spec.reg_args, **spec.reg_kwargs)(_make_trampoline(spec, worker))
    return srv


def build_reference_server(kind, server_name, specs):
    """The same server with the ORIGINAL functions registered (tests: the schemas must be identical)."""
    with _BUILD_LOCK:
        srv = _new_server_object(kind, server_name)
        for spec in specs:
            srv.tool(*spec.reg_args, **spec.reg_kwargs)(spec.fn)
    return srv


def handle_message_source_sha256():
    from mcp.server.lowlevel.server import Server
    return hashlib.sha256(inspect.getsource(Server._handle_message).encode()).hexdigest()


def _keep_warnings(srv):
    """Per server instance: mcp 1.27.1's Server._handle_message without its warnings.catch_warnings(record=True)
    (which swallows the tool's Python warnings into a list and, entered from several threads, can leave the
    process's warning display pointing at a dead list).  Everything else is the library's code, unchanged."""
    import types as _pytypes
    import mcp.types as types
    from mcp.server.lowlevel.server import Server
    from mcp.shared.session import RequestResponder
    low = srv._mcp_server
    if type(low)._handle_message is not Server._handle_message:
        raise McpTransportError("%s overrides _handle_message: re-verify the MCP transport" % type(low).__name__)
    sha = handle_message_source_sha256()
    if sha != _HANDLE_MESSAGE_SHA256:
        raise McpTransportError("mcp's Server._handle_message changed (sha256 %s, verified %s): re-verify the MCP "
                                "transport or use %s=direct"
                                % (sha, _HANDLE_MESSAGE_SHA256, ENV))
    lowlevel_logger = logging.getLogger("mcp.server.lowlevel.server")

    # Adapted from the MCP Python SDK 1.27.1 (https://github.com/modelcontextprotocol/python-sdk),
    # mcp/server/lowlevel/server.py Server._handle_message; MIT License, Copyright (c) 2024
    # Anthropic, PBC.  See NOTICE.md.
    async def _handle_message(self, message, session, lifespan_context, raise_exceptions=False):
        match message:
            case RequestResponder(request=types.ClientRequest(root=req)) as responder:
                with responder:
                    await self._handle_request(message, req, session, lifespan_context, raise_exceptions)
            case types.ClientNotification(root=notify):
                await self._handle_notification(notify)
            case Exception():
                lowlevel_logger.error(f"Received exception from stream: {message}")
                await session.send_log_message(
                    level="error",
                    data="Internal Server Error",
                    logger="mcp.server.exception_handler",
                )
                if raise_exceptions:
                    raise message

    low._handle_message = _pytypes.MethodType(_handle_message, low)
