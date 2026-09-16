#!/usr/bin/env python3
"""Per-request GPU prefix-cache "no-store" (``skip_writing_prefix_cache``).

Backport of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks PR #95 (2026-09-16
upstream-check round). Three-file anchored patch, ported close to verbatim:
every anchor below was verified byte-for-byte against this image's installed
vLLM before being written here.

Problem (all line refs = upstream's own fork, consistent with this one):

  vLLM has a READ-side opt-out (``SamplingParams.skip_reading_prefix_cache``
  -> ``Request.skip_reading_prefix_cache`` -> ``KVCacheManager.get_computed_
  blocks``) but no WRITE-side one. ``cache_salt`` namespaces and still
  stores. ``BlockPool.free_blocks`` appends HASHED blocks to the back of the
  free queue (LRU) and prepends UNHASHED ones (LIFO); ``get_new_blocks``
  pops from the front. A one-off batch/eval request's blocks are therefore
  hashed, queue behind the owner's idle long conversation, and the owner's
  blocks are what gets evicted next: the batch job re-orders the LRU in its
  own favour. With a no-store flag its blocks carry no hash, go to the
  FRONT, and are the very next ids recycled.

Mechanism: suppress the HASH INSERTION, keep every piece of bookkeeping.

  The two request-driven ``_insert_block_hash`` sites in block_pool.py --
  ``cache_full_blocks`` and ``cache_partial_block`` -- return early for a
  no-store request. NOT the obvious ``allocate_slots`` one-liner (``if not
  self.enable_caching or delay_cache_blocks``): skipping
  ``coordinator.cache_blocks`` skips ``SingleTypeKVCacheManager.cache_blocks``'
  last line ``self.num_cached_block[request_id] = num_full_blocks``, and
  membership in ``num_cached_block`` is the running-request sentinel read by
  ``get_num_blocks_to_allocate`` and ``KVCacheCoordinator.
  allocate_new_computed_blocks``. With the block_pool placement
  ``num_cached_block`` advances identically for every manager on every step.
  Every consumer already tolerates an allocated full block without a hash:
  that is exactly what the fork's sparse retention (``reachable_block_mask``)
  produces.

Semantics (write-only, GPU prefix cache only):
  * lookups stay enabled: a no-store request may still read-hit, and reading
    touches blocks (refreshes their LRU position). Combine with
    ``skip_reading_prefix_cache`` for zero cache interaction.
  * a no-store request that is preempted resumes from whatever it could READ
    (nothing, if its prefix was cold).
  * KV connectors / CPU offload are not touched (none active on this kit).

Surface:
  ``SamplingParams.skip_writing_prefix_cache: bool | None`` (typed) or
  ``vllm_xargs: {"skip_writing_prefix_cache": 1}`` on /v1/chat/completions,
  /v1/completions, /v1/responses (already forwarded to
  ``SamplingParams.extra_args`` by stock vLLM -- verified against this
  image; zero entrypoint edits). Values are STRICT -- bool, int 0/1, str
  "0"/"1" -- and validated in ``SamplingParams.__post_init__`` (API-server
  side; a bad value is an HTTP 400 via the ValueError handler, never a
  silent no-op). ``vllm_xargs`` is typed ``dict[str, str|int|float|list]``;
  pydantic v2 coerces JSON ``true``/``false`` to ``1``/``0`` there, which is
  exactly the intended meaning, so booleans work too. ``Request`` is
  materialised in the engine-core input thread, so its resolver never
  raises; an unparseable value there (unreachable through the API) logs
  once and stores normally.

Kill switch: ``GLM53_APC_NO_STORE`` exactly ``0`` or ``1`` (unset = 1,
enabled -- nothing changes unless a caller asks for the per-request flag).
``0`` = a valid ``1`` from the client is ignored (logged once); malformed
values are still rejected. Anything else fails closed at import (boot
failure; the launcher refuses it before the pair is stopped).

Receipts in the server log (both ``logger.info_once``):
  ``[glm53-apc-no-store] first request resolved skip_writing_prefix_cache=1``
      the flag reached the engine (emitted at resolution, so a warm request
      that never stores anything still leaves proof);
  ``[glm53-apc-no-store] suppressing prefix-cache store (full site)`` /
  ``(partial site)`` -- the store path was actually cut.

Fail closed if any anchor drifts: every anchor in every file is checked
BEFORE anything is written; a file that already carries the MARK must carry
every generated snippet verbatim and still parse (else refuse); complete
files are skipped; each write is a temp-file + os.replace (never a
truncated target). Same style as overlay/patch_default_max_new_tokens.py.
"""

from __future__ import annotations

import ast
import os
import stat
import sys
import tempfile
from pathlib import Path

MARK = "# [glm53-apc-no-store]"
NO_STORE_KEY = "skip_writing_prefix_cache"

_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
SAMPLING_PARAMS_PY = Path(
    os.environ.get("GLM53_SAMPLING_PARAMS_PY", f"{_VLLM}/sampling_params.py")
)
REQUEST_PY = Path(os.environ.get("GLM53_REQUEST_PY", f"{_VLLM}/v1/request.py"))
BLOCK_POOL_PY = Path(
    os.environ.get("GLM53_BLOCK_POOL_PY", f"{_VLLM}/v1/core/block_pool.py")
)

# ---------------------------------------------------------------- anchors ----

# sampling_params.py ----------------------------------------------------------

# 1. the read-side field: add its write-side sibling.
SP_FIELD_OLD = "    skip_reading_prefix_cache: bool | None = None\n"
SP_FIELD_NEW = '''    skip_reading_prefix_cache: bool | None = None
    skip_writing_prefix_cache: bool | None = None  # [glm53-apc-no-store]
    """If True, this request's KV blocks are never inserted into the GPU
    prefix cache (overlay/patch_apc_no_store.py). Lookups are unaffected: the
    request may still read-hit. Its blocks stay unhashed, so
    ``BlockPool.free_blocks`` recycles them LIFO from the front of the free
    queue instead of queueing them behind -- and thereby evicting -- other
    sessions' cached blocks. For one-off batch lanes whose prefix will never be
    reused. Also reachable as ``vllm_xargs: {"skip_writing_prefix_cache": 1}``
    (``extra_args``); accepted values are bool, int 0/1 and str "0"/"1"
    (pydantic coerces a JSON boolean in ``vllm_xargs`` to 1/0)."""
'''

# 2. module-level helpers, inserted before the SamplingParams class.
SP_CLASS_ANCHOR = "\nclass SamplingParams(\n"
SP_HELPERS = '''
# [glm53-apc-no-store] helper-begin
_GLM53_NO_STORE_KEY = "skip_writing_prefix_cache"
_GLM53_NO_STORE_ENV = "GLM53_APC_NO_STORE"


def _glm53_parse_no_store(value, where):  # [glm53-apc-no-store]
    """Strict 0/1 parser for the per-request no-store flag.

    Accepts bool, int 0/1 and str "0"/"1" -- nothing else. ``bool("0")`` is
    True, floats are ambiguous, "yes"/"true" are not part of the contract, so
    all of those raise ValueError (an HTTP 400 through the OpenAI server's
    ValueError handler when raised from SamplingParams.__post_init__).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value in ("0", "1"):
        return value == "1"
    raise ValueError(
        f"{where} must be 0 or 1 (int), \\"0\\"/\\"1\\" (str) or a JSON boolean; "
        f"got {value!r}."
    )


def _glm53_no_store_env():  # [glm53-apc-no-store]
    """GLM53_APC_NO_STORE: unset -> enabled; exactly "0"/"1"; else fail closed."""
    import os as _os

    raw = _os.environ.get(_GLM53_NO_STORE_ENV)
    if raw is None:
        return True
    if raw in ("0", "1"):
        return raw == "1"
    raise ValueError(
        f"{_GLM53_NO_STORE_ENV} must be exactly 0 or 1 (got {raw!r}); "
        "unset it to keep the per-request no-store flag honoured."
    )


_GLM53_NO_STORE_ENABLED = _glm53_no_store_env()


def _glm53_validate_no_store_params(params):  # [glm53-apc-no-store]
    """Reject a malformed flag where the request is still the client's.

    Runs from ``SamplingParams.__post_init__`` -- i.e. in the API-server
    process for /v1/chat/completions etc. (``to_sampling_params`` ->
    ``from_optional``) and in the caller's process for the offline ``LLM``
    API -- BEFORE the kill switch is consulted, so ``GLM53_APC_NO_STORE=0``
    never turns a typo into a silent no-op. The typed field is normalised to
    a bool; ``extra_args`` is left as sent.
    """
    typed = params.skip_writing_prefix_cache
    if typed is not None:
        params.skip_writing_prefix_cache = _glm53_parse_no_store(
            typed, "SamplingParams.skip_writing_prefix_cache"
        )
    extra = params.extra_args
    if extra and _GLM53_NO_STORE_KEY in extra:
        _glm53_parse_no_store(
            extra[_GLM53_NO_STORE_KEY], 'vllm_xargs["skip_writing_prefix_cache"]'
        )


def _glm53_resolve_no_store(sampling_params, request_id):  # [glm53-apc-no-store]
    """Engine-side resolution: typed field first, then ``extra_args``.

    Never raises: ``Request`` is materialised in the engine-core input thread
    (``EngineCore.preprocess_add_request``), where an exception is not a
    request-scoped error. A value that fails the strict parser here is
    unreachable through the OpenAI API (``_glm53_validate_no_store_params``
    already rejected it with a 400); if it happens anyway it is logged once and
    the request stores normally. Malformed values are rejected BEFORE the kill
    switch is applied; the switch only decides whether a valid 1 is honoured.
    """
    if sampling_params is None:
        return False
    raw = getattr(sampling_params, _GLM53_NO_STORE_KEY, None)
    where = "SamplingParams.skip_writing_prefix_cache"
    if raw is None:
        extra = getattr(sampling_params, "extra_args", None)
        if not extra or _GLM53_NO_STORE_KEY not in extra:
            return False
        raw = extra[_GLM53_NO_STORE_KEY]
        where = 'vllm_xargs["skip_writing_prefix_cache"]'
    try:
        value = _glm53_parse_no_store(raw, where)
    except ValueError as exc:
        logger.warning(
            "[glm53-apc-no-store] %s -- request %s stores normally "
            "(this value should have been rejected at the API boundary)",
            exc,
            request_id,
        )
        return False
    if not value:
        return False
    # ``*_once`` is keyed on (msg, args): keep the request id OUT of the args
    # so these really are one line per process; the per-request id is a debug
    # line.
    if not _GLM53_NO_STORE_ENABLED:
        logger.info_once(
            "[glm53-apc-no-store] ignoring skip_writing_prefix_cache=1: "
            "GLM53_APC_NO_STORE=0 (first occurrence; later ones not logged)"
        )
        logger.debug("[glm53-apc-no-store] ignored for request %s", request_id)
        return False
    logger.info_once(
        "[glm53-apc-no-store] first request resolved skip_writing_prefix_cache=1 "
        "(lookups unaffected, prefix-cache stores suppressed; later "
        "occurrences not logged)"
    )
    logger.debug("[glm53-apc-no-store] request %s is no-store", request_id)
    return True
# [glm53-apc-no-store] helper-end

'''

# 3. the tail of SamplingParams.__post_init__: validate after the read-side
#    auto-set so a bad value is a 400, not a silent no-op.
SP_POST_OLD = (
    "            self.skip_reading_prefix_cache = self.prompt_logprobs is not None\n"
)
SP_POST_NEW = (
    "            self.skip_reading_prefix_cache = self.prompt_logprobs is not None\n"
    "        _glm53_validate_no_store_params(self)  # [glm53-apc-no-store]\n"
)

# v1/request.py ---------------------------------------------------------------

# 4. resolve the flag once, next to the read-side one.
REQ_RESOLVE_OLD = (
    "        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()\n"
)
REQ_RESOLVE_NEW = (
    "        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()\n"
    "        # [glm53-apc-no-store] write-side sibling; resolved once, never raises\n"
    "        self.skip_writing_prefix_cache = self.get_skip_writing_prefix_cache()\n"
)

# 5. the resolver, after get_skip_reading_prefix_cache.
REQ_METHOD_OLD = """        return False

    def is_finished(self) -> bool:
"""
REQ_METHOD_NEW = '''        return False

    def get_skip_writing_prefix_cache(self) -> bool:  # [glm53-apc-no-store]
        """Whether this request's blocks must never enter the prefix cache.

        Typed ``SamplingParams.skip_writing_prefix_cache`` first, then
        ``SamplingParams.extra_args`` (the ``vllm_xargs`` route). Strict 0/1
        values; the kill switch GLM53_APC_NO_STORE=0 makes a valid 1 a logged
        no-op. Never raises (see _glm53_resolve_no_store). PoolingParams are
        out of scope (chat/completions only).
        """
        from vllm.sampling_params import _glm53_resolve_no_store

        return _glm53_resolve_no_store(self.sampling_params, self.request_id)

    def is_finished(self) -> bool:
'''

# v1/core/block_pool.py -------------------------------------------------------

# 6. proof-of-life helper above the class (module logger already exists).
BP_CLASS_ANCHOR = "\nclass BlockPool:\n"
BP_HELPER = '''
def _glm53_log_nostore(request, site) -> None:  # [glm53-apc-no-store]
    """Proof-of-life: log the first time a no-store request suppresses a store.

    Pair with the resolution receipt logged by Request: that one proves the
    flag reached the engine, this one proves the store path was cut. A warm
    no-store request that has nothing new to store legitimately emits only
    the first.
    """
    # ``info_once`` is keyed on (msg, args): the site is a bounded set (two
    # values), the request id is deliberately not an arg.
    logger.info_once(
        "[glm53-apc-no-store] suppressing prefix-cache store (%s site); "
        "lookups still enabled; later occurrences not logged",
        site,
    )
    logger.debug(
        "[glm53-apc-no-store] %s store suppressed for request %s",
        site,
        getattr(request, "request_id", "<unknown>"),
    )

'''

# 7. cache_full_blocks: the first request-driven _insert_block_hash site.
#    Returning here also skips the BlockStored event: nothing was stored.
BP_FULL_OLD = """        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
"""
BP_FULL_NEW = """        if num_cached_blocks >= num_full_blocks:
            return
        if getattr(request, "skip_writing_prefix_cache", False):  # [glm53-apc-no-store]
            # No-store request: insert no hash, emit no BlockStored event.
            # The caller (SingleTypeKVCacheManager.cache_blocks) still advances
            # ``num_cached_block`` -- the running-request sentinel read by
            # get_num_blocks_to_allocate and KVCacheCoordinator.
            # allocate_new_computed_blocks -- so allocation accounting is
            # unchanged. These blocks keep ``block_hash is None`` and are
            # recycled LIFO from the front of the free queue by
            # ``free_blocks`` below, ahead of anything cached.
            _glm53_log_nostore(request, "full")
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
"""

# 8. cache_partial_block: the second site (fine-grained partial tail).
#    Returning None also stops a producer registration for this request.
BP_PARTIAL_OLD = """        if block.is_null:
            return None

        assert block_size > self.hash_block_size
"""
BP_PARTIAL_NEW = """        if block.is_null:
            return None
        if getattr(request, "skip_writing_prefix_cache", False):  # [glm53-apc-no-store]
            # No-store request: no partial entry. Returning None also keeps
            # this request out of any partial-tail *producer* bookkeeping a
            # manager (e.g. mamba) does on a registered hash, which keeps
            # get_num_blocks_to_allocate / allocate_new_blocks in agreement.
            # The *reader* side of the partial-hit CoW is untouched: lookups
            # remain enabled for no-store requests.
            _glm53_log_nostore(request, "partial")
            return None

        assert block_size > self.hash_block_size
"""

# (label, old, new) per file, applied in order. ``old`` must occur exactly once
# in the pristine text; the helper anchors are re-inserted in front of their
# anchor line.
PLAN = {
    "sampling_params.py": (
        SAMPLING_PARAMS_PY,
        (
            ("sampling-field", SP_FIELD_OLD, SP_FIELD_NEW),
            ("sampling-helpers", SP_CLASS_ANCHOR, SP_HELPERS + SP_CLASS_ANCHOR),
            ("sampling-post-init", SP_POST_OLD, SP_POST_NEW),
        ),
        ("logger = init_logger(__name__)\n",),
    ),
    "request.py": (
        REQUEST_PY,
        (
            ("request-resolve", REQ_RESOLVE_OLD, REQ_RESOLVE_NEW),
            ("request-method", REQ_METHOD_OLD, REQ_METHOD_NEW),
        ),
        (),
    ),
    "block_pool.py": (
        BLOCK_POOL_PY,
        (
            ("block-pool-helper", BP_CLASS_ANCHOR, BP_HELPER + BP_CLASS_ANCHOR),
            ("block-pool-full", BP_FULL_OLD, BP_FULL_NEW),
            ("block-pool-partial", BP_PARTIAL_OLD, BP_PARTIAL_NEW),
        ),
        ("logger = init_logger(__name__)\n",),
    ),
}


def expected_marks(edits) -> int:
    """How many MARK occurrences a completely patched file carries."""
    return sum(new.count(MARK) - old.count(MARK) for _, old, new in edits)


# ------------------------------------------------------------------ apply ----


def _parses(text: str, path: Path) -> None:
    try:
        ast.parse(text, str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: does not parse: {exc}") from None


def preflight(name: str, path: Path, edits, requires) -> str | None:
    """Return the patched text, or None if the file is already complete.

    Raises SystemExit on a missing file, a drifted anchor, or a file that
    carries the MARK without every one of this overlay's generated snippets
    (verbatim, exactly once) -- a partially applied, edited or truncated
    target is refused rather than skipped as "already done".
    """
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    text = path.read_text()
    have = text.count(MARK)
    want = expected_marks(edits)
    if have:
        missing = [label for label, _old, new in edits if text.count(new) != 1]
        if have != want or missing:
            raise SystemExit(
                f"{path}: carries {have} '{MARK}' marker(s) (complete = {want}) "
                f"and lacks the verbatim snippet(s) {missing}; refusing to patch "
                "a partially modified file (restore the pristine file first)"
            )
        _parses(text, path)
        return None
    for needle in requires:
        if text.count(needle) != 1:
            raise SystemExit(f"{path}: prerequisite {needle.strip()!r} not unique")
    for label, old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected one {label} target, found {n}")
        text = text.replace(old, new, 1)
    if text.count(MARK) != want:
        raise SystemExit(f"{path}: internal error, marker count after apply")
    _parses(text, path)
    return text


def atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file + os.replace: the target is never left
    truncated, even if this process dies mid-write.

    ``mkstemp`` creates the temp file mode 0600 (owner-only) regardless of
    the target's real permissions -- ``os.replace`` then installs that mode
    verbatim, since replace is a rename, not a content copy. Built as root
    (Docker build stage) this was invisible; the container runs this
    process as a non-root user, which lost read access to the target and
    crashed the whole engine at first request with a bare PermissionError.
    Preserve the target's original mode before replacing it -- same fix
    already applied in this project's patch_spinwait.py."""
    orig_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".glm53", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, orig_mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    staged: dict[str, tuple[Path, str]] = {}
    for name, (path, edits, requires) in PLAN.items():
        patched = preflight(name, path, edits, requires)
        if patched is None:
            print(f"{path.name}: {MARK} already present - skipping")
        else:
            staged[name] = (path, patched)
    # Every anchor in every file has been verified; only now write anything.
    for name, (path, patched) in staged.items():
        atomic_write(path, patched)
        print(f"patched {path.name} ({name}: {expected_marks(PLAN[name][1])} marks)")
    if staged:
        print(
            "patched no-store overlay (per-request GPU prefix-cache no-store via "
            f"SamplingParams.{NO_STORE_KEY} / vllm_xargs; kill switch "
            "GLM53_APC_NO_STORE)"
        )
    else:
        print("no-store overlay: already applied, nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
