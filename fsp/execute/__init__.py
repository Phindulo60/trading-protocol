"""Execution bridge: turn journalled FSP signals into live orders.

The bridge is opt-in (``FSP_EXECUTE=1``) and decoupled from the mt4-executor
engine: it writes a ``buy``/``sell`` row into the shared Supabase ``commands``
queue, which the engine already polls and fills via MetaApi. No shared imports,
no inbound port -- one HTTPS POST per executed signal.
"""

from fsp.execute.bridge import ExecutionConfig, maybe_execute, size_position

__all__ = ["ExecutionConfig", "maybe_execute", "size_position"]
