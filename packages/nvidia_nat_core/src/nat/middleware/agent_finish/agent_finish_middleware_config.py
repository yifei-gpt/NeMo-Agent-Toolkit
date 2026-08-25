from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class AgentFinishMiddlewareConfig(FunctionMiddlewareBaseConfig, name="agent_finish"):
    """Turns a `finish` call into the answer of the agent that made it."""
