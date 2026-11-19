from .spec import ENV_VERSION as __version__


def __getattr__(name):
    if name == "AuditEnv":
        from .env import AuditEnv

        return AuditEnv
    raise AttributeError(name)
