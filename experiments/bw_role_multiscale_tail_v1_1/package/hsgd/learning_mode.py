"""Separate strict engineering gates from ordinary efficient learning."""


def state():
    import torch
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def configure(strict):
    import torch
    torch.use_deterministic_algorithms(strict, warn_only=False)
    require(state(),strict)


def require(binding,strict):
    if binding.get("deterministic_algorithms") is not strict or binding.get("deterministic_warn_only") is not False:
        raise RuntimeError("wrong execution mode: expected deterministic_algorithms="+str(strict)+", warn_only=False")


def same_platform(a,b):
    ignored={"deterministic_algorithms","deterministic_warn_only"}
    if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}:
        raise RuntimeError("preflight/training platform, dependency or actual runtime changed")
