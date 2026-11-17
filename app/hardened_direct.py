"""v0.2.15: opt-in citation / Unicode / entity-focus hardening.

All earlier pipeline modes stay available unchanged.
"""
from app.relation_direct import check_relations


def check_hardened(checker, english, chinese, requested_model=None,
                   requested_verifier=None, mode='dual'):
    return check_relations(checker, english, chinese, requested_model,
                           requested_verifier, mode, hardened=True)
