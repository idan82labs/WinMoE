# Rollback Rule - TPS Optimization Loop

Reject if tok/s regresses, output becomes incoherent, or the change introduces instability (crashes, OOM).
Revert GGUF patches with patch_k_experts.py. Revert CLI flags by returning to last accepted config.
