"""Verify the hand-rolled state-vector simulator against a brute-force
construction of the full 2^q x 2^q circuit matrix, then smoke-test the
RainbowPrompt wiring for every gate type / fusion mode."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_gate import QuantumFeatureMap, RelevanceGate, entropy_sparsity, _cnot_perm
from prompt import RainbowPrompt

torch.manual_seed(0)
OK = '  ok'


# ---------------------------------------------------------------- reference
def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def ry_mat(a):
    c, s = math.cos(a / 2), math.sin(a / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def cnot_mat(q, ctrl, tgt):
    n = 2 ** q
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        cbit = (i >> (q - 1 - ctrl)) & 1
        j = i ^ (1 << (q - 1 - tgt)) if cbit else i
        M[j, i] = 1.0
    return M


def reference_state(angles, theta, q, lq):
    """Brute force: build the circuit matrix and apply it to |0...0>."""
    psi = torch.zeros(2 ** q, dtype=torch.float64)
    psi[0] = 1.0
    for l in range(lq):
        psi = kron_list([ry_mat(float(angles[j])) for j in range(q)]) @ psi
        psi = kron_list([ry_mat(float(theta[l, j])) for j in range(q)]) @ psi
        for j in range(q - 1):
            psi = cnot_mat(q, j, j + 1) @ psi
    return psi


# ---------------------------------------------------------------- 1. circuit
print('1. state-vector simulator vs brute-force circuit matrix')
for q, lq in [(2, 1), (3, 2), (5, 2), (6, 1)]:
    fm = QuantumFeatureMap(in_dim=7, n_qubits=q, n_layers=lq).double()
    v = torch.randn(4, 7, dtype=torch.float64)
    got = fm(v)
    angles = torch.tanh(fm.proj(v * fm.in_scale)) * math.pi
    for b in range(v.shape[0]):
        want = reference_state(angles[b], fm.theta, q, lq)
        assert torch.allclose(got[b], want, atol=1e-10), (q, lq, b)
    print(f'{OK}  q={q} lq={lq}  max err {(got[0] - reference_state(angles[0], fm.theta, q, lq)).abs().max():.2e}')

print('2. unitarity: states stay on the unit sphere')
fm = QuantumFeatureMap(768, n_qubits=8, n_layers=2)
psi = fm(torch.randn(16, 768))
norms = psi.norm(dim=-1)
assert torch.allclose(norms, torch.ones(16), atol=1e-5), norms
print(f'{OK}  ||psi|| in [{norms.min():.6f}, {norms.max():.6f}], shape {tuple(psi.shape)}')

print('3. CNOT permutation is an involution')
for q in (2, 4, 6):
    for j in range(q - 1):
        p = _cnot_perm(q, j, j + 1)
        assert torch.equal(p[p], torch.arange(2 ** q))
print(f'{OK}  checked q in (2, 4, 6)')

print('4. fidelity is a valid probability and gradients flow')
gate = RelevanceGate('quantum', 768, n_qubits=6, q_layers=2, tau=1.0)
feat = torch.randn(8, 768, requires_grad=True)
keys = torch.randn(5, 768, requires_grad=True)
alpha, p = gate(feat, keys)
assert p.min() >= -1e-6 and p.max() <= 1 + 1e-6, (p.min().item(), p.max().item())
assert torch.allclose(alpha.sum(-1), torch.ones(8), atol=1e-5)
(alpha * torch.randn_like(alpha)).sum().backward()
assert gate.qmap.theta.grad is not None and gate.qmap.theta.grad.abs().sum() > 0
assert gate.qmap.proj.weight.grad.abs().sum() > 0
assert keys.grad.abs().sum() > 0
print(f'{OK}  p in [{p.min():.4f}, {p.max():.4f}], theta.grad |.|={gate.qmap.theta.grad.abs().sum():.4f}')

print('5. identical sample and task give fidelity 1')
gate = RelevanceGate('quantum', 16, n_qubits=5, q_layers=2)
v = torch.randn(3, 16)
_, p = gate(v, v)
assert torch.allclose(p.diagonal(), torch.ones(3), atol=1e-5), p.diagonal()
print(f'{OK}  diag = {[round(float(x), 6) for x in p.diagonal()]}')

print('6. entropy sparsity: uniform > peaked, and it is differentiable')
uni = torch.full((1, 8), 1 / 8)
peak = torch.tensor([[0.93] + [0.01] * 7])
assert entropy_sparsity(uni) > entropy_sparsity(peak)
print(f'{OK}  H(uniform)={entropy_sparsity(uni):.4f} > H(peaked)={entropy_sparsity(peak):.4f}')

print('7. parameter count of each gate type (768-d input)')
for gt in ('cosine', 'mlp', 'attention', 'quantum'):
    g = RelevanceGate(gt, 768, n_qubits=8, q_layers=2, hidden=64)
    n = sum(p.numel() for p in g.parameters())
    print(f'{OK}  {gt:<10} {n:>8,}')


# ---------------------------------------------------------------- RainbowPrompt
print('8. RainbowPrompt forward, all gate types x fusion modes')
L, D, T, TOP = 20, 64, 4, 1
IDX = list(range(6))
SELF = [0, 1, 2]


def build(gate_type, fusion):
    return RainbowPrompt(
        length=L, embed_dim=D, pool_size=T, top_k=TOP, prompt_tune_idx=IDX,
        n_tasks=T, D1=8, D2=12, relation_type='attention', use_linear=True,
        KI_iter=2, self_attn_idx=SELF, gate_type=gate_type, fusion=fusion,
        n_qubits=4, q_layers=2, gate_hidden=8)


B = 5
for gate_type in ('mean', 'cosine', 'mlp', 'attention', 'quantum'):
    for fusion in ('none', 'infer', 'both'):
        if gate_type == 'mean' and fusion != 'none':
            continue
        m = build(gate_type, fusion)
        cls = torch.randn(B, D)
        alpha = None
        if gate_type != 'mean':
            alpha, _ = m.gate(cls, m.base_key[:T])

        for t in range(T):
            a = alpha[:, :t + 1] / alpha[:, :t + 1].sum(-1, keepdim=True) if alpha is not None else None
            for layer in IDX:
                for ptype in ('Unique', 'Rainbow'):
                    out = m(None, layer=layer, cls_features=cls, task_id=t, cur_id=t,
                            train=True, p_type=ptype, alpha=a)
                    k, v = out['batched_prompt']
                    assert k.shape == (B, L // 2, D), (gate_type, fusion, t, layer, k.shape)
                    assert v.shape == (B, L // 2, D)
                # inference path
                out = m(None, layer=layer, cls_features=cls, task_id=t, cur_id=t,
                        train=False, p_type='Rainbow', alpha=a)
                k, v = out['batched_prompt']
                assert k.shape == (B, L // 2, D), (gate_type, fusion, t, layer, k.shape)
        print(f'{OK}  gate={gate_type:<10} fusion={fusion:<6} -> prompt {tuple(k.shape)}')

print('9b. theta is inert at q_layers=1 but live at q_layers=2')
for lq, want_grad in ((1, False), (2, True)):
    g = RelevanceGate('quantum', 32, n_qubits=4, q_layers=lq)
    a, _ = g(torch.randn(6, 32), torch.randn(4, 32))
    (a * torch.randn_like(a)).sum().backward()
    mag = float(g.qmap.theta.grad.abs().sum())
    assert (mag > 1e-4) == want_grad, (lq, mag)
    print(f'{OK}  q_layers={lq}: theta.grad |.| = {mag:.3e} ({"live" if want_grad else "inert"})')

print('9. backward through the full prompt path (gate=quantum, fusion=both)')
m = build('quantum', 'both')
cls = torch.randn(B, D)
alpha, _ = m.gate(cls, m.base_key[:3])
loss = 0
for layer in IDX:
    out = m(None, layer=layer, cls_features=cls, task_id=2, cur_id=2,
            train=True, p_type='Rainbow', alpha=alpha)
    loss = loss + out['batched_prompt'][0].pow(2).mean() + out['batched_prompt'][1].pow(2).mean()
loss.backward()
named = dict(m.named_parameters())
for key in ('gate.qmap.theta', 'gate.qmap.proj.weight', 'base_knowledge_0', 'base_key',
            'query_matcher.0.weight', 'fc1.0.weight'):
    g = named[key].grad
    assert g is not None and g.abs().sum() > 0, f'no gradient on {key}'
    print(f'{OK}  {key:<26} grad |.| = {g.abs().sum():.4e}')

print('10. buffers are populated after training passes')
assert m.stored_evolved.abs().sum() > 0, 'stored_evolved empty'
assert m.stored_rainbow_prompts.abs().sum() > 0, 'stored_rainbow_prompts empty (self-attn layers)'
print(f'{OK}  stored_evolved {tuple(m.stored_evolved.shape)}, '
      f'stored_rainbow_prompts {tuple(m.stored_rainbow_prompts.shape)}')

print('10b. task 0 is usable at eval: no zero prompts after the first task')
for fusion in ('infer', 'both'):
    m0 = build('quantum', fusion)
    cls = torch.randn(B, D)
    a0, _ = m0.gate(cls, m0.base_key[:1])
    for layer in IDX:  # train task 0 only
        m0(None, layer=layer, cls_features=cls, task_id=0, cur_id=0,
           train=True, p_type='Rainbow', alpha=a0)
    for layer in IDX:
        k, v = m0(None, layer=layer, cls_features=cls, task_id=0, cur_id=0,
                  train=False, p_type='Rainbow', alpha=a0)['batched_prompt']
        assert k.abs().sum() > 0, f'zero prompt at layer {layer} (fusion={fusion})'
        assert v.abs().sum() > 0, f'zero prompt at layer {layer} (fusion={fusion})'
    print(f'{OK}  fusion={fusion:<6} all {len(IDX)} layers non-zero after task 0')

print('11. baseline path is untouched (gate=mean reproduces argmax routing)')
torch.manual_seed(123)
base = build('mean', 'none')
cls = torch.randn(B, D)
for t in range(T):
    for layer in IDX:
        base(None, layer=layer, cls_features=cls, task_id=t, cur_id=t,
             train=True, p_type='Rainbow', alpha=None)
out = base(None, layer=0, cls_features=cls, task_id=1, cur_id=3, train=False, p_type='Rainbow')
k, v = out['batched_prompt']
assert k.shape == (B, L // 2, D)
assert (k[0] == k[1]).all(), 'baseline must give every sample in the batch the same prompt'
print(f'{OK}  all samples share one prompt, as in the original argmax routing')

print('\nALL TESTS PASSED')
