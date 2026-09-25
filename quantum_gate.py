# ------------------------------------------
# Quantum-gated relevance for RainbowPrompt.
#
# Implements the QGTM module of
#   Li et al., "Quantum-Gated Task-interaction Knowledge Distillation for
#   Pre-trained Model-based Class-Incremental Learning", CVPR 2026,
# adapted from adapter-based CIL to the prompt-evolving setting.
#
# Classical counterparts (cosine / mlp / attention) are provided in the same
# interface so that the ablation of Tab. 4 in that paper can be reproduced from
# a single code path.
# ------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

GATE_TYPES = ('mean', 'cosine', 'mlp', 'attention', 'quantum')


def _cnot_perm(n_qubits, ctrl, tgt):
    """Index permutation realising CNOT(ctrl -> tgt) on a state vector.

    CNOT is an involution, so the same permutation applies in both directions
    and psi[perm] is the transformed state.
    """
    idx = torch.arange(2 ** n_qubits, dtype=torch.long)
    ctrl_bit = (idx >> (n_qubits - 1 - ctrl)) & 1
    return torch.where(ctrl_bit.bool(), idx ^ (1 << (n_qubits - 1 - tgt)), idx)


class QuantumFeatureMap(nn.Module):
    """Shallow parameterised circuit U(v; theta) of Eq. (5)-(9).

        U(v; theta) = prod_l [ U_ent . U_var^(l)(theta) . U_enc^(l)(v) ]

    The circuit uses only R_y rotations and CNOT gates, so every amplitude stays
    real: the state vector is simulated with plain real tensors, autograd flows
    through it directly and no QML runtime is needed.

    Note on n_layers: theta is INERT when n_layers == 1. Since R_y(a) R_y(b) =
    R_y(a + b), a single repetition gives

        <psi(h)|phi(s)> = <0| R_y(th + a_h)^d U_ent^d U_ent R_y(th + a_s) |0>
                        = <0| R_y(a_s - a_h) |0>,

    so theta cancels exactly and receives zero gradient. The entangling chain
    only breaks this cancellation between repetitions, so n_layers >= 2 is
    required for the variational part of Eq. (8) to do anything at all.
    """

    def __init__(self, in_dim, n_qubits=8, n_layers=2):
        super().__init__()
        assert n_qubits >= 2, 'need at least 2 qubits for the entangling chain'
        if n_layers < 2:
            logger.warning(
                'QuantumFeatureMap: q_layers=1 makes the variational angles cancel out of '
                'the fidelity and receive zero gradient; use q_layers >= 2 unless this is '
                'a deliberate ablation.')
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # classical feature -> q-dimensional angle vector.
        # The input arrives L2-normalised (unit norm over in_dim), so each
        # component is O(1/sqrt(in_dim)). Feeding that straight into a
        # default-initialised Linear yields angles ~ 0, the circuit collapses to
        # the identity and every fidelity degenerates to 1. Rescaling to unit
        # component variance keeps the encoded angles spread over the sphere.
        self.in_scale = math.sqrt(in_dim)
        self.proj = nn.Linear(in_dim, n_qubits)
        # variational rotation angles, updated jointly with the classical parts
        self.theta = nn.Parameter(torch.rand(n_layers, n_qubits) * 2 * math.pi)

        self.register_buffer(
            'perms',
            torch.stack([_cnot_perm(n_qubits, j, j + 1) for j in range(n_qubits - 1)]),
            persistent=False,
        )

    def _apply_ry(self, psi, angle, j):
        """R_y(angle) on qubit j. angle is (B,) or broadcastable to it."""
        B = psi.shape[0]
        left, right = 2 ** j, 2 ** (self.n_qubits - 1 - j)
        psi = psi.view(B, left, 2, right)
        half = angle / 2
        c = torch.cos(half).view(-1, 1, 1)
        s = torch.sin(half).view(-1, 1, 1)
        a0, a1 = psi[:, :, 0, :], psi[:, :, 1, :]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], dim=2).reshape(B, -1)

    def forward(self, v):
        """(B, in_dim) -> (B, 2 ** n_qubits) unit-norm state vector."""
        B = v.shape[0]
        angles = torch.tanh(self.proj(v * self.in_scale)) * math.pi  # rotations in [-pi, pi]

        psi = v.new_zeros(B, 2 ** self.n_qubits)
        psi[:, 0] = 1.0  # |0...0>

        for l in range(self.n_layers):
            for j in range(self.n_qubits):  # U_enc, Eq. (5)
                psi = self._apply_ry(psi, angles[:, j], j)
            for j in range(self.n_qubits):  # U_var, Eq. (6)
                psi = self._apply_ry(psi, self.theta[l, j].expand(B), j)
            for j in range(self.n_qubits - 1):  # U_ent, Eq. (7)
                psi = psi[:, self.perms[j]]
        return psi


class RelevanceGate(nn.Module):
    """Sample-to-task relevance coefficients alpha, shape (B, T).

    gate_type:
        cosine     plain cosine similarity
        mlp        one shared encoder on both branches, then inner product
        attention  separate W^Q / W^K projections, scaled dot product
        quantum    fidelity in the 2**q-dimensional Hilbert space (QGTM)

    'mean' is handled by the caller, which keeps the original uniform-average
    path untouched so that the baseline stays bit-for-bit reproducible.
    """

    def __init__(self, gate_type, in_dim, n_qubits=8, q_layers=2, tau=1.0, hidden=64):
        super().__init__()
        if gate_type not in GATE_TYPES:
            raise ValueError(f'unknown gate_type {gate_type}, expected one of {GATE_TYPES}')
        self.gate_type = gate_type
        self.tau = tau

        if gate_type == 'quantum':
            self.qmap = QuantumFeatureMap(in_dim, n_qubits, q_layers)
        elif gate_type == 'mlp':
            self.enc = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        elif gate_type == 'attention':
            self.wq = nn.Linear(in_dim, hidden, bias=False)
            self.wk = nn.Linear(in_dim, hidden, bias=False)
            self.scale = hidden ** -0.5

    def scores(self, sample_feat, task_feats):
        """(B, D) x (T, D) -> (B, T) unnormalised relevance."""
        h = F.normalize(sample_feat, dim=-1)
        s = F.normalize(task_feats, dim=-1)
        if self.gate_type == 'quantum':
            return (self.qmap(h) @ self.qmap(s).t()) ** 2  # Eq. (10), fidelity
        if self.gate_type == 'cosine':
            return h @ s.t()
        if self.gate_type == 'mlp':
            return self.enc(h) @ self.enc(s).t()
        return (self.wq(h) @ self.wk(s).t()) * self.scale

    def forward(self, sample_feat, task_feats):
        p = self.scores(sample_feat, task_feats)
        alpha = F.softmax(p / self.tau, dim=-1)  # Eq. (12)
        return alpha, p


def entropy_sparsity(alpha, eps=1e-8):
    """Sparsity term on the relevance vector.

    Eq. (11) of the paper defines L_s = ||alpha||_1, but alpha is a softmax
    output, so its L1 norm is identically 1 and carries no gradient. We use the
    entropy instead, which expresses the stated intent -- concentrate on the
    most relevant past tasks -- and is differentiable.
    """
    return -(alpha * (alpha + eps).log()).sum(-1).mean()
