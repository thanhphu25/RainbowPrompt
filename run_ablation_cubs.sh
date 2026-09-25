#!/usr/bin/env bash
# Ablation of the relevance gate on CUB-200, 20 tasks.
# Mirrors Tab. 4 of the QKD paper: the same fusion mechanism is driven by four
# different task recognizers, so the quantum circuit is compared against its
# classical counterparts rather than against the original uniform average.
#
# A0 is the untouched RainbowPrompt baseline. Keep --seed fixed across the whole
# table, then re-run the winner with three seeds for the final numbers.
set -e

PY=${PY:-python}
SEED=${SEED:-10961}
DEV=${DEV:-cuda:0}
TAU=${TAU:-0.1}

COMMON="cub200_RainbowPrompt --model vit_base_patch16_224 --device ${DEV} --epochs 20 \
        --batch-size 32 --length 20 --top_k 1 --size 20 --D1 28 --D2 56 --warm_up 5 \
        --use_linear True --relation_type attention --balancing 0.01 \
        --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --self_attn_idx 0 1 2 3 4 5 \
        --num_tasks 20 --seed ${SEED}"

run () {  # run <tag> <extra args...>
    tag=$1; shift
    echo "===== ${tag} ====="
    ${PY} main.py ${COMMON} --output_dir ./out_CUBS20_${tag} --trial CUBS20_${tag} "$@"
}

# A0  original RainbowPrompt: uniform mean at training, argmax routing at test
run A0_baseline   --gate_type mean

# A1  is a learned routing mechanism worth anything at all?
run A1_cosine     --gate_type cosine    --fusion both --gate_tau ${TAU}

# A2/A3  classical recognizers of comparable capacity
run A2_mlp        --gate_type mlp       --fusion both --gate_tau ${TAU} --gate_hidden 64
run A3_attention  --gate_type attention --fusion both --gate_tau ${TAU} --gate_hidden 64

# A4  quantum gating (QGTM). q_layers must be >= 2, see quantum_gate.py
run A4_quantum    --gate_type quantum   --fusion both --gate_tau ${TAU} --n_qubits 8 --q_layers 2

# A5  + entropy sparsity on the relevance vector
run A5_quantum_ls --gate_type quantum   --fusion both --gate_tau ${TAU} --n_qubits 8 --q_layers 2 \
                  --lam_s 0.05

# A6  alpha used only for inference routing, trained by the auxiliary task loss.
#     Separates "better aggregation" from "better test-time routing".
run A6_inferonly  --gate_type quantum   --fusion infer --gate_tau ${TAU} --n_qubits 8 --q_layers 2 \
                  --lam_route 0.1
