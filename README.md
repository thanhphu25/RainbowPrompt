## RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning [ICCV 2025]

**[RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_RainbowPrompt_Diversity-Enhanced_Prompt-Evolving_for_Continual_Learning_ICCV_2025_paper.pdf)**  
*Kiseong Hong, Gyeong-hyeon Kim, Eunwoo Kim*  
IEEE/CVF International Conference on Computer Vision (ICCV), 2025


![RainbowPrompt Overview](./Overview.png)


## Abstract
Prompt-based continual learning provides a rehearsal-free solution by tuning small sets of parameters while keeping pre-trained models frozen. To meet the complex demands of sequential tasks, it is crucial to integrate task-specific knowledge within prompts effectively. However, existing works rely on either fixed learned prompts (i.e., prompts whose representations remain unchanged during new task learning) or on prompts generated from an entangled task-shared space, limiting the representational diversity of the integrated prompt. To address this issue, we propose a novel prompt-evolving mechanism to adaptively aggregate base prompts (i.e., task-specific prompts) into a unified prompt while ensuring diversity. By transforming and aligning base prompts, both previously learned and newly introduced, our approach continuously evolves accumulated knowledge to facilitate learning new tasks. We further introduce a learnable probabilistic gate that adaptively determines which layers to activate during the evolution process. We validate our method on image classification and video action recognition tasks in class-incremental learning, achieving average gains of 9.07% and 7.40% over existing methods across all scenarios.

---

## Environment & Setup

```bash
conda create -n rainbow python=3.9 -y
conda activate rainbow

pip install -r requirements.txt
```

> **Note**  
> All experiments were conducted under this environment.  
> Minor version differences may lead to slightly different results.

---

## Training

```bash
# CIFAR-100 (10 tasks & 20 tasks)
bash run_cifar100.sh

# ImageNet-R (10 tasks & 20 tasks)
bash run_imr.sh

# CUB-200-2011 (10 tasks & 20 tasks)
bash run_cubs.sh
```

> **Note**  
> For each setting, we conducted **three independent runs with different random seeds**  
> and reported the **average performance** in the paper.  
> The default seed provided in this repository corresponds to **one of the seeds used**.

---

## Results

- **Training logs** (e.g., loss, accuracy, task-wise performance) are saved under the `logs/` directory.
- **Model checkpoints** are stored in the directory specified by the `output_dir` argument in the parser.

---

## Quantum-gated relevance (branch `QUANT`)

This branch adds an optional relevance gate between the accumulated base prompts
and the incoming sample, ported from the QGTM module of
*Li et al., "Quantum-Gated Task-interaction Knowledge Distillation for PTM-based
Class-Incremental Learning", CVPR 2026*.

In the original method the evolved prompts are combined by a uniform average
(Eq. 5) and, at test time, a single task is picked by an argmax over key
similarity. Both are replaced here by coefficients `alpha_i` that measure how
relevant each accumulated task is to the current sample:

    p_i    = |<psi(h; theta) | phi(e_i; theta)>|^2      fidelity in a 2^q Hilbert space
    alpha  = softmax(p / tau)
    prompt = sum_i alpha_i * evolved_prompt_i

`psi` and `phi` are produced by a shallow R_y / CNOT circuit, simulated exactly
in [`quantum_gate.py`](quantum_gate.py). The circuit only contains real-valued
gates, so the state vector stays real and autograd flows through it without a
QML runtime.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--gate_type` | `mean` | `mean` (original), `cosine`, `mlp`, `attention`, `quantum` |
| `--fusion` | `both` | `both` = weighted aggregation + routing, `infer` = routing only, `none` = original |
| `--n_qubits` | `8` | width of the quantum feature map |
| `--q_layers` | `2` | repetitions of the variational block, **must be >= 2** (see below) |
| `--gate_tau` | `1.0` | softmax temperature; `0.1` is a far better starting point for 20 tasks |
| `--gate_hidden` | `64` | hidden width of the `mlp` / `attention` gates |
| `--lam_s` | `0.0` | weight of the entropy sparsity term |
| `--lam_route` | `0.0` | auxiliary task-routing loss, required when `--fusion infer` |

`--gate_type mean` leaves every original code path untouched, so the baseline
stays reproducible from this branch.

### Two departures from the paper

* **`L_s = ||alpha||_1` (Eq. 11) has no gradient.** `alpha` is a softmax output,
  so its L1 norm is identically 1. `--lam_s` applies the entropy of `alpha`
  instead, which expresses the same intent and is differentiable.
* **`q_layers = 1` makes `theta` inert.** Because `R_y(a) R_y(b) = R_y(a + b)`,
  a single repetition lets the variational angles cancel out of the fidelity
  exactly, leaving them with zero gradient. Only the entangling chain between
  repetitions breaks the cancellation, hence the `>= 2` requirement.

### Running

```bash
python test_quantum_gate.py     # simulator vs brute-force circuit matrix, wiring checks
bash run_ablation_cubs.sh       # A0 baseline .. A6, one output_dir per variant
```

---

## Acknowledgement

This repository is built upon the codebase of **[DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch)**. 
We thank the authors for their valuable research and for making their code publicly available.

---

## Citation

If you found our work useful for your research, please cite our work:

```bibtex
@InProceedings{Hong_2025_ICCV,
    author    = {Hong, Kiseong and Kim, Gyeong-hyeon and Kim, Eunwoo},
    title     = {RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1130-1140}
}
```

---

