# Quantum Full Adder

## What it does

Given oracles $U_a$ and $U_p$ encoding arrays $a$ and $p$ of length $2^n$ with $m$-bit values, prepare:

$$|\psi\rangle = \frac{1}{\sqrt{\sum_x a_x^2}} \sum_x a_x \cdot e^{i \cdot 2\pi \cdot p_x / 2^m} |x\rangle$$

## Running

```bash
source .venv/bin/activate
pip3 install -r requirements.txt
python test_state_prep.py
```

## Test Results

```bash
=== Oracle Tests ===
  PASS

=== Iteration Count Tests ===
  PASS

=== Small Exhaustive Tests (n=2, m=3) ===
  5 cases in 0.02s ... PASS  (avg target fidelity=0.991572)

=== Boundary Tests ===
  PASS  (avg target fidelity=1.000000)

=== Phase-Only Tests (uniform amplitude) ===
  4 cases in 0.00s ... PASS  (avg target fidelity=1.000000)

=== Amplitude-Only Tests (zero phase) ===
  4 cases in 0.00s ... PASS  (avg target fidelity=0.982239)

=== Random Tests ===
  n=2 m=3 (8 trials): PASS  (0.02s total, 0.003s/trial, avg target fidelity=0.988630)
  n=2 m=4 (6 trials): PASS  (0.02s total, 0.003s/trial, avg target fidelity=0.985736)
  n=3 m=3 (4 trials): PASS  (0.11s total, 0.028s/trial, avg target fidelity=0.979883)
  n=3 m=4 (3 trials): PASS  (0.08s total, 0.027s/trial, avg target fidelity=0.986394)

ALL TESTS PASSED
```