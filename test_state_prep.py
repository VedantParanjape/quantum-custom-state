import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from quantum_state_prep import build_oracle
from quantum_state_prep import compute_expected_state, compute_target_state, optimal_iterations, prepare_state, state_fidelity


FIDELITY_THRESHOLD = 0.99


#helpers
def verify_oracle(n, m, values):
    """Verify oracle maps |x⟩|0⟩ → |x⟩|values[x]⟩ for all x."""
    oracle = build_oracle(n, m, values)
    x_reg = QuantumRegister(n, "x")
    val_reg = QuantumRegister(m, "v")

    for x in range(2**n):
        prep = QuantumCircuit(x_reg, val_reg)
        for k in range(n):
            if (x >> k) & 1:
                prep.x(x_reg[k])
        full = prep.compose(oracle)
        sv = Statevector(full)
        probs = sv.probabilities_dict(decimals=10)
        result = max(probs, key=probs.get)
        assert probs[result] > 0.999, f"Not pure for x={x}"

        # Parse: val is the high bits, x is the low bits
        val_out = int(result[:m], 2)
        x_out = int(result[m:], 2)
        if x_out != x or val_out != values[x]:
            return False, f"x={x}: got x_out={x_out} val_out={val_out}, expected val={values[x]}"
    return True, ""


def check_state_prep(n, m, a, p, label=""):
    """Prepare state and check fidelity against expected."""
    result = prepare_state(n, m, a, p)
    expected = compute_expected_state(n, m, a, p)
    target = compute_target_state(n, m, a, p)
    fid = state_fidelity(result, expected)
    target_fid = state_fidelity(result, target)
    ok = fid >= FIDELITY_THRESHOLD
    if not ok:
        print(f"  FAIL {label} fidelity={fid:.6f} (threshold={FIDELITY_THRESHOLD})")
        print(f"    a={a}, p={p}")
        print(f"    result  = {np.round(result, 4)}")
        print(f"    expected= {np.round(expected, 4)}")
    return ok, fid, target_fid


#oracle tests
def test_oracle():
    print("=== Oracle Tests ===")
    ok = True

    # n=2, m=2
    for values in [[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]]:
        passed, msg = verify_oracle(2, 2, values)
        if not passed:
            print(f"  FAIL oracle n=2 m=2 values={values}: {msg}")
            ok = False

    # n=1, m=3
    for values in [[0, 7], [5, 3], [0, 0]]:
        passed, msg = verify_oracle(1, 3, values)
        if not passed:
            print(f"  FAIL oracle n=1 m=3 values={values}: {msg}")
            ok = False

    # n=3, m=2
    values = [0, 1, 2, 3, 3, 2, 1, 0]
    passed, msg = verify_oracle(3, 2, values)
    if not passed:
        print(f"  FAIL oracle n=3 m=2: {msg}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


#small exhaustive tests
def test_small_exhaustive():
    print("=== Small Exhaustive Tests (n=2, m=3) ===")
    n, m = 2, 3
    ok = True
    count = 0
    target_fids = []

    test_cases = [
        ([1, 2, 3, 4], [0, 0, 0, 0], "increasing amp, zero phase"),
        ([2, 2, 2, 2], [0, 2, 4, 6], "uniform amp, varying phase"),
        ([1, 3, 2, 4], [1, 0, 3, 2], "mixed amp and phase"),
        ([4, 4, 4, 4], [0, 0, 0, 0], "max amp, zero phase"),
        ([1, 1, 1, 1], [0, 0, 0, 0], "unit amp, zero phase"),
    ]

    t0 = time.time()
    for a, p, label in test_cases:
        passed, fid, tfid = check_state_prep(n, m, a, p, label)
        ok &= passed
        count += 1
        target_fids.append(tfid)

    elapsed = time.time() - t0
    avg_tfid = np.mean(target_fids)
    print(f"  {count} cases in {elapsed:.2f}s ... {'PASS' if ok else 'FAIL'}  (avg target fidelity={avg_tfid:.6f})")
    return ok


#boundary tests
def test_boundary():
    print("=== Boundary Tests ===")
    ok = True
    target_fids = []

    #Single non-zero amplitude
    n, m = 2, 3
    for idx in range(4):
        a = [0] * 4
        a[idx] = 3
        p = [0] * 4
        passed, fid, tfid = check_state_prep(n, m, a, p, f"single a[{idx}]=3")
        ok &= passed
        target_fids.append(tfid)

    #All equal amplitudes
    passed, fid, tfid = check_state_prep(2, 3, [2, 2, 2, 2], [0, 0, 0, 0], "uniform amp")
    ok &= passed
    target_fids.append(tfid)

    #Max valid amplitude (2^(m-1))
    passed, fid, tfid = check_state_prep(2, 3, [4, 4, 4, 4], [0, 0, 0, 0], "max amp")
    ok &= passed
    target_fids.append(tfid)

    # Single non-zero with phase
    a = [0, 0, 3, 0]
    p = [0, 0, 5, 0]
    passed, fid, tfid = check_state_prep(2, 3, a, p, "single with phase")
    ok &= passed
    target_fids.append(tfid)

    avg_tfid = np.mean(target_fids)
    print(f"  {'PASS' if ok else 'FAIL'}  (avg target fidelity={avg_tfid:.6f})")
    return ok


#phase-only tests
def test_phase_only():
    print("=== Phase-Only Tests (uniform amplitude) ===")
    ok = True
    target_fids = []

    #n=2, m=3: uniform amp, different phase patterns
    n, m = 2, 3
    amp = [3, 3, 3, 3]

    phase_patterns = [
        [0, 1, 2, 3],
        [0, 2, 4, 6],
        [7, 5, 3, 1],
        [0, 0, 4, 4],
    ]

    t0 = time.time()
    for p in phase_patterns:
        passed, fid, tfid = check_state_prep(n, m, amp, p, f"phases={p}")
        ok &= passed
        target_fids.append(tfid)

    elapsed = time.time() - t0
    avg_tfid = np.mean(target_fids)
    print(f"  {len(phase_patterns)} cases in {elapsed:.2f}s ... {'PASS' if ok else 'FAIL'}  (avg target fidelity={avg_tfid:.6f})")
    return ok


#amplitude-only tests
def test_amplitude_only():
    print("=== Amplitude-Only Tests (zero phase) ===")
    ok = True
    target_fids = []

    n, m = 2, 3
    zero_p = [0, 0, 0, 0]

    amp_patterns = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 4, 1, 4],
        [1, 1, 4, 4],
    ]

    t0 = time.time()
    for a in amp_patterns:
        passed, fid, tfid = check_state_prep(n, m, a, zero_p, f"amps={a}")
        ok &= passed
        target_fids.append(tfid)

    elapsed = time.time() - t0
    avg_tfid = np.mean(target_fids)
    print(f"  {len(amp_patterns)} cases in {elapsed:.2f}s ... {'PASS' if ok else 'FAIL'}  (avg target fidelity={avg_tfid:.6f})")
    return ok


#random tests
def test_random():
    print("=== Random Tests ===")
    rng = random.Random(42)
    ok = True

    configs = [
        (2, 3, 8),
        (2, 4, 6),
        (3, 3, 4),
        (3, 4, 3),
    ]

    for n, m, k in configs:
        max_a = 2**(m - 1)
        max_p = 2**m - 1
        t0 = time.time()
        passed_count = 0
        target_fids = []

        for trial in range(k):
            a = [rng.randint(1, max_a) for _ in range(2**n)]
            p = [rng.randint(0, max_p) for _ in range(2**n)]
            passed, fid, tfid = check_state_prep(n, m, a, p, f"n={n} m={m} trial={trial}")
            if passed:
                passed_count += 1
            ok &= passed
            target_fids.append(tfid)

        elapsed = time.time() - t0
        avg = elapsed / k
        avg_tfid = np.mean(target_fids)
        status = "PASS" if passed_count == k else "FAIL"
        print(f"  n={n} m={m} ({k} trials): {status}  ({elapsed:.2f}s total, {avg:.3f}s/trial, avg target fidelity={avg_tfid:.6f})")

    return ok


#iteration count test
def test_iteration_count():
    print("=== Iteration Count Tests ===")
    ok = True

    #High success probability → 0 iterations
    k = optimal_iterations(2, 3, [4, 4, 4, 4])
    if k != 0:
        print(f"  FAIL: expected 0 iterations for max amplitude, got {k}")
        ok = False

    #All zeros → 0 iterations
    k = optimal_iterations(2, 3, [0, 0, 0, 0])
    if k != 0:
        print(f"  FAIL: expected 0 iterations for all zeros, got {k}")
        ok = False

    #Low probability → positive iterations
    k = optimal_iterations(4, 4, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if k <= 0:
        print(f"  FAIL: expected positive iterations for sparse input, got {k}")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = True

    ok &= test_oracle()
    print()
    ok &= test_iteration_count()
    print()
    ok &= test_small_exhaustive()
    print()
    ok &= test_boundary()
    print()
    ok &= test_phase_only()
    print()
    ok &= test_amplitude_only()
    print()
    ok &= test_random()

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)
