# Plan: Fix Topological Charge Definition

## Context

Comparing our `topo_charge()` in `src/observables.cpp` against the Qlattice reference implementation in `/thfs1/home/fengxu/zzl/Qlattice/`. The goal is to make our definition match Qlattice exactly.

## Qlattice Reference (correct definition)

### Step 1: Clover leaf (`gf_clover_leaf_no_comm`)

For each (mu, nu) pair, compute the **sum of 4 oriented plaquettes** (the clover), then multiply by 0.25:

```
C_mu_nu(x) = (1/4) * [ P(mu,nu) + P(-mu,-nu) + P(nu,-mu) + P(-nu,mu) ]
```

Where the 4 paths are:
1. `(+mu, +nu, -mu, -nu)` — forward-forward
2. `(-mu, -nu, +mu, +nu)` — backward-backward
3. `(+nu, -mu, -nu, +mu)` — right then back
4. `(-nu, +mu, +nu, -mu)` — left then forward

This gives a matrix C that is **NOT** anti-Hermitianized yet.

### Step 2: Anti-Hermitian projection (`clf_topology_density`)

```cpp
arr[i] = 0.5 * (C[i] - C[i]†)    // anti-Hermitian part, NO traceless projection
```

Note: **No** trace subtraction here. Just `(C - C†)/2`.

### Step 3: Topological charge density

```cpp
fac = -1/(4*pi^2)
q(x) = fac * [ -Tr(arr[1]*arr[4]) + Tr(arr[2]*arr[3]) + Tr(arr[5]*arr[0]) ]
```

Where the index mapping is:
- arr[0] = F_01, arr[1] = F_02, arr[2] = F_03
- arr[3] = F_12, arr[4] = F_13, arr[5] = F_23

So: `q = -1/(4*pi^2) * [ -Tr(F02*F13) + Tr(F03*F12) + Tr(F23*F01) ]`

### Total Q:
```
Q = sum_x q(x)    // no extra factor of 8
```

---

## Our Current Code: Problems Found

### Bug 1: Clover leaf construction uses wrong paths

Our `clover_F()` (line 36-79) manually constructs 4 plaquettes, but with **incorrect paths** for leaves 2-4. Specifically:

- Leaf 2 (line 54-57): The path `U_nu†(s-nu) * U_mu(s-nu) * U_nu(s-nu+mu) * U_mu†(s)` is wrong — this is `(-nu, +mu, +nu, -mu)`, but should be `(+nu, -mu, -nu, +mu)`.
- Leaf 3 and 4 have similar ordering errors.

The Qlattice uses Wilson lines along specific direction sequences, which automatically handles the path ordering correctly.

### Bug 2: `F = (Q - Q†) * (-i/8)` mixes anti-Hermitian projection with normalization incorrectly

Our code computes: `F = (Q - Q†) * (-i/8)` where Q is the sum of 4 plaquettes.

Qlattice computes:
1. `C = (1/4) * sum(4 plaquettes)` — averaged clover
2. `arr = (1/2) * (C - C†)` — anti-Hermitian part

So Qlattice's `arr = (1/2) * ((1/4)*Q - (1/4)*Q†) = (Q - Q†)/8`.

Our code: `F = (Q - Q†) * (-i/8)`.

**The extra factor of `-i` is wrong.** Qlattice does NOT multiply by `-i`. The field strength should be `(Q - Q†)/8`, which is already anti-Hermitian. Multiplying by `-i` makes it Hermitian, which changes the trace products.

### Bug 3: Normalization factor

Our code uses: `Q = (1/(16*pi^2)) * 8 * sum[...]`

Qlattice uses: `Q = (-1/(4*pi^2)) * sum[...]`

These should be equivalent: `-1/(4*pi^2) = -4/(16*pi^2)`. But our code uses `+8/(16*pi^2) = +1/(2*pi^2)`, which is **wrong by a factor of -2**.

### Bug 4: Epsilon tensor sign convention

Our signs: `+Tr(F01*F23) - Tr(F02*F13) + Tr(F03*F12)`

Qlattice signs: `-Tr(F02*F13) + Tr(F03*F12) + Tr(F23*F01)`

Rewriting Qlattice: `+Tr(F01*F23) - Tr(F02*F13) + Tr(F03*F12)` — **same signs**, just different order. This part is correct.

---

## Summary of Bugs

| Issue | Our Code | Qlattice (correct) | Impact |
|-------|----------|-------------------|--------|
| Extra `-i` in F | `(Q-Q†)*(-i/8)` | `(Q-Q†)/8` | Changes trace products completely |
| Normalization | `+8/(16*pi^2)` | `-1/(4*pi^2) = -4/(16*pi^2)` | Wrong by factor -2 |
| Clover paths | Manual, some wrong | Wilson line sequences | Some leaves have wrong link ordering |

## Fix

Rewrite `topo_charge()` to match Qlattice exactly:

1. Compute 6 clover leaves `C[i]` using correct 4-plaquette paths with `*0.25`
2. Anti-Hermitianize: `A[i] = 0.5 * (C[i] - C[i]†)` — **no `-i`, no trace subtraction**
3. `q = -1/(4*pi^2) * [-Tr(A[1]*A[4]) + Tr(A[2]*A[3]) + Tr(A[5]*A[0])]`
4. `Q = sum_x q(x)` — **no extra factor of 8**

### Files to modify
- `src/observables.cpp`: rewrite `clover_F()` and `topo_charge()`

### Verification
- On a cold (unit) config: Q should be exactly 0
- After gradient flow on a thermalized config: Q should approach an integer
