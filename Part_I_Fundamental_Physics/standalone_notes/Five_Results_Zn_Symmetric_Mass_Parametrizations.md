# Five Results on Zₙ-Symmetric Mass Parametrizations

Steve Bico Mujjabi, MD · Independent researcher · Kampala, Uganda

## Status of this document

Expanded from an earlier two-result note. This is an elementary algebra and geometry
note, not evidence for a physical model. On 2026-08-20, Result 1 was corrected: positive
values of the squared masses do not imply positive amplitudes, so the original square-root
step was invalid on part of its stated domain. Results 1--4 are known or immediate
consequences of roots-of-unity identities and a stated modelling axiom. Result 5 is a
conditional geometric lemma that still needs an independent mathematical check and a
novelty review before it is described as publishable. Nothing here claims any connection
to the physical fine-structure constant, vacuum structure, cosmology, or biology — see
the closing note.

---

## Setup, used throughout

For integer n ≥ 2, real θ, and c > 0, define n amplitudes

Aₖ = 1 + c·cos(θ + 2πk/n),  k = 0, ..., n−1,  and masses mₖ = M·Aₖ², M > 0.

Two Fourier identities recur throughout:

**(F1)** Σₖ cos(θ + 2πk/n) = 0, for every n ≥ 2 and every θ.
*Proof:* real part of e^(iθ)Σₖ e^(2πik/n); the sum of n-th roots of unity vanishes for
every n ≥ 2.

**(F2)** Σₖ cos²(θ + 2πk/n) = n/2, for every n ≥ 3 (or n=1) and every θ; **fails** for n=2.
*Proof:* cos²x = 1/2+1/2cos(2x); apply (F1)-type reasoning to the doubled angles, which
requires e^(4πi/n) ≠ 1, i.e. n ∤ 2 — true for every n except n=2. (This is Result 2 below,
stated here because Results 3–5 depend on it.)

---

## Result 1 — Koide's Relation Holds Exactly for the Brannen Form on Its Positive-Amplitude Domain (n=3, c=√2)

**Theorem.** For all M>0 and theta such that Aₖ > 0 for every k,
Q = Σmₖ / (Σ√mₖ)² = 2/3, identically.

*Proof.* Using (F1), (F2) at n=3: Σmₖ = M·Σ(1+2√2cosφₖ+2cos²φₖ) = M(3+0+3) = 6M.
Because Aₖ > 0, √mₖ = √M·Aₖ. Hence Σ√mₖ = √M·Σ(1+√2cosφₖ) = √M(3+0) = 3√M.
So Q = 6M/9M = 2/3. ∎

**Why the positivity condition is necessary.** The weaker condition mₖ > 0 only says
Aₖ != 0; it does not prevent Aₖ < 0, whereas √(Aₖ²) = |Aₖ|. For example, at M=1,
theta=pi, and c=√2, the three amplitudes are approximately
(-0.414214, 1.707107, 1.707107). All three masses are positive, but
Q is approximately 0.409365 rather than 2/3. Result 5 gives the relevant open
positive-amplitude domain when c=√2.

*Attribution:* this construction is due to C.A. Brannen (2010), building on Y. Koide's
1983 empirical mass relation. This document restates the proof; it does not claim
priority for the construction itself.

---

## Result 2 — The Fourier Sum Rule (F2) Fails Uniquely at n=2

Stated and proved above as (F2). n=2 is the sole integer ≥2 for which the amplitude
sum-of-squares fails to be θ-independent — an unconditional, clean algebraic exclusion.

---

## Result 3 — The Universal Mass Sum Rule (any n≥3, any θ, any c=√2)

**Theorem.** With c=√2, for every n≥3 and every θ: Σₖ mₖ = 2nM, independent of θ.

*Proof.* Σₖ Aₖ² = Σₖ(1 + 2√2cosφₖ + 2cos²φₖ) = n + 2√2·0 + 2·(n/2) = 2n, using (F1) and
(F2). So Σmₖ = M·Σ Aₖ² = 2nM. ∎

Note this generalizes Result 1's numerator computation to arbitrary n — the total mass
of any such n-fold family is always exactly 2nM, regardless of orientation.

---

## Result 4 — The Coefficient c=√2 Is Forced by a Stated Variance Condition

This result is **conditional** on adopting the following as a modeling axiom — it is not
derived from anything more fundamental, and should be read as "if you require this
property, then c=√2 follows," not as "c=√2 is somehow necessary."

**Axiom (stated, not derived):** the amplitude distribution satisfies Var(Aₖ) = ⟨Aₖ⟩²
(coefficient of variation equal to 1) over the n modes, for a given n≥3.

**Theorem.** Given the axiom, c=√2 uniquely (positive root).

*Proof.* By (F1), ⟨Aₖ⟩ = (1/n)Σ(1+c·cosφₖ) = 1. By (F2), ⟨Aₖ²⟩ = (1/n)Σ(1+2c·cosφₖ+c²cos²φₖ)
= 1 + c²/2. So Var(Aₖ) = ⟨Aₖ²⟩−⟨Aₖ⟩² = c²/2. Setting this equal to ⟨Aₖ⟩²=1: c²/2=1, c=√2
(taking the positive root). ∎

**Honest caveat:** the axiom itself (why coefficient of variation should equal 1) is a
modeling choice presented in the earlier manuscript series as a "maximum entropy"
condition, without derivation from any more basic physical principle. This proof shows
the axiom *implies* c=√2 correctly; it does not show the axiom is true of anything in
nature.

---

## Result 5 — n=3 Is the Unique n≥3 with an Open Domain of Strictly Positive Amplitudes (general proof)

An earlier version of this claim (in the larger manuscript series this note is drawn
from) was checked only numerically, case by case, for small n. Here is a full, general
proof for all n≥3 at once.

**Theorem.** With c=√2 fixed (Result 4), there exists an open interval of θ for which
Aₖ>0 for all k=0,...,n−1 **if and only if n=3**.

*Proof.* Aₖ>0 ⟺ cos(θ+2πk/n) > −1/√2 ⟺ φₖ = θ+2πk/n (mod 2π) lies outside the closed
arc F = [3π/4, 5π/4], which has angular width π/2. Equivalently, all n points {φₖ} must
lie inside the complementary "safe" arc S, of width 2π − π/2 = 3π/2.

The n points φ₀,...,φₙ₋₁ are equally spaced with common spacing 2π/n; they partition the
circle into n gaps, each of width exactly 2π/n. Any single arc that avoids all n points
must be contained within one such gap; conversely, the *minimal* arc containing all n
points (i.e., the complement of the single largest empty gap) has width exactly
2π − 2π/n = 2π(n−1)/n — and since all n gaps are equal, this minimal width does not
depend on θ, only its rotational position does.

For all n points to fit inside S (width 3π/2) with room to spare (a genuine open
interval of valid θ, not a single boundary point), we need the strict inequality:

2π(n−1)/n < 3π/2  ⟺  n < 4.

Combined with n≥3, this forces n=3 exactly. At n=4, the inequality becomes equality
(2π(3)/4 = 3π/2), which is a single degenerate configuration (two amplitudes sit exactly
at zero, not strictly positive, and only for one specific θ, not an open interval) — so
n=4 is excluded as well, cleanly. For n≥5 the minimal-span requirement already exceeds
the safe zone, so no valid θ exists at all — consistent with, and generalizing, what the
earlier manuscript series found by direct numerical search at n=5,7,9,11. ∎

This replaces a case-by-case numerical check in the earlier manuscript series with a
closed-form all-n proof. That is an internal improvement, not a claim of mathematical
novelty or publication priority.

---

## What is not claimed

None of the above establishes: why these particular values of n, c, or the amplitude
form itself should describe anything physical; why nature (if it does at all) would
select n=3 specifically over simply not instantiating this structure; or any connection
to the fine-structure constant, vacuum medium physics, cosmology, or any domain outside
pure algebra. An earlier, much larger manuscript series attempted several of those
physical extensions and did not survive independent scrutiny — see the companion audit
document for the full account of what was tested and what happened.

## Publication status

This note is not ready to submit as a mathematics paper. Results 1--4 are elementary
identities or conditional algebra and do not establish a new mathematical contribution.
Result 5 may be worth an independent check only after a literature/priority search and
after it is presented separately from the physical manuscript series. See
`MATHEMATICAL_TRIAGE_2026-08-20.md` for the current decision record.

## References

- Koide, Y. *Phys. Rev. D* 28, 252 (1983).
- Brannen, C.A. *Found. Phys.* 40, 1681–1699 (2010).
- Workman, R.L. et al. (PDG). *PTEP* 2022, 083C01 (2022).
