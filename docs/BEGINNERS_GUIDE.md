# A Beginner's Guide to Our GPU QR Speed Challenge

This document explains, from scratch, what we're doing and why — no prior knowledge of
linear algebra or GPUs assumed. By the end you'll understand the problem, the tricks that
made our code fast, and the clever-sounding ideas that turned out to be dead ends.

---

## 1. The 30-second version

There's a public competition (run by a community called **GPU MODE**) where people write the
**fastest possible code** for a specific math task and submit it to a leaderboard. Our task is
called **`qr`**: take a big pile of square grids of numbers ("matrices") and compute their
**QR decomposition** as fast as possible on a specific, very powerful GPU (an NVIDIA **B200**).

We started by beating the previous best time (7.13 milliseconds). We're now at **~4.03 ms**,
which is about **11× faster than the standard library**, and we're pushing toward **2.5 ms**.

---

## 2. What is a matrix, and what is QR decomposition?

A **matrix** is just a grid of numbers — like a spreadsheet. A **square** matrix has the same
number of rows and columns (e.g., 512×512).

**QR decomposition** rewrites a matrix `A` as the product of two simpler matrices:

```
A  =  Q  ×  R
```

- **Q** is an *orthogonal* matrix — think of it as a clean set of perpendicular arrows
  (directions at right angles to each other, each of length 1). It represents a pure rotation/reflection.
- **R** is *upper-triangular* — everything below its diagonal is zero:

```
  R looks like this:        (the *'s are numbers, the 0's are exactly zero)
    *  *  *  *
    0  *  *  *
    0  0  *  *
    0  0  0  *
```

**Why anyone cares:** QR is one of the workhorses of scientific computing. It's used to solve
systems of equations, fit lines/curves to data (least squares), and as a building block inside
bigger algorithms. It shows up constantly, so making it fast is genuinely useful.

**How it's computed (the "Householder" method):** Imagine you want to make all the numbers
*below the diagonal* become zero, one column at a time. The Householder method does this with a
sequence of **reflections** — each reflection is like holding a mirror at just the right angle so
that a whole column of numbers collapses onto a single axis (zeroing out the rest). Do this column
by column and you've turned `A` into the triangular `R`, while the reflections together make up `Q`.

---

## 3. The exact thing we must output (and why it stops us from cheating)

Here's a subtle but important rule. The competition's checker does **not** ask for `Q` and `R`
directly. It asks for the **compact form** that standard libraries (LAPACK / PyTorch's
`torch.geqrf`) produce:

- **`H`** — the original matrix, but with the reflection vectors packed into the space below the
  diagonal (where the zeros of `R` would be). Nothing is wasted.
- **`tau`** — a short list of scalar numbers, one per reflection, needed to reconstruct `Q`.

This `(H, tau)` format is specific to the **Householder** method. There are *other* ways to compute
QR (Gram-Schmidt, Cholesky-QR, randomized methods) that are sometimes faster — but they produce
`Q` and `R` directly and **cannot** produce this `(H, tau)` format. So the output rule quietly
**forces us to use Householder.** No shortcuts around the algorithm itself.

Two more rules:
- **FP32 precision** — we must use 32-bit floating-point numbers (normal single precision).
- **Accuracy is checked every single run.** So we can't secretly use sloppy, low-precision math
  to go faster — the checker would catch the errors and reject us.

---

## 4. A crash course in "why is GPU code fast or slow?"

A **GPU** (graphics processing unit) is a chip that can do *thousands* of calculations at the same
time. It's fantastic when you need to do the *same* operation on *lots* of data — like running QR
on hundreds of matrices at once ("**batched**" work).

Two things govern speed, and beginners usually only think about the first:

1. **Compute** — the actual arithmetic. More number-crunching = more time.
2. **Overhead** — this is the sneaky one. Every time the CPU tells the GPU "do this one operation,"
   there's a fixed setup cost. We call each such operation a **kernel launch**. Think of it like
   sending a separate text message for every tiny errand: even if each errand is quick, the
   back-and-forth adds up. **Doing 100 tiny operations is often slower than 1 big operation**, even
   if the total math is identical.

A couple more terms you'll see below:
- **Kernel** = one GPU program / operation.
- **Shared memory** = a tiny but *extremely fast* scratchpad built into the GPU chip. Reading from
  it is far faster than reading from the GPU's main memory (which is large but comparatively slow).
  A huge amount of GPU optimization is just "**keep the data in the fast scratchpad and stop going
  back to slow memory.**"
- **Matrix multiply (matmul)** = the operation GPUs are *most* optimized for. Turning your work
  into big matmuls is usually a win.

---

## 5. The 7 test shapes — why one strategy can't win them all

The leaderboard scores us on **7 different shapes**. A "shape" = how big each matrix is (`n × n`)
and how many of them there are (the **batch**). They are deliberately very different:

| Matrix size | How many (batch) | Standard library | **Ours** | Speed-up |
|---|---|---|---|---|
| 32×32     | 20  | 0.33 ms   | **0.027 ms** | ~12× |
| 176×176   | 40  | 21.8 ms   | **0.63 ms**  | ~34× |
| 352×352   | 40  | 51.2 ms   | **1.63 ms**  | ~31× |
| 512×512   | 640 | 1073 ms   | **13.3 ms**  | ~81× |
| 1024×1024 | 60  | 240 ms    | **11.7 ms**  | ~21× |
| 2048×2048 | 8   | 77 ms     | 76.8 ms      | ~1× (tie) |
| 4096×4096 | 2   | 52 ms     | 52.2 ms      | ~1× (tie) |

Notice the extremes: **many small matrices** (640 at once) versus **a couple of giant ones** (just 2).
Those need completely different approaches. With lots of matrices we have tons of parallel work to
fill the GPU; with only 2 giant matrices there's almost no batch parallelism, and the standard
library (which is already excellent at single huge matrices) is very hard to beat. That's why we
**tie** on the two biggest shapes — we honestly can't beat NVIDIA's own library there yet, so we
just call it.

**How the final score works:** the leaderboard combines the 7 times with a **geometric mean**
(multiply them all, take the 7th root). Unlike a normal average, the geomean treats each shape's
*relative* improvement equally — so making a slow shape 2× faster matters just as much as making a
fast shape 2× faster. This is why we care about *every* shape, not just the slowest.

---

## 6. The optimizations that worked

Our baseline is `torch.geqrf` (PyTorch calling NVIDIA's general-purpose **cuSOLVER** library).
It's good, but it's a one-size-fits-all tool. Here's how we beat it.

### (a) Dispatch — use the right tool for each shape
Before doing anything, we **look at the shape** (matrix size and batch) and route to the best
strategy for that case:
- **Tiny matrices** (e.g., 32×32) → a special compact kernel.
- **Medium matrices with a decent batch** (176 up to 1024) → our custom fast kernel (the star of the show).
- **Giant matrices** (2048, 4096) → just use `torch.geqrf`, because we can't beat it there.

This is the single most important idea: *don't* try to win every case with one piece of code.
(Analogy: a good cook doesn't use the same knife for everything.)

### (b) Fused panel — keep the work on the fast scratchpad
We factor the matrix a **panel** at a time (a block of columns). The naive way performs each
column's reflection as a separate operation that reads and writes the slow main memory over and
over. That's death by a thousand memory trips.

Instead, we wrote a custom kernel (in **Triton**, a Python-like language for writing GPU kernels)
that **loads the whole panel into shared memory once**, does *all* the column steps right there in
the fast scratchpad, and writes the result back *once*. "**Fused**" means we merged many little
steps into one kernel — so it's also **one launch instead of dozens** (remember the overhead from
Section 4). This was one of our biggest wins.

### (c) Trisolve WY — turn a loop of tiny ops into one smart op
Householder reflections can be applied one at a time, but that's slow. There's a standard trick
called the **WY representation**: bundle a whole block of reflections into a compact form so that
applying them becomes a couple of big **matrix multiplies** (which GPUs love — Section 4).

The catch: *building* that bundle the obvious way needs a **loop of many tiny matrix multiplies**
— and each tiny multiply is its own kernel launch (lots of overhead). We found that the bundle can
instead be built with a **single triangular solve** (one standard operation), replacing the whole
loop. The result: roughly **10× fewer kernel launches**. Same math, far less overhead.

### (d) Fused subtract — combine three steps into one
After factoring a panel, we update the rest of the matrix with the formula:

```
new_block  =  old_block  −  Y × W      (a matrix multiply, then a subtraction)
```

The naive way is **three** separate operations: (1) do the multiply, (2) do the subtraction,
(3) copy the answer back into place — three kernels, three sets of overhead, plus extra memory traffic.

PyTorch has a single operation called **`baddbmm`** that does "**multiply, then subtract, in place**"
all at once. Switching to it **deleted two whole kernels and a copy**. This was our most recent
improvement (we call it **v19**) and it took us from **4.73 ms → 4.03 ms**.

---

## 7. Ideas that did NOT work out (and why that's still valuable)

A big part of optimization is *ruling things out with evidence*. Here are the promising-sounding
ideas we tried and rejected — each one taught us something about why our current design is good.

### ❌ BF16x9 — "exact fast math" that didn't fit our shapes
Modern GPUs have **tensor cores**: special hardware that does matrix multiplies *very* fast, but
normally in low precision (which we're not allowed to use — Section 3). There's a clever trick
("**BF16x9**", a form of Ozaki/error-correction) that runs **9 low-precision passes** and combines
them to get a result that is **bit-for-bit identical to true FP32** — fast *and* accurate. It
sounded like a perfect fit.

We got it working (it really is exact, ~2× faster on the right inputs). **But** it only pays off
when the matrix multiplies are "**fat**" (wide blocks of work). Our design uses **narrow** blocks
(see the next point for why), and on narrow blocks BF16x9 is actually *slower*. On the giant
matrices, where the multiplies *are* fat, the time is dominated by a different part of the algorithm
(the panel), so speeding up the multiply barely helped (~1.5%). **Conclusion: a real, measured dead
end for our specific shapes** — useful to know, so we stop chasing it.

### ❌ Wider blocks — broke the thing that made us fast
The obvious response to "BF16x9 needs fat blocks" is: *make the blocks wider!* We tried. But a wide
panel **no longer fits in the fast shared-memory scratchpad** (Section 4). So the kernel was forced
to keep going back to slow main memory on every step — and the panel got **~5× slower**. This was an
important confirmation: our current **narrow-block, fully-on-chip** design is the **sweet spot**, not
an accident. Pushing one knob (block width) just breaks another (staying on-chip).

### ❌ Sloppy precision / reshaping the problem
Tempting beginner ideas that the rules forbid: using low precision to go fast (the accuracy checker
rejects it), or swapping in a different QR algorithm (the `(H, tau)` output format forbids it —
Section 3).

### 🔬 Currently being explored (jury still out)
Because the easy wins are exhausted, reaching 2.5 ms needs harder, structural changes. Three
experiments are running in parallel right now:
1. A **"megakernel"** — fuse the *entire* factorization into one giant kernel that uses tensor cores
   internally, so the matrix multiply never becomes a slow, narrow, separate operation. (The hardest
   idea, but the most likely path to 2.5 ms — it's what the top competitors do.)
2. **Attacking the two giant shapes** (2048, 4096) — where the multiplies *are* fat, so the exact
   BF16x9 trick might finally win if we can also speed up the panel.
3. A **faster panel** — careful tuning of the existing on-chip kernel for a safe, smaller gain.

---

## 8. Where we stand

- **Current best: ~4.03 ms**, about **11× faster** than the standard library, live on the leaderboard.
- We already beat the original target (7.13 ms) by a comfortable margin.
- The next goal is **2.5 ms**. It's known to be possible (about a dozen people have done it), but it's
  no longer a quick tweak — it needs the kind of structural rewrite described above.

---

## 9. Mini-glossary

- **Matrix** — a grid of numbers.
- **QR decomposition** — rewriting a matrix as a clean rotation (`Q`) times a triangular matrix (`R`).
- **Householder reflection** — the "mirror" trick used to zero out numbers below the diagonal.
- **`(H, tau)`** — the compact, packed way of storing the answer that the competition requires.
- **GPU / B200** — the massively-parallel chip we run on.
- **Batch** — doing the same operation on many matrices at once.
- **Kernel** — one GPU operation/program. **Kernel launch** — the (costly) act of starting one.
- **Shared memory** — the GPU's small, very fast on-chip scratchpad.
- **Triton** — a Python-like language for writing custom GPU kernels.
- **Matmul** — matrix multiply, the operation GPUs are most optimized for.
- **Tensor cores** — special GPU hardware for ultra-fast (normally low-precision) matrix multiplies.
- **FP32** — standard 32-bit floating-point precision (what we're required to use).
- **Geometric mean** — the way the 7 shape-times are combined into one score.
- **Dispatch** — choosing different code paths based on the input shape.
