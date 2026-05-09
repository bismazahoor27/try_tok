# Mathematical Toolkit for Evaluating Chemical Tokenizers

## Notation (fixed throughout)

| Symbol | Meaning |
|---|---|
| $\mathcal{C}$ | Corpus — sequence of $N$ tokens $t_1, \dots, t_N$ produced by tokenizer $\mathcal{T}$ |
| $\mathcal{V}$ | Vocabulary with $\|\mathcal{V}\| = V$ |
| $c(t)$ | Unigram count of token $t$ in corpus |
| $p(t) = c(t)/N$ | Unigram probability of token $t$ |
| $M$ | Number of molecules in corpus |
| $L_m$ | Number of tokens molecule $m$ produces |
| $A_m$ | Number of heavy atoms in molecule $m$ |
| $\log$ | Base-2 unless stated; entropies are in **bits** |

---

## 1. Unigram (Shannon) Entropy

$$H_1(\mathcal{T}) = -\sum_{t \in \mathcal{V}} p(t)\,\log_2 p(t)$$

### What it measures
The average number of bits needed to encode a single token if you assume tokens are drawn independently from the unigram distribution. It is the classic measure of "spread" or "uncertainty" in the token distribution. A high $H_1$ means the corpus uses many tokens roughly equally; a low $H_1$ means a few tokens dominate.

### Bounds
$$0 \le H_1 \le \log_2 V$$

- $H_1 = 0$ — degenerate: one token covers the entire corpus (zero information content per draw because there is no surprise).
- $H_1 = \log_2 V$ — uniform: every token is equally probable (maximum possible information per draw).

### Good vs bad for a chemical tokenizer

| Situation | What it means |
|---|---|
| $H_1 \approx \log_2 V$ | Vocabulary is well-utilised. Every token in $\mathcal{V}$ carries unique information. This is the *ideal* case. |
| $H_1 \ll \log_2 V$ | Most tokens are rare; a few tokens (e.g. single-atom linkers) dominate the corpus. The vocabulary budget is being wasted. BPE with aggressive `min_freq` often causes this. |
| $H_1$ decreases after merging linker runs | Merging creates very frequent super-tokens that dominate, reducing spread. Whether this is good depends on downstream use. |
| $H_1$ is the *same* across two tokenizers at different $V$ | The larger-$V$ tokenizer is less efficient — same average information but more parameters. |

### Estimator bias
The plug-in estimator $\hat{H}_1 = -\sum \hat{p}(t)\log\hat{p}(t)$ is negatively biased:
$$\mathbb{E}[\hat{H}_1] - H_1 \approx -\frac{V-1}{2N\ln 2} \quad \text{(Miller–Madow correction)}$$

**Fix:** use $\hat{H}_1 + (V-1)/(2N\ln 2)$. If your corpus is very small relative to $V$ (e.g. during prototyping), use the Nemenman–Shafee–Bialek (NSB) Bayesian estimator instead.

---

## 2. Rényi Entropy

$$H_\alpha(\mathcal{T}) = \frac{1}{1-\alpha}\,\log_2\!\left(\sum_{t \in \mathcal{V}} p(t)^\alpha\right), \qquad \alpha \ge 0,\ \alpha \ne 1$$

### Special cases

| $\alpha$ | Name | Formula | Emphasises |
|---|---|---|---|
| $\alpha \to 0$ | Hartley / max-entropy | $\log_2 V_{\text{eff}}$ (counts support) | Number of *any* tokens used |
| $\alpha \to 1$ | Shannon | $H_1$ (L'Hôpital limit) | Balanced average |
| $\alpha = 2$ | Collision entropy | $-\log_2 \sum_t p(t)^2$ | Probability of two draws matching |
| $\alpha \to \infty$ | Min-entropy | $-\log_2 \max_t p(t)$ | The single most frequent token |

### Monotonicity (a sanity check you must code)
$$H_0 \ge H_1 \ge H_2 \ge \cdots \ge H_\infty$$

This is always true. If your code violates it, there is a bug.

### Rényi efficiency (Zouhar et al. 2023)
$$\eta_\alpha = \frac{H_\alpha}{\log_2 V} \in [0, 1]$$

This normalises the entropy to the vocabulary size so tokenizers with different $V$ can be fairly compared.

### What it measures
Rényi entropy with parameter $\alpha$ is a *one-parameter family* that lets you smoothly tune sensitivity from rare tokens ($\alpha < 1$) to frequent tokens ($\alpha > 1$). It answers: "if I care more about the tail / head of the distribution, how much information is there?"

### Good vs bad for a chemical tokenizer

| $\alpha$ | What you are probing | Chemistry interpretation |
|---|---|---|
| $\alpha = 0.5$ | Tail-heavy — rare fragment tokens | Tests whether infrequent but chemically meaningful substructures (e.g. rare ring systems) are represented |
| $\alpha = 1$ | Balanced | Overall vocabulary utilisation |
| $\alpha = 2$ | Head-heavy | Dominated by backbone/linker tokens |
| $\alpha = \infty$ | Worst-case | How dominant is the single most common token? (usually a linker) |

**Key experiment:** sweep $\alpha$ and correlate $\eta_\alpha$ with downstream property-prediction performance. The $\alpha$ that gives the highest correlation tells you which part of the distribution matters most for your task. Zouhar et al. found $\alpha \approx 1$–2 best for NMT. Chemistry may differ.

---

## 3. Tsallis Entropy

$$S_q(\mathcal{T}) = \frac{1}{q-1}\!\left(1 - \sum_{t} p(t)^q\right), \qquad q \in \mathbb{R}$$

### Special cases
- $q \to 1$: reduces to Shannon entropy $-\sum p \ln p$ (in nats).
- $q = 2$: $S_2 = 1 - \sum p^2 = \text{Gini–Simpson diversity index}$.

### Non-additivity (the defining property)
For two independent random variables $X$ and $Y$:
$$S_q(X, Y) = S_q(X) + S_q(Y) + (1-q)\,S_q(X)\,S_q(Y)$$

Shannon entropy is *additive* (the $(1-q)$ cross-term is zero at $q=1$). Tsallis entropy is not. This makes it useful for systems with long-range correlations or power-law distributions — which molecular fragment frequencies often are.

### Relation to Rényi
$$H_\alpha = \frac{1}{1-\alpha}\,\log_2\!\left[1 - (\alpha-1)\,S_\alpha\right]$$

They carry the same information but have different algebraic properties. Tsallis is better for optimisation objectives (differentiable, no log-of-sum); Rényi is better for theoretical bounds (additive for product distributions).

### Good vs bad for a chemical tokenizer
- $S_2$ (Gini–Simpson) is directly interpretable: "probability that two randomly drawn tokens are *different*." $S_2 \approx 1$ means high diversity; $S_2 \approx 0$ means the corpus is dominated by one token.
- Because fragment frequencies follow a power law (a few scaffold-type tokens are very common), $q < 1$ gives high weight to rare tokens and may better reflect the structural diversity of the vocabulary than Shannon entropy does.

---

## 4. Sharma–Mittal Entropy (two-parameter generalisation)

$$H_{\alpha,\beta} = \frac{1}{1-\beta}\!\left[\!\left(\sum_t p(t)^\alpha\right)^{(1-\beta)/(1-\alpha)} \!\!- 1\right]$$

### Recovers all previous measures

| Parameter values | Reduces to |
|---|---|
| $\beta \to 1$ (any $\alpha$) | Rényi $H_\alpha$ |
| $\beta = \alpha$ | Tsallis $S_\alpha$ |
| $\alpha, \beta \to 1$ | Shannon $H_1$ |

### Why use it
Sharma–Mittal gives you a 2-D landscape over $(\alpha, \beta)$. Plot $H_{\alpha,\beta}$ for your tokenizer as a heat map. Compare heat maps across tokenizers. Regions where your tokenizer is "hotter" (higher entropy) than competitors show where it has more information. This is a novel angle for a paper — Zouhar et al. only used Rényi (1-D). Extending to the 2-D Sharma–Mittal plane and showing which $(\alpha,\beta)$ region best predicts chemistry-LM quality is a publishable contribution.

---

## 5. Min-Entropy and Collision Entropy

$$H_\infty = -\log_2 \max_t p(t), \qquad H_2 = -\log_2 \sum_t p(t)^2$$

### Min-entropy $H_\infty$
- The information content of the *most predictable* token.
- Equivalently: $-\log_2$ of the probability of guessing the next token correctly in the best case (you always predict the mode).
- A **lower bound** on "useful information per token" — if $H_\infty$ is small, the most common token is so dominant that on average you are learning very little per draw.

**Chemistry interpretation:** In SMILES-derived tokenizers, linker atoms (`C`, `N`, `O`) are extremely common. If a single token (e.g. chain-carbon `*C*`) has $p \approx 0.3$, then $H_\infty \approx 1.74$ bits even if $H_1 = 8$ bits. That gap is a warning: one token is trivially predictable and is diluting the average.

### Collision entropy $H_2$
$$H_2 = -\log_2 \sum_t p(t)^2 = -\log_2 P(\text{two i.i.d. draws are equal})$$

- $2^{H_2}$ is the **effective vocabulary size**: the $V$ that a *uniform* distribution would need to have the same collision probability. Compare this to the true $V$; if $2^{H_2} \ll V$, most of the vocabulary is under-used.

**Chemistry interpretation:** A tokenizer that merges rare fragments into one super-token artificially reduces $V$ but raises $2^{H_2}$ toward $V$ — a good sign. A tokenizer with $2^{H_2} / V \approx 0.1$ is wasting 90% of its vocabulary capacity.

---

## 6. Conditional Entropy and Entropy Rate

For a stationary token process:
$$H(T_n \mid T_{<n}) = -\!\!\sum_{t_{<n},\,t_n}\!\! p(t_{<n}, t_n)\,\log_2 p(t_n \mid t_{<n})$$

### Block entropy
$$H_n = -\!\sum_{w \in \mathcal{V}^n}\! p(w)\,\log_2 p(w) \quad \text{(entropy of } n\text{-grams)}$$

### Entropy rate
$$h = \lim_{n \to \infty} \frac{H_n}{n} = \lim_{n \to \infty} H(T_n \mid T_{<n})$$

Always: $h \le H_1$. Equality holds only if tokens are i.i.d. (no sequential structure).

The gap $H_1 - h$ measures **redundancy**: information that the unigram model ignores but that a sequence model exploits.

### Estimation in practice
Train a small language model (KN-smoothed 5-gram or a small Transformer, 2–4 layers) on the training corpus. Evaluate on a held-out split:
$$\hat{h} = -\frac{1}{N_{\text{test}}}\sum_{i=1}^{N_{\text{test}}} \log_2 \hat{p}(t_i \mid t_{<i}) \quad \text{(bits per token)}$$

$$\text{Perplexity} = 2^{\hat{h}}$$

### Good vs bad for a chemical tokenizer

| Metric | Good tokenizer | Bad tokenizer |
|---|---|---|
| $H_1 - h$ | Small — tokens already carry most information; little context needed | Large — tokens are ambiguous; model must memorise long-range patterns to decode molecules |
| $h$ at fixed fertility | Low — more information compressed per token | High — tokens are individually surprising but not structurally meaningful |
| PPL | Low on held-out — model can predict the next fragment | High — structure is not captured in the token units |

**Chemistry interpretation:** r-fragSMILES splits at chemical bond points, so adjacent tokens in a fragSMILES string are chemically bonded fragments. A good tokenizer should make these transitions predictable (low $h$) compared to atom-level SMILES, where the next character could be any atom.

---

## 7. Excess Entropy (Predictive Information)

$$E = \sum_{n=1}^{\infty}\big[H(T_n \mid T_{<n}) - h\big] = \lim_{n \to \infty}\big[H_n - n\,h\big]$$

Equivalently, the mutual information between past and future halves of the sequence:
$$E = I(T_{<0};\, T_{\ge 0})$$

### What it measures
Total "memory" stored in the sequence. How much does knowing all past tokens reduce uncertainty about all future tokens, beyond the steady-state rate $h$? It is the area between the block-entropy curve $H_n/n$ and its asymptote $h$.

### Practical estimator
1. Compute $H_n$ for $n = 1, 2, 3, \dots, 10$ using $n$-gram counts.
2. Fit the linear model $H_n = h\,n + E$ by OLS.
3. The intercept is $\hat{E}$; the slope is $\hat{h}$.

### Good vs bad for a chemical tokenizer

| $E$ value | Interpretation |
|---|---|
| Low $E$ | Token boundaries are well-chosen: each token is largely self-contained. The LM has little long-range work to do. **Desirable.** |
| High $E$ | Tokens split across chemically meaningful units (e.g. a ring system split between two tokens). The model must track long dependencies to decode. **Undesirable.** |

**Chemistry interpretation:** If your BPE is merging linker chains but leaving ring atoms as individual tokens, consecutive ring-atom tokens will have high mutual information (they always co-occur in the same ring), inflating $E$. A tokenizer that keeps whole rings as single tokens will have lower $E$. You can use this to *diagnose* which merges are most important.

---

## 8. Mutual Information Between Adjacent Tokens

$$I(T_i;\, T_{i+1}) = \sum_{a,b} p(a,b)\,\log_2\frac{p(a,b)}{p(a)\,p(b)} = 2H_1 - H_2^{(\text{pair})}$$

where $H_2^{(\text{pair})} = -\sum_{a,b} p(a,b)\log_2 p(a,b)$ is the bigram block entropy.

### Pointwise mutual information (PMI)
$$\text{PMI}(a,b) = \log_2\frac{p(a,b)}{p(a)\,p(b)}$$

High positive PMI: $(a,b)$ co-occur much more than chance — BPE *should* merge them.
Negative PMI: $(a,b)$ repel — they almost never appear together.

### Normalised PMI (NPMI) — bounded in $[-1, 1]$
$$\text{NPMI}(a,b) = \frac{\text{PMI}(a,b)}{-\log_2 p(a,b)} \in [-1, 1]$$

NPMI = 1: the two tokens always co-occur. NPMI = 0: independent. NPMI = −1: mutually exclusive.

### Good vs bad for a chemical tokenizer

**Use NPMI to audit BPE merge decisions:**
- Sort all candidate pairs by NPMI. If BPE merged low-NPMI pairs early (because they were frequent), that is a flaw — frequency ≠ co-occurrence surprise.
- The highest-NPMI pairs that BPE *did not* merge are your best candidates for MDL-based or entropy-regularised merging.

**Chemistry interpretation:** Adjacent atoms in a ring almost always appear together in the same ring context → high NPMI. A tokenizer that does not merge them wastes sequence length and forces the LM to learn ring closure implicitly.

---

## 9. Per-Token (Pointwise) Information

### Surprisal under a contextual LM
$$s_i = -\log_2 \hat{p}(t_i \mid t_{<i}) \quad \text{(bits)}$$

High surprisal = this token was unexpected given context (high information delivered).
Low surprisal = this token was predictable (low information delivered, possibly redundant).

### Unigram information content (Resnik)
$$\text{IC}(t) = -\log_2 p(t)$$

Rare tokens have high IC by definition. But IC does not account for context.

### Average information contributed by token type $t$
$$\bar{s}(t) = -\mathbb{E}\!\left[\log_2 \hat{p}(T_i \mid T_{<i}) \mid T_i = t\right] \cdot p(t)$$

This weights contextual surprisal by how often $t$ actually occurs. Tokens with high $\bar{s}(t)$ are both informative and common — these are your most valuable tokens.

### Good vs bad for a chemical tokenizer

| Token type | Expected $s_i$ | Interpretation |
|---|---|---|
| Linker (`*C*`, `*N*`) | Low — very predictable | These tokens carry little information; they confirm structure rather than specify it |
| Rare ring system | High — contextually surprising | These tokens carry real chemical information |
| Scaffold head token | Medium | Determines the class of molecule; important but somewhat predictable from dataset distribution |

**Actionable:** Plot the distribution of $s_i$ for each tokenizer. A good tokenizer concentrates surprisal on chemically-meaningful tokens (scaffolds, functional groups) and drives it toward zero for structural connectors.

---

## 10. Compression Metrics

Let $B$ = total bits to losslessly encode the corpus using optimal coding under $\hat{p}$. Then:

### Bits per token
$$\frac{B}{N} = \hat{h}$$

### Bits per molecule
$$\frac{B}{M}$$

### Bits per heavy atom
$$\frac{B}{\sum_m A_m}$$

This normalises by molecular complexity and allows fair comparison between molecules of different sizes. A tokenizer that is concise for small molecules but verbose for large ones will be penalised here.

### Bits per SMILES character
$$\frac{B}{\sum_m |\text{SMILES}_m|}$$

**Critical for cross-format comparison.** SELFIES strings are longer than SMILES for the same molecule. If you compare bits/token across SMILES-based and SELFIES-based tokenizers, you must normalise by the underlying character count, not by token count.

### Fertility (tokens per molecule)
$$F = \frac{1}{M}\sum_{m=1}^{M} L_m = \frac{N}{M}$$

Lower fertility at fixed downstream quality = better tokenizer. Every extra token costs a Transformer attention step.

### Compression ratio vs atom-level baseline
$$\rho = \frac{F_{\text{atom-level}}}{F_{\text{tokenizer}}}$$

$\rho > 1$: your tokenizer produces shorter sequences than atom-level tokenization. $\rho < 1$ should never happen for a BPE-derived tokenizer (it would mean the merged vocabulary is expanding sequences).

### The compression chain (memorise this)
$$N\,h \;\le\; N\,H_1 \;\le\; N\,\log_2 V$$

- $N\,h$: optimal bits using a perfect contextual model.
- $N\,H_1$: optimal bits under the i.i.d. assumption.
- $N\,\log_2 V$: bits if you used fixed-length coding (naive baseline).

Good tokenizers push $N\,h$ down. The gap $N(H_1 - h)$ is the saving from using a contextual model on top of the tokenizer.

---

## 11. Source-Coding Upper Bounds

### Shannon's source coding theorem
For any uniquely decodable code with codelengths $\ell(t)$:
$$\mathbb{E}[\ell(T)] \ge H_1(T)$$

with equality achievable by arithmetic coding in the limit of long sequences.

### Kraft inequality
Any prefix code satisfies:
$$\sum_{t \in \mathcal{V}} 2^{-\ell(t)} \le 1$$

### The three bounds on information per token

| Bound | Formula | Tightness |
|---|---|---|
| Trivial vocabulary bound | $\log_2 V$ | Loose; just counts the vocabulary |
| Statistical bound | $H_1(T)$ | Tight under i.i.d. assumption |
| Operational (contextual) bound | $h$ | The true, achievable limit |

### How to find the empirical upper bound for your corpus
1. Train held-out LMs at many vocab sizes $V \in \{256, 512, 1k, 2k, \dots, 32k\}$.
2. Record $\hat{h}(V)$ for each.
3. Plot $\hat{h}$ vs $V$. You will see a curve that drops steeply then *plateaus*.
4. The plateau value $h^*$ is the **empirical upper bound on information per token for this corpus**. Beyond the corresponding $V^*$, you gain no information by adding vocabulary — you just add parameters.

**Chemistry significance:** This directly answers "what vocabulary size is optimal for r-fragSMILES BPE on this molecular dataset?" without needing a downstream task.

---

## 12. Vocabulary-Structure Metrics

### Zipf's law
Rank tokens by descending frequency. Rank-$r$ frequency:
$$f_r \propto r^{-s}$$

Fit $s$ and $R^2$ by OLS on $\log f_r$ vs $\log r$.

- $s \approx 1$: classic Zipf (natural language, SMILES). Efficient BPE vocabulary.
- $s \ll 1$: flat distribution — rare and common tokens are equally frequent. Unusual; may indicate vocabulary is too large or corpus is too small.
- $s \gg 1$: extremely head-heavy — a few tokens dominate completely.

**Chemistry note:** Fragment tokens tend to follow $s \approx 0.8$–$1.2$. Linker tokens will be at the head; rare heterocycles at the tail. Check that the tail does not drop off too steeply (which would mean rare-but-important chemical substructures are being dropped below `min_freq`).

### Heaps' law
How many unique token types $V(n)$ appear in the first $n$ tokens:
$$V(n) \approx K\,n^{\beta_H}, \quad \beta_H \in (0, 1)$$

- $\beta_H$ close to 1: vocabulary grows fast — the corpus has high lexical diversity. You need a large $V$.
- $\beta_H$ close to 0: vocabulary saturates quickly — a small $V$ suffices.

**Chemistry note:** Molecular datasets have lower Heaps' exponents than natural language text because the chemical vocabulary is closed. ZINC-250k typically gives $\beta_H \approx 0.4$–$0.6$ for fragment-level BPE. If your `min_freq` cutoff is too aggressive, you will see $V(n)$ plateau early and $\beta_H$ will appear artificially small.

### Type-token ratio
$$\text{TTR} = V / N$$

Low TTR: few types cover many tokens — efficient. High TTR: nearly every token is unique — corpus is too small for reliable statistics.

### Coverage at threshold
$$\text{Cov}(k) = \sum_{t:\, c(t) \ge k} p(t)$$

Fraction of corpus mass explained by tokens with count $\ge k$. Compute for $k = 5, 10, 50$.

- If $\text{Cov}(50) > 0.95$: 95% of your training corpus is covered by tokens that appeared at least 50 times — `min_freq=50` is safe.
- If $\text{Cov}(50) < 0.8$: you are discarding 20%+ of the corpus by ignoring rare tokens — lower `min_freq` or increase corpus size.

### OOV rate on held-out
$$\text{OOV} = \frac{\#\text{tokens not in }\mathcal{V}}{N_{\text{test}}}$$

A scaffold-OOD held-out set (Bemis–Murcko split) will have higher OOV than a random split. Report both. A tokenizer that has low random-split OOV but high scaffold-OOD OOV is overfitting to training scaffolds.

---

## 13. Effective Vocabulary Sizes

From any entropy $H_\alpha$, define an "effective vocabulary size":
$$V_{\text{eff}}^{(\alpha)} = 2^{H_\alpha}$$

This is the size a *uniform* distribution would need to have the same Rényi-$\alpha$ entropy as your actual distribution.

| Effective size | Formula | Interpretation |
|---|---|---|
| $V_{\text{eff}}^{(1)} = 2^{H_1}$ | Unigram perplexity | Number of "equally likely" tokens under Shannon entropy |
| $V_{\text{eff}}^{(2)} = 2^{H_2} = 1/\sum p^2$ | Inverse Simpson | Effective number of distinct tokens by collision probability |
| $V_{\text{eff}}^{(\infty)} = 2^{H_\infty} = 1/\max p$ | Inverse max-prob | How dominant is the top token? |

Always report these alongside raw $V$. The ratio $V_{\text{eff}}^{(1)} / V$ is the "vocabulary utilisation ratio." 

**Target:** $V_{\text{eff}}^{(1)} / V > 0.5$ for a well-designed tokenizer. A ratio near 0.1 means 90% of the vocabulary is so rarely used it might as well not exist.

---

## 14. Chemistry-Specific Semantic Information

This section is novel and specific to chemical tokenizers.

### Setup
For each token $t$, let $\mathcal{F}_t$ = set of distinct molecular fragment subgraphs (RDKit-canonicalised) that token $t$ is mapped to across the corpus. Each occurrence of $t$ in the corpus corresponds to one fragment. Compute the conditional distribution:
$$p(f \mid t) = \frac{c(t, f)}{c(t)}, \quad f \in \mathcal{F}_t$$

### Semantic capacity
$$C_t = \log_2 |\mathcal{F}_t|$$

The maximum information (in bits) that token $t$ can carry about molecular structure, assuming each fragment is equally likely. This is the theoretical ceiling.

**Chemistry interpretation:** If $C_t = 0$, the token always means exactly one fragment — it is unambiguous (ideal). If $C_t = 5$, the token can mean 32 different fragments depending on context — it is highly ambiguous.

### Semantic ambiguity (entropy of meaning)
$$H^{\text{sem}}(t) = -\!\sum_{f \in \mathcal{F}_t}\! p(f \mid t)\,\log_2 p(f \mid t)$$

Unlike $C_t$ (which is uniform-weighted), this accounts for the actual distribution over fragments. $H^{\text{sem}}(t) < C_t$ whenever some fragments are more common than others.

### Corpus-average semantic ambiguity
$$\bar{H}^{\text{sem}} = \sum_t p(t)\,H^{\text{sem}}(t) = H(F \mid T)$$

This is the conditional entropy of the fragment given the token — how much chemical meaning is lost by looking at only the token label.

### Mutual information between token and fragment
$$I(T; F) = H(F) - H(F \mid T) = H(F) - \bar{H}^{\text{sem}}$$

- $I(T; F) = H(F)$: tokens are perfectly informative about fragments. The tokenizer is a bijection from tokens to fragments. **Ideal.**
- $I(T; F) \ll H(F)$: tokens are ambiguous — knowing the token tells you little about which fragment it represents.

### Fano's inequality (a hard lower bound on decoding error)
$$P_e \ge \frac{H(F \mid T) - 1}{\log_2(|\mathcal{F}| - 1)}$$

Even with perfect downstream context, the probability of wrong fragment identification is lower-bounded by semantic ambiguity alone. This gives a *theoretical* lower bound on the error rate of any model that must infer molecular structure from token sequences — no matter how large the LM.

**Chemistry interpretation:** If $\bar{H}^{\text{sem}} = 3$ bits and $|\mathcal{F}| = 100$, then $P_e \ge (3-1)/\log_2(99) \approx 0.31$. Your model cannot get below 31% fragment-identification error on ambiguous tokens, regardless of training. This is a fundamental limit imposed by the tokenizer, not the model.

### What to do with these metrics
1. Compute $H^{\text{sem}}(t)$ for every token. Plot histogram. Tokens with high $H^{\text{sem}}$ are the "bad" tokens.
2. Check: do these high-ambiguity tokens tend to be linkers (which appear in many chemical contexts) or ring atoms (which tend to appear in one or two ring types)?
3. A modified merge selection criterion (semantic-aware BPE, see §15 extension) penalises merges that create high-$H^{\text{sem}}$ tokens.

---

## 15. MDL Objective for Tokenizer Training

Minimum Description Length (MDL) provides a principled way to choose vocabulary size without setting `max_merges` or `min_freq` as hyperparameters.

### Two-part code
$$L(\mathcal{C}, \mathcal{V}) = \underbrace{L(\mathcal{V})}_{\text{cost of the vocabulary itself}} + \underbrace{L(\mathcal{C} \mid \mathcal{V})}_{\text{cost of encoding the corpus given vocabulary}}$$

### Vocabulary cost
Each BPE merge creates a new token by combining two existing tokens. If at step $k$ there are $|\mathcal{V}_k|$ tokens in the vocabulary, storing one merge costs $2\log_2 |\mathcal{V}_k|$ bits (two parent indices). Summing over all merges:
$$L(\mathcal{V}) = \sum_{k=1}^{|\mathcal{V}| - |\mathcal{V}_0|} 2\log_2 |\mathcal{V}_k|$$

where $\mathcal{V}_0$ is the base vocabulary (individual atoms/characters).

### Corpus cost under unigram model
$$L(\mathcal{C} \mid \mathcal{V}) = -\sum_{t} c(t)\,\log_2 \hat{p}(t) = N\,\hat{H}_1(\mathcal{T})$$

### The greedy MDL merge rule
Accept merge $(a, b) \to ab$ if and only if:
$$\Delta L = \underbrace{\Delta L(\mathcal{V})}_{= 2\log_2 V_{k} > 0} + \underbrace{\Delta L(\mathcal{C} \mid \mathcal{V})}_{< 0} < 0$$

**Computing $\Delta L(\mathcal{C} \mid \mathcal{V})$ in closed form.** Let $c_{ab}$ = bigram count of pair $(a,b)$ before merge. After merging:
- $c'(a) = c(a) - c_{ab}$
- $c'(b) = c(b) - c_{ab}$
- $c'(ab) = c_{ab}$ (new token)
- $N' = N - c_{ab}$ (sequence shortens)

Therefore:
$$\Delta L(\mathcal{C} \mid \mathcal{V}) = -N'\hat{H}_1' + N\hat{H}_1$$

where $\hat{H}_1' = -\sum_t \frac{c'(t)}{N'}\log_2\frac{c'(t)}{N'}$ is recomputed with updated counts. In practice, compute this delta efficiently using only the changed terms.

### Why this is better than frequency-based BPE

| BPE (standard) | MDL BPE |
|---|---|
| Merges most frequent pair | Merges pair that most reduces total description length |
| Requires `max_merges` hyperparameter | Stops automatically when $\Delta L \ge 0$ |
| Requires `min_freq` hyperparameter | No threshold needed |
| Can over-merge or under-merge | Theoretically optimal trade-off between vocab cost and corpus cost |
| Greedy, but in frequency space | Greedy, but in information space |

**Chemistry significance:** Standard BPE may merge `CC` (two carbons) thousands of times because it is frequent, even if it adds little structural information. MDL will only merge `CC` if the corpus cost saving outweighs the vocabulary cost. This tends to preserve chemically meaningful units.

---

## 16. Information Bottleneck Objective

Frame the entire tokenization problem as a compression problem with a relevance constraint.

### Setup
- $X$: the molecule (e.g. as a molecular graph or Morgan fingerprint).
- $T$: the token sequence produced by tokenizer $\mathcal{T}$.
- $Y$: downstream label (property, bioactivity, etc.).

### IB Lagrangian
$$\mathcal{L}_{\text{IB}}(\mathcal{T}) = I(T; X) - \beta\,I(T; Y)$$

Minimise $I(T; X)$ (compression — use as few bits as possible to represent the molecule) while maximising $I(T; Y)$ (relevance — preserve information about the downstream target).

$\beta > 0$ controls the trade-off:
- $\beta \to 0$: maximise compression, ignore task. Equivalent to minimising fertility.
- $\beta \to \infty$: maximise task relevance, ignore compression. Equivalent to using the molecule directly.

### The IB curve
The **IB curve** (information plane) plots $I(T; Y)$ vs $I(T; X)$ for varying $\beta$ or across tokenizer families. It is the theoretical Pareto front.

**This is your cleanest comparison figure.** Each tokenizer at each vocab size is one point in $(I(T;X), I(T;Y))$ space. Tokenizers on the Pareto frontier dominate all others.

### Practical estimation
- $I(T; Y) \approx$ MI between token-sequence embedding and label (compute via mutual information neural estimation, MINE, or via $k$-NN MI estimators).
- $I(T; X) \approx$ cross-entropy of an autoencoder that reconstructs the molecular fingerprint from the token sequence.
- For a simpler proxy: use downstream prediction $R^2$ or AUROC as $I(T; Y)$ and fertility as $I(T; X)^{-1}$.

---

## 17. Cross-Tokenizer Comparison Metrics

When comparing two tokenizers $\mathcal{T}_A$ and $\mathcal{T}_B$ on a common corpus, you need divergence measures.

### KL divergence (asymmetric)
$$D_{\text{KL}}(p_A \| p_B) = \sum_x p_A(x)\,\log_2\frac{p_A(x)}{p_B(x)}$$

"Extra bits needed to encode $p_A$-distributed data using a code optimised for $p_B$." Not symmetric: $D_{\text{KL}}(p_A \| p_B) \ne D_{\text{KL}}(p_B \| p_A)$.

### Jensen–Shannon divergence (symmetric, bounded)
$$\text{JSD}(p_A, p_B) = \tfrac{1}{2}D_{\text{KL}}(p_A \| m) + \tfrac{1}{2}D_{\text{KL}}(p_B \| m), \quad m = \tfrac{1}{2}(p_A + p_B)$$

$\text{JSD} \in [0, 1]$ bits (with base-2 log). $\text{JSD} = 0$: identical distributions. $\text{JSD} = 1$: completely disjoint supports.

**Chemistry use:** Compute JSD between the fragment-distribution $p_A(f)$ and $p_B(f)$ of two tokenizers over a shared fragment vocabulary. High JSD means the two tokenizers are "seeing" chemically different things in the same corpus.

### Normalised Compression Distance (Cilibrasi–Vitányi 2005)
$$\text{NCD}(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

where $C(\cdot)$ is a compressor (gzip, zstd) applied to the tokenized byte stream.

$\text{NCD} \in [0, 1]$. NCD ≈ 0: the two tokenized representations are nearly identical. NCD ≈ 1: the two are maximally dissimilar.

**Use:** cluster tokenizers by NCD of their output on the same molecule set. Tokenizers that cluster together (low NCD) are substitutable; tokenizers far apart are capturing different structure.

---

## 18. Statistical Rigor for Correlations and Comparisons

Across $K$ tokenizer configurations $\{\mathcal{T}_k\}_{k=1}^K$ you have intrinsic metric $X_k$ and downstream score $Y_k$.

### Correlation coefficients

**Pearson** (linear, sensitive to outliers):
$$r = \frac{\sum_{k=1}^K (X_k - \bar{X})(Y_k - \bar{Y})}{\sqrt{\sum(X_k - \bar{X})^2}\,\sqrt{\sum(Y_k - \bar{Y})^2}}$$

**Spearman** (rank-based, robust to monotone non-linearity):
Compute ranks $R(X_k)$ and $R(Y_k)$, then apply Pearson to the ranks.
$$r_s = 1 - \frac{6\sum d_k^2}{K(K^2-1)}, \quad d_k = R(X_k) - R(Y_k)$$

**Kendall's $\tau$** (pair-based, most robust):
$$\tau = \frac{\#\text{concordant pairs} - \#\text{discordant pairs}}{\binom{K}{2}}$$

For A* papers, report all three. Discrepancies between Pearson and Spearman reveal non-linearity (common in entropy-vs-performance relationships).

### Bootstrap confidence intervals for $r$
1. Resample $(X_k, Y_k)$ pairs with replacement $B = 10{,}000$ times.
2. Compute $r^{(b)}$ for each resample.
3. Report the 2.5th and 97.5th percentile as the 95% CI.

### Paired tokenizer comparison (Wilcoxon signed-rank test)
When comparing tokenizer $A$ vs $B$ across $K$ tasks:
1. Compute $d_k = Y_k^A - Y_k^B$ for each task $k$.
2. Rank $|d_k|$ by magnitude.
3. Compute $W^+ = \sum_{d_k > 0} \text{rank}(|d_k|)$.
4. Under $H_0: $ median$\,(d_k) = 0$, $W^+$ follows a known distribution.

Use this instead of a $t$-test because downstream scores are not normally distributed.

### Multiple testing correction
If you test 20 intrinsic metrics for correlation with downstream performance, expect 1 false positive at $p < 0.05$ by chance. Use Benjamini–Hochberg FDR correction, not Bonferroni (which is too conservative for correlated metrics).

---

## 19. Quick-Reference Cheat Sheet

| Quantity | Formula | Bound | Chemistry use |
|---|---|---|---|
| Shannon $H_1$ | $-\sum p\log p$ | $[0, \log V]$ | Vocabulary utilisation |
| Rényi $H_\alpha$ | $\frac{1}{1-\alpha}\log\sum p^\alpha$ | $H_\alpha \downarrow$ in $\alpha$ | Tail/head sensitivity; correlate with downstream |
| Tsallis $S_q$ | $\frac{1}{q-1}(1-\sum p^q)$ | $\ge 0$ | Power-law distribution analysis |
| Sharma–Mittal $H_{\alpha,\beta}$ | §4 formula | generalises all | 2-D entropy landscape |
| Min-entropy $H_\infty$ | $-\log\max p$ | $\le H_1$ | Linker dominance check |
| Collision $H_2$ | $-\log\sum p^2$ | $\le H_1$ | Effective vocabulary size |
| Entropy rate $h$ | $\lim H_n/n$, estimated by LM cross-entropy | $\le H_1$ | True bits per token; find saturation plateau |
| Excess entropy $E$ | $\lim(H_n - nh)$ | $\ge 0$ | Long-range chemical dependencies |
| MI $I(T_i; T_{i+1})$ | $2H_1 - H_2^{(\text{pair})}$ | $\ge 0$ | Identifies pairs BPE should merge |
| NPMI$(a,b)$ | $\text{PMI}/(- \log p(a,b))$ | $[-1,1]$ | Rank candidate merges |
| Surprisal $s_i$ | $-\log\hat{p}(t_i \mid t_{<i})$ | $\ge 0$ | Per-token information delivery |
| Fertility $F$ | $N/M$ | $\ge 1$ | Sequence length cost |
| $V_{\text{eff}}^{(1)}$ | $2^{H_1}$ | $\le V$ | Vocabulary utilisation ratio |
| $V_{\text{eff}}^{(2)}$ | $1/\sum p^2$ | $\le V$ | Collision-based effective size |
| Semantic ambiguity $\bar{H}^{\text{sem}}$ | $\sum_t p(t) H^{\text{sem}}(t)$ | $\ge 0$ | How often does one token mean multiple fragments? |
| Fano bound | $P_e \ge \frac{H(F\mid T)-1}{\log(\|\mathcal{F}\|-1)}$ | — | Hard lower bound on fragment-decoding error |
| MDL cost | $L(\mathcal{V}) + N\hat{H}_1$ | minimise | Hyperparameter-free vocabulary selection |
| IB objective | $I(T;X) - \beta I(T;Y)$ | Pareto frontier | Task-relevant compression |
| JSD | $\frac{1}{2}D_\text{KL}(p_A\|m) + \frac{1}{2}D_\text{KL}(p_B\|m)$ | $[0,1]$ | Dissimilarity between tokenizer distributions |
| NCD | $(C(xy) - \min C)/ \max C$ | $[0,1]$ | Kolmogorov-style tokenizer distance |

---

## 20. Sanity Checks (Code These as Unit Tests)

Before running any analysis, implement these tests. They catch ~90% of implementation bugs.

| Test | Condition | Formula to verify |
|---|---|---|
| **Uniform distribution** | $p(t) = 1/V$ for all $t$ | $H_\alpha = \log_2 V$ for all $\alpha$; $S_q = (V^{1-q}-1)/(q-1)$ (in nats) |
| **Delta distribution** | $p(t^*) = 1$, all others 0 | $H_\alpha = 0$ for all $\alpha$; $H^{\text{sem}} = 0$ |
| **Monotonicity** | Always | $H_0 \ge H_1 \ge H_2 \ge H_\infty$ |
| **Shannon limit of Rényi** | $\alpha = 1 \pm 10^{-6}$ | $H_{1\pm\epsilon} \approx H_1$ (within numerical tolerance) |
| **Block entropy sub-additivity** | $n$-grams | $H_n \le n\,H_1$; equality iff i.i.d. |
| **MI non-negativity** | Any $X, Y$ | $I(X;Y) \ge 0$ |
| **MI upper bound** | Any $X, Y$ | $I(X;Y) \le \min(H(X), H(Y))$ |
| **JSD symmetry** | $p_A, p_B$ | $\text{JSD}(p_A, p_B) = \text{JSD}(p_B, p_A)$ |
| **JSD range** | With base-2 log | $\text{JSD} \in [0, 1]$ |
| **Tsallis–Rényi consistency** | Any distribution, any $\alpha = q$ | $H_\alpha = \frac{1}{1-\alpha}\log_2[1-(\alpha-1)S_\alpha]$ |
| **Miller–Madow correction sign** | Any undersampled corpus | Corrected $\hat{H}_1 \ge$ uncorrected $\hat{H}_1$ |
| **Fano bound range** | Any $H(F\mid T)$ | $P_e \in [0, 1]$ |

---

## 21. Summary of What Makes a Chemical Tokenizer "Good"

A chemical tokenizer optimised for LM training should simultaneously achieve:

1. **High $H_1 / \log_2 V$** — vocabulary is efficiently used; no dead tokens.
2. **Low $h$** (entropy rate) — tokens carry maximal structural information; LM prediction is easy.
3. **Low $E$** (excess entropy) — tokens are semantically self-contained; minimal long-range dependencies needed.
4. **High $I(T;F)$** (semantic MI) — each token maps reliably to one chemical fragment; no ambiguity.
5. **Low $\bar{H}^{\text{sem}}$** — tokens have low semantic ambiguity; Fano error bound is tight.
6. **Low fertility $F$** — sequences are short; efficient attention use.
7. **Low OOV on scaffold-OOD** — vocabulary generalises to unseen chemical space.
8. **$h$ plateaus at $V^*$** — adding more vocabulary beyond $V^*$ gives no information gain; stay at $V^*$.
9. **High $\eta_\alpha$ (Rényi efficiency)** at the $\alpha$ that correlates with your downstream task.
10. **Pareto-dominant** in $(I(T;X), I(T;Y))$ space — best task-relevant compression.

No existing chemical tokenizer achieves all ten simultaneously. That is the gap your paper fills.
