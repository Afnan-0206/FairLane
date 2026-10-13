# FairLane Limitations & Known Constraints (Draft)

## 1. Horizon-Dependence of FIFO Under Overload
In Experiment 1, the Flooder arrival rate ($\lambda = 83.3$ req/s) exceeds maximum continuous batch service capacity ($\mu \approx 12.5$ req/s).
- **Unbounded Queue**: The FIFO queue grows without bound ($Q(t) \propto (\lambda - \mu)t$).
- **Horizon Sensitivity**: The measured $160.37\times$ slowdown is directly tied to the 300-request ($104.0\text{ s}$) simulation horizon. When scaled to 600 requests ($208\text{ s}$), victim slowdown reaches $313.82\times$; at 1200 requests ($416\text{ s}$), it reaches $633.77\times$. The reported slowdown under unbounded FIFO is strictly horizon-dependent.

## 2. Break-On-No-Fit Head-of-Line Blocking
Strict work-conservation under continuous batching enforces `break-on-no-fit`:
- If the tenant selected by the policy submits a request that requires more KV cache than currently available, batch formation halts immediately.
- **Asymmetric Request Vulnerability**: When a tenant with large prompts (e.g., Flooder at 1024 tokens or RAG at 4000 tokens) is selected, smaller requests (Victim at 128 tokens) that would fit in the remaining KV cache are blocked.
- **Diagnostic Numbers**: In Experiment 1, Victim experienced an average of **199.4** break blocks under VTC versus **38.8** under Round-Robin. When skip-on-no-fit is permitted, VTC victim TTFT decreases from $1,201.97\text{ ms}$ to $655.30\text{ ms}$.

## 3. Finite-Trace Lifetime Token Conservation
In discrete-event traces executed to completion:
- By conservation of tokens, $\text{TotalTokensServed}_i \equiv \text{TotalTokensSubmitted}_i$.
- In Experiment 2, because the Chat tenant submitted 200 requests (38,400 tokens) and the RAG tenant submitted 50 requests (206,400 tokens), lifetime token shares for all work-conserving policies are mathematically constrained to 15.7% (Chat) and 84.3% (RAG).
- VTC equalizes service rates *during the concurrent backlogged window*, draining Chat requests first (reducing Chat TTFT to $2,209.6\text{ ms}$ vs $5,249.3\text{ ms}$ under FIFO). Demonstrating steady-state 50/50 token throughput requires infinite continuous demand streams for both tenants.

## 4. Simulator-to-Hardware Fidelity Gap
- **Step Latency Modeling**: The discrete simulator uses a linear roofline approximation ($T_{\text{step}} = T_{\text{base}} + T_{\text{tok}} \times N_{\text{tokens}}$). It does not capture Tensor Core saturation transitions, CUDA graph launch overheads, or memory bandwidth inflection points.
- **KV Cache Fragmentation**: Assumes continuous token availability without block-level fragmentation (PagedAttention block table overheads are omitted).
- **Prefix Caching**: The current simulator does not model radix tree prefix caching. Prefix hits would alter the effective prefill token cost for tenants with common system prompts.

## 5. Upstream Reference Status
- **Kubernetes GAIE PR #2031**: Flow-control narrative remains marked as **UNVERIFIED** due to shallow clone constraints. Production flow-control interfaces are referenced against `llm-d-router` (`pkg/epp/framework/plugins/flowcontrol`).
- **AIBrix Architecture**: Operates as a distributed router-level load balancer across pods/nodes, not an intra-instance queue-level continuous batching scheduler.
