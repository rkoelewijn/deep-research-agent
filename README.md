# Deep Research Agent on TRQA-Wiki2 (BM25 + Search-R1)

Codebase for **NWI-I00041 Information Retrieval Research Project** (Radboud University):
- **Dataset Subset:** `TRQA-Wiki2` (`validation`: 1,083 queries, `test`: 1,258 queries)
- **Corpus:** Partial Wikipedia Corpus (~5.9M indexed documents)
- **Retriever:** BM25 (Pyserini / Lucene)
- **Deep Research Agent:** Search-R1 (`Qwen2.5-7B`)

This repository follows the Radboud `cluster-skeleton-python` structure for the **`csedu`** SLURM cluster.

---

## 1. Repository Structure

```text
deep-research-agent/
├── configs/
│   └── wiki2_bm25.yaml                           # Dataset, BM25, Single-Step RAG, and Search-R1 settings
├── data -> $NFS_USER_DIR/deep-research-agent/data # Symlink to /vol/csedu-nobackup (TRQA, BM25 index, hf_cache, jdk-21)
├── logs -> $NFS_USER_DIR/deep-research-agent/logs # Symlink to /vol/csedu-nobackup (SLURM outputs, TREC runs, JSONL logs)
├── venv -> /scratch/$USER/virtual_environments/deep-research-agent # Symlink to node-local /scratch venv
├── experiments/
│   ├── step1_1_bm25_cluster_csedu.sh             # SLURM job (CPU-only, 16 threads) for Step 1.1 Table 4
│   ├── step1_1_bm25_local.sh                     # Local runner for Step 1.1
│   ├── step1_2a_single_step_cluster_csedu.sh     # SLURM job (2x RTX 2080 Ti, float16) for Step 1.2a Table 8 Baseline
│   ├── step1_2a_single_step_local.sh             # Local runner for Step 1.2a
│   ├── step1_2b_search_r1_cluster_csedu.sh       # SLURM job (2x RTX 2080 Ti, float16) for Step 1.2b Table 8 Search-R1
│   └── step1_2b_search_r1_local.sh               # Local runner for Step 1.2b
├── scripts/
│   ├── prepare_cluster.sh                        # Creates /vol/ dirs, symlinks data & logs, installs portable OpenJDK 21
│   ├── setup_virtual_environment.sh              # Creates /scratch venv, links venv/, installs requirements.txt
│   ├── sync_csedu.sh                             # Syncs /scratch venv across all csedu compute nodes
│   ├── reset_venv.sh                             # Recreates the virtual environment cleanly
│   ├── get_gpu_shell.sh                          # Requests an interactive GPU shell on csedu
│   ├── download_data.sh                          # Downloads mahtaa/trqa (Wiki2 val+test) & partial BM25 index into data/
│   └── download_remote_logs.sh                   # rsyncs logs/ from csedu cluster to local logs/cluster/
├── src/
│   ├── data/loader.py                            # Loads TRQA-Wiki2 queries and qrels (validation & test)
│   ├── retrieval/bm25_session.py                 # BM25SessionRetriever: Pyserini LuceneSearcher + top-100 cache + session state
│   ├── single_step/rag_runner.py                 # Single-Step RAG (K=50) prompt builder & Qwen2.5-7B generator
│   ├── search_r1/                                # Multi-turn Search-R1 loop, stopping criteria, and modular Phase 2 hooks
│   └── evaluation/                               # Recall@K, EM, SM_10%..90%, trajectory Union P/R/F1, and JSONL checkpoints
├── run_step1_1_bm25.py                           # Step 1.1 entrypoint (Table 4 reproduction)
├── run_step1_2a_single_step.py                   # Step 1.2a entrypoint (Table 8 Single-Step Retriever K=50)
├── run_step1_2b_search_r1.py                     # Step 1.2b entrypoint (Table 8 Search-R1 + BM25)
└── analyze_trajectories.py                       # Stagnation & failure taxonomy analyzer on validation/test trajectories
```

---

## 2. Setting Up the `csedu` Cluster Environment

1. **Disable pip caching in `~/.profile`** (so your home disk does not fill up):
   ```bash
   echo 'export PIP_NO_CACHE_DIR=off' >> ~/.profile
   source ~/.profile
   ```
2. **Create your user directory on `/vol/csedu-nobackup/`**:
   ```bash
   mkdir -p /vol/csedu-nobackup/project/$USER
   chmod 700 /vol/csedu-nobackup/project/$USER
   ```
3. **Configure `.env` and create cluster symlinks (`data`, `logs`, and portable OpenJDK 21)**:
   ```bash
   cp .env.example .env
   ./scripts/prepare_cluster.sh
   ```
4. **Create the `/scratch` virtual environment and sync across `csedu` nodes**:
   ```bash
   ./scripts/setup_virtual_environment.sh
   ./scripts/sync_csedu.sh
   ```
   *(Whenever you add a dependency to `requirements.txt`, activate `venv`, run `pip install -r requirements.txt`, and re-run `./scripts/sync_csedu.sh`.)*
5. **Download TRQA-Wiki2 (`validation` & `test`) and the 5.9M Partial Wikipedia BM25 Index**:
   ```bash
   ./scripts/download_data.sh
   ```

---

## 3. Running Phase 1 Experiments on `csedu` (SLURM)

### Step 1.1: Standalone BM25 Retrieval (Table 4)
Runs multi-threaded Pyserini BM25 retrieval (`top-1000`) on `validation` (1,083 queries) and `test` (1,258 queries) on CPU, caches the top-100 full documents per query for Step 1.2a, and outputs `logs/step1_1_table4_results.md`:
```bash
sbatch experiments/step1_1_bm25_cluster_csedu.sh
```

### Step 1.2a: Single-Step Retriever `BM25 (K=50) + Qwen2.5-7B-Instruct` (Table 8 Baseline)
Requests 2x RTX 2080 Ti GPUs (`float16`), feeds the top $K=50$ BM25 passages to `Qwen/Qwen2.5-7B-Instruct` on both `validation` and `test` with line-by-line JSONL checkpointing (`--resume`), and outputs `logs/step1_2a_table8_results.md`:
```bash
sbatch experiments/step1_2a_single_step_cluster_csedu.sh
```

### Step 1.2b: Deep Research Agent `Search-R1 (Qwen2.5-7B) + BM25` (Table 8 DRA)
Requests 2x RTX 2080 Ti GPUs (`float16`), runs the multi-turn `<think>`/`<search>`/`<information>`/`<answer>` loop with `BM25SessionRetriever` ($k=3$ per search) on `validation` and `test` with `--resume`, and outputs `logs/step1_2b_table8_results.md`:
```bash
sbatch experiments/step1_2b_search_r1_cluster_csedu.sh
```

---

## 4. Observing & Analyzing Results Locally

Pull all `.out`, `.md`, and `.jsonl` files from the cluster to `logs/cluster/` on your local machine:
```bash
./scripts/download_remote_logs.sh
```

Run the trajectory & stagnation analyzer to inspect duplicate query rates, zero-new-document turns, per-turn recall curves, and retrieval vs. aggregation failure breakdowns:
```bash
python analyze_trajectories.py --logs-dir logs
```
