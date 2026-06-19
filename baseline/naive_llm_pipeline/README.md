# Naive LangGraph Baseline (ChatGPT)

This baseline is intentionally naive for token-cost benchmarking.
It reads raw CSV text, splits into large chunks, and sends each chunk directly to ChatGPT.

## Why this baseline
- Establishes a high-token reference point.
- Useful to compare against your later metadata-driven and local-agent approaches.

## Setup
1. Create a Python environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Set environment variables:
   - OPENAI_API_KEY=your_key
   - Optional: OPENAI_MODEL=gpt-4o

## Run
python baseline.py

## Files
- baseline.py: Entry point
- graph.py: LangGraph workflow definition
- nodes.py: Node logic
- state.py: Typed state
- prompts.py: Prompt templates
- config.py: Runtime config

## Suggested real dataset for this thesis
Use the NYC TLC Yellow Taxi Trip Records (monthly parquet files).
- Why: large, messy, mixed numeric/categorical/time fields, realistic preprocessing costs.
- Link: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Suggested smaller backup dataset
The UCI Adult Income dataset.
- Why: mixed feature types, missing values, easy for controlled experiments.
- Link: https://archive.ics.uci.edu/dataset/2/adult

## Tool ideas to build yourself (not premade)
1. Token Budget Estimator
   - Input: raw row samples or chunks
   - Output: estimated tokens before model call
2. Metadata-First Profiler
   - Scans only headers, dtypes, null rates, cardinality sketches
3. Chunk Router
   - Decides which chunks need LLM analysis vs simple local rules
4. Prompt Compressor
   - Rewrites raw chunk into compact statistical summaries
5. Local Small-Task Executor
   - Runs deterministic transforms locally (imputation, encoding, type casting)
6. Prompt-Result Cache
   - Content hash key to avoid repeated API calls on same chunk
7. Quality Gate Evaluator
   - Compares naive vs optimized outputs on data quality metrics
