# Deliverable 3: Resource Mapping Document

## 3.1 Resource Requirements Analysis

| RNN Characteristic          | Cloud Metric to Measure    | How to Determine (My Values)                                                                                              |
| :-------------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Number of parameters**    | Memory required (GB)       | `5.3M` (from `model.count_parameters()`). (5.3M \* 4 bytes) ≈ `20.35 MB`.                                                 |
| **Training iterations**     | Compute hours needed       | Measured `2.5 hours` on a simulated GPU instance.                                                                         |
| **Batch size**              | Memory footprint per batch | Profiled during training. Peak was `~1536 MB`.                                                                            |
| **Inference latency**       | Requests per second        | Benchmarked `~48.72 ms` average latency.                                                                                  |
| **Model checkpoint size**   | Storage requirements (GB)  | `20.35 MB` (from `os.path.getsize()`).                                                                                    |
| **Expected request volume** | Compute hours per month    | _Assumption:_ Medium scenario (10,000 req/day). (10,000 \* 48.72ms) = `487.2s` = `0.135 hours` of active compute per day. |

## 3.2 Right-Sizing and Justification

### Training Instance: `g4dn.xlarge` (GPU)

- **Justification:** Our RNN training is computationally intensive.
- **Memory:** The `g4dn.xlarge` has 16 GB of RAM, which easily covers our peak simulated usage of `~1.5 GB` (with a 2x headroom recommendation of 3 GB).
- **Compute:** It provides a NVIDIA T4 GPU, which is ideal for accelerating the matrix multiplications in an RNN, drastically reducing training time from days (on CPU) to hours (our 2.5-hour measurement).

### Inference Instance: `c5.large` (CPU)

- **Justification:** Inference is latency-sensitive but not as parallelizable as training. A single request only needs a fast CPU core, not a full GPU.
- **Memory:** The `c5.large` has 4 GB of RAM, which is more than enough for our 3 GB (peak + headroom) recommendation.
- **Compute:** The latency of `~49 ms` is well within acceptable limits for a "next word" suggestion feature. A GPU instance would be idle most of the time and cost 5-6x more.
- **Cost-Effectiveness:** At ~$0.085/hr, this instance is very cheap to run 24/7. For 10,000 requests/day, the total active compute is only ~8 minutes, but the instance must be "on" to accept requests.
