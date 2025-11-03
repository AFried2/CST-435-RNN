# instrumented_rnn.py
import time
import os
import psutil
import json
import random
import torch
import torch.nn as nn

# --- 1. Placeholder RNN Model ---
# (Replace this with your actual RNN model)
class PlaceholderRNN(nn.Module):
    def __init__(self, vocab_size=1000, embed_size=128, hidden_size=256):
        super(PlaceholderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        print(f"Model initialized with {self.count_parameters():,} parameters.")

    def forward(self, x, h):
        embed = self.embedding(x)
        out, h = self.rnn(embed, h)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, h
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- 2. Instrumentation Functions (from Activity) ---

def measure_training_time():
    """Simulates and measures training time."""
    print("Starting (simulated) training...")
    start_time = time.time()
    # SIMULATED TRAINING:
    # In a real script, your model.fit() or training loop would be here.
    # We will simulate a 2.5 hour training job.
    time.sleep(2.5) # Using 2.5s to simulate 2.5 hours
    # ---
    training_duration_seconds = time.time() - start_time
    training_duration_hours = training_duration_seconds / 3600  # Convert simulation to hours
    
    # FOR THIS SIMULATION, WE'LL HARDCODE THE 2.5 HOURS
    training_duration_hours = 2.5
    print(f"Training finished in {training_duration_hours:.2f} hours")
    return training_duration_hours

def measure_inference_time(model, num_runs=100):
    """Measures average inference latency."""
    print(f"Measuring inference time over {num_runs} runs...")
    latencies = []
    
    # Prepare dummy data for inference
    # (Replace with your actual data preprocessing)
    dummy_input = torch.randint(0, 1000, (1, 10)) # batch_size=1, seq_len=10
    dummy_hidden = (torch.zeros(2, 1, model.hidden_size),
                    torch.zeros(2, 1, model.hidden_size))

    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            # SIMULATED INFERENCE:
            output, dummy_hidden = model(dummy_input, dummy_hidden)
            # ---
            latency = time.time() - start
            latencies.append(latency)

    avg_latency_sec = sum(latencies) / len(latencies)
    avg_latency_ms = avg_latency_sec * 1000
    print(f"Average inference time: {avg_latency_ms:.2f} ms")
    return avg_latency_ms

def get_memory_usage():
    """Gets peak memory usage."""
    # This is a simple proxy. For real training, you'd want to
    # sample this during the training loop to find the peak.
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    
    # SIMULATION: Hardcode a more realistic peak memory
    mem_mb = 1536.0 # e.g., 1.5 GB peak
    print(f"Peak memory usage (simulated): {mem_mb:.2f} MB")
    return mem_mb

def get_model_size(model, model_path="rnn_model.pth"):
    """Saves model and gets its size."""
    # Save the model
    torch.save(model.state_dict(), model_path)
    
    # Get size
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"Model size on disk: {model_size_mb:.2f} MB")
    return model_size_mb

def get_dataset_size(path="./data"):
    """Simulates getting dataset size."""
    # In reality, you'd use the os.walk function from the activity.
    # We will simulate a 5 GB dataset.
    dataset_size_gb = 5.0
    print(f"Dataset size (simulated): {dataset_size_gb:.2f} GB")
    return dataset_size_gb

def generate_cost_metrics_report(training_hours, inference_latency_ms,
                                   model_size_mb, dataset_size_gb,
                                   peak_memory_mb):
    """
    Generate a comprehensive metrics report for cost calculation
    (Function from Section 4.2)
    """
    report = {
        "training_hours": round(training_hours, 2),
        "inference_latency_ms": round(inference_latency_ms, 2),
        "inferences_per_second": round(1000 / inference_latency_ms, 2),
        "inferences_per_hour": round(3600 / (inference_latency_ms / 1000), 2),
        "model_size_mb": round(model_size_mb, 2),
        "dataset_size_gb": round(dataset_size_gb, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "recommended_ram_gb": round((peak_memory_mb / 1024) * 2, 2) # 2x headroom
    }

    # Save to JSON for documentation
    report_path = "cost_metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSuccessfully generated metrics report at: {report_path}")
    return report

# --- 3. Main execution ---
def main():
    print("=== Starting RNN Cost Instrumentation ===")
    
    # Initialize model
    model = PlaceholderRNN()
    
    # Run measurements
    training_hours = measure_training_time()
    inference_latency_ms = measure_inference_time(model)
    peak_memory_mb = get_memory_usage()
    model_size_mb = get_model_size(model)
    dataset_size_gb = get_dataset_size()
    
    # Generate report
    report = generate_cost_metrics_report(
        training_hours=training_hours,
        inference_latency_ms=inference_latency_ms,
        model_size_mb=model_size_mb,
        dataset_size_gb=dataset_size_gb,
        peak_memory_mb=peak_memory_mb
    )
    
    print("\n--- METRICS REPORT ---")
    print(json.dumps(report, indent=2))
    print("=======================================")

if __name__ == "__main__":
    main()