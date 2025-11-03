# cost_model.py
import json
import pandas as pd

# --- 1. Pricing Constants ---
# Based on cost_research.md (AWS, us-east-1)
PRICING = {
    "compute_training_gpu_hr": 0.526,  # g4dn.xlarge (On-Demand)
    "compute_training_gpu_spot_hr": 0.158, # g4dn.xlarge (Spot - for optimization)
    "compute_inference_cpu_hr": 0.085, # c5.large (On-Demand)
    "storage_ssd_gb_month": 0.08,
    "data_transfer_egress_gb": 0.09,
    "load_balancer_hr": 0.0225,
}

# --- 2. Load Metrics from Report ---
try:
    with open("cost_metrics_report.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    print("Error: cost_metrics_report.json not found.")
    print("Please run instrumented_rnn.py first.")
    exit()

# --- 3. Cost Model Formulas ---

def calculate_training_cost(metrics, pricing):
    """Calculates the one-time cost of training the model."""
    compute_cost = metrics["training_hours"] * pricing["compute_training_gpu_hr"]
    
    # Assume dataset is stored for 1 month during training/dev
    dataset_storage_cost = metrics["dataset_size_gb"] * pricing["storage_ssd_gb_month"]
    
    # Assume 1-time data ingress (free) and no egress during training
    data_transfer_cost = 0.0 
    
    total = compute_cost + dataset_storage_cost + data_transfer_cost
    
    return {
        "compute_cost": compute_cost,
        "storage_cost": dataset_storage_cost,
        "data_transfer_cost": data_transfer_cost,
        "total_training_cost": total
    }

def calculate_inference_cost(metrics, pricing, requests_per_day):
    """Calculates the ongoing monthly cost of deployment."""
    # --- Compute Cost ---
    # We need one c5.large instance running 24/7
    hours_per_month = 24 * 30
    compute_cost = hours_per_month * pricing["compute_inference_cpu_hr"]
    
    # --- Load Balancer Cost ---
    lb_cost = hours_per_month * pricing["load_balancer_hr"]

    # --- Storage Cost ---
    # Store the model file (20.35 MB) + 5GB for logs
    storage_gb = (metrics["model_size_mb"] / 1024) + 5
    storage_cost = storage_gb * pricing["storage_ssd_gb_month"]
    
    # --- Data Transfer Cost ---
    # Assume each request/response is 5 KB
    kb_per_request = 5
    gb_per_request = kb_per_request / (1024**3)
    total_gb_egress = (requests_per_day * 30) * gb_per_request
    data_transfer_cost = total_gb_egress * pricing["data_transfer_egress_gb"]
    
    # --- Total ---
    total = compute_cost + storage_cost + data_transfer_cost + lb_cost
    
    return {
        "compute_cost": compute_cost,
        "storage_cost": storage_cost,
        "data_transfer_cost": data_transfer_cost,
        "load_balancer_cost": lb_cost,
        "total_monthly_cost": total
    }

def calculate_key_metrics(monthly_cost, requests_per_day):
    """Calculates standard industry metrics."""
    requests_per_month = requests_per_day * 30
    if requests_per_month == 0:
        return {
            "cost_per_inference": float('inf'),
            "inferences_per_dollar": 0
        }
        
    cost_per_inference = monthly_cost / requests_per_month
    inferences_per_dollar = 1 / cost_per_inference
    
    return {
        "cost_per_inference": cost_per_inference,
        "inferences_per_dollar": inferences_per_dollar
    }

# --- 4. Main Report Generation ---
def main():
    print("=== Running Cost Model Calculations ===")
    
    # --- Training ---
    training_costs = calculate_training_cost(metrics, PRICING)
    print("\n--- 1. TRAINING COSTS (One-time) ---")
    print(f"   Compute:        ${training_costs['compute_cost']:.2f} ({metrics['training_hours']} hours * ${PRICING['compute_training_gpu_hr']}/hr)")
    print(f"   Storage:        ${training_costs['storage_cost']:.2f} ({metrics['dataset_size_gb']} GB * ${PRICING['storage_ssd_gb_month']}/GB-mo)")
    print(f"   Total Training: ${training_costs['total_training_cost']:.2f}")

    # --- Sensitivity Analysis ---
    scenarios = {
        "Low (100 req/day)": 100,
        "Medium (10,000 req/day)": 10000,
        "High (1,000,000 req/day)": 1000000
    }
    
    report_data = []

    print("\n--- 2. INFERENCE COSTS (Monthly) ---")
    for scenario, req_count in scenarios.items():
        inference_costs = calculate_inference_cost(metrics, PRICING, req_count)
        key_metrics = calculate_key_metrics(inference_costs["total_monthly_cost"], req_count)
        
        print(f"\n   Scenario: {scenario}")
        print(f"   - Compute:        ${inference_costs['compute_cost']:.2f}")
        print(f"   - Storage:        ${inference_costs['storage_cost']:.2f}")
        print(f"   - Data Transfer:  ${inference_costs['data_transfer_cost']:.2f}")
        print(f"   - Load Balancer:  ${inference_costs['load_balancer_cost']:.2f}")
        print(f"   - TOTAL MONTHLY:  ${inference_costs['total_monthly_cost']:.2f}")
        print(f"   - Cost/Inference: ${key_metrics['cost_per_inference'] * 100:.6f} cents")
        print(f"   - Inferences/$1:  {key_metrics['inferences_per_dollar']:,.0f}")
        
        report_data.append({
            "Scenario": scenario,
            "Requests/Day": req_count,
            "Total Monthly Cost ($)": inference_costs['total_monthly_cost'],
            "Cost per 1000 Inferences ($)": key_metrics['cost_per_inference'] * 1000
        })

    # --- 3. Visualization Data ---
    print("\n--- 3. Data for Visualization ---")
    df = pd.DataFrame(report_data)
    print("Cost vs. Request Volume:")
    print(df.to_markdown(index=False))
    
    pie_data = calculate_inference_cost(metrics, PRICING, 10000) # Medium scenario
    print("\nCost Breakdown (Medium Scenario):")
    pie_df = pd.DataFrame([
        {"Component": "Compute", "Cost": pie_data["compute_cost"]},
        {"Component": "Load Balancer", "Cost": pie_data["load_balancer_cost"]},
        {"Component": "Storage", "Cost": pie_data["storage_cost"]},
        {"Component": "Data Transfer", "Cost": pie_data["data_transfer_cost"]},
    ])
    pie_df["Percentage"] = (pie_df["Cost"] / pie_df["Cost"].sum()) * 100
    print(pie_df.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()