# Deliverable 1: Cost Research Documentation

**Cloud Provider:** Amazon Web Services (AWS)
**Region:** US-East (N. Virginia, `us-east-1`)
**Date:** October 30, 2025

## 1. Primary Cost Components

| Cost Component          | Service / Instance    | Pricing                    | Why It Matters                                                     |
| :---------------------- | :-------------------- | :------------------------- | :----------------------------------------------------------------- |
| **Compute (Training)**  | `g4dn.xlarge` (GPU)   | ~$0.526 / hour (On-Demand) | Training requires a GPU for reasonable speed.                      |
| **Compute (Inference)** | `c5.large` (CPU)      | ~$0.085 / hour (On-Demand) | Inference is less intensive and can run on a cheaper CPU instance. |
| **Storage**             | EBS `gp3` (SSD)       | ~$0.08 / GB-month          | Storing the model checkpoints, logs, and the dataset.              |
| **Data Transfer**       | Data Egress           | ~$0.09 / GB (first 10TB)   | Cost to send results (predictions) back to the user.               |
| **Additional Services** | Elastic Load Balancer | ~$0.0225 / hour            | Needed to distribute inference requests.                           |

## 2. Pricing Sources (URLs)

- **AWS Pricing Calculator:** `https://calculator.aws/`
- **EC2 On-Demand Pricing:** `https://aws.amazon.com/ec2/pricing/on-demand/`
- **EBS Pricing:** `https://aws.amazon.com/ebs/pricing/`
- **Data Transfer Pricing:** `https://aws.amazon.com/ec2/pricing/internet-data-transfer/`

## 3. Assumptions Made

- **Pricing:** All prices are examples based on `us-east-1` On-Demand rates and can change.
- **Training:** We will use a **Spot Instance** for training, which could be up to 70-90% cheaper (e.g., ~$0.16/hr) but can be interrupted. The cost model will use the On-Demand price for a "worst-case" TCO.
- **Inference:** We will use an **On-Demand** instance for stable, continuous deployment. A **Reserved Instance** could save ~40% for a 1-year commitment.
- **Data Ingress:** Assumed to be $0.00 (free) as is standard.
- **Storage:** Assumed 50 GB of persistent SSD storage for the dataset, model, and logs.
