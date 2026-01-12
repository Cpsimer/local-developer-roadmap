# Exact Next Steps: Local Developer Roadmap Deployment

**Document Date:** January 12, 2026
**Status:** Phase 1 Planning → Phase 2 Hardware Validation
**Urgency:** Critical Path Items Identified

---

## Executive Summary

Your `local-developer-roadmap` repository contains comprehensive planning documentation for a distributed AI IDP spanning:
- **AI Desktop (Schlimers-server)**: RTX 5070 Ti + Ryzen 9900X
- **XPS 15 (Stationary Workhorse)**: i9-9980HK + GTX 1650 (Proxmox LXC host)
- **XPS 13 (Developer Workstation)**: Intel Core Ultra 7 + NPU
- **Jetson Orin Nano Super**: Edge inference node

**Current State:**
- ✅ Strategic planning complete (Perplexity LABS 001, Research 002)
- ✅ Single-user optimization documented (Perplexity-Labs-003)
- ⚠️ No active deployment / production verification
- ⚠️ RTX 5070 Ti benchmarks unvalidated (hardware not yet released as of Jan 2026)
- ⚠️ Performance claims require empirical verification

**Critical Path to Production:**
1. **Hardware Validation** (Week 1-2): Benchmark RTX 5070 Ti vs projected specs
2. **Docker Infrastructure Setup** (Week 2-3): Deploy vLLM + Triton on AI Desktop
3. **LXC Integration** (Week 3-4): Connect XPS 15 Proxmox services with GPU services
4. **Jetson Edge Integration** (Week 4-5): USB 4.0 RPC latency testing + model distribution
5. **Knowledge Management Pipeline** (Week 5-6): n8n + Obsidian automation

---

## Part 1: Impact Analysis & Strategic Goals

### Business Goals (Why This Matters)

| Goal | Impact | Success Metric | Timeline |
|------|--------|---------------|-----------|
| **Privacy-First AI Development** | Eliminate cloud dependency, full data sovereignty | Zero external API calls for inference | Week 1-2 |
| **Maximum Hardware Utilization** | Transform idle Ryzen 9900X → 80%+ utilization | CPU throughput: 144-200 tok/s @ 10 threads | Week 3-4 |
| **Reproducible Research** | Automated experiment tracking & knowledge capture | MLflow + n8n + Obsidian sync daily | Week 5-6 |
| **Real-time Model Optimization** | Deploy quantized models in minutes, not hours | Model switching: <30 seconds vs 5-10 min enterprise | Week 2-3 |
| **Scalable Inference Pipeline** | Support 32-64 concurrent inference requests | vLLM PagedAttention: 32+ sequences @ RTX 5070 Ti | Week 3-4 |

### Technical Objectives (What Success Looks Like)

**Tier 1 (Must-Have by Week 4):**
- [ ] vLLM serving Llama 3.1 8B FP8 at realistic throughput (60-100 tok/s, not optimistic 140-170)
- [ ] Triton Inference Server configured for multi-model serving
- [ ] CPU inference engine (llama.cpp) utilizing Ryzen 9900X at 100+ tok/s aggregate
- [ ] GPU metrics exported to Prometheus (DCGM Exporter)
- [ ] Reverse proxy (NGINX) on XPS 15 → AI Desktop inference endpoints

**Tier 2 (Should-Have by Week 6):**
- [ ] NeMo training pipeline for custom model fine-tuning
- [ ] MLflow experiment tracking integrated with inference pipeline
- [ ] n8n workflow automation connecting Git commits → Obsidian notes
- [ ] Jetson Orin Nano edge inference with USB 4.0 layer splitting
- [ ] Grafana dashboards for performance monitoring

**Tier 3 (Could-Have by Week 8):**
- [ ] Run:ai dynamic GPU fractionation for multi-tenant workloads
- [ ] NVIDIA AI Enterprise evaluation and licensing
- [ ] Speculative decoding (CPU draft + GPU verify) for 2x speedup
- [ ] OpenVINO NPU inference on XPS 13 Intel AI Boost

---

## Part 2: Detailed Action Plan (Week-by-Week)

### Week 1: Hardware Validation & Baseline Benchmarking

#### Goal
Establish realistic performance bounds for all hardware. Identify gaps between documented claims and actual performance.

#### Tasks (Priority Order)

**1.1: RTX 5070 Ti Baseline Benchmarking** [CRITICAL]

```bash
# On AI Desktop (Schlimers-server):

# Step 1: Verify CUDA runtime and drivers
ubuntu@schlimers:~$ nvidia-smi
# Expected: RTX 5070 Ti, Driver 580.95.05, CUDA 13.0.2

# Step 2: Install vLLM and base dependencies
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# Step 3: Download Llama 3.1 8B FP8 quantized model
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --cache-dir /mnt/models/llama-3.1-8b \
  --resume-download

# Step 4: Run baseline vLLM benchmark
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/models/llama-3.1-8b-fp8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 32 \
  --port 8000 &

# Step 5: Benchmark with curls (in parallel)
for i in {1..5}; do
  time curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama-3.1-8b-instruct",
      "prompt": "Explain quantum computing in simple terms.",
      "max_tokens": 100,
      "temperature": 0.7
    }' &
done

# Record:
# - Time to first token (TTFT) - measure in milliseconds
# - Total generation time
# - GPU memory used (nvidia-smi dmon)
# - GPU utilization (%)
# - Token throughput (tokens/generation_time)
```

**Expected Results (NOT Optimistic Claims):**
- TTFT: 80-150ms (NOT 22ms)
- Throughput: 60-100 tok/s (NOT 140-170 tok/s)
- GPU Memory: 12-14GB / 16GB
- GPU Utilization: 85-95%

**Deliverable:** `benchmarks/week1_rtx5070ti_baseline.csv`

**Uncertainty Flags:**
- ⚠️ RTX 5070 Ti not yet released (as of January 2026) - benchmarks are projections
- ⚠️ GDDR7 memory bandwidth (896 GB/s) significantly lower than H100 HBM3e (3.35 TB/s)
- ⚠️ Practical performance will be 40-60% of datacenter GPU equivalents

---

**1.2: Ryzen 9900X CPU Inference Benchmarking** [HIGH]

```bash
# On AI Desktop CPU cores (10/12 dedicated to inference):

# Step 1: Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Step 2: Download Llama 3.2 3B Q4_K_M (quantized)
huggingface-cli download TheBloke/Llama-3.2-3B-Instruct-GGUF \
  --cache-dir /mnt/models/llama-3.2-3b-q4 \
  --resume-download

# Step 3: Benchmark single-threaded
time ./main \
  -m /mnt/models/llama-3.2-3b-q4/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -n 100 \
  -p "Explain quantum computing in simple terms." \
  -t 1 \
  -ngl 0

# Step 4: Benchmark with max threads
time ./main \
  -m /mnt/models/llama-3.2-3b-q4/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -n 100 \
  -p "Explain quantum computing in simple terms." \
  -t 10 \
  -ngl 0

# Step 5: Benchmark parallel batching
time ./main \
  -m /mnt/models/llama-3.2-3b-q4/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -n 100 \
  -p "Explain quantum computing in simple terms." \
  -t 10 \
  -np 8 \
  -ngl 0
```

**Expected Results:**
- Single-threaded: 15-25 tok/s
- 10-threaded: 120-150 tok/s
- 10-threaded + 8 parallel: 144-200 tok/s aggregate (matches documentation)

**Deliverable:** `benchmarks/week1_ryzen9900x_baseline.csv`

---

**1.3: Jetson Orin Nano Baseline** [MEDIUM]

```bash
# On Jetson Orin Nano via SSH:

jetsondocker run --rm --runtime nvidia \
  -v /mnt/models:/models \
  nvcr.io/nvidia/jetson-inference:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/llama-3.2-1b-q4 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 2 \
    --port 8001

# Measure: tokens/s for 1B Q4 model (expect 5-10 tok/s)
```

**Deliverable:** `benchmarks/week1_jetson_baseline.csv`

---

**1.4: Create Benchmarking Script** [MEDIUM]

```python
# benchmarks/benchmark_suite.py
import time
import requests
import statistics
import csv
from datetime import datetime

def benchmark_inference_endpoint(url, model_name, prompt, iterations=5):
    """Benchmark inference endpoint with consistent methodology."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'endpoint': url,
        'ttfts': [],
        'throughputs': [],
        'total_times': []
    }
    
    for i in range(iterations):
        start = time.perf_counter()
        response = requests.post(f"{url}/v1/completions", json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "stream": False
        })
        total_time = time.perf_counter() - start
        
        data = response.json()
        tokens = data['usage']['completion_tokens']
        
        results['ttfts'].append(total_time * 1000)  # ms
        results['throughputs'].append(tokens / total_time)
        results['total_times'].append(total_time)
    
    # Aggregate statistics
    results['stats'] = {
        'ttft_p50': statistics.median(results['ttfts']),
        'ttft_p95': sorted(results['ttfts'])[int(0.95 * len(results['ttfts']))],
        'throughput_mean': statistics.mean(results['throughputs']),
        'throughput_stdev': statistics.stdev(results['throughputs'])
    }
    
    return results

if __name__ == "__main__":
    # Write to CSV
    endpoints = [
        ("http://localhost:8000", "llama-3.1-8b-fp8"),
        ("http://localhost:8003", "llama-3.2-3b-q4"),
        ("http://jetson.local:8001", "llama-3.2-1b-q4")
    ]
    
    with open(f"benchmarks/{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'endpoint', 'model', 'ttft_p50_ms', 'ttft_p95_ms', 'throughput_tok_s', 'stdev'])
        writer.writeheader()
        
        for url, model in endpoints:
            results = benchmark_inference_endpoint(url, model, "Explain quantum computing.")
            writer.writerow({
                'timestamp': results['timestamp'],
                'endpoint': url,
                'model': model,
                'ttft_p50_ms': f"{results['stats']['ttft_p50']:.1f}",
                'ttft_p95_ms': f"{results['stats']['ttft_p95']:.1f}",
                'throughput_tok_s': f"{results['stats']['throughput_mean']:.1f}",
                'stdev': f"{results['stats']['throughput_stdev']:.1f}"
            })
```

**Deliverable:** `benchmarks/benchmark_suite.py`

---

**1.5: Document Findings in Analysis Report** [HIGH]

Create `WEEK1_FINDINGS.md` with:
- Actual vs. documented performance claims
- Confidence levels for each metric
- Identified gaps requiring correction
- Hardware limitations discovered

**Acceptance Criteria:**
- All three inference endpoints (AI Desktop GPU, AI Desktop CPU, Jetson) benchmarked
- Data saved as CSV with consistent format
- Analysis identifies which claims from Perplexity-Labs-003 require revision
- Report includes confidence intervals and methodology

---

### Week 2: Docker Infrastructure & vLLM Deployment

#### Goal
Set up production-ready inference services on AI Desktop with corrected configurations based on Week 1 benchmarks.

#### Tasks

**2.1: Create Production docker-compose.yml** [CRITICAL]

Based on Week 1 findings, replace overoptimistic settings:

```yaml
# /home/ubuntu/ai-idp/docker-compose.yml
version: '3.8'
services:

  # CORRECTED vLLM Configuration
  vllm-primary:
    image: vllm/vllm-openai:v0.7.1
    container_name: vllm-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HF_TOKEN}  # From .env file
    volumes:
      - /mnt/models/llama-3.1-8b-fp8:/models:ro
      - /home/ubuntu/ai-idp/logs:/app/logs
    ports:
      - "8000:8000"
    command: >
      /bin/bash -c "python -m vllm.entrypoints.openai.api_server
      --model /models/llama-3.1-8b-instruct
      --quantization fp8
      --kv-cache-dtype fp8
      --dtype float16
      --max-model-len 16384
      --gpu-memory-utilization 0.80
      --max-num-seqs 32
      --max-num-batched-tokens 4096
      --enable-chunked-prefill
      --enable-prefix-caching
      --block-size 16
      --served-model-name llama-3.1-8b
      --api-key ${VLLM_API_KEY}
      --host 0.0.0.0
      --port 8000
      --disable-log-stats"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # llama.cpp CPU inference
  llamacpp-cpu:
    image: ghcr.io/ggerganov/llama.cpp:server
    container_name: llamacpp-inference
    volumes:
      - /mnt/models/llama-3.2-3b-q4:/models:ro
    ports:
      - "8003:8080"
    command: >
      -m /models/llama-3.2-3b-instruct-Q4_K_M.gguf
      -c 8192
      -t 10
      -tb 64
      -np 8
      --cont-batching
      --log-disable
      --api-key ${LLAMACPP_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Triton Inference Server
  triton-server:
    image: nvcr.io/nvidia/tritonserver:25.05-py3
    container_name: triton-inference
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - /mnt/models/triton:/models
    ports:
      - "8001:8000"  # HTTP
      - "8002:8001"  # gRPC
      - "8003:8002"  # Metrics
    command: tritonserver --model-repository=/models --log-verbose=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    restart: unless-stopped

  # DCGM GPU Metrics Exporter
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
    container_name: dcgm-exporter
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "9400:9400"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    restart: unless-stopped

networks:
  default:
    name: ai-idp-network
    driver: bridge
```

**Deliverable:** `/home/ubuntu/ai-idp/docker-compose.yml`

**Key Corrections from Perplexity-Labs-003:**
1. ✅ `--gpu-memory-utilization 0.80` (not 0.85) - safer margin for OOM
2. ✅ `--max-num-seqs 32` (not 128) - realistic for single-user, reduces memory pressure
3. ✅ `--max-num-batched-tokens 4096` - prevents L3 cache thrashing
4. ✅ API keys use secure generation: `openssl rand -hex 32`
5. ✅ Healthchecks added for service monitoring

---

**2.2: Create .env Configuration File** [HIGH]

```bash
# /home/ubuntu/ai-idp/.env
# SECURITY: Store securely, restrict file permissions (chmod 600)

# NVIDIA NGC API
NGC_API_KEY=your_ngc_key_here
HF_TOKEN=your_huggingface_token

# API Security
VLLM_API_KEY=$(openssl rand -hex 32)
LLAMACPP_API_KEY=$(openssl rand -hex 32)
TRITON_API_KEY=$(openssl rand -hex 32)

# Model Paths
MODEL_CACHE_DIR=/mnt/models
LOG_DIR=/home/ubuntu/ai-idp/logs

# Resource Limits
GPU_MEMORY_UTILIZATION=0.80
GPU_MAX_SEQS=32
CPU_THREADS=10
CPU_PARALLEL=8
```

**Deliverable:** `/home/ubuntu/ai-idp/.env` (with secure permissions)

---

**2.3: Model Preparation** [HIGH]

```bash
# /home/ubuntu/ai-idp/scripts/prepare_models.sh

#!/bin/bash
set -e

echo "[$(date)] Starting model preparation..."

MODEL_DIR="/mnt/models"
mkdir -p $MODEL_DIR/{llama-3.1-8b-fp8,llama-3.2-3b-q4,triton}

# Load environment
source /home/ubuntu/ai-idp/.env

# Download Llama 3.1 8B FP8 (for vLLM GPU)
echo "[$(date)] Downloading Llama 3.1 8B FP8..."
huggingface-cli download \
  TheBloke/Llama-3.1-8B-Instruct-GGUF \
  --cache-dir $MODEL_DIR/llama-3.1-8b-fp8 \
  --resume-download \
  --token $HF_TOKEN

# Download Llama 3.2 3B Q4 (for llama.cpp CPU)
echo "[$(date)] Downloading Llama 3.2 3B Q4..."
huggingface-cli download \
  TheBloke/Llama-3.2-3B-Instruct-GGUF \
  llama-3.2-3b-instruct-Q4_K_M.gguf \
  --cache-dir $MODEL_DIR/llama-3.2-3b-q4 \
  --resume-download \
  --token $HF_TOKEN

# Download Llama 3.2 1B Q4 (for Jetson)
echo "[$(date)] Downloading Llama 3.2 1B Q4..."
huggingface-cli download \
  TheBloke/Llama-3.2-1B-Instruct-GGUF \
  llama-3.2-1b-instruct-Q4_K_M.gguf \
  --cache-dir $MODEL_DIR/llama-3.2-1b-q4 \
  --resume-download \
  --token $HF_TOKEN

echo "[$(date)] Model preparation complete!"
echo "Total models cached: $(find $MODEL_DIR -type f | wc -l)"
```

**Deliverable:** `/home/ubuntu/ai-idp/scripts/prepare_models.sh` (executable)

---

**2.4: Deploy Services** [CRITICAL]

```bash
# On AI Desktop (Schlimers-server):

cd /home/ubuntu/ai-idp

# Create logs directory
mkdir -p logs

# Load environment variables
set -a
source .env
set +a

# Start services (detached mode)
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f vllm-primary
# Wait for: "INFO:     Application startup complete"
```

**Expected Output:**
```
NAME             STATUS          PORTS
vllm-gpu         Up 2 minutes    0.0.0.0:8000->8000/tcp
llamacpp-cpu     Up 2 minutes    0.0.0.0:8003->8080/tcp
triton-inference Up 1 minute     0.0.0.0:8001->8000/tcp, 0.0.0.0:8002->8001/tcp
dcgm-exporter    Up 1 minute     0.0.0.0:9400->9400/tcp
```

**Deliverable:** All containers running and healthy

---

**2.5: Validate Endpoints** [HIGH]

```bash
# Test vLLM
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "llama-3.1-8b",
    "prompt": "Explain quantum computing:",
    "max_tokens": 50
  }' | jq .

# Test llama.cpp
curl http://localhost:8003/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "n_predict": 50
  }' | jq .

# Test Triton health
curl http://localhost:8001/v2/health/ready

# Test DCGM metrics
curl http://localhost:9400/metrics | head -20
```

**Acceptance Criteria:**
- ✅ vLLM responds with valid completions within 5 seconds
- ✅ llama.cpp responds with valid completions within 5 seconds
- ✅ Triton health endpoint returns 200
- ✅ DCGM exports GPU metrics in Prometheus format

---

**2.6: Performance Validation Against Week 1 Baseline** [HIGH]

Run benchmarking suite from Week 1 with docker-compose services:

```bash
python3 benchmarks/benchmark_suite.py
```

Compare results to Week 1 baseline. Expect:
- vLLM TTFT: 80-150ms (consistent with Week 1 standalone)
- llama.cpp throughput: 144-200 tok/s (consistent)
- No performance regression from containerization

**Deliverable:** `benchmarks/week2_production_validation.csv`

---

### Week 3: XPS 15 Integration (Reverse Proxy & LXC Services)

#### Goal
Connect XPS 15 Proxmox infrastructure to AI Desktop GPU services via NGINX reverse proxy and establish monitoring pipeline.

#### Tasks

**3.1: NGINX Reverse Proxy Configuration** [HIGH]

On XPS 15 LXC 1 (Sclimers-Gateway) - NGINX reverse proxy:

```nginx
# /etc/nginx/sites-available/ai-idp.conf

upstream vllm_backend {
    server schlimers-server:8000;
    keepalive 32;
}

upstream llamacpp_backend {
    server schlimers-server:8003;
    keepalive 32;
}

upstream triton_backend {
    server schlimers-server:8001;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name vllm.sclimers.local llamacpp.sclimers.local triton.sclimers.local;
    
    ssl_certificate /etc/letsencrypt/live/sclimers.local/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sclimers.local/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # vLLM endpoint
    location ~ ^/vllm/(.*) {
        proxy_pass http://vllm_backend/$1;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        
        # Authentication
        proxy_set_header Authorization $http_authorization;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # llama.cpp endpoint
    location ~ ^/llamacpp/(.*) {
        proxy_pass http://llamacpp_backend/$1;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
    
    # Triton endpoint
    location ~ ^/triton/(.*) {
        proxy_pass http://triton_backend/$1;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
    
    # Metrics endpoint (restricted to localhost)
    location /metrics {
        allow 127.0.0.1;
        allow 192.168.1.0/24;  # Local network
        deny all;
        
        proxy_pass http://vllm_backend/metrics;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}
```

**Deliverable:** `/etc/nginx/sites-available/ai-idp.conf` (linked to sites-enabled)

**Validation:**
```bash
# On XPS 15:
nginx -t  # Syntax check
sudo systemctl reload nginx

# Test from XPS 13:
curl -k https://vllm.sclimers.local/vllm/health
```

---

**3.2: Prometheus Scrape Configuration** [MEDIUM]

On XPS 15 LXC 5 (Sclimers-Observation) - update Prometheus config:

```yaml
# /etc/prometheus/prometheus.yml (append to scrape_configs)

scrape_configs:
  # Existing configs...
  
  - job_name: 'ai-desktop-gpu'
    static_configs:
      - targets: ['schlimers-server:9400']
    scrape_interval: 15s
    metrics_path: '/metrics'
  
  - job_name: 'vllm-inference'
    static_configs:
      - targets: ['schlimers-server:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
  
  - job_name: 'llamacpp-inference'
    static_configs:
      - targets: ['schlimers-server:8003']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

**Deliverable:** Updated `/etc/prometheus/prometheus.yml`

**Validation:**
```bash
# On XPS 15:
sudo systemctl reload prometheus

# Check targets
curl http://localhost:9090/api/v1/targets
```

---

**3.3: Grafana Dashboard Creation** [MEDIUM]

Create inference monitoring dashboard in Grafana (XPS 15 LXC 6):

```json
// POST http://localhost:3000/api/dashboards/db
{
  "dashboard": {
    "title": "AI IDP Inference Monitoring",
    "panels": [
      {
        "title": "GPU Utilization (%)",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_UTIL"
        }]
      },
      {
        "title": "GPU Memory Usage (GB)",
        "targets": [{
          "expr": "DCGM_FI_DEV_FB_USED / 1024 / 1024 / 1024"
        }]
      },
      {
        "title": "vLLM Throughput (tok/s)",
        "targets": [{
          "expr": "rate(vllm_tokens_generated_total[1m])"
        }]
      },
      {
        "title": "Inference Latency (P95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, vllm_request_duration_seconds_bucket)"
        }]
      }
    ]
  }
}
```

**Deliverable:** Grafana dashboard exported as JSON

---

### Week 4: Jetson Integration & Edge Deployment

#### Goal
Validate USB 4.0 RPC latency and deploy edge inference model on Jetson Orin Nano.

#### Tasks

**4.1: USB 4.0 Latency Profiling** [CRITICAL]

This addresses the "2-5ms minimum, not 0.4ms" issue from the critique:

```python
# benchmarks/jetson_usb4_latency.py
import time
import requests
import statistics
from datetime import datetime

def measure_rpc_latency():
    """Measure round-trip RPC latency from AI Desktop → Jetson."""
    
    jetson_url = "http://192.168.1.50:8001"  # Jetson static IP
    latencies = []
    
    for i in range(100):
        start = time.perf_counter()
        try:
            response = requests.get(f"{jetson_url}/health", timeout=5)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            if response.status_code == 200:
                latencies.append(elapsed)
        except Exception as e:
            print(f"Error: {e}")
    
    if latencies:
        print(f"USB 4.0 RPC Latency:")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  P50: {statistics.median(latencies):.2f}ms")
        print(f"  P95: {sorted(latencies)[int(0.95*len(latencies))]:.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        print(f"  Mean: {statistics.mean(latencies):.2f}ms")
        print(f"  StdDev: {statistics.stdev(latencies):.2f}ms")
    else:
        print("No successful measurements.")

if __name__ == "__main__":
    measure_rpc_latency()
```

**Expected Results (Realistic):**
- Min: 1.5-2ms
- P50: 2-3ms
- P95: 3-5ms
- Max: 8-15ms

**Deliverable:** `benchmarks/jetson_usb4_latency_report.md`

---

**4.2: Deploy Jetson Edge Model** [HIGH]

```bash
# On Jetson Orin Nano:

# Install dependencies
sudo apt-get update
sudo apt-get install -y docker.io git

# Download model
huggingface-cli download \
  TheBloke/Llama-3.2-1B-Instruct-GGUF \
  llama-3.2-1b-instruct-Q4_K_M.gguf \
  --cache-dir /mnt/models

# Deploy via docker-compose
docker-compose -f jetson-docker-compose.yml up -d
```

**Deliverable:** Jetson inference endpoint responding at `/jetson-orin:8001`

---

### Week 5: Knowledge Management Integration

#### Goal
Automate n8n → Obsidian pipeline for experiment tracking and knowledge capture.

#### Tasks (Abbreviated - see `/USER-PROGRAMS-INTEGRATION.md` for full spec)

**5.1: n8n Workflow Setup** [MEDIUM]
**5.2: Obsidian Vault Configuration** [MEDIUM]
**5.3: Automated Git Commit → Note Pipeline** [HIGH]

---

### Week 6: Testing & Validation

#### Goal
Full system integration testing and performance validation.

#### Acceptance Criteria

```bash
✅ All inference endpoints healthy (passing healthchecks)
✅ Prometheus collecting metrics from all sources
✅ Grafana dashboards displaying real-time performance
✅ Jetson RPC latency measured and documented
✅ n8n automating Obsidian notes from Git commits
✅ MLflow experiment tracking integrated with inference
✅ All performance metrics ≥ 80% of conservative projections
```

---

## Part 3: Critical Gaps & Risk Mitigation

### Identified Issues from Repository Analysis

| Issue | Severity | Mitigation | Owner | Timeline |
|-------|----------|-----------|-------|----------|
| RTX 5070 Ti benchmarks unvalidated (hardware not released) | 🔴 CRITICAL | Establish baseline Week 1; use conservative estimates | You | Week 1 |
| 70B Q4 hybrid config physically impossible with 16GB VRAM | 🔴 CRITICAL | Remove from roadmap; recommend 8B or wait for 40GB GPUs | You | Week 1 |
| API keys generated with predictable timestamp | 🟡 MEDIUM | Switch to `openssl rand -hex 32` | Engineering | Week 2 |
| No thermal monitoring for sustained 85% GPU util | 🟡 MEDIUM | Deploy DCGM Exporter + alerts in Grafana | Engineering | Week 2 |
| No backup strategy for 500GB model cache | 🟡 MEDIUM | Set up 2TB NAS backup with daily rsync | Infrastructure | Week 3 |
| USB 4.0 RPC latency 0.4ms claim unverified | 🟠 HIGH | Measure actual latency Week 4; expect 2-5ms | Engineering | Week 4 |
| Docker socket exposed without security opts | 🟠 HIGH | Add `--security-opt no-new-privileges` | Security | Week 2 |
| No UPS/power budget documentation | 🟡 MEDIUM | Document power draw; recommend UPS sizing | Operations | Week 1 |

---

## Part 4: Success Metrics & Milestones

### Definition of Done (by Week 6)

**Functionality:**
- [ ] vLLM serving Llama 3.1 8B at stable throughput (60-100 tok/s)
- [ ] Triton multi-model serving with <10ms endpoint latency
- [ ] llama.cpp CPU inference utilizing Ryzen 9900X at 100+ tok/s
- [ ] Jetson edge inference with working USB 4.0 RPC
- [ ] Prometheus + Grafana monitoring live with 7-day retention
- [ ] n8n automating knowledge capture to Obsidian

**Performance:**
- [ ] GPU utilization: >80% during inference
- [ ] CPU utilization: 50-70% (headroom for system tasks)
- [ ] Memory efficiency: VRAM fragmentation <15%
- [ ] Inference latency P95: <200ms for 8B models

**Reliability:**
- [ ] All services pass healthchecks continuously
- [ ] Zero unplanned downtime in Week 5-6
- [ ] Automatic restart on failure (via docker-compose)
- [ ] Metrics retention: 30 days minimum

**Security:**
- [ ] All API keys stored in `.env` (not in logs/git)
- [ ] NGINX TLS enabled for all external endpoints
- [ ] Local network isolation verified
- [ ] No hardcoded credentials in any configs

---

## Part 5: Dependency & Resource Checklist

### Required Hardware
- ✅ RTX 5070 Ti (Schlimers-server)
- ✅ Ryzen 9900X (Schlimers-server)
- ✅ 128GB RAM (Schlimers-server)
- ✅ XPS 15 Proxmox LXC host
- ✅ Jetson Orin Nano Super
- ⚠️ 2.5GbE network (verify UniFi infrastructure)
- ⚠️ NAS storage for models (500GB+)
- ⚠️ UPS (for power stability during training)

### Required Software
- ✅ Docker Engine + Docker Compose
- ✅ NVIDIA CUDA 13.0.2 + cuDNN
- ✅ Prometheus + Grafana
- ✅ NGINX
- ✅ Python 3.10+
- ✅ Hugging Face CLI
- ✅ vLLM, llama.cpp, Triton

### Credentials Needed
- 🔐 Hugging Face API token (for model downloads)
- 🔐 NVIDIA NGC API key (for enterprise containers)
- 🔐 TLS certificates for HTTPS (self-signed or Let's Encrypt)

---

## Part 6: Handoff & Next Phase Planning

### After Week 6: Phase 2 Roadmap

Once core infrastructure is validated, consider:

**Phase 2A: Advanced Optimization** (Weeks 7-8)
- Speculative decoding implementation (CPU draft + GPU verify)
- OpenVINO NPU inference on XPS 13
- NeMo training pipeline for custom fine-tuning

**Phase 2B: Enterprise Features** (Weeks 9-10)
- Run:ai dynamic GPU fractionation
- NVIDIA AI Enterprise evaluation
- Multi-user isolation with Kubernetes

**Phase 2C: Scaling** (Weeks 11-12)
- Additional GPU node (H100/RTX 6000) for training
- Distributed inference across Jetson cluster
- Hybrid cloud burst compute (optional)

---

## Summary: Immediate Actions

**TODAY (January 12, 2026):**
1. Review this document with full context
2. Create `/home/ubuntu/ai-idp` directory on Schlimers-server
3. Copy `docker-compose.yml` and `.env.template` to repo
4. Schedule Week 1 benchmarking timeline

**Week 1 (Jan 13-19):**
1. Baseline RTX 5070 Ti vLLM performance
2. Baseline Ryzen 9900X llama.cpp throughput
3. Document findings vs. optimistic claims

**Week 2 (Jan 20-26):**
1. Deploy corrected docker-compose.yml
2. Validate all three inference endpoints
3. Begin performance monitoring with Prometheus

---

**Document Status:** FINAL PLANNING PHASE
**Next Review:** After Week 1 benchmarking (January 19, 2026)
**Prepared By:** AI Analysis System
**For:** Cpsimer (Operations Manager)
