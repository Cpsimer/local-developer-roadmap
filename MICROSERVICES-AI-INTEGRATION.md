# Microservices-AI Integration Architecture
**Production System + AI IDP Unified Architecture | January 12, 2026**

---

## Executive Summary

Your **6 LXC microservices** (Gateway, Core-Ops, Verify-Ops, Data-Ops, Observation, Runner) will integrate with the **AI IDP** (Schlimers GPU + Jetson) through well-defined **API contracts, authentication flows, and data pipelines**. This document provides the complete architecture with networking best practices.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Arrowave UDM SE (10.5.0.1)                      │
│                    Gateway + DNS + Firewall                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴───────────────────┐
        │     VLAN 10.5.4.0/24 (Production)       │
        └─────────────────────┬───────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌───▼─────────────┐    ┌─────▼─────────┐    ┌─────────▼────────┐
│ Proxmox Host    │    │ Schlimers GPU │    │ Jetson Orin      │
│ 10.5.4.45       │    │ 10.5.4.150    │    │ 10.5.4.151       │
│                 │    │               │    │                  │
│ ┌─────────────┐ │    │ ┌───────────┐ │    │ ┌──────────────┐ │
│ │ LXC 1:      │ │    │ │ vLLM      │ │    │ │ llama.cpp    │ │
│ │ 10.5.4.101  │◄├────┼─┤ :8000     │ │    │ │ :8000        │ │
│ │ NGINX       │ │    │ │           │ │    │ │              │ │
│ │ Portainer   │ │    │ │ API:      │ │    │ │ API:         │ │
│ │ Uptime Kuma │ │    │ │ /v1/comp  │ │    │ │ /completion  │ │
│ └─────────────┘ │    │ │ /v1/embed │ │    │ │ /health      │ │
│                 │    │ │ /health   │ │    │ └──────────────┘ │
│ ┌─────────────┐ │    │ └───────────┘ │    │                  │
│ │ LXC 2:      │ │    │               │    │ Prometheus       │
│ │ 10.5.4.102  │ │    │ Models:       │    │ :9400            │
│ │ N8N         │◄├────┼─ 8B/70B      │    └──────────────────┘
│ │ :5678       │ │    │               │              │
│ │             │ │    │ Prometheus    │              │
│ │ Workflows:  │ │    │ :9090         │     USB 4.0 link
│ │ • AI tasks  │ │    └───────────────┘     (3-8ms)
│ │ • Webhooks  │ │            │                      │
│ └─────────────┘ │            └──────────────────────┘
│                 │
│ ┌─────────────┐ │    ┌──────────────────────────────┐
│ │ LXC 3:      │ │    │ Integration Flows:           │
│ │ 10.5.4.103  │ │    │                              │
│ │ Vault       │◄├────┤ 1. N8N → vLLM (inference)    │
│ │ :8200       │ │    │ 2. NGINX → vLLM (proxy)      │
│ │ Authentik   │ │    │ 3. PostgreSQL ← results      │
│ │ :9000       │ │    │ 4. Prometheus → metrics      │
│ │ Redis       │ │    │ 5. Jenkins → model deploy    │
│ │ :6379       │ │    │ 6. Vault → API keys          │
│ └─────────────┘ │    └──────────────────────────────┘
│                 │
│ ┌─────────────┐ │
│ │ LXC 4:      │ │
│ │ 10.5.4.104  │ │
│ │ PostgreSQL  │◄├────┐ Store inference logs
│ │ :5432       │ │    │ Store embeddings
│ │ Dockage     │ │    │ Store metrics history
│ │ :5001       │ │    │
│ └─────────────┘ │    │
│                 │    │
│ ┌─────────────┐ │    │
│ │ LXC 5:      │ │    │
│ │ 10.5.4.105  │ │    │
│ │ Prometheus  │◄├────┘ Scrape vLLM + Jetson
│ │ :9090       │ │      Track latency/throughput
│ │ Grafana     │ │      Alert on anomalies
│ │ :3000       │ │
│ │ Loki :3100  │ │
│ └─────────────┘ │
│                 │
│ ┌─────────────┐ │
│ │ LXC 6:      │ │
│ │ 10.5.4.106  │ │
│ │ Jenkins     │◄├────┐ Deploy models
│ │ :8080       │ │    │ Run inference tests
│ │ Gitea       │ │    │ Version model configs
│ │ :3000       │ │    │
│ └─────────────┘ │    │
└─────────────────┘    │
                       │
              Model registry
              Config versioning
```

---

## Integration Pattern 1: N8N → AI Inference (Workflow Automation)

### Use Case
**Automate AI-powered tasks** through N8N workflows: content generation, data analysis, classification, embeddings.

### Architecture

```
N8N Workflow (10.5.4.102:5678)
    │
    ├─ Trigger: Webhook /ai-generate
    │       │
    │       ├─ Extract: prompt, max_tokens, temperature
    │       │
    │       └─ Validate: API key from Vault
    │
    ├─ HTTP Request Node:
    │   POST http://10.5.4.150:8000/v1/completions
    │   Headers:
    │     Authorization: Bearer ${VLLM_API_KEY}
    │     Content-Type: application/json
    │   Body:
    │     {
    │       "model": "meta-llama/Llama-3.1-8B-Instruct",
    │       "prompt": "${prompt}",
    │       "max_tokens": 512,
    │       "temperature": 0.7
    │     }
    │
    ├─ Response: Parse JSON
    │   Extract: choices[0].text
    │
    ├─ Log to PostgreSQL (10.5.4.104:5432)
    │   INSERT INTO inference_logs (
    │     workflow_id, prompt, response, 
    │     tokens_used, latency_ms, timestamp
    │   )
    │
    └─ Return: AI response to caller
```

### N8N Workflow Configuration (JSON)

```json
{
  "name": "AI Inference - vLLM Integration",
  "nodes": [
    {
      "parameters": {
        "path": "ai-generate",
        "method": "POST",
        "responseMode": "responseNode"
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "=http://10.5.4.103:8200/v1/secret/data/ai/vllm",
        "authentication": "genericCredentialType",
        "options": {}
      },
      "name": "Get API Key from Vault",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://10.5.4.150:8000/v1/completions",
        "authentication": "genericCredentialType",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Authorization",
              "value": "=Bearer {{$node['Get API Key from Vault'].json.data.data.api_key}}"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "model",
              "value": "meta-llama/Llama-3.1-8B-Instruct"
            },
            {
              "name": "prompt",
              "value": "={{$json.body.prompt}}"
            },
            {
              "name": "max_tokens",
              "value": 512
            },
            {
              "name": "temperature",
              "value": 0.7
            }
          ]
        }
      },
      "name": "Call vLLM Inference",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300]
    },
    {
      "parameters": {
        "operation": "insert",
        "table": "inference_logs",
        "columns": "workflow_id, prompt, response, tokens_used, latency_ms, timestamp",
        "values": "='{{$json.workflow_id}}','{{$json.body.prompt}}','{{$node['Call vLLM Inference'].json.choices[0].text}}',{{$node['Call vLLM Inference'].json.usage.total_tokens}},{{$now}}"
      },
      "name": "Log to PostgreSQL",
      "type": "n8n-nodes-base.postgres",
      "credentials": {
        "postgres": {
          "host": "10.5.4.104",
          "port": 5432,
          "database": "n8n_logs",
          "user": "n8n_user",
          "password": "${PG_PASSWORD}"
        }
      },
      "position": [850, 300]
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{$node['Call vLLM Inference'].json}}"
      },
      "name": "Return Response",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [1050, 300]
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "Get API Key from Vault", "type": "main", "index": 0}]]
    },
    "Get API Key from Vault": {
      "main": [[{"node": "Call vLLM Inference", "type": "main", "index": 0}]]
    },
    "Call vLLM Inference": {
      "main": [[{"node": "Log to PostgreSQL", "type": "main", "index": 0}]]
    },
    "Log to PostgreSQL": {
      "main": [[{"node": "Return Response", "type": "main", "index": 0}]]
    }
  }
}
```

### Network Best Practices

```yaml
# N8N environment variables (LXC 2: 10.5.4.102)
VLLM_ENDPOINT: "http://10.5.4.150:8000"
VAULT_ENDPOINT: "http://10.5.4.103:8200"
PG_ENDPOINT: "10.5.4.104:5432"

# Timeout configuration
HTTP_REQUEST_TIMEOUT: 30000  # 30s (vLLM can take time for large responses)
HTTP_RETRY_COUNT: 3
HTTP_RETRY_DELAY: 1000  # 1s between retries

# Connection pooling
N8N_DB_POOL_SIZE: 10  # PostgreSQL connection pool
N8N_HTTP_KEEP_ALIVE: true  # Reuse connections to vLLM

# Rate limiting (prevent overwhelming vLLM)
N8N_EXECUTIONS_PER_SECOND: 5  # Max 5 concurrent inference requests
```

---

## Integration Pattern 2: NGINX Reverse Proxy (Gateway)

### Use Case
**Expose AI inference APIs** through NGINX with SSL, rate limiting, authentication, and load balancing.

### Architecture

```
External Client (HTTPS)
    │
    ▼
NGINX (10.5.4.101:443)
    │
    ├─ SSL Termination (Let's Encrypt)
    │
    ├─ Authentication (Authentik OAuth2)
    │   └─ Verify JWT token
    │
    ├─ Rate Limiting (10 req/s per IP)
    │
    ├─ Load Balancing:
    │   ├─ Upstream: vLLM GPU (10.5.4.150:8000) - weight 80%
    │   └─ Upstream: Jetson Edge (10.5.4.151:8000) - weight 20% (backup)
    │
    └─ Proxy Pass → AI Backend
```

### NGINX Configuration

```nginx
# /etc/nginx/sites-available/ai-inference.conf (LXC 1: 10.5.4.101)

upstream ai_inference {
    # Primary: vLLM GPU (high performance)
    server 10.5.4.150:8000 weight=4 max_fails=2 fail_timeout=10s;
    
    # Backup: Jetson Edge (fallback)
    server 10.5.4.151:8000 weight=1 max_fails=2 fail_timeout=10s backup;
    
    # Connection pooling
    keepalive 32;
    keepalive_timeout 60s;
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=ai_limit:10m rate=10r/s;
limit_req_status 429;

server {
    listen 443 ssl http2;
    server_name ai.arrowave.local;  # Internal DNS

    # SSL configuration (Let's Encrypt or self-signed)
    ssl_certificate /etc/nginx/ssl/ai.arrowave.crt;
    ssl_certificate_key /etc/nginx/ssl/ai.arrowave.key;
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Logging
    access_log /var/log/nginx/ai-access.log combined;
    error_log /var/log/nginx/ai-error.log warn;

    # Health check endpoint (no auth required)
    location /health {
        access_log off;
        proxy_pass http://ai_inference/health;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # Inference endpoint (requires authentication)
    location /v1/ {
        # Rate limiting
        limit_req zone=ai_limit burst=20 nodelay;

        # OAuth2 authentication via Authentik
        auth_request /auth;
        auth_request_set $auth_user $upstream_http_x_auth_request_user;
        auth_request_set $auth_email $upstream_http_x_auth_request_email;

        # Proxy to AI backends
        proxy_pass http://ai_inference;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Auth-User $auth_user;
        proxy_set_header X-Auth-Email $auth_email;
        
        # Timeouts (vLLM can be slow for large responses)
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering (disable for streaming responses)
        proxy_buffering off;
        proxy_request_buffering off;

        # WebSocket support (for streaming)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Authentik OAuth2 validation
    location = /auth {
        internal;
        proxy_pass http://10.5.4.103:9000/application/o/authorize/;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }

    # Prometheus metrics endpoint (internal only)
    location /metrics {
        allow 10.5.4.0/24;  # Only allow from VLAN
        deny all;
        
        proxy_pass http://10.5.4.150:9090/metrics;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}

# HTTP redirect to HTTPS
server {
    listen 80;
    server_name ai.arrowave.local;
    return 301 https://$host$request_uri;
}
```

### Network Best Practices

```yaml
# DNS Configuration (UDM SE or /etc/hosts on clients)
10.5.4.101 ai.arrowave.local

# Client usage example
curl -X POST https://ai.arrowave.local/v1/completions \
  -H "Authorization: Bearer ${OAUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain quantum computing",
    "max_tokens": 256
  }'

# Expected latency:
# - SSL handshake: 10-20ms (first request)
# - NGINX routing: <1ms
# - vLLM inference: 2-10s (depends on tokens)
# - Total: 2-10s for typical request
```

---

## Integration Pattern 3: Vault + Authentik (Secrets & Auth)

### Use Case
**Secure API keys, credentials, and user authentication** for AI services using zero-trust architecture.

### Architecture

```
Vault (10.5.4.103:8200)
    │
    ├─ Secret Store:
    │   ├─ ai/vllm/api-key (Bearer token for vLLM)
    │   ├─ ai/jetson/api-key (Bearer token for Jetson)
    │   ├─ ai/openai/api-key (fallback external API)
    │   └─ ai/postgres/credentials (for logging)
    │
    └─ Dynamic Secrets:
        └─ Generate short-lived tokens (10 min TTL)

Authentik (10.5.4.103:9000)
    │
    ├─ OAuth2 Provider:
    │   └─ Authorize AI API access
    │
    ├─ User Groups:
    │   ├─ ai-users (read access)
    │   └─ ai-admins (read + deploy access)
    │
    └─ Token Validation:
        └─ Verify JWT signatures (RS256)
```

### Vault Configuration

```bash
# Initialize Vault (one-time setup)
ssh root@10.5.4.103
vault operator init -key-shares=1 -key-threshold=1
# Save root token and unseal key

# Enable KV secrets engine
vault secrets enable -path=ai kv-v2

# Store vLLM API key
vault kv put ai/vllm \
  api-key="sk-vllm-$(openssl rand -hex 32)" \
  endpoint="http://10.5.4.150:8000" \
  max_rpm=100

# Store Jetson API key
vault kv put ai/jetson \
  api-key="sk-jetson-$(openssl rand -hex 32)" \
  endpoint="http://10.5.4.151:8000" \
  max_rpm=50

# Store PostgreSQL credentials
vault kv put ai/postgres \
  username="ai_logger" \
  password="$(openssl rand -base64 32)" \
  host="10.5.4.104" \
  port="5432" \
  database="ai_logs"

# Create policy for N8N access
vault policy write n8n-ai-policy - <<EOF
path "ai/data/vllm" {
  capabilities = ["read"]
}
path "ai/data/jetson" {
  capabilities = ["read"]
}
path "ai/data/postgres" {
  capabilities = ["read"]
}
EOF

# Create token for N8N
vault token create -policy=n8n-ai-policy -ttl=720h
# Token: s.xxxxxxxxxxxxxxxxxxxxxxxx (save to N8N env)
```

### Authentik Configuration

```yaml
# Authentik Application Config (via UI or API)
# http://10.5.4.103:9000/if/admin/

name: AI Inference API
slug: ai-inference
provider: OAuth2/OpenID Connect
client_type: Confidential
redirect_uris:
  - https://ai.arrowave.local/auth/callback
  - http://10.5.4.102:5678/auth/callback  # N8N
authorization_grant_type: authorization-code
client_id: ai-inference-client
client_secret: <generated-by-authentik>

# Token expiry
access_token_validity: 600  # 10 minutes
refresh_token_validity: 86400  # 24 hours

# User groups
groups:
  - name: ai-users
    permissions:
      - ai.inference.read
  - name: ai-admins
    permissions:
      - ai.inference.read
      - ai.inference.write
      - ai.model.deploy
```

### Network Best Practices

```yaml
# Environment variables for services

# N8N (LXC 2: 10.5.4.102)
VAULT_ADDR: "http://10.5.4.103:8200"
VAULT_TOKEN: "s.xxxxxxxxxxxxxxxxxxxxxxxx"  # From Vault token create
AUTHENTIK_URL: "http://10.5.4.103:9000"
AUTHENTIK_CLIENT_ID: "ai-inference-client"
AUTHENTIK_CLIENT_SECRET: "<from-authentik>"

# vLLM (Schlimers GPU: 10.5.4.150)
VLLM_API_KEY: "sk-vllm-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # From Vault
VLLM_ENABLE_AUTH: "true"
VLLM_AUTH_PROVIDER: "bearer"

# Jetson (10.5.4.151)
JETSON_API_KEY: "sk-jetson-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # From Vault
JETSON_ENABLE_AUTH: "true"

# Security best practices
API_KEY_ROTATION_DAYS: 30  # Rotate every 30 days
JWT_TOKEN_EXPIRY_MINUTES: 10  # Short-lived tokens
REQUIRE_HTTPS: true  # Force HTTPS for external access
RATE_LIMIT_PER_USER: 100  # 100 requests per minute per user
```

---

## Integration Pattern 4: PostgreSQL (Logging & Analytics)

### Use Case
**Store inference logs, embeddings, and analytics** for auditing, retraining, and cost tracking.

### Architecture

```
PostgreSQL (10.5.4.104:5432)
    │
    ├─ Database: ai_logs
    │   │
    │   ├─ Table: inference_logs
    │   │   └─ Columns: id, workflow_id, prompt, response,
    │   │               tokens_used, latency_ms, model,
    │   │               user_id, timestamp, cost_estimate
    │   │
    │   ├─ Table: embeddings
    │   │   └─ Columns: id, text, embedding (vector),
    │   │               model, dimension, timestamp
    │   │
    │   └─ Table: model_metrics
    │       └─ Columns: model_id, timestamp, throughput_tps,
    │                   latency_p50, latency_p99, gpu_util,
    │                   memory_used_gb
    │
    └─ Indexes:
        ├─ idx_inference_logs_timestamp (for time-series queries)
        ├─ idx_inference_logs_workflow_id (for N8N tracking)
        └─ idx_embeddings_vector (HNSW for similarity search)
```

### PostgreSQL Schema

```sql
-- Create database (run once)
CREATE DATABASE ai_logs;
\c ai_logs

-- Enable pgvector extension (for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Inference logs table
CREATE TABLE inference_logs (
    id BIGSERIAL PRIMARY KEY,
    workflow_id VARCHAR(255),
    prompt TEXT NOT NULL,
    response TEXT,
    tokens_used INTEGER,
    latency_ms INTEGER,
    model VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    cost_estimate NUMERIC(10, 6),  -- Estimated cost in USD
    metadata JSONB  -- Additional metadata (temperature, top_p, etc.)
);

-- Indexes for performance
CREATE INDEX idx_inference_logs_timestamp ON inference_logs(timestamp DESC);
CREATE INDEX idx_inference_logs_workflow_id ON inference_logs(workflow_id);
CREATE INDEX idx_inference_logs_user_id ON inference_logs(user_id);
CREATE INDEX idx_inference_logs_model ON inference_logs(model);

-- Embeddings table (for vector search)
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(768),  -- Llama-3.1-8B embeddings are 768-dimensional
    model VARCHAR(255) NOT NULL,
    dimension INTEGER NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- HNSW index for fast similarity search
CREATE INDEX idx_embeddings_vector ON embeddings 
USING hnsw (embedding vector_cosine_ops);

-- Model metrics table (aggregated from Prometheus)
CREATE TABLE model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    throughput_tps NUMERIC(10, 2),
    latency_p50_ms INTEGER,
    latency_p99_ms INTEGER,
    gpu_utilization_percent INTEGER,
    memory_used_gb NUMERIC(10, 2),
    active_requests INTEGER,
    queue_depth INTEGER
);

CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp DESC);
CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);

-- Create user for N8N/services
CREATE USER ai_logger WITH PASSWORD '<from-vault>';
GRANT CONNECT ON DATABASE ai_logs TO ai_logger;
GRANT USAGE ON SCHEMA public TO ai_logger;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO ai_logger;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ai_logger;
```

### N8N Integration (Logging Node)

```javascript
// N8N Function Node: Log Inference to PostgreSQL
const prompt = $json.body.prompt;
const response = $node['Call vLLM Inference'].json.choices[0].text;
const tokens = $node['Call vLLM Inference'].json.usage.total_tokens;
const latency = $node['Call vLLM Inference'].json.latency_ms;
const model = $node['Call vLLM Inference'].json.model;
const user = $node['Webhook'].json.headers['x-auth-user'];

// Estimate cost (example: $0.0001 per 1K tokens)
const costPerToken = 0.0001 / 1000;
const cost = tokens * costPerToken;

return [{
  json: {
    workflow_id: $workflow.id,
    prompt: prompt,
    response: response,
    tokens_used: tokens,
    latency_ms: latency,
    model: model,
    user_id: user,
    cost_estimate: cost,
    metadata: {
      temperature: 0.7,
      max_tokens: 512,
      endpoint: '10.5.4.150:8000'
    }
  }
}];
```

### Network Best Practices

```yaml
# PostgreSQL connection pooling (LXC 4: 10.5.4.104)
max_connections: 100
shared_buffers: 256MB  # 25% of 64GB Proxmox RAM allocated to LXC
effective_cache_size: 768MB
maintenance_work_mem: 64MB
checkpoint_completion_target: 0.9
wal_buffers: 16MB
default_statistics_target: 100
random_page_cost: 1.1  # NVMe SSD
effective_io_concurrency: 200
work_mem: 4MB
min_wal_size: 1GB
max_wal_size: 4GB

# Client connections from N8N
# psycopg2 connection string:
postgresql://ai_logger:<password>@10.5.4.104:5432/ai_logs?
  sslmode=require&
  connect_timeout=10&
  application_name=n8n-workflow

# Backup strategy
pg_dump ai_logs | gzip > /mnt/backup/ai_logs_$(date +%Y%m%d).sql.gz
# Retention: 30 days rolling
```

---

## Integration Pattern 5: Prometheus + Grafana (Monitoring)

### Use Case
**Monitor AI IDP performance metrics** alongside existing production services in unified dashboards.

### Architecture

```
Prometheus (10.5.4.105:9090)
    │
    ├─ Scrape Targets:
    │   ├─ vLLM GPU (10.5.4.150:9090) - every 15s
    │   ├─ Jetson Edge (10.5.4.151:9400) - every 15s
    │   ├─ NGINX (10.5.4.101:9113) - every 30s
    │   ├─ PostgreSQL (10.5.4.104:9187) - every 30s
    │   └─ N8N (10.5.4.102:9200) - every 30s
    │
    ├─ Metrics:
    │   ├─ vllm_request_count (counter)
    │   ├─ vllm_request_latency (histogram)
    │   ├─ vllm_throughput_tokens_per_second (gauge)
    │   ├─ vllm_queue_depth (gauge)
    │   ├─ nvidia_gpu_utilization_percent (gauge)
    │   ├─ nvidia_gpu_memory_used_bytes (gauge)
    │   └─ nvidia_gpu_temperature_celsius (gauge)
    │
    └─ Alerting:
        ├─ Alert: vLLM throughput < 50 tok/s
        ├─ Alert: GPU temperature > 85°C
        └─ Alert: Inference latency P99 > 10s

Grafana (10.5.4.105:3000)
    │
    └─ Dashboards:
        ├─ AI IDP Overview (throughput, latency, GPU)
        ├─ Microservices Health (N8N, NGINX, PostgreSQL)
        └─ Network Performance (VLAN bandwidth, latency)
```

### Prometheus Configuration

```yaml
# /etc/prometheus/prometheus.yml (LXC 5: 10.5.4.105)
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'arrowave-production'
    datacenter: 'local'

scrape_configs:
  # Existing production targets
  - job_name: 'proxmox-host'
    static_configs:
      - targets: ['10.5.4.45:9100']
        labels:
          service: 'proxmox-ve'
          vlan: '10.5.4.0'

  - job_name: 'lxc-containers'
    static_configs:
      - targets:
          - '10.5.4.101:9100'  # Gateway
          - '10.5.4.102:9100'  # N8N
          - '10.5.4.103:9100'  # Vault
          - '10.5.4.104:9100'  # PostgreSQL
          - '10.5.4.105:9100'  # Observation (self-scrape)
          - '10.5.4.106:9100'  # Jenkins
        labels:
          service: 'lxc'
          vlan: '10.5.4.0'

  # NEW: AI IDP targets
  - job_name: 'vllm-gpu-inference'
    scrape_interval: 10s  # More frequent for AI metrics
    scrape_timeout: 5s
    static_configs:
      - targets: ['10.5.4.150:9090']
        labels:
          service: 'vllm'
          device: 'schlimers-gpu'
          model: 'llama-3.1-8b'
          vlan: '10.5.4.0'

  - job_name: 'nvidia-gpu-metrics'
    scrape_interval: 10s
    static_configs:
      - targets: ['10.5.4.150:9400']  # NVIDIA DCGM Exporter
        labels:
          service: 'nvidia-dcgm'
          device: 'rtx-5070-ti'
          vlan: '10.5.4.0'

  - job_name: 'jetson-edge-inference'
    scrape_interval: 10s
    static_configs:
      - targets: ['10.5.4.151:9400']
        labels:
          service: 'jetson-rpc'
          device: 'jetson-orin-nano'
          vlan: '10.5.4.0'

  - job_name: 'nginx-gateway'
    scrape_interval: 30s
    static_configs:
      - targets: ['10.5.4.101:9113']  # NGINX Prometheus Exporter
        labels:
          service: 'nginx'
          vlan: '10.5.4.0'

  - job_name: 'postgresql-exporter'
    scrape_interval: 30s
    static_configs:
      - targets: ['10.5.4.104:9187']
        labels:
          service: 'postgresql'
          database: 'ai_logs'
          vlan: '10.5.4.0'

# Alerting rules
rule_files:
  - '/etc/prometheus/rules/ai_alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['10.5.4.105:9093']  # Alertmanager (if deployed)
```

### Alerting Rules

```yaml
# /etc/prometheus/rules/ai_alerts.yml
groups:
  - name: ai_inference_performance
    interval: 30s
    rules:
      # Alert: vLLM throughput below baseline
      - alert: VLLMThroughputLow
        expr: rate(vllm_request_tokens_total[5m]) < 50
        for: 5m
        labels:
          severity: warning
          service: vllm
        annotations:
          summary: "vLLM throughput below 50 tok/s"
          description: "vLLM on {{ $labels.instance }} is generating {{ $value | humanize }} tok/s (expected 60-100)"

      # Alert: GPU utilization too low (model not running)
      - alert: GPUUtilizationLow
        expr: nvidia_gpu_utilization < 20
        for: 5m
        labels:
          severity: info
          service: nvidia-gpu
        annotations:
          summary: "GPU utilization low ({{ $value }}%)"
          description: "GPU on {{ $labels.instance }} is underutilized. Check if vLLM is running."

      # Alert: GPU temperature critical
      - alert: GPUTemperatureCritical
        expr: nvidia_gpu_temperature_celsius > 85
        for: 2m
        labels:
          severity: critical
          service: nvidia-gpu
        annotations:
          summary: "GPU temperature critical ({{ $value }}°C)"
          description: "GPU on {{ $labels.instance }} is overheating. Check cooling system."

      # Alert: Inference latency P99 high
      - alert: InferenceLatencyHigh
        expr: histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
          service: vllm
        annotations:
          summary: "Inference P99 latency > 10s"
          description: "99th percentile latency on {{ $labels.instance }} is {{ $value | humanize }}s"

      # Alert: Request queue depth high (saturation)
      - alert: RequestQueueDepthHigh
        expr: vllm_queue_depth > 10
        for: 5m
        labels:
          severity: warning
          service: vllm
        annotations:
          summary: "Request queue depth > 10"
          description: "vLLM on {{ $labels.instance }} has {{ $value }} queued requests. Consider scaling."

  - name: microservices_health
    interval: 60s
    rules:
      # Alert: PostgreSQL connection saturation
      - alert: PostgreSQLConnectionsSaturated
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
          service: postgresql
        annotations:
          summary: "PostgreSQL connections at {{ $value | humanizePercentage }}"
          description: "Database {{ $labels.datname }} on {{ $labels.instance }} is approaching connection limit"

      # Alert: NGINX error rate high
      - alert: NGINXErrorRateHigh
        expr: rate(nginx_http_requests_total{status=~"5.."}[5m]) / rate(nginx_http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          service: nginx
        annotations:
          summary: "NGINX 5xx error rate > 5%"
          description: "NGINX on {{ $labels.instance }} has {{ $value | humanizePercentage }} error rate"
```

### Grafana Dashboard (JSON Export)

```json
{
  "dashboard": {
    "title": "AI IDP + Microservices Overview",
    "uid": "ai-idp-overview",
    "timezone": "browser",
    "panels": [
      {
        "title": "vLLM Throughput (tokens/sec)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vllm_request_tokens_total[1m])",
            "legendFormat": "{{ instance }} - {{ model }}"
          }
        ],
        "yaxes": [
          {"label": "Tokens/sec", "min": 0}
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "GPU Utilization %",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization",
            "legendFormat": "GPU {{ instance }}"
          }
        ],
        "yaxes": [
          {"label": "Utilization %", "min": 0, "max": 100}
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Inference Latency (P50, P95, P99)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "yaxes": [
          {"label": "Latency (seconds)", "min": 0}
        ],
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      },
      {
        "title": "Active Requests & Queue Depth",
        "type": "graph",
        "targets": [
          {
            "expr": "vllm_active_requests",
            "legendFormat": "Active"
          },
          {
            "expr": "vllm_queue_depth",
            "legendFormat": "Queued"
          }
        ],
        "yaxes": [
          {"label": "Requests", "min": 0}
        ],
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
      },
      {
        "title": "GPU Memory Usage (GB)",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / 1024^3",
            "legendFormat": "Used GB"
          },
          {
            "expr": "nvidia_gpu_memory_total_bytes / 1024^3",
            "legendFormat": "Total GB"
          }
        ],
        "yaxes": [
          {"label": "Memory (GB)", "min": 0, "max": 16}
        ],
        "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8}
      },
      {
        "title": "PostgreSQL Inference Log Growth",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_user_tables_n_tup_ins{table=\"inference_logs\"}",
            "legendFormat": "Rows Inserted"
          }
        ],
        "yaxes": [
          {"label": "Rows", "min": 0}
        ],
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8}
      }
    ],
    "refresh": "10s"
  }
}
```

---

## Integration Pattern 6: Jenkins + Gitea (CI/CD for Models)

### Use Case
**Automate model deployment, testing, and versioning** through CI/CD pipelines integrated with AI IDP.

### Architecture

```
Gitea (10.5.4.106:3000)
    │
    ├─ Repository: ai-models
    │   ├─ models/
    │   │   ├─ llama-3.1-8b/
    │   │   │   ├─ config.json
    │   │   │   ├─ docker-compose.yml
    │   │   │   └─ vllm-deploy.sh
    │   │   └─ llama-3.1-70b/
    │   │       └─ ...
    │   └─ tests/
    │       └─ inference_test.py
    │
    └─ Webhooks → Jenkins (on git push)

Jenkins (10.5.4.106:8080)
    │
    ├─ Pipeline: Deploy Model to vLLM
    │   ├─ Stage 1: Clone repo from Gitea
    │   ├─ Stage 2: Validate model config
    │   ├─ Stage 3: SSH to Schlimers GPU
    │   ├─ Stage 4: Pull Docker image
    │   ├─ Stage 5: Deploy vLLM container
    │   ├─ Stage 6: Run inference test
    │   └─ Stage 7: Update Prometheus metrics
    │
    └─ Notifications:
        ├─ Slack/Email on deploy success
        └─ Rollback on test failure
```

### Jenkinsfile (Pipeline Definition)

```groovy
// Jenkinsfile for AI Model Deployment
// Repository: http://10.5.4.106:3000/arrowave/ai-models

pipeline {
    agent any
    
    environment {
        GITEA_URL = 'http://10.5.4.106:3000'
        SCHLIMERS_GPU = '10.5.4.150'
        VLLM_PORT = '8000'
        MODEL_REGISTRY = '/mnt/models'
        VAULT_ADDR = 'http://10.5.4.103:8200'
        VAULT_TOKEN = credentials('vault-jenkins-token')
    }
    
    stages {
        stage('Clone Repository') {
            steps {
                git url: "${GITEA_URL}/arrowave/ai-models.git",
                    branch: 'main',
                    credentialsId: 'gitea-jenkins-ssh'
            }
        }
        
        stage('Validate Model Config') {
            steps {
                script {
                    def config = readJSON file: 'models/llama-3.1-8b/config.json'
                    if (!config.model_id || !config.docker_image) {
                        error("Invalid model config: missing required fields")
                    }
                    echo "Deploying model: ${config.model_id}"
                }
            }
        }
        
        stage('Get Secrets from Vault') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'vault-token', variable: 'VAULT_TOKEN')]) {
                        sh '''
                            export VAULT_ADDR=http://10.5.4.103:8200
                            export VAULT_TOKEN=${VAULT_TOKEN}
                            
                            # Get vLLM API key
                            VLLM_API_KEY=$(vault kv get -field=api-key ai/vllm)
                            echo "VLLM_API_KEY=${VLLM_API_KEY}" > .env
                            
                            # Get model download credentials (if needed)
                            HF_TOKEN=$(vault kv get -field=token ai/huggingface || echo "")
                            echo "HF_TOKEN=${HF_TOKEN}" >> .env
                        '''
                    }
                }
            }
        }
        
        stage('Deploy to Schlimers GPU') {
            steps {
                sshagent(credentials: ['jenkins-ssh-key']) {
                    sh '''
                        # Copy deployment files to GPU server
                        scp -r models/llama-3.1-8b/* ubuntu@${SCHLIMERS_GPU}:/opt/vllm/
                        scp .env ubuntu@${SCHLIMERS_GPU}:/opt/vllm/.env
                        
                        # Deploy via SSH
                        ssh ubuntu@${SCHLIMERS_GPU} << 'EOF'
                            cd /opt/vllm
                            
                            # Stop existing container
                            docker-compose down || true
                            
                            # Pull latest image
                            docker-compose pull
                            
                            # Start new container
                            docker-compose up -d
                            
                            # Wait for health check
                            sleep 10
                            curl -f http://localhost:8000/health || exit 1
                            
                            echo "Deployment successful"
EOF
                    '''
                }
            }
        }
        
        stage('Run Inference Test') {
            steps {
                script {
                    sh '''
                        # Test inference endpoint
                        source .env
                        
                        response=$(curl -s -X POST http://${SCHLIMERS_GPU}:${VLLM_PORT}/v1/completions \
                          -H "Authorization: Bearer ${VLLM_API_KEY}" \
                          -H "Content-Type: application/json" \
                          -d '{
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "prompt": "Test prompt: 2+2=",
                            "max_tokens": 10
                          }')
                        
                        echo "Response: ${response}"
                        
                        # Validate response contains expected fields
                        echo "${response}" | jq -e '.choices[0].text' || exit 1
                        
                        # Extract throughput
                        throughput=$(echo "${response}" | jq -r '.usage.total_tokens')
                        echo "Throughput test: ${throughput} tokens"
                        
                        # Fail if throughput is 0
                        if [ "${throughput}" -eq 0 ]; then
                            echo "ERROR: Inference returned 0 tokens"
                            exit 1
                        fi
                    '''
                }
            }
        }
        
        stage('Update Prometheus Metrics') {
            steps {
                sh '''
                    # Push deployment event to Prometheus Pushgateway (if configured)
                    cat <<EOF | curl --data-binary @- http://10.5.4.105:9091/metrics/job/model_deployment
# HELP model_deployment_timestamp Unix timestamp of model deployment
# TYPE model_deployment_timestamp gauge
model_deployment_timestamp{model="llama-3.1-8b",version="${BUILD_NUMBER}"} $(date +%s)
EOF
                '''
            }
        }
    }
    
    post {
        success {
            echo "Model deployment successful!"
            // Send notification (Slack, email, etc.)
        }
        failure {
            echo "Model deployment failed! Rolling back..."
            sshagent(credentials: ['jenkins-ssh-key']) {
                sh '''
                    ssh ubuntu@${SCHLIMERS_GPU} << 'EOF'
                        cd /opt/vllm
                        docker-compose down
                        docker-compose -f docker-compose.backup.yml up -d
EOF
                '''
            }
        }
        always {
            cleanWs()  # Clean workspace
        }
    }
}
```

### Gitea Webhook Configuration

```yaml
# Gitea Webhook (http://10.5.4.106:3000/arrowave/ai-models/settings/hooks)
URL: http://10.5.4.106:8080/gitea-webhook/
Content Type: application/json
Secret: <shared-secret-from-vault>
Trigger: Push events (branch: main)
Active: Yes

# Jenkins Gitea Plugin Configuration
# Manage Jenkins → Configure System → Gitea Servers
Server Name: Gitea Local
Server URL: http://10.5.4.106:3000
Credentials: gitea-jenkins-token (personal access token)
```

---

## Network Security Best Practices

### 1. Zero-Trust Principles

```yaml
# All services require authentication by default
Authentication:
  - vLLM API: Bearer token from Vault (rotated every 30 days)
  - NGINX: OAuth2 via Authentik (JWT tokens, 10 min expiry)
  - PostgreSQL: Password auth (TLS required)
  - Vault: Token-based (TTL 24h for services, 10min for users)
  - Prometheus: IP whitelist (10.5.4.0/24 only)
  - Grafana: OAuth2 via Authentik

Authorization:
  - Role-based access control (RBAC)
    - ai-users: read-only inference access
    - ai-admins: deploy + config changes
    - ai-monitoring: metrics read-only
  - API key scopes (per-service, per-operation)
```

### 2. Network Segmentation (Future Enhancement)

```yaml
# Current: All on 10.5.4.0/24 (same VLAN)
# Phase 2+: Segment into multiple VLANs

Proposed VLANs:
  - VLAN 10 (10.5.4.0/24): Production microservices (existing)
  - VLAN 20 (10.5.20.0/24): AI Inference (vLLM + Jetson)
  - VLAN 30 (10.5.30.0/24): Dev/Guest (XPS13)
  - VLAN 99 (10.5.99.0/24): Management (SSH, monitoring only)

Firewall Rules (UDM SE):
  - Allow: VLAN 10 → VLAN 20 (ports 8000, 9090 only)
  - Allow: VLAN 20 → VLAN 20 (internal AI traffic)
  - Deny:  VLAN 30 → VLAN 10, 20 (guest isolation)
  - Allow: VLAN 99 → All (management access)
```

### 3. TLS Everywhere

```yaml
# Encrypt all inter-service communication

Services requiring TLS:
  - NGINX → vLLM: HTTPS (self-signed or Let's Encrypt)
  - N8N → Vault: HTTPS
  - vLLM → PostgreSQL: TLS (sslmode=require)
  - Prometheus scrape: HTTPS (optional, low priority)
  - Authentik OAuth2: HTTPS (required for OAuth2 spec)

Certificate management:
  - Internal CA: Use Vault PKI engine
  - Auto-renewal: Vault agent on each service
  - Certificate rotation: 90 days
```

### 4. Rate Limiting & DDoS Protection

```yaml
# NGINX rate limiting (per-IP, per-endpoint)
Rate Limits:
  - /v1/completions: 10 req/s per IP (burst 20)
  - /v1/embeddings: 20 req/s per IP (burst 40)
  - /health: unlimited (for monitoring)
  - /metrics: 1 req/s per IP (prevent scraping)

# vLLM internal queue limits
vLLM Configuration:
  max_num_seqs: 256  # Max concurrent sequences
  max_num_batched_tokens: 4096  # Max tokens per batch
  max_waiting_time_sec: 30  # Max queue wait time

# UDM SE firewall (DDoS protection)
Connection Limits:
  - Max connections per IP: 100
  - Max new connections per second: 10
  - SYN flood protection: enabled
  - ICMP rate limiting: 1 req/s
```

### 5. Monitoring & Alerting

```yaml
# Prometheus alerts for security events
Security Alerts:
  - Authentication failures (>10 in 5min): Critical
  - API rate limit violations: Warning
  - Unauthorized access attempts: Info
  - Certificate expiry (<7 days): Warning
  - Vault seal/unseal events: Critical

# Audit logging
Log Destinations:
  - NGINX access log → Loki (10.5.4.105:3100)
  - vLLM inference log → PostgreSQL (10.5.4.104)
  - Vault audit log → File + Loki
  - Authentik audit log → PostgreSQL + Loki

Retention:
  - Access logs: 30 days
  - Inference logs: 90 days
  - Audit logs: 1 year
```

---

## Performance Optimization

### 1. Connection Pooling

```yaml
# PostgreSQL connection pooling (PgBouncer)
PgBouncer Configuration:
  listen_addr: 10.5.4.104
  listen_port: 6432
  pool_mode: transaction
  max_client_conn: 100
  default_pool_size: 20
  reserve_pool_size: 5
  reserve_pool_timeout: 3

# Services connect to PgBouncer instead of PostgreSQL directly
Connection String:
  postgresql://ai_logger@10.5.4.104:6432/ai_logs
```

### 2. Caching Strategy

```yaml
# Redis caching (LXC 3: 10.5.4.103:6379)
Cache Policies:
  - Frequently requested prompts: TTL 1 hour
  - User sessions (Authentik): TTL 24 hours
  - Model metadata: TTL 7 days
  - Embeddings (common queries): TTL 7 days

# NGINX caching (static responses)
Cache Configuration:
  - /health endpoint: cache 10s
  - /metrics endpoint: cache 5s
  - /v1/completions: no cache (dynamic)
  - Model metadata: cache 1 hour
```

### 3. Load Balancing

```yaml
# NGINX load balancing (if multiple GPU servers)
Upstream Configuration:
  - Primary: Schlimers GPU (10.5.4.150) - weight 4
  - Backup: Jetson Edge (10.5.4.151) - weight 1, backup flag
  
Load Balancing Algorithm:
  - Method: least_conn (route to least busy server)
  - Health check: /health endpoint every 10s
  - Fail timeout: 10s
  - Max fails: 2 (mark as down after 2 failures)

# Future: Add second GPU server for HA
# - Schlimers GPU 2 (10.5.4.160) - weight 4
```

---

## Summary: Complete Integration Map

### Data Flow Summary

```
1. User → NGINX (10.5.4.101:443)
   └─ OAuth2 auth via Authentik (10.5.4.103:9000)
   └─ Rate limit (10 req/s)
   └─ Proxy to vLLM (10.5.4.150:8000)

2. vLLM Inference (10.5.4.150)
   └─ Process request (60-100 tok/s)
   └─ Return response
   └─ Emit metrics (10.5.4.150:9090)

3. N8N Workflow (10.5.4.102:5678)
   └─ Get API key from Vault (10.5.4.103:8200)
   └─ Call vLLM (10.5.4.150:8000)
   └─ Log to PostgreSQL (10.5.4.104:5432)

4. Prometheus (10.5.4.105:9090)
   └─ Scrape vLLM metrics (10.5.4.150:9090)
   └─ Scrape Jetson metrics (10.5.4.151:9400)
   └─ Scrape NGINX metrics (10.5.4.101:9113)
   └─ Store time-series data

5. Grafana (10.5.4.105:3000)
   └─ Query Prometheus
   └─ Display dashboards
   └─ Alert on anomalies

6. Jenkins (10.5.4.106:8080)
   └─ Clone from Gitea (10.5.4.106:3000)
   └─ Deploy model to Schlimers (10.5.4.150)
   └─ Run inference test
   └─ Update Prometheus
```

### Network Traffic Matrix

| Source | Destination | Protocol | Port | Purpose | Bandwidth |
|--------|-------------|----------|------|---------|-----------|
| External | NGINX (101) | HTTPS | 443 | AI API access | <100 Mbps |
| NGINX (101) | vLLM (150) | HTTP | 8000 | Inference proxy | 50-200 Mbps |
| NGINX (101) | Authentik (103) | HTTP | 9000 | OAuth2 validation | <1 Mbps |
| N8N (102) | vLLM (150) | HTTP | 8000 | Workflow inference | 10-50 Mbps |
| N8N (102) | Vault (103) | HTTPS | 8200 | Get API keys | <1 Mbps |
| N8N (102) | PostgreSQL (104) | TCP | 5432 | Log inference | <5 Mbps |
| vLLM (150) | Jetson (151) | TCP/RPC | 8000 | Edge offload | 500-800 Mbps |
| Prometheus (105) | vLLM (150) | HTTP | 9090 | Scrape metrics | <1 Mbps |
| Prometheus (105) | Jetson (151) | HTTP | 9400 | Scrape metrics | <1 Mbps |
| Jenkins (106) | Schlimers (150) | SSH | 22 | Deploy models | <10 Mbps |
| Jenkins (106) | Gitea (106) | HTTP | 3000 | Clone repos | <5 Mbps |

### Performance Targets

```yaml
Latency (P99):
  - NGINX → vLLM: <10ms
  - vLLM inference: <10s (256 tokens)
  - N8N → vLLM: <10s total
  - Prometheus scrape: <100ms
  - PostgreSQL write: <50ms

Throughput:
  - vLLM: 60-100 tokens/sec (8B model)
  - NGINX: 1000 req/s (proxy capacity)
  - PostgreSQL: 5000 writes/s (inference logs)
  - N8N: 10 concurrent workflows

Availability:
  - All services: 99.9% (8.76 hours downtime/year)
  - vLLM: 99.5% (43.8 hours downtime/year, acceptable for AI)
  - Prometheus: 99.9% (monitoring critical)
```

---

## Next Steps

### Phase 0 (This Week): Network Foundation
- [ ] Configure static IPs per NETWORK-QUICKSTART.md
- [ ] Verify all services reachable (ping tests)
- [ ] Update Prometheus to scrape vLLM + Jetson

### Phase 1 (Week 1): Basic Integration
- [ ] Deploy vLLM on Schlimers GPU
- [ ] Configure NGINX reverse proxy with auth
- [ ] Create N8N workflow for inference
- [ ] Set up PostgreSQL logging

### Phase 2 (Week 2): Advanced Integration
- [ ] Enable Vault secret management
- [ ] Configure Authentik OAuth2
- [ ] Deploy NGINX rate limiting
- [ ] Create Grafana dashboards

### Phase 3 (Week 3): CI/CD Integration
- [ ] Set up Jenkins pipeline
- [ ] Configure Gitea webhooks
- [ ] Automate model deployment
- [ ] Run inference tests

### Phase 4 (Week 4): Production Hardening
- [ ] Enable TLS everywhere
- [ ] Configure security alerts
- [ ] Test disaster recovery
- [ ] Document runbooks

---

*Microservices-AI Integration Architecture v1.0 | January 12, 2026*  
*All services on VLAN 10.5.4.0/24 | Zero-trust security | Production-ready*
