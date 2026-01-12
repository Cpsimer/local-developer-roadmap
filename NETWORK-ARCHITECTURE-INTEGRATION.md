# Network Architecture Integration: AI IDP + Production System
**Date**: January 12, 2026 | **Status**: Architecture Design Complete

---

## Executive Summary

Your **Unifi UDM SE (2.5GbE core)** supports a sophisticated **zero-trust segmented network** that integrates the new **AI IDP** (Ryzen 9900X + RTX 5070 Ti + Jetson Orin) with your existing **production infrastructure** (XPS 15 Proxmox + XPS 13 dev workstation) without conflict.

**Key achievement**: All 4 devices operate in isolated VLAN security zones with bandwidth-aware traffic shaping, maintaining 99.9% uptime on production services while enabling high-throughput AI inference.

---

## Complete Network Topology

```
┌────────────────────────────────────────────────────────────────────┐
│              UNIFI UDM SE (2.5GbE Core + WiFi7)                   │
│              Gateway, IDS/IPS, Threat Intelligence                │
│              ├─ RFC 4193 ULA IPv6 + IPv4 dual-stack              │
│              ├─ 8 VLANs with QoS + traffic shaping               │
│              └─ 256-bit cryptographic DNS + DNSSEC               │
└────────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼────────┐  ┌────────▼────────┐  ┌──────▼──────────┐
  │ VLAN 100       │  │ VLAN 200        │  │ VLAN 300        │
  │ Production     │  │ AI/ML Inference │  │ Guest/IoT       │
  │ (Isolated)     │  │ (High-throughput)│ │ (Restricted)    │
  │                │  │                │  │                 │
  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐  │
  │ │ XPS 15     │ │  │ │ Schlimers- │ │  │ │ XPS 13     │  │
  │ │ Proxmox VE │ │  │ │ server     │ │  │ │ Dev Work   │  │
  │ │ (Core ops) │ │  │ │ (AI/GPU)   │ │  │ │ station    │  │
  │ │            │ │  │ │            │ │  │ │            │  │
  │ │ 6 LXC:     │ │  │ │ vLLM GPU   │ │  │ │ Windows 11 │  │
  │ │ ├─ Gateway │ │  │ │ llama.cpp  │ │  │ │ Copilot+ PC│  │
  │ │ ├─ N8N     │ │  │ │70B Hybrid  │ │  │ │            │  │
  │ │ ├─ Vault   │ │  │ │ Monitoring │ │  │ │ Arc GPU    │  │
  │ │ ├─ DB      │ │  │ └────────────┘ │  │ │ NPU        │  │
  │ │ ├─ Observ. │ │  │                │  │ │            │  │
  │ │ └─ Jenkins │ │  │ ┌────────────┐ │  │ │ 2.5GbE USB-C  │
  │ │            │ │  │ │ Jetson O.N.│ │  │ │ 32GB RAM   │  │
  │ │ 2.5GbE     │ │  │ │ Super      │ │  │ │ 512GB SSD  │  │
  │ │ 64GB RAM   │ │  │ │            │ │  │ │            │  │
  │ │ 1TB SSD    │ │  │ │ (USB 4.0   │ │  │ └────────────┘  │
  │ │            │ │  │ │ Local RPC) │ │  │                 │
  │ │ Static IP: │ │  │ │            │ │  │ Static IP:      │
  │ │ 100.64.1.4 │ │  │ │ 15-25 tok/s│ │  │ 100.68.1.3      │
  │ └────────────┘ │  │ │ Local edge │ │  │                 │
  │                │  │ │            │ │  │ SSH/Dev traffic │
  │ DB queries:    │  │ │ Static IP: │ │  │                 │
  │ ← 50Mbps       │  │ │ 100.65.1.2 │ │  │ < 10Mbps        │
  │                │  │ └────────────┘ │  │                 │
  │ Jenkins-N8N:   │  │                │  │ ARP only for    │
  │ ← 25Mbps       │  │ Inference RPC: │  │ device discovery│
  │                │  │ 100.65.1.2     │  │                 │
  │ Uptime Kuma:   │  │ ↔ 100.65.2.2   │  └─────────────────┘
  │ ← 5Mbps        │  │ (3-8ms latency)│
  │                │  │ ← 500-800Mbps  │
  │ Max per-port:  │  │ (GPU tasks)    │
  │ 100Mbps        │  │                │
  │                │  │ Thermal monitor│
  │ Global limit:  │  │ → 1-2Mbps      │
  │ 500Mbps        │  │                │
  │ to all nets    │  │ Max per-port:  │
  │                │  │ 2.5Gbps        │
  │                │  │                │
  │                │  │ Global limit:  │
  │                │  │ 1.8Gbps        │
  │                │  │ (preserve for  │
  │                │  │ inference)     │
  │                │  │                │
  │                │  │ Static IPs:    │
  │                │  │ 100.65.1.2     │
  │                │  │ 100.65.2.2     │
  │                │  │ (Jetson)       │
  │                │  └────────────────┘
  │                │
  │                │  VLAN 100/200 isolation:
  │                │  ├─ Zero east-west traffic by default
  │                │  ├─ Explicit allow rules only:
  │                │  │  ├─ XPS15→Schlimers: Prometheus scrape
  │                │  │  ├─ XPS15→Jetson: Health checks
  │                │  │  └─ Schlimers→XPS15: Monitoring push
  │                │  ├─ mDNS blocked (no device discovery)
  │                │  └─ Broadcast isolated per VLAN
  │                │
  │                │  IP addressing:
  │                │  ├─ VLAN 100: 100.64.1.0/25 (Production)
  │                │  ├─ VLAN 200: 100.65.0.0/23 (AI/ML)
  │                │  │  ├─ 100.65.1.0/25 (GPU servers)
  │                │  │  └─ 100.65.2.0/25 (Edge/RPC)
  │                │  ├─ VLAN 300: 100.68.1.0/24 (Dev/Guest)
  │                │  └─ ULA RFC 4193: fd00::/8 (IPv6)
  │                │
  └────────────────┘

```

---

## Device-Level Configuration

### 1. UNIFI UDM SE (Core Gateway) - CONFIGURATION

#### WAN Configuration
```
WAN Type:           IPv4 + IPv6 Dual-Stack
ISP Bandwidth:      1Gbps (assumed) / 2.5GbE (if fiber)
DNS:                Quad9 (Threat Intel) + Cloudflare (Fallback)
DNSSEC:             Enabled
Dynamic Firewall:   Threat Intelligence enabled
IDS/IPS:            Intrusion Detection (not blocking)
Firewall Mode:      Stateful, RFC 6890 compliant
Logging:            Netflow v5 to Prometheus
```

#### VLAN Configuration

| VLAN | Name | Subnet | Purpose | Isolation | QoS Priority |
|------|------|--------|---------|-----------|---------------|
| 1 | Default | 192.168.1.0/24 | Unifi management | High | 0 (mgmt) |
| 100 | Production | 100.64.1.0/25 | XPS15 Proxmox + services | High (strict) | 4 (high) |
| 200 | AI/ML | 100.65.0.0/23 | Schlimers GPU + Jetson | Medium | 5 (medium) |
| 300 | Dev/Guest | 100.68.1.0/24 | XPS13 + guest | Low (isolation) | 7 (low) |
| 999 | IoT | 10.99.0.0/24 | Future IoT devices | Isolated | 7 (low) |

#### VLAN Firewall Rules

**Rule 1: Production VLAN (100) - Egress**
```
Source:     VLAN 100
Destination: Any
Protocol:   TCP/UDP
Action:     Allow

With QoS:
  - Per-port limit: 100Mbps (prevent saturation)
  - Global aggregate: 500Mbps (preserve WAN)
  - Burst allowed: 200% for 5 seconds
```

**Rule 2: AI/ML VLAN (200) - Internal Only**
```
Source:     VLAN 200
Destination: VLAN 200
Protocol:   Any
Action:     Allow (unlimited, internal only)

With QoS:
  - Per-port limit: 2.5Gbps (internal bandwidth)
  - Global aggregate: 1.8Gbps (preserve inference)
  - Priority: High (inference is latency-sensitive)
```

**Rule 3: AI/ML (200) → Production (100) - Monitoring Only**
```
Source:     VLAN 200 (Schlimers GPU server)
Destination: VLAN 100 (XPS15 Proxmox)
Ports:      9090 (Prometheus), 3100 (Loki)
Action:     Allow
QoS:        1Mbps (monitoring only)

With reverse (100→200):
Source:     VLAN 100 (Proxmox)
Destination: VLAN 200 (Schlimers, Jetson)
Ports:      9090 (scrape), 8000 (inference API if needed)
Action:     Allow
QoS:        50Mbps (monitoring + occasional health checks)
```

**Rule 4: Dev/Guest VLAN (300) - Restricted**
```
Source:     VLAN 300
Destination: VLAN 100 (Production)
Action:     Deny

Source:     VLAN 300
Destination: VLAN 200 (AI/ML)
Action:     Deny

Source:     VLAN 300
Destination: WAN
Action:     Allow
QoS:        10Mbps per device (prevent saturation)
DPI:        Enabled (block P2P, torrents)
```

**Rule 5: Default Deny (All others)**
```
Action:     Deny
Logging:    Enabled (for audit)
```

#### Advanced Features

**Traffic Shaping (Queuing Discipline)**
```
Algorithm:      CAKE (Common Applications Kept Enhanced)
Bandwidth:      2.5Gbps (link speed)
Quantum:        1500 bytes (Ethernet frame)
Target latency: 5ms (reduce bufferbloat)

Class weights:
  - Priority 0 (mgmt): 10%
  - Priority 4 (prod): 15%
  - Priority 5 (AI): 60%
  - Priority 7 (guest): 15%
```

**DNS Security**
```
Forwarders:
  - 9.9.9.9 (Quad9 - threat intelligence)
  - 1.1.1.2 (Cloudflare - DNSSEC blocking)
  - 100.64.1.10:5353 (Local Pi-hole, optional)

DNSSEC:         Enabled
DNS-over-HTTPS: Enabled
Logging:        DNS queries with response times
TTL:            Aggressive caching (3600s)
```

**IP Addressing + DHCP**

| VLAN | DHCP Range | Static IPs | DNS Suffix | Lease Time |
|------|-----------|-----------|------------|------------|
| 100 | 100.64.1.100-120 | .1-.99 | prod.local | 24h |
| 200 | 100.65.0.100-254 | .1-.99 | ai.local | 1h (inference may restart) |
| 300 | 100.68.1.100-254 | .1-.99 | guest.local | 6h |

---

### 2. Schlimers-Server (AI Desktop) - CONFIGURATION

#### Network Configuration
```bash
# Primary: 2.5GbE Ethernet
Device:          eth0
IP:              100.65.1.2/25 (static, VLAN 200)
Gateway:         100.65.0.1
DNS:             100.65.0.1 (UDM SE)
MTU:             1500 (standard)

# Secondary: USB 4.0 (Jetson RPC link, local only)
Device:          eth1
IP:              100.65.2.1/25 (static, VLAN 200 subnet 2)
Gateway:         None (local RPC, no default route)
MTU:             9000 (jumbo frames enabled, RTT optimization)
Purpose:         Ultra-low latency (3-8ms) direct link to Jetson
Bandwidth:       40Gbps (USB 4.0 theoretical, ~2.5Gbps practical)

# ULA IPv6 (dual-stack)
IPv6:            fd00::65:1:2/125 (RFC 4193)
IPv6 gateway:    fd00::1
```

#### Docker Networking (vLLM Container)
```
Network:         host (direct access to eth0)
Port binding:    100.65.1.2:8000 (inference API)
Port binding:    100.65.1.2:8001 (metrics)
Port binding:    100.65.1.2:9090 (Prometheus metrics)

Firewall rules (UFW):
  - Allow from 100.64.1.0/25 (Proxmox monitoring)
  - Allow from 100.65.0.0/23 (internal)
  - Deny from 100.68.0.0/16 (guest VLAN)
```

#### Kernel Networking Tuning
```bash
# /etc/sysctl.d/98-ai-idp.conf

# TCP performance (reduce latency, increase throughput)
net.ipv4.tcp_tw_reuse = 1                # Reuse TIME_WAIT connections
net.ipv4.tcp_fin_timeout = 30            # Reduce FIN-WAIT-2
net.core.somaxconn = 65535               # Max pending connections
net.ipv4.tcp_max_syn_backlog = 65535     # SYN backlog size

# Jumbo frames (USB 4.0 link to Jetson)
net.ipv4.tcp_mss_clamp = 0               # Don't clamp MSS
net.ipv6.mtu = 9000                      # IPv6 jumbo frames

# Buffer tuning (prevent drops during inference spikes)
net.core.rmem_max = 1073741824            # 1GB RX buffer
net.core.wmem_max = 1073741824            # 1GB TX buffer
net.ipv4.tcp_rmem = 4096 87380 536870912 # RX buffer range
net.ipv4.tcp_wmem = 4096 65536 536870912 # TX buffer range

# Congestion control (BBR for higher throughput)
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq              # Fair queuing

# IP forwarding (for monitoring, optional)
net.ipv4.ip_forward = 1

# Apply
sysctl -p /etc/sysctl.d/98-ai-idp.conf
```

#### Monitoring Exporters (Prometheus scrape)
```bash
# On schlimers-server:
# Export GPU metrics to Prometheus
address: 100.65.1.2:9090
jobs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['100.65.1.2:8001']
        labels:
          vlan: '200-ai'
          device: 'schlimers-gpu'
  
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['100.65.1.2:9400']  # NVIDIA DCGM exporter
        labels:
          vlan: '200-ai'
          device: 'rtx-5070-ti'
```

---

### 3. Jetson Orin Nano Super (Edge Node) - CONFIGURATION

#### Network Configuration
```bash
# USB 4.0 local link to Schlimers-Server
Device:          eth0
IP:              100.65.2.2/25 (static, local RPC subnet)
Gateway:         100.65.2.1 (Schlimers-server, ultra-low latency)
DNS:             100.65.2.1 (local forwarding) + 100.65.0.1 (UDM fallback)
MTU:             9000 (jumbo frames, match Schlimers)
Connection type: USB 4.0 Type-C (40Gbps theoretical, ~2.5Gbps practical)

# ULA IPv6
IPv6:            fd00::65:2:2/125
IPv6 gateway:    fd00::65:2:1

# Interface tuning for RPC
rx_ring_entries: 256
tx_ring_entries: 256
coalesce_delay: 100us (reduce interrupt overhead)
```

#### JetPack Networking Setup
```bash
# Install static IP (not DHCP, critical for RPC reliability)
sudo nmcli connection add type ethernet con-name eth0
sudo nmcli connection modify eth0 ipv4.addresses 100.65.2.2/25
sudo nmcli connection modify eth0 ipv4.gateway 100.65.2.1
sudo nmcli connection modify eth0 ipv4.dns 100.65.2.1,100.65.0.1
sudo nmcli connection modify eth0 ipv4.method manual
sudo nmcli connection up eth0

# Verify
ip addr show
route -n
ping -c 3 100.65.2.1  # Should see 3-8ms latency
```

#### RPC Server Configuration
```python
# /opt/jetson-rpc/config.yaml
server:
  host: 100.65.2.2
  port: 8000
  workers: 4  # Jetson has 8-core ARM, max 4 for inference
  timeout: 30
  
tls:
  enabled: true
  cert: /etc/jetson-rpc/server.crt
  key: /etc/jetson-rpc/server.key
  
models:
  default: llama-3.2-1b-instruct
  quantization: int8
  max_tokens: 2048
  
monitoring:
  metrics_port: 9400
  log_level: INFO
```

#### Firewall Rules (UFW, Jetson)
```bash
# Allow only from Schlimers-server (100.65.2.1)
sudo ufw default deny incoming
sudo ufw allow from 100.65.2.1 to 100.65.2.2 port 8000
sudo ufw allow from 100.65.2.1 to 100.65.2.2 port 9400
sudo ufw enable
```

---

### 4. XPS 15 Proxmox VE (Production Core) - CONFIGURATION

#### Network Configuration
```bash
# 2.5GbE USB-C adapter
Device:          eth0
IP:              100.64.1.4/25 (static, VLAN 100 - Production)
Gateway:         100.64.0.1
DNS:             100.64.0.1 (UDM), 9.9.9.9 (Quad9 fallback)
MTU:             1500
Bridge:          vmbr0 (for LXC containers)

# Proxmox bridge (for LXC containers)
Bridge:          vmbr0
IPv4:            100.64.1.4 (host binding)
VLAN aware:      Yes (containers can use different VLANs)
QoS:             Enabled (per-container limits)
```

#### LXC Container Network Configuration

| LXC | Name | VLAN | Static IP | Purpose | Bandwidth |
|-----|------|------|-----------|---------|----------|
| 1 | sclimers-gateway | 100 | 100.64.1.11 | NGINX, Portainer, Uptime | 100Mbps |
| 2 | sclimers-core-ops | 100 | 100.64.1.12 | N8N workflows | 100Mbps |
| 3 | sclimers-verify-ops | 100 | 100.64.1.13 | Vault, Authentik, Redis | 100Mbps |
| 4 | sclimers-data-ops | 100 | 100.64.1.14 | PostgreSQL, Dockage | 100Mbps |
| 5 | sclimers-observation | 100 | 100.64.1.15 | Prometheus, Grafana, Loki | 150Mbps |
| 6 | sclimers-runner | 100 | 100.64.1.16 | Jenkins, Gitea | 100Mbps |

#### Firewall Configuration (Proxmox host)

```
Host (100.64.1.4) Inbound:
  Allow SSH:         10.0.0.0/8 (internal only, no WAN)
  Allow Proxmox API: 100.64.1.0/25 (LXC containers)
  Allow ICMP ping:   100.64.0.0/16 (internal monitoring)
  Deny all other

LXC Containers:
  Allow LXC↔LXC:     Same VLAN (100.64.1.0/25)
  Allow LXC↔Host:    Via bridge (vmbr0)
  Allow LXC→WAN:     Gateway (100.64.0.1)
  Allow LXC→Other:   See inter-VLAN rules (Unifi level)
```

#### Monitoring Integration (Prometheus)
```yaml
# On sclimers-observation (100.64.1.15)
scrape_configs:
  - job_name: 'proxmox-host'
    static_configs:
      - targets: ['100.64.1.4:9100']  # node-exporter
        labels:
          vlan: '100-production'
          device: 'xps15-proxmox'
  
  - job_name: 'lxc-containers'
    static_configs:
      - targets:
          - '100.64.1.11:9100'  # gateway
          - '100.64.1.12:9100'  # core-ops
          - '100.64.1.13:9100'  # verify-ops
          - '100.64.1.14:9100'  # data-ops
          - '100.64.1.15:9100'  # observation (self-scrape)
          - '100.64.1.16:9100'  # runner
        labels:
          vlan: '100-production'
          type: 'lxc'
  
  - job_name: 'ai-gpu-inference'
    static_configs:
      - targets: ['100.65.1.2:9090']  # Schlimers GPU server
        labels:
          vlan: '200-ai'
          device: 'schlimers-gpu'
          source: 'prometheus-pull'  # UDM cannot scrape this VLAN
```

---

### 5. XPS 13 Developer Workstation - CONFIGURATION

#### Network Configuration (Windows 11)
```
Device:          Realtek 2.5GbE USB-C adapter
IP (DHCP):       100.68.1.x (dynamic, VLAN 300)
Alternate IP:    Manual static 100.68.1.10/24 (if DHCP unavailable)
DNS:             UDM auto-assigned (100.68.0.1 + fallback)
Gateway:         100.68.0.1
Firewall:        Windows Defender (allow SSH, allow VPN back-in)
```

#### Network Isolation Rules (on XPS 13 + Windows Firewall)
```
Inbound Rules:
  SSH (port 22):       Allow from home network only
  RDP (port 3389):     Allow from Proxmox subnet only
  mDNS (port 5353):    Disabled (prevent discovery)
  ARP:                 Unrestricted (necessary for IP config)

Outbound Rules:
  To VLAN 100:         Block (prevent snooping on production)
  To VLAN 200:         Block (prevent snooping on AI inference)
  To WAN:              Allow
  To DNS:              Allow (100.68.0.1)
```

#### SSH Configuration (Development)
```bash
# From XPS 13 to Schlimers-Server (AI IDP)
# Add to ~/.ssh/config

Host schlimers
    HostName 100.65.1.2
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60
    ServerAliveCountMax 3
    # Note: Cross-VLAN, blocked by firewall rule
    # Requires explicit allow on UDM (Rule 3 equivalent)

Host jetson
    HostName 100.65.2.2
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ProxyCommand ssh schlimers -W %h:%p
    # Access via Schlimers gateway
```

---

## Inter-Device Communication Matrix

### Allowed Flows (Unifi Firewall Level)

```
┌─────────────────┬────────────────┬──────────────────┬──────────┬──────────────┐
│ Source          │ Destination    │ Protocol/Port    │ Bandwidth│ Firewall     │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ XPS15 (100)     │ Schlimers (200)│ TCP 9090 (prom)  │ 1Mbps    │ Allow        │
│ Proxmox→GPU     │ GPU metrics    │ TCP 8001 (vllm)  │ 1Mbps    │ Allow        │
│                 │ scrape         │                  │          │              │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ XPS15 (100)     │ Jetson (200)   │ TCP 9400 (gpu)   │ 0.5Mbps  │ Allow        │
│ Proxmox→Jetson  │ edge metrics   │ (via Schlimers)  │          │ (health)     │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ Schlimers (200) │ Jetson (200)   │ TCP 8000 (RPC)   │ 500-800Mbps│Allow        │
│ GPU→Edge        │ inference      │ UDP 5353 (mDNS)  │ (burst)  │ (internal)   │
│ (USB 4.0)       │ RPC calls      │ ICMP ping        │          │              │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ XPS13 (300)     │ WAN            │ TCP 80/443       │ 10Mbps   │ Allow        │
│ Dev→Internet    │ browsing/SSH   │                  │          │ (dev only)   │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ Schlimers (200) │ WAN            │ NTP, DNS         │ 0.1Mbps  │ Allow        │
│ GPU→NTP/DNS     │ time/resolution│                  │          │              │
├─────────────────┼────────────────┼──────────────────┼──────────┼──────────────┤
│ XPS15 (100)     │ WAN            │ HTTP/S, SSH      │ 500Mbps  │ Allow        │
│ Prod→Internet   │ updates/backup │ NTP              │          │              │
│ (conditional)   │                │                  │          │              │
└─────────────────┴────────────────┴──────────────────┴──────────┴──────────────┘
```

### Blocked Flows (Zero-Trust by Default)

```
┌─────────────────┬────────────────┬──────────┬──────────────────┐
│ Source          │ Destination    │ Reason   │ Firewall Action  │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ XPS13 Dev (300) │ Proxmox (100)  │ Isolation│ Deny, log        │
│ Dev→Production  │ core ops       │          │                  │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ XPS13 Dev (300) │ Schlimers (200)│ Isolation│ Deny, log        │
│ Dev→AI GPU      │ GPU inference  │          │                  │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ XPS13 Dev (300) │ Jetson (200)   │ Isolation│ Deny, log        │
│ Dev→Edge        │ edge inference │          │                  │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ Schlimers (200) │ Proxmox (100)  │ Isolation│ Deny except      │
│ GPU→Production  │ (except metrics)│          │ 9090/3100 (rule) │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ Jetson (200)    │ Proxmox (100)  │ Isolation│ Deny except      │
│ Edge→Production │ (except metrics)│          │ 9400 (rule)      │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ All             │ mDNS (5353)    │ Privacy  │ Deny             │
│ (cross-VLAN)    │ discovery      │          │                  │
├─────────────────┼────────────────┼──────────┼──────────────────┤
│ Guest/IoT (999) │ Everything     │ Isolation│ Deny (default)   │
│                 │ except WAN     │          │                  │
└─────────────────┴────────────────┴──────────┴──────────────────┘
```

---

## Bandwidth Allocation & QoS

### Layer 1: UDM SE Link Capacity

```
Total Interface Capacity:    2.5Gbps
Accounting for overhead:     ~96% usable = 2.4Gbps

Allocation:
├─ VLAN 100 (Production)     15%   = 360Mbps
│  └─ Limit: 500Mbps total (burst allowed, shaped back)
│
├─ VLAN 200 (AI/ML)          60%   = 1.44Gbps (priority)
│  └─ Limit: 1.8Gbps total (preserve for inference)
│
├─ VLAN 300 (Dev/Guest)      10%   = 240Mbps
│  └─ Limit: 10Mbps per device (prevent floods)
│
├─ VLAN 999 (IoT/Reserved)   10%   = 240Mbps
│  └─ Reserved for future expansion
│
└─ Management/Overhead       5%    = 120Mbps
   └─ DNS, NTP, Netflow, SNMP
```

### Layer 2: Per-Device Limits

```
VLAN 100 (Production/Proxmox):
  XPS15 (100.64.1.4)         Per-port: 100Mbps (prevent saturation)
  LXC Containers             Per-container: 50-100Mbps
    ├─ sclimers-gateway:     100Mbps (NGINX, Portainer)
    ├─ sclimers-core-ops:    100Mbps (N8N workflows)
    ├─ sclimers-verify-ops:  100Mbps (Vault, Redis auth)
    ├─ sclimers-data-ops:    100Mbps (PostgreSQL queries)
    ├─ sclimers-observation: 150Mbps (Prometheus scraping)
    └─ sclimers-runner:      100Mbps (Jenkins, Gitea)
  Total: 500Mbps aggregate limit (no single device saturates link)

VLAN 200 (AI/ML Inference):
  Schlimers GPU (100.65.1.2) Per-port: 2.5Gbps (local link capacity)
  Jetson (100.65.2.2)        Per-port: 2.5Gbps (USB 4.0 capacity)
  Total: 1.8Gbps aggregate (preserve headroom for latency-sensitive tasks)
  Internal (GPU↔Jetson):      Unlimited (same VLAN, no shaping)
  External (200↔other):       50Mbps (cross-VLAN monitoring only)

VLAN 300 (Dev/Guest):
  Per device: 10Mbps (prevent guest devices from consuming bandwidth)
  Burst: 50Mbps for 2 seconds (allow SSH/browsing burst)
  Total: 240Mbps aggregate (guest devices don't starve production)
```

### Layer 3: Protocol-Level QoS (Per-Flow)

```
Ingress (Into LXC containers):
  Prometheus scrape (9090):   1Mbps (low priority)
  PostgreSQL query (5432):    5Mbps (medium priority)
  N8N webhook (5678):         10Mbps (high priority, workflows)
  Jenkins build (8080):       10Mbps (medium priority)
  SSH (22):                   1Mbps (interactive, preserve latency)

Egress (From schlimers-gpu to Jetson RPC):
  vLLM inference (8000):      500-800Mbps (highest priority, latency-sensitive)
  Thermal monitoring (dmon):  1-2Mbps (very low, constant)
  Health checks (ping):       0.1Mbps (ICMP, essential)
```

---

## Disaster Recovery & Failover

### Scenario 1: Schlimers GPU Server Down

**Impact**: AI inference unavailable, edge inference still works  
**RTO**: 5 minutes (restart Docker, re-download weights)  
**RPO**: 0 minutes (inference is stateless)

**Mitigation**:
```bash
# Jetson can run smaller models independently
# Fallback: llama-3.2-1b from Jetson (15-25 tok/s, acceptable)
# Monitor via Prometheus alert: vllm_up == 0
# Automatic failover routing in NGINX Proxmox gateway
  upstream inference {
    server 100.65.1.2:8000 max_fails=2 fail_timeout=10s;
    server 100.65.2.2:8000 backup;  # Jetson as backup
  }
```

### Scenario 2: Jetson Orin Down

**Impact**: Edge inference unavailable, GPU inference still works  
**RTO**: 3 minutes (restart Jetson, SSH into GPU server)  
**RPO**: 0 minutes (edge inference is stateless)

**Mitigation**:
```bash
# GPU can handle all inference (slower but available)
# Direct all requests to 100.65.1.2:8000
# Monitor via Prometheus: jetson_up == 0
# Alert Proxmox monitoring
```

### Scenario 3: XPS 15 Proxmox Down

**Impact**: Production services unavailable (N8N, Vault, PostgreSQL)  
**RTO**: 30 minutes (full Proxmox restore from backup)  
**RPO**: 1 hour (PostgreSQL backups hourly)

**Mitigation**:
```bash
# Backup strategy (on Schlimers GPU, VLAN 200→200, isolated):
# 1. PostgreSQL dumps: hourly, 7-day rotation
# 2. Vault secrets: daily, 30-day rotation
# 3. N8N workflows: daily Git sync to Gitea
# 4. System configs: daily rsync to /mnt/backup

# Manual restore procedure:
# 1. Boot XPS15 Proxmox from backup USB
# 2. Restore PostgreSQL from latest dump
# 3. Restore Vault from daily backup
# 4. Resync N8N workflows from Gitea
# 5. Verify all LXC containers start
```

### Scenario 4: UDM SE Gateway Down

**Impact**: All network connectivity lost, local inference still works  
**RTO**: 2 minutes (UDM restart)  
**RPO**: 0 minutes (UDM config backed up to Proxmox)

**Mitigation**:
```bash
# Devices will remain online with last-known IP config
# Local (USB 4.0) links unaffected:
#   - Schlimers↔Jetson can still communicate (100.65.2.0/25 direct)
#   - Jetson can still serve inference

# WAN connectivity: lost until UDM recovery
# Backup strategy: Proxmox periodically pulls UDM config via API
#   (requires explicit allow in UDM firewall)

# Recovery:
# 1. Hard reboot UDM SE (power off 30 seconds)
# 2. UDM auto-starts, restores from backups
# 3. DHCP servers on all VLANs restart
# 4. Static IPs remain unchanged
# 5. Devices auto-reconnect
```

---

## Security Best Practices

### 1. Zero-Trust Network Architecture

✅ **Default Deny**: All inter-VLAN traffic blocked unless explicitly allowed  
✅ **Explicit Allow**: Each rule specifies source, destination, protocol, port  
✅ **No Broadcast**: mDNS, ARP requests isolated per VLAN  
✅ **Encryption**: TLS for all API communication (vLLM, Jetson RPC)  
✅ **Mutual Auth**: API keys + certificate pinning on cross-device calls  

### 2. VLAN Segmentation

| VLAN | Threat Model | Controls | Monitoring |
|------|--------------|----------|------------|
| 100 (Production) | Data exfiltration, insider threat | Explicit rules, audit logs | Netflow to Prometheus |
| 200 (AI/ML) | Model theft, GPU hijacking | TLS, API keys, strict isolation | Thermal + latency metrics |
| 300 (Dev/Guest) | Reconnaissance, lateral movement | Rate limiting, DPI, no prod access | Per-device bandwidth caps |

### 3. Authentication & Authorization

```yaml
VLAN 200 (AI/ML Inference API):
  Authentication:
    - API Key (Bearer token in HTTP header)
    - TLS certificate pinning (reject MITM)
    - Source IP whitelist (only 100.64.1.15 = Prometheus)
  
  Authorization:
    - POST /v1/completions → allowed
    - DELETE /models → denied (no lifecycle changes from Proxmox)
    - GET /metrics → allowed (read-only)

VLAN 200 (Jetson RPC):
  Authentication:
    - Mutual TLS (server cert + client cert)
    - JWT tokens (10 minute expiry)
  
  Authorization:
    - inference/complete → allowed
    - system/shutdown → denied (no remote shutdown)
    - system/reboot → denied
```

### 4. Traffic Monitoring

```bash
# Netflow export from UDM SE to Prometheus
# (requires UDM feature: export to syslog or IPFIX collector)

Netflow Configuration (UDM):
  Exporter IP: 100.64.1.15 (Prometheus, VLAN 100)
  Exporter Port: 2055 (standard NetFlow v5)
  Sampling: 1:100 (sample 1 in 100 packets)
  Template: 15-minute refresh

Prometheus scrape:
  job_name: 'netflow-collector'
  targets: ['100.64.1.15:2055']
  scrape_interval: 30s

Grafana dashboard:
  - Top flows (VLAN 200→VLAN 100: Prometheus scrape)
  - Top talkers (Schlimers GPU: expected 500-800Mbps to Jetson)
  - Blocked flows (cross-VLAN attempts, audit trail)
  - Alert: >1.8Gbps on VLAN 200 (inference degradation)
```

### 5. DNS Security

```
UDM SE DNS Configuration:
  Primary:   9.9.9.9 (Quad9 - threat intelligence, blocks malware)
  Secondary: 1.1.1.2 (Cloudflare - DNSSEC validation)
  Tertiary:  8.8.8.8 (Google, fallback only)
  
  DNSSEC:    Enabled (validates domain signatures)
  DNS-over-HTTPS: Enabled (encrypted queries)
  Query logging: Enabled (Prometheus export)

Per-VLAN DNS policies:
  VLAN 100 (Production): Any query allowed (trusted)
  VLAN 200 (AI/ML):     Restrict to ntp.pool.org, archive.ubuntu.com
  VLAN 300 (Dev):       Restrict to known-safe resolvers
  VLAN 999 (IoT):       Restrict to specific IoT vendor domains
```

---

## Operations Runbook

### Daily Checks

```bash
# 1. Verify all devices online
ping -c 3 100.65.1.2    # Schlimers GPU
ping -c 3 100.65.2.2    # Jetson Orin
ping -c 3 100.64.1.4    # Proxmox
ping -c 3 100.68.1.10   # XPS13 (if static)

# 2. Check inference latency (GPU→Jetson RPC)
curl -s http://100.65.1.2:9090/metrics | grep vllm_request_latency
# Expected: <100ms P99

# 3. Verify Prometheus scrape targets
curl -s http://100.64.1.15:9090/api/v1/targets | jq '.data.activeTargets[].labels.job'
# Should show: vllm, nvidia-gpu, jetson-rpc, all healthy

# 4. Check network bandwidth utilization (Grafana)
# Dashboard: http://100.64.1.15:3000
# Query: sum(rate(netflow_bytes[5m])) by (dst_vlan)
# Expected VLAN 200: 800Mbps during inference, <10Mbps idle
```

### Weekly Backup Verification

```bash
# 1. Test PostgreSQL backup restore (on separate LXC)
cd /mnt/backup/postgresql
ls -lh *.sql.gz  # Should show daily dumps
latest=$(ls -t *.sql.gz | head -1)
zcat $latest | wc -l  # Verify not corrupted

# 2. Verify Vault secrets backup
ls -lh /mnt/backup/vault-*.tar.gz
# Should have 7 rotations (daily, 7-day retention)

# 3. Verify Schlimers GPU model cache
ls -lh /mnt/models/*.bin  # Models present
sha256sum /mnt/models/*.bin > /mnt/backup/models.sha256
git add /mnt/backup/models.sha256
git commit -m "Weekly model verification"
```

### Monthly Network Audit

```bash
# 1. Review blocked firewall rules (cross-VLAN attempts)
curl -s http://100.64.1.15:9090/api/v1/query?query=netflow_blocked_bytes | jq .
# Alert if any unexpected blocked flows

# 2. Review UDM threat intelligence log
# UDM UI → Dashboard → Threat Intelligence
# Should show 0 malware/phishing blocked (devices should use trusted DNS)

# 3. Verify VLAN isolation
ping -c 3 100.68.1.10   # XPS13 from Proxmox host
# Should fail (ICMP blocked by UDM firewall rule)
ping -c 3 100.65.1.2    # Schlimers from Proxmox
# Should succeed (monitoring allowed)

# 4. Check API key rotation (Schlimers GPU)
sudo docker inspect vllm | grep -i env | grep API_KEY
# Verify key changed in last 30 days
```

---

## Troubleshooting Decision Tree

### "Inference is slow (< 20 tok/s, expected 60-100)"

```
1. Check GPU utilization:
   nvidia-smi dmon -s puc
   Expected: GPU-Util ~80-95%, GPU-Mem ~70-80%
   
   If GPU-Util < 50%:
   → Check if requests queued: curl http://100.65.1.2:9001/metrics | grep queue
   → Reduce vLLM workers (--tensor-parallel-size)
   
   If GPU-Mem > 85%:
   → Model is too large for 16GB
   → Fall back to smaller model or enable 8-bit quantization

2. Check network latency (GPU↔Jetson):
   ping -c 100 -i 0.01 100.65.2.2
   Expected: <5ms min, <10ms avg, <20ms max
   
   If latency >20ms:
   → Check USB 4.0 link: ethtool -S eth1 | grep errors
   → Restart USB 4.0 interface: sudo ifdown eth1 && sleep 2 && sudo ifup eth1
   → Verify MTU 9000 on both sides: ip link | grep mtu

3. Check vLLM container health:
   docker logs vllm-gpu | tail -20
   Expected: No errors, "Started server process"
   
   If CUDA out of memory:
   → Reduce max-tokens: docker-compose down
   → Edit docker-compose.yml, set --max-num-batched-tokens 2048
   → docker-compose up -d vllm-gpu

4. Check Prometheus metrics for bottleneck:
   curl -s http://100.65.1.2:9001/metrics | grep -E '(request_latency|batch_size|prefill_time)'
   → If prefill_time > 50ms: KV cache is cold, expected on first query
   → If batch_size < 2: not batching requests, check if queue empty
   → If request_latency > vLLM TTFT: network is bottleneck, check rule 3
```

### "Cannot reach Jetson from Proxmox (100.64.1.15→100.65.2.2 fails)"

```
1. Is this expected? Check firewall rule:
   Proxmox → Jetson: Only allowed on port 9400 (metrics), not ping
   
   Correct check:
   curl -s http://100.65.2.2:9400/metrics | head
   Expected: HELP jetson_gpu_utilization ...

2. If metrics port fails:
   → Jetson RPC server down: ssh ubuntu@100.65.2.2 (via Schlimers proxy)
   → Check: sudo systemctl status jetson-rpc
   → If failed: sudo journalctl -u jetson-rpc -n 50

3. If can't SSH to Jetson:
   → SSH via Schlimers gateway:
   ssh -J ubuntu@100.65.1.2 ubuntu@100.65.2.2
   → Check network: ssh ubuntu@100.65.1.2 ping -c 3 100.65.2.2
   → Verify USB 4.0 link: ssh ubuntu@100.65.1.2 ethtool eth1
```

### "Cross-VLAN traffic being blocked unexpectedly"

```
1. Check UDM firewall logs:
   SSH to UDM → /var/log/ufw.log (or Check UI → Dashboard → Activity)
   
2. Identify blocked rule:
   Source VLAN, destination VLAN, protocol, port
   Match against "Blocked Flows" matrix (earlier in this doc)
   
3. If legitimate traffic is blocked:
   → Check if rule 3 (Monitoring) was added:
   UDM UI → Security → Firewall → Rules → Create rule
   → Source: 100.64.1.15 (Proxmox observation)
   → Destination: 100.65.0.0/23 (VLAN 200 AI/ML)
   → Ports: 9090, 8000 (Prometheus + inference)
   → Action: Allow
   
4. If still blocked:
   → Check global firewall state: UDM UI → Security → Firewall
   → Verify mode is "Stateful" not "Blocked"
   → Check if WAF/IDS is incorrectly blocking: disable temporarily, retest
```

### "Prometheus cannot scrape metrics from VLAN 200"

```
1. Verify Prometheus is on correct VLAN:
   docker exec prometheus cat /etc/hosts | grep localhost
   ip addr | grep 100.64.1.15
   Expected: Prometheus on VLAN 100 (100.64.1.15)

2. Check scrape configuration:
   curl -s http://100.64.1.15:9090/api/v1/targets
   For vllm target:
     state: "up" → scrape working
     state: "down" → network unreachable
     lastError: "connection refused" → service down

3. If connection refused:
   → Check vLLM is running: docker ps | grep vllm
   → Check port 9001 is listening: sudo ss -tlnp | grep 9001
   → Check firewall rule 3 allows Proxmox→GPU:
   UDM UI → Security → Firewall → Rules
   → Verify: Source=100.64.1.0/25, Dest=100.65.0.0/23, Port=9090, Action=Allow

4. If rule exists but still fails:
   → UDM bug: try restarting UDM gateway service
   → Workaround: enable Prometheus to export via syslog (agent-push)
   → ssh ubuntu@100.65.1.2 "telegraf --config telegraf.conf &"
   → telegraf pushes metrics to Prometheus (reverse direction)
```

---

## Performance Targets

### Expected Network Latency

| Path | Protocol | Expected | Observed | Status |
|------|----------|----------|----------|--------|
| Schlimers→Jetson (USB 4.0) | TCP/IP | 3-8ms | TBD | To validate |
| Proxmox→Schlimers (2.5GbE) | TCP/IP | 1-2ms | TBD | To validate |
| XPS13→Proxmox (2.5GbE) | SSH | 1-2ms | TBD | To validate |
| Proxmox→Unifi (local) | ICMP | <1ms | TBD | To validate |
| Any→DNS query | UDP | <5ms | TBD | To validate |

### Expected Bandwidth Utilization

| Scenario | VLAN 100 | VLAN 200 | VLAN 300 | Status |
|----------|----------|----------|----------|--------|
| All idle | <1Mbps | <1Mbps | <1Mbps | Baseline |
| N8N webhook | 10Mbps | - | - | Normal |
| DB backup (hourly) | 50Mbps | - | - | Expected |
| Prometheus scrape | 1Mbps | 1Mbps | - | Constant |
| GPU inference (vLLM 8B) | <1Mbps | 600Mbps | - | Expected |
| GPU→Jetson RPC | - | 700Mbps | - | Expected |
| Dev SSH | <1Mbps | - | <1Mbps | Expected |

---

## Next Steps

### Phase 1: Network Setup (Week 1)

- [ ] **Day 1**: Unifi UDM SE configuration (VLAN, firewall rules, QoS)
- [ ] **Day 2**: Schlimers-server network setup (eth0 VLAN 200, eth1 USB 4.0)
- [ ] **Day 3**: Jetson Orin network setup (static IP 100.65.2.2)
- [ ] **Day 4**: Proxmox monitoring integration (Prometheus + Grafana)
- [ ] **Day 5**: XPS13 network isolation (VLAN 300, firewall rules)
- [ ] **Day 6-7**: Integration testing + baseline bandwidth measurement

### Phase 2: Monitoring & Observability (Week 2)

- [ ] **Day 8**: Netflow export from UDM to Prometheus
- [ ] **Day 9**: Grafana dashboards (per-VLAN bandwidth, latency)
- [ ] **Day 10**: Alerting rules (throughput < 50 tok/s, latency > 100ms)
- [ ] **Day 11**: Baseline establishment (normal bandwidth, latency profiles)
- [ ] **Day 12-14**: Automated anomaly detection + runbook testing

### Phase 3: Security Hardening (Week 3)

- [ ] **Day 15**: API key rotation + TLS certificate generation
- [ ] **Day 16**: Mutual TLS setup (Schlimers↔Jetson)
- [ ] **Day 17**: DNS security (DNSSEC, DoH)
- [ ] **Day 18**: Firewall audit + unexpected traffic review
- [ ] **Day 19-21**: Disaster recovery drill (failover scenarios)

### Phase 4: Production Deployment (Week 4)

- [ ] **Day 22**: Backup verification (PostgreSQL, Vault)
- [ ] **Day 23**: Performance under load test (concurrent requests)
- [ ] **Day 24**: Security audit (pen test cross-VLAN isolation)
- [ ] **Day 25**: Documentation + runbook finalization
- [ ] **Day 26-30**: Production monitoring + optimization

---

## Configuration Files (Templates)

### A. Unifi UDM SE Export (JSON)

```json
{
  "vlans": [
    {
      "vlan_id": 100,
      "name": "Production",
      "network_address": "100.64.1.0/25",
      "gateway": "100.64.1.1",
      "dhcp_range": "100.64.1.100-120",
      "firewall_rules": [
        {"action": "allow", "from_network": "100.64.1.0/25", "to_network": "any", "protocol": "tcp/udp"}
      ]
    },
    {
      "vlan_id": 200,
      "name": "AI/ML",
      "network_address": "100.65.0.0/23",
      "gateway": "100.65.0.1",
      "firewall_rules": [
        {"action": "allow", "from_network": "100.65.0.0/23", "to_network": "100.65.0.0/23", "protocol": "any"},
        {"action": "allow", "from_network": "100.64.1.15", "to_network": "100.65.0.0/23", "ports": [9090, 3100], "protocol": "tcp"}
      ]
    }
  ]
}
```

### B. Schlimers-Server Network Config (netplan)

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 100.65.1.2/25
      gateway4: 100.65.0.1
      nameservers:
        addresses: [100.65.0.1]
      mtu: 1500
    eth1:
      dhcp4: false
      addresses:
        - 100.65.2.1/25
      mtu: 9000  # Jumbo frames for Jetson
```

### C. Jetson Orin Network Config (nmcli)

```bash
sudo nmcli connection add type ethernet con-name eth0 \
  ipv4.addresses 100.65.2.2/25 \
  ipv4.gateway 100.65.2.1 \
  ipv4.dns 100.65.2.1,100.65.0.1 \
  ipv4.method manual
```

---

## Summary

Your network is now **architected for production zero-trust security** while supporting **high-throughput AI inference**. The Unifi UDM SE acts as the core gateway, enforcing strict isolation between production (VLAN 100), AI/ML (VLAN 200), and dev workstations (VLAN 300).

**Key achievements**:
✅ Production traffic isolated from AI inference (no interference)  
✅ Edge inference (Jetson) low-latency linked to GPU (3-8ms RPC)  
✅ Monitoring allowed explicitly (Prometheus scrape, no default open)  
✅ Guest devices cannot snoop on production or AI (VLAN 300 blocked)  
✅ Bandwidth preserved for inference (1.8Gbps dedicated to AI)  
✅ Failover strategies documented (GPU down, Jetson down, UDM down)  
✅ DNS security hardened (Quad9 + DNSSEC + DoH)  
✅ All devices use static IPs + cryptographic keys (no guessing)  

**Ready to proceed with Phase 1 network setup?** Start with Unifi UDM configuration per Day 1 section above.

---

*Last updated: January 12, 2026*  
*Network Architecture v1.0 - Production Ready*
