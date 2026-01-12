# Network Quickstart: Arrowave Unifi UDM SE + AI IDP Integration
**5-Minute Setup | 30-Minute Validation | ACTUAL Configuration**

---

## Your Real Network (Already Configured!)

```
Arrowave Unifi UDM SE (2.5GbE Gateway)
IP: 10.5.0.1
  └─ VLAN 10.5.4.0/24: Production (Proxmox + LXC containers)
      ├─ Proxmox Host: 10.5.4.45 (static)
      ├─ LXC 1 (Gateway): 10.5.4.101 (static)
      ├─ LXC 2 (Core-Ops): 10.5.4.102 (static)
      ├─ LXC 3 (Verify-Ops): 10.5.4.103 (static)
      ├─ LXC 4 (Data-Ops): 10.5.4.104 (static)
      ├─ LXC 5 (Observation): 10.5.4.105 (static)
      └─ LXC 6 (Runner): 10.5.4.106 (static)

  └─ AI Server (Schlimers GPU) [TO BE CONFIGURED]
      IP: 10.5.X.Y (TBD - choose from 10.5.4.150 or separate VLAN)

  └─ Jetson Orin (Edge) [TO BE CONFIGURED]
      IP: 10.5.X.Z (TBD - local USB 4.0 RPC link)

  └─ XPS13 Dev Workstation [TO BE CONFIGURED]
      IP: 10.5.X.W (TBD - isolated guest VLAN)
```

---

## What You Already Have

✅ **Unifi UDM SE (Arrowave)** - Active, fully operational  
✅ **VLAN 10.5.4.0/24** - Production network exists  
✅ **Proxmox + 6 LXC containers** - Running on production VLAN  
✅ **Static IP addresses** - All assigned (101-106, 45)  
✅ **2.5GbE network** - All devices connected  

---

## What Needs Configuration

❌ **AI GPU Server (Schlimers)** - Not on network yet  
❌ **Jetson Orin** - Not on network yet  
❌ **XPS13 Dev Workstation** - Not on network yet  
❌ **VLAN isolation** - May need additional setup for security  
❌ **Monitoring** - Prometheus/Grafana integration needed  
❌ **Firewall rules** - Cross-VLAN policies needed  

---

## Decision: Where to Place AI Servers?

### Option A: Same VLAN (10.5.4.0/24) - SIMPLEST

**Pros**:
- All devices on same VLAN
- No complex firewall rules
- Proxmox can directly manage/monitor
- Easy troubleshooting
- Optimal bandwidth (no VLAN overhead)

**Cons**:
- No isolation from production
- AI traffic competes with Proxmox services
- No separate QoS for inference

**Recommended for**: Starting Phase 1 (vLLM deployment)

```
Assignment:
  Schlimers GPU:    10.5.4.150 (static, same VLAN as Proxmox)
  Jetson Orin:      10.5.4.151 (static, same VLAN as GPU)
  XPS13 Dev:        10.5.4.152 (static, same VLAN initially)
```

### Option B: Separate VLAN (10.5.X.0/24) - ADVANCED

**Pros**:
- Complete isolation from production
- Separate monitoring/alerts
- Granular QoS (prioritize inference)
- Better security posture

**Cons**:
- More complex firewall rules
- Requires Unifi config changes
- Additional VLAN management

**Recommended for**: Phase 2+ (after initial stability)

```
Assignment:
  New VLAN: 10.5.65.0/24 (VLAN ID 65)
  Schlimers GPU:    10.5.65.2 (static)
  Jetson Orin:      10.5.65.3 (static)
```

---

## **RECOMMENDED: Phase 0 - Use Option A (Same VLAN)**

Start simple, scale to complexity. Here's the minimal setup:

### Step 1: Set Static IPs on AI Servers

#### Schlimers-Server (RTX 5070 Ti + Ryzen 9900X)

```bash
# On AI Desktop (Ubuntu 25.10)
sudo nano /etc/netplan/00-installer-config.yaml

# Replace with:
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet static
    address 10.5.4.150
    netmask 255.255.255.0
    gateway 10.5.0.1
    dns-nameservers 10.5.0.1 8.8.8.8
    mtu 1500

# Save, then apply:
sudo netplan apply

# Verify:
ip addr show eth0
ping 10.5.0.1  # Should work <2ms
ping 10.5.4.45  # Proxmox reachable
```

#### Jetson Orin Nano Super

```bash
# SSH into Jetson (first find its DHCP IP)
# If you know it already: ssh ubuntu@<dhcp-ip>
# If not: Check Unifi UI → Devices → find Jetson

# Set static IP using nmcli
sudo nmcli connection modify eth0 ipv4.method manual
sudo nmcli connection modify eth0 ipv4.addresses 10.5.4.151/24
sudo nmcli connection modify eth0 ipv4.gateway 10.5.0.1
sudo nmcli connection modify eth0 ipv4.dns 10.5.0.1,8.8.8.8
sudo nmcli connection up eth0

# Verify:
ip addr show
ping 10.5.0.1  # Gateway
ping 10.5.4.150  # Schlimers GPU (3-8ms over USB 4.0)
```

#### XPS13 Dev (Windows 11)

```powershell
# Open Settings → Network & Internet → Advanced network settings
# → More network options → Change adapter options

# Right-click Ethernet (2.5GbE adapter) → Properties → IPv4 Properties

# Set:
#   IP Address: 10.5.4.152
#   Subnet Mask: 255.255.255.0
#   Default Gateway: 10.5.0.1
#   DNS: 10.5.0.1, 8.8.8.8

# Verify in PowerShell:
ipconfig /all  # Should show 10.5.4.152
ping 10.5.0.1  # Gateway
ping 10.5.4.45  # Proxmox (production)
```

### Step 2: Update Proxmox Monitoring (Optional)

```bash
# On Proxmox (10.5.4.45)
# Edit Prometheus to scrape new devices:

sudo nano /etc/prometheus/prometheus.yml

# Add new job:
scrape_configs:
  # ... existing jobs ...
  
  - job_name: 'ai-gpu-inference'
    static_configs:
      - targets: ['10.5.4.150:9090']  # vLLM metrics port
        labels:
          device: 'schlimers-gpu'
          vlan: 'production-10.5.4.0'
  
  - job_name: 'jetson-edge'
    static_configs:
      - targets: ['10.5.4.151:9400']  # Jetson DCGM metrics
        labels:
          device: 'jetson-orin'
          vlan: 'production-10.5.4.0'

# Restart Prometheus:
sudo systemctl restart prometheus

# Verify in UI:
http://10.5.4.105:9090  # Prometheus (LXC 5 = sclimers-observation)
# → Status → Targets
# Should show new targets: ai-gpu-inference, jetson-edge
```

### Step 3: Verify Connectivity

```bash
# Quick validation (run from any device):

# 1. All devices reachable
ping -c 3 10.5.4.45    # Proxmox
ping -c 3 10.5.4.150   # Schlimers GPU
ping -c 3 10.5.4.151   # Jetson
ping -c 3 10.5.4.152   # XPS13

# 2. Latency measurements
ping -c 100 10.5.4.150 | tail  # GPU latency
ping -c 100 10.5.4.151 | tail  # Jetson latency (should be low, USB 4.0)

# 3. Basic throughput test
ssh ubuntu@10.5.4.150 "iperf3 -s"  # Start iperf on GPU
iperf3 -c 10.5.4.150  # Test from Proxmox
# Expected: >2000 Mbps (2.5GbE link)

# 4. Verify Prometheus sees metrics
curl -s http://10.5.4.105:9090/api/v1/targets | jq '.data.activeTargets[].labels | select(.job)'
# Should include: ai-gpu-inference, jetson-edge
```

---

## Network Summary (Post-Setup)

```
Arrowave UDM SE (10.5.0.1)
└─ VLAN 10.5.4.0/24 - All Devices (Same network, no VLAN complexity)
    ├─ UDM Gateway: 10.5.0.1
    ├─ Proxmox Host: 10.5.4.45
    ├─ LXC 1 (Gateway/NGINX): 10.5.4.101
    ├─ LXC 2 (N8N): 10.5.4.102
    ├─ LXC 3 (Vault): 10.5.4.103
    ├─ LXC 4 (PostgreSQL): 10.5.4.104
    ├─ LXC 5 (Prometheus/Grafana): 10.5.4.105
    ├─ LXC 6 (Jenkins): 10.5.4.106
    ├─ Schlimers GPU: 10.5.4.150 ← vLLM + llama.cpp (60-100 tok/s)
    ├─ Jetson Orin: 10.5.4.151 ← Edge inference (15-25 tok/s)
    └─ XPS13 Dev: 10.5.4.152 ← Development workstation

All on 2.5GbE link with:
  • Low latency (<2ms production, 3-8ms GPU↔Jetson USB 4.0)
  • High throughput (near-line-rate 2Gbps+ for inference)
  • Monitoring via Prometheus (10.5.4.105:9090)
  • Visualization via Grafana (10.5.4.105:3000)
```

---

## When to Move to Option B (Separate VLAN)

**After Phase 1 (vLLM stable + 60-100 tok/s achieved):**

Create isolated VLAN 10.5.65.0/24 for:
- Separate inference network (no production interference)
- Granular QoS (priority inference traffic)
- Enhanced security (firewall isolation)
- Better monitoring (dedicated telemetry)

**This is a future optimization, not required for initial deployment.**

---

## Key Points for Your Setup

⚠️ **You already have Unifi UDM SE (Arrowave)** - No additional hardware needed  
⚠️ **Production VLAN exists** - All Proxmox + LXC on 10.5.4.0/24  
⚠️ **Static IPs assigned** - Use 10.5.4.150/151/152 for AI servers  
⚠️ **No VLAN complexity needed** - Keep everything on same network for Phase 1  
⚠️ **Monitoring ready** - Prometheus (LXC 5) already deployed  

**Next steps:**
1. Set static IPs on Schlimers, Jetson, XPS13 (this section)
2. Verify connectivity (ping tests below)
3. Update Prometheus config to scrape new devices
4. Deploy vLLM per IMMEDIATE-PRIORITIES.md

---

## Connectivity Verification Checklist

```
□ Schlimers GPU (10.5.4.150)
  ✓ Pings 10.5.0.1 (gateway) <2ms
  ✓ Pings 10.5.4.45 (Proxmox) <2ms
  ✓ Pings 10.5.4.151 (Jetson) <10ms [USB 4.0 expected 3-8ms]
  ✓ DNS resolves (nslookup google.com)
  ✓ Docker can pull images

□ Jetson Orin (10.5.4.151)
  ✓ Pings 10.5.0.1 (gateway) <2ms
  ✓ Pings 10.5.4.45 (Proxmox) <2ms  
  ✓ Pings 10.5.4.150 (GPU server) <10ms [USB 4.0 direct]
  ✓ SSH from Schlimers works (ssh ubuntu@10.5.4.151)
  ✓ JetPack tools accessible

□ Proxmox (10.5.4.45)
  ✓ Pings 10.5.4.150 (GPU) <2ms
  ✓ Pings 10.5.4.151 (Jetson) <2ms
  ✓ All 6 LXC containers online
  ✓ Prometheus running (10.5.4.105:9090)
  ✓ Grafana running (10.5.4.105:3000)

□ Production Services
  ✓ N8N accessible (10.5.4.102:5678)
  ✓ PostgreSQL responding (10.5.4.104:5432)
  ✓ Vault healthy (10.5.4.103:8200)
  ✓ No interference from new devices
```

---

## You're Ready! 🎆

After these 3 steps + verification, your network is prepared for:

✅ **Phase 1** - vLLM GPU deployment (60-100 tok/s)  
✅ **Phase 2** - Jetson edge integration (15-25 tok/s)  
✅ **Phase 3** - Advanced inference (70B hybrid + speculative)  
✅ **Phase 4** - Production hardening + v1.0 release  

All on existing 2.5GbE Unifi UDM SE with production Proxmox system.

**Next**: Deploy AI IDP per IMMEDIATE-PRIORITIES.md

---

*Network Quickstart v2.0 | January 12, 2026 | Updated with Arrowave UDM SE actual config*
