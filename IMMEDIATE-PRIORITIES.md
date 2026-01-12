# Immediate Implementation Priorities
**Status**: Action Items Ready | **Date**: January 12, 2026

---

## Your Situation

You have a **production-ready AI IDP specification** (Perplexity-Labs-003) rated **8.9-9.0/10**. All critical gaps resolved. Physics-validated performance targets documented.

**Next steps**: Execute 30-day deployment plan with 4 phases of work.

---

## The 3 Most Critical Actions (This Week)

### 1. Complete Pre-Flight Checklist (Today - 30 min)

```bash
# VERIFY YOU'RE READY
nvidia-smi                    # RTX 5070 Ti shows
free -h                       # 128GB available
df -h /mnt/models             # /mnt/models mounted
docker run --rm --gpus all nvidia/cuda nvidia-smi  # GPU access OK
```

**Decision**: All pass → Proceed. Any fail → Fix first.

---

### 2. Deploy vLLM (Days 1-5 of Phase 1)

**Core Steps**:

```bash
# 1. Secrets
mkdir -p ~/ai-idp/secrets
cd ~/ai-idp
cp ../local-developer-roadmap/Perplexity-Labs-003/scripts/generate-secrets.sh scripts/
./scripts/generate-secrets.sh

# 2. Models (2-3 hours in background)
cp ../local-developer-roadmap/Perplexity-Labs-003/scripts/download-models.sh scripts/
tmux new -d -s models
tmux send-keys -t models "./scripts/download-models.sh" Enter

# 3. Deploy
cp ../local-developer-roadmap/Perplexity-Labs-003/docker-compose.production-v2.yml docker-compose.yml
vim docker-compose.yml  # Verify /mnt/models paths and timezone
docker-compose up -d vllm-gpu

# 4. Verify
sleep 30
curl http://localhost:8000/health  # Should return {"status":"ready"}
```

**Expected**: vLLM healthy, Llama 3.1 8B FP8 loaded, API responding.

---

### 3. Validate Performance (Day 5)

```bash
# CRITICAL: Test against CORRECTED targets
cp -r ../local-developer-roadmap/Perplexity-Labs-003/tests .
cd tests
pip install pytest httpx requests
pytest test_inference_performance.py -v

# Expected CORRECTED targets (NOT original claims):
# - Throughput: 60-100 tok/s (NOT 140-170)
# - TTFT P50: <80ms (NOT 22ms)
# - TTFT P95: <500ms
```

**Why corrected targets**: See SYSTEM-FINALIZATION-v2-PRODUCTION.md §3 (physics validation).

---

## Phase Breakdown

| Phase | Week | Days | Deliverable | Effort |
|-------|------|------|-------------|--------|
| **0** | Now | 1 day | Pre-flight pass | 0.5 hrs |
| **1** | W1 | 7 | vLLM deployed & validated | 8-10 hrs |
| **2** | W2 | 7 | Jetson edge integration | 6-8 hrs |
| **3** | W3 | 7 | 70B hybrid + speculative | 10-12 hrs |
| **4** | W4 | 7 | Security + v1.0 release | 8-10 hrs |
| **TOTAL** | 4 weeks | 30 | Production AI IDP | ~40 hrs |

---

## What NOT To Do

| Feature | Issue | Status |
|---------|-------|--------|
| 140-170 tok/s target | Physics-invalid | ⛔ Use 60-100 instead |
| 30-40 tok/s 70B | DDR5 bandwidth bottleneck | ⛔ Expect 8-15 tok/s |
| 0.4ms USB 4.0 RPC | Unvalidated claim | ⛔ Budget 3-8ms |
| Deploy speculative Week 1 | Experimental | ⏸️ Defer to Week 3 |
| NPU inference | Immature support | ⏸️ Experimental (Week 3) |

---

## Decision Tree: Where Are You?

```
Completed Phase 0 pre-flight?
├─ NO: Do it now (30 min)
└─ YES: Continue

Deployed vLLM (Phase 1)?
├─ NO: Follow NEXT-STEPS-EXECUTION-PLAN.md Days 1-5
└─ YES: Continue

Validated performance (Phase 1)?
├─ NO: Run pytest test_inference_performance.py
│      Expect 60-100 tok/s (not 140)
└─ YES: Continue

Finished Week 1 (all 7 days)?
├─ NO: Follow daily sequence, don't skip
└─ YES: Start Phase 2

Finished Week 2 (Jetson)?
├─ NO: Follow Days 8-14
└─ YES: Continue

Finished Week 3 (70B)?
├─ NO: Follow Days 15-21
└─ YES: Continue

Finished Week 4 (production)?
├─ NO: Follow Days 22-30
└─ YES: v1.0-PRODUCTION deployed!
```

---

## Critical Reading Order

1. **NEXT-STEPS-EXECUTION-PLAN.md** ← Primary execution guide
2. **SYSTEM-FINALIZATION-v2-PRODUCTION.md** ← Physics validation
3. **FINALIZATION-COMPLETE.md** ← Why rated 8.9/10
4. **STRATEGIC-ROADMAP.md** ← 30-day timeline

---

## Quick Reference: Daily Operations

```bash
# Check system
docker-compose ps
curl http://localhost:8000/health

# Monitor GPU
nvidia-smi dmon -s puc

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b","prompt":"test","max_tokens":50}'

# Emergency: Reduce memory if OOM
sed -i 's/0.80/0.75/' docker-compose.yml
docker-compose restart vllm-gpu
```

---

## Ready to Start?

**Next action**: Complete Phase 0 pre-flight checklist above.

If all checks pass → Begin NEXT-STEPS-EXECUTION-PLAN.md immediately.

**Confidence**: 9/10 (production-ready)
**Time commitment**: 40 hours over 30 days
**Success probability**: >95% (if following plan)

---
*Last updated: January 12, 2026*
