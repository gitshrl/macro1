# macro1 Architecture

## System Overview

```
┌─────────────────────────────────────────────────┐
│  Job Management (external API/dashboard)        │
│  - Creates jobs in MongoDB (bots.jobs)           │
│  - Pushes job messages to RabbitMQ queues        │
│  - Exposes JOB_STATUS_API, JOB_LOG_API           │
└──────────────────────┬──────────────────────────┘
                       │ job message (JSON)
                       ▼
┌─────────────────────────────────────────────────┐
│  RabbitMQ (10.10.0.44:5673)                     │
│  - Queue per device (e.g. "social1", "social2") │
│  - Job message = full job document from MongoDB  │
└──────────────────────┬──────────────────────────┘
                       │ pika subscribe/consume
                       ▼
┌─────────────────────────────────────────────────┐
│  Job Orchestrator (macro1)          ← BUILD THIS│
│                                                  │
│  Spawns one worker per emulator (parallel).      │
│  Each worker:                                    │
│  1. Subscribe to its own RabbitMQ queue           │
│  2. Consume job → build goal for ReActAgent      │
│  3. update_jobstatus(jobid, "In Progress")       │
│  4. Run ReActAgent with goal on its emulator     │
│  5. On success:                                  │
│     - update_jobstatus(jobid, "OK")              │
│     - create_job_log(jobid, "OK")                │
│     - open_url(JOB_SUCCESS_URL + jobid)          │
│  6. On error:                                    │
│     - update_jobstatus(jobid, "Error", detail)   │
│     - create_job_log(jobid, "Error", detail)     │
│     - open_url(JOB_FAILED_URL + jobid)           │
└──────────────────────┬──────────────────────────┘
                       │ ADB + uiautomator2
                       │ (one connection per emulator)
                       ▼
┌─────────────────────────────────────────────────┐
│  Emulators (Android 14, API 34)     ← DONE      │
│                                                  │
│  Each emulator has its own:                      │
│  - RabbitMQ queue (e.g. "social1")               │
│  - Worker process/thread                         │
│  - ReActAgent instance                           │
│  - Environment (ADB + u2) connection             │
│                                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│  │  social1   │ │  social2   │ │  social3   │ │  social4   │
│  │  emu-5554  │ │  emu-5556  │ │  emu-5558  │ │  emu-5560  │
│  │  worker 1  │ │  worker 2  │ │  worker 3  │ │  worker 4  │
│  │  queue: q1 │ │  queue: q2 │ │  queue: q3 │ │  queue: q4 │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘
│  - 20 actions tested, all passing                │
└─────────────────────────────────────────────────┘
```

## Job Flow

### Old System (rule-based macros)
```
RabbitMQ → job (with action_list_url) → action_translate() → ADB (uiautomator v1)
```
- Predefined macro steps fetched from action_list_url
- Giant switch statement in action_translate()
- No AI, every step is scripted

### New System (AI-driven, macro1)
```
RabbitMQ → job (with goal) → ReActAgent (VLM decides steps) → Environment (u2)
```
- Job is translated into a natural language goal
- ReActAgent uses VLM (Qwen 3.5) to see screen + decide actions
- 20 actions available, agent picks the right ones autonomously

## Job Message Format (from MongoDB bots.jobs)

```json
{
  "_id": {"$oid": "69aab49b51fdf035cd02a378"},
  "platform": "instagram",
  "bot": {
    "id": 32559,
    "username": "siska_mhrn99",
    "password": "mhrani9900",
    "phone": null
  },
  "channel": {
    "name": "device09",           // maps to emulator
    "ip": "10.10.1.22",
    "device_type": "SM-A03"
  },
  "action": "like",               // like, comment, follow, posting, repost
  "detail": {
    "booster_id": 19784,
    "method": "postpage",
    "url": "https://www.instagram.com/p/DTo6AG2k9Mu/",
    "username": "siska_mhrn99",
    "password": "mhrani9900",
    "name": "Siska Maharani",
    "random": 1,                  // wait seconds before starting
    "profile_url": "https://www.instagram.com/siska_mhrn99/"
  },
  "created_at": {"$date": 1772795042728}
}
```

## Job → Goal Translation

Each device queue receives mixed job types. The orchestrator translates job fields
into a natural language goal for ReActAgent based on action + platform:

| Action   | Platform  | Goal Example |
|----------|-----------|-------------|
| like     | instagram | "Open this Instagram post URL and like it: {url}" |
| like     | twitter   | "Open this tweet URL and like it: {url}" |
| comment  | instagram | "Open this Instagram post URL and comment '{text}': {url}" |
| follow   | instagram | "Open this Instagram profile URL and follow: {profile_url}" |
| follow   | youtube   | "Open this YouTube channel URL and subscribe: {url}" |
| posting  | instagram | "Open Instagram, create a new post with the photo in gallery, caption: {text}" |
| repost   | twitter   | "Open this tweet URL and repost it: {url}" |

Example queue flow for social1:
```
[social1 queue] → like (IG) → comment (IG) → follow (YT) → like (TW) → ...
```
Each job processed sequentially on the same emulator, but different emulators run in parallel.

## Job Status API (existing, from old orchestrator)

### update_jobstatus(jobid, status, posturl, detail, error_detail, exec_time)
```
PATCH {JOB_STATUS_API}{jobid}
Body: {
  "status": "In Progress" | "OK" | "Error" | "Canceled",
  "detail": {
    "posturl": "",
    "error": "detail message",
    "error_detail": "stack trace",
    "exec_time": minutes
  }
}
```

### create_job_log(jobid, status, posturl, detail)
```
POST {JOB_LOG_API}
Body: {
  "jobid": "...",
  "status": "OK" | "Error",
  "detail": {"error": "...", "posturl": "..."}
}
```

### Browser feedback on device
```
Success → open_url(JOB_SUCCESS_URL + jobid)
Failed  → open_url(JOB_FAILED_URL + jobid)
```

## Environment Variables Needed

```env
# VLM (existing)
VLM_API_KEY=...
VLM_BASE_URL=https://openrouter.ai/api/v1

# MongoDB (existing)
MONGO_HOST=10.10.0.44
MONGO_PORT=27078
MONGO_USER=root
MONGO_PASS=...

# RabbitMQ (existing)
RABBIT_HOST=10.10.0.44
RABBIT_PORT=5673
RABBIT_USER=user
RABBIT_PASS=pass

# Job APIs (needed from old system)
JOB_STATUS_API=http://???/api/v1/update-jobstatus/
JOB_LOG_API=http://???/api/v1/job-logs
JOB_SUCCESS_URL=http://???/job-success/
JOB_FAILED_URL=http://???/job-failed/

# Device mapping
IP_ADDR=10.10.1.xx
```

## Device Mapping & Parallel Execution

Old system uses `channel.name` (e.g. "device01"-"device09") mapped to physical devices.
New system uses emulator serial numbers:

```
social1 → emulator-5554 → worker 1 → queue "social1"
social2 → emulator-5556 → worker 2 → queue "social2"
social3 → emulator-5558 → worker 3 → queue "social3"
social4 → emulator-5560 → worker 4 → queue "social4"
```

Each emulator gets its own worker (process or thread) that:
- Listens to its own RabbitMQ queue
- Has its own ReActAgent + Environment instance
- Processes jobs sequentially per emulator, but all 4 run in parallel
- Jobs are isolated — one emulator failing doesn't affect others

## Components Status

| Component | Status |
|-----------|--------|
| Environment (ADB + u2) | DONE - 20 actions, tested on real emulator |
| ReActAgent (VLM) | DONE - parse + step + stuck detection |
| Action space | DONE - 20 actions |
| Tests (unit) | DONE - 131 passing |
| Tests (integration) | DONE - 20 passing on emulator |
| RabbitMQ consumer | TODO |
| Job → Goal translator | TODO |
| Job status updater | TODO (port from old orchestrator) |
| Device mapping | TODO |
