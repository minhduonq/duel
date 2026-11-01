#!/usr/bin/env bash
# Run N copies of render-service-js on a vast.ai instance, on 127.0.0.1:BASE_PORT, BASE_PORT+1, ...
# `python vast.py render` uploads the service to ~/render and runs this script; `python vast.py tunnel` forwards the ports.
# The renderer draws WebGL on the CPU (SwiftShader) and the pages of one Chrome share one GPU thread, so throughput
# grows with processes, not RENDER_POOL_SIZE, up to the container's CPU quota (cat /sys/fs/cgroup/cpu.max).
# Each copy holds ~140 threads; vast containers cap threads (cat /sys/fs/cgroup/pids.max), so keep N modest.
set -euo pipefail

DIR="${RENDER_DIR:-$HOME/render}"
N="${RENDER_INSTANCES:-8}"
BASE="${RENDER_BASE_PORT:-8101}"
cd "$DIR"
command -v node >/dev/null || . /opt/nvm/nvm.sh

lock="$(sha256sum package-lock.json)"
if [ "$(cat .lock-sha 2>/dev/null)" != "$lock" ]; then
    npm ci --omit=dev --no-audit --no-fund
    echo "$lock" > .lock-sha
fi
npx puppeteer browsers install chrome --install-deps >/dev/null

pkill -9 -f "^node src/server.js" || true
pkill -9 -f "^$HOME/.cache/puppeteer/chrome" || true
for i in $(seq 0 $((N - 1))); do
    port=$((BASE + i))
    PORT=$port STATIC_PORT=$((port + 1000)) RENDER_POOL_SIZE=2 RENDER_QUEUE_CAPACITY=256 RENDER_QUEUE_WAIT_MS=600000 \
        setsid nohup node src/server.js > "render-$port.log" 2>&1 < /dev/null &
    sleep 2
done
sleep 10

ok=0
for i in $(seq 0 $((N - 1))); do
    curl -s -m 3 "http://127.0.0.1:$((BASE + i))/health" | grep -q ok && ok=$((ok + 1))
done
echo "[render] $ok/$N healthy on ports $BASE-$((BASE + N - 1)); threads $(cat /sys/fs/cgroup/pids.current 2>/dev/null)/$(cat /sys/fs/cgroup/pids.max 2>/dev/null), cpu quota $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
