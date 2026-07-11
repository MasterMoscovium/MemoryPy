// ═══════════════════════════════════════════════════════════
//  MEMORYPY — LIVE PYTHON BACKEND UI
// ═══════════════════════════════════════════════════════════

// ── State ──
let running = true;
let decayModel = 'adaptive';
let decayLambda = 0.05;
let isSpawningObstacle = false;

// ── Canvas Setup ──
const simCanvas = document.getElementById('simCanvas');
const simCtx = simCanvas.getContext('2d');
let gridW = 0, gridH = 0, res = 0.1, cellPx = 2;

function resizeSimCanvas() {
    const container = document.getElementById('sim-container');
    simCanvas.width = container.clientWidth;
    simCanvas.height = container.clientHeight;
}
window.addEventListener('resize', resizeSimCanvas);
resizeSimCanvas();

// ── Background Particles ──
const particlesCanvas = document.getElementById('particles-canvas');
const pCtx = particlesCanvas.getContext('2d');
const particles = [];
function initParticles() {
    particlesCanvas.width = window.innerWidth;
    particlesCanvas.height = window.innerHeight;
    particles.length = 0;
    for (let i = 0; i < 40; i++) {
        particles.push({
            x: Math.random() * particlesCanvas.width,
            y: Math.random() * particlesCanvas.height,
            vx: (Math.random() - 0.5) * 0.2,
            vy: (Math.random() - 0.5) * 0.2,
            r: Math.random() * 2 + 0.5,
            alpha: Math.random() * 0.2 + 0.05,
            hue: Math.random() > 0.5 ? 185 : 270
        });
    }
}
function renderParticles() {
    pCtx.clearRect(0, 0, particlesCanvas.width, particlesCanvas.height);
    for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = particlesCanvas.width;
        if (p.x > particlesCanvas.width) p.x = 0;
        if (p.y < 0) p.y = particlesCanvas.height;
        if (p.y > particlesCanvas.height) p.y = 0;
        pCtx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${p.alpha})`;
        pCtx.beginPath(); pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2); pCtx.fill();
    }
    requestAnimationFrame(renderParticles);
}
initParticles();
renderParticles();

// ── Metrics History ──
const MAX_HISTORY = 100;
let coverageHistory = [];
let memoryHistory = [];
let currentTimestep = 0;

// ── Controls ──
const FORMULAS = {
    exponential: { text: 'R = e<sup>−λ·Δt</sup>', desc: 'Ebbinghaus curve — rapid initial decay' },
    power_law: { text: 'R = (Δt+1)<sup>−β</sup>', desc: "Jost's Law — slower long-term decay" },
    adaptive: { text: 'R = e<sup>−λ/S · Δt</sup>', desc: 'S grows with revisits (spaced repetition)' },
    none: { text: 'R = 1.0', desc: 'Perfect memory, standard SLAM' },
    threshold: { text: 'R = 0 if Δt > τ, else 1', desc: 'Binary forgetting' },
};

function changeModel(model) {
    decayModel = model;
    const f = FORMULAS[model];
    document.getElementById('formula-display').innerHTML = f.text;
    document.getElementById('formula-desc').textContent = f.desc;
    document.getElementById('badge-model').textContent = `${model.replace('_', '-')} λ=${decayLambda.toFixed(3)}`;
    
    fetch('/api/set_decay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rate: decayLambda, model: model })
    });
    renderDecayCurves();
}

function updateLambda(val) {
    decayLambda = parseFloat(val);
    document.getElementById('val-lambda').textContent = decayLambda.toFixed(3);
    document.getElementById('badge-model').textContent = `${decayModel.replace('_', '-')} λ=${decayLambda.toFixed(3)}`;
    
    fetch('/api/set_decay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rate: decayLambda })
    });
    renderDecayCurves();
}

function setSpeed(s) {
    document.querySelectorAll('.speed-pill').forEach(p => {
        p.classList.toggle('active', parseFloat(p.dataset.speed) === s);
    });
    fetch('/api/set_speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: s })
    });
}

function toggleSim() {
    fetch('/api/toggle', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
            running = d.running;
            document.getElementById('btn-toggle').innerHTML = running ? '⏸ Pause' : '▶ Resume';
        });
}

function clearObstacles() {
    fetch('/api/clear_obstacles', { method: 'POST' });
}

function wipeMemory() {
    fetch('/api/clear_memory', { method: 'POST' });
    coverageHistory = [];
    memoryHistory = [];
}

function toggleObstacleMode() {
    isSpawningObstacle = !isSpawningObstacle;
    const btn = document.getElementById('btn-obstacle');
    btn.innerHTML = isSpawningObstacle ? '🎯 Click Map' : '🚧 Obstacle';
    simCanvas.style.cursor = isSpawningObstacle ? 'crosshair' : 'default';
}

simCanvas.addEventListener('click', (e) => {
    if (!isSpawningObstacle || gridW === 0) return;
    const rect = simCanvas.getBoundingClientRect();
    const cellW = simCanvas.width / gridW;
    const cellH = simCanvas.height / gridH;
    
    const col = Math.floor((e.clientX - rect.left) / cellW);
    const row = Math.floor((e.clientY - rect.top) / cellH);
    
    // Convert to world coordinates (matching Python backend logic)
    const worldX = (col + 0.5) * res;
    const worldY = (gridH - 1 - row + 0.5) * res;
    
    fetch('/api/spawn_obstacle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: worldX - 0.5, y: worldY - 0.5, w: 1.0, h: 1.0 })
    });
    
    toggleObstacleMode();
});

// ── WebSocket Rendering ──
let cachedGT = null;   // Cache ground truth (only sent every 10 frames)
let fpsFrames = 0, fpsLast = performance.now(), fpsDisplay = 0;

function renderGrid(data) {
    const W = data.w;
    const H = data.h;
    res = data.res;

    if (W !== gridW || H !== gridH) {
        gridW = W;
        gridH = H;
    }

    // Cache gt when sent
    if (data.gt) cachedGT = data.gt;

    // FPS counter
    fpsFrames++;
    const now = performance.now();
    if (now - fpsLast >= 1000) {
        fpsDisplay = fpsFrames;
        fpsFrames = 0;
        fpsLast = now;
    }
    const fpsEl = document.getElementById('fps-counter');
    if (fpsEl) fpsEl.textContent = `${fpsDisplay} FPS`;

    // Use ImageData for blazing fast pixel rendering
    const imgData = simCtx.createImageData(W, H);
    const pixels = imgData.data;

    let known = 0, total = 0, decayed = 0;

    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const i = r * W + c;
            const v = data.grid[i];
            const gt = cachedGT ? cachedGT[i] : 0;
            const lastObs = data.last_obs[i];

            // ImageData is top-left origin, our grid row 0 is bottom
            const drawRow = H - 1 - r;
            const pi = (drawRow * W + c) * 4;

            if (gt === 0) total++;
            if (v !== -1 && gt === 0) known++;

            let red, green, blue;

            if (gt === 1) {
                red = 50; green = 10; blue = 70;
            } else if (v === -1) {
                red = 5; green = 8; blue = 16;
            } else if (v < 0.35) {
                const dt = (lastObs >= 0) ? (data.t - lastObs) : 999;
                if (dt > 30) decayed++;
                const freshness = Math.max(0, 1 - dt / 150);
                red = Math.round(5 + 15 * (1 - freshness) + 120 * (1 - freshness));
                green = Math.round(60 + 170 * freshness);
                blue = Math.round(80 + 175 * freshness);
            } else if (v > 0.6) {
                const intensity = Math.min((v - 0.6) / 0.4, 1);
                red = Math.round(100 + 155 * intensity);
                green = 10;
                blue = Math.round(100 + 129 * intensity);
            } else {
                red = 40; green = 25; blue = 65;
            }

            pixels[pi]     = red;
            pixels[pi + 1] = green;
            pixels[pi + 2] = blue;
            pixels[pi + 3] = 255;
        }
    }

    // Draw the pixel grid scaled up to the canvas
    const offscreen = new OffscreenCanvas(W, H);
    const offCtx = offscreen.getContext('2d');
    offCtx.putImageData(imgData, 0, 0);

    simCtx.imageSmoothingEnabled = false;
    simCtx.clearRect(0, 0, simCanvas.width, simCanvas.height);
    simCtx.drawImage(offscreen, 0, 0, simCanvas.width, simCanvas.height);

    const cellW = simCanvas.width / W;
    const cellH = simCanvas.height / H;

    // Update Stats
    const coverage = total > 0 ? known / total : 0;
    document.getElementById('stat-coverage').textContent = (coverage * 100).toFixed(1) + '%';
    document.getElementById('stat-cells').textContent = known;
    document.getElementById('stat-decayed').textContent = decayed;
    document.getElementById('stat-frontiers').textContent = data.frontiers ? data.frontiers.length : 0;

    if (data.t % 5 === 0 && data.t !== currentTimestep) {
        currentTimestep = data.t;
        coverageHistory.push(coverage);
        memoryHistory.push(known);
        if (coverageHistory.length > MAX_HISTORY) {
            coverageHistory.shift();
            memoryHistory.shift();
        }
        renderCoverageGraph();
        renderMemoryGraph();
    }

    // Frontiers
    if (data.frontiers) {
        simCtx.shadowColor = '#ffab00';
        simCtx.shadowBlur = 8;
        simCtx.fillStyle = '#ffab00';
        for (const f of data.frontiers) {
            const fx = (f[1] + 0.5) * cellW;
            const fy = (H - 1 - f[0] + 0.5) * cellH;
            simCtx.beginPath();
            simCtx.arc(fx, fy, Math.max(2, cellW * 0.8), 0, Math.PI * 2);
            simCtx.fill();
        }
        simCtx.shadowBlur = 0;
    }

    // Robot
    if (data.est) {
        const est = data.est;
        const rx = (est[0] / res + 0.5) * cellW;
        const ry = (H - 1 - est[1] / res + 0.5) * cellH;
        const rr = Math.max(5, cellW * 2.5);

        simCtx.shadowColor = '#ffffff';
        simCtx.shadowBlur = 20;
        simCtx.fillStyle = '#ffffff';
        simCtx.beginPath(); simCtx.arc(rx, ry, rr, 0, Math.PI * 2); simCtx.fill();

        simCtx.shadowBlur = 0;
        simCtx.fillStyle = '#ffff00';
        simCtx.beginPath(); simCtx.arc(rx, ry, rr * 0.5, 0, Math.PI * 2); simCtx.fill();

        simCtx.strokeStyle = '#ffffff';
        simCtx.lineWidth = 2;
        simCtx.beginPath();
        simCtx.moveTo(rx, ry);
        simCtx.lineTo(rx + Math.cos(-est[2]) * rr * 3, ry + Math.sin(-est[2]) * rr * 3);
        simCtx.stroke();
    }
}

// ── Comparison Grids ──
function renderComparisons(comps, t) {
    if (!comps) return;
    for (const [name, compData] of Object.entries(comps)) {
        const canvas = document.getElementById(`canv-${name}`);
        if (!canvas) continue;

        const ctx = canvas.getContext('2d');
        const W = compData.w;
        const H = compData.h;

        // Match aspect ratio
        const parent = canvas.parentElement;
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight - 20;

        // Use ImageData for fast rendering
        const imgData = ctx.createImageData(W, H);
        const pixels = imgData.data;

        for (let r = 0; r < H; r++) {
            for (let c = 0; c < W; c++) {
                const i = r * W + c;
                const v = compData.grid[i];
                const lastObs = compData.last_obs[i];
                const drawRow = H - 1 - r;
                const pi = (drawRow * W + c) * 4;

                let red, green, blue;
                if (v === -1) {
                    red = 5; green = 8; blue = 16;
                } else if (v < 0.35) {
                    const dt = (lastObs >= 0) ? (t - lastObs) : 999;
                    const freshness = Math.max(0, 1 - dt / 150);
                    red = Math.round(5 + 15 * (1 - freshness) + 120 * (1 - freshness));
                    green = Math.round(60 + 170 * freshness);
                    blue = Math.round(80 + 175 * freshness);
                } else if (v > 0.6) {
                    const intensity = Math.min((v - 0.6) / 0.4, 1);
                    red = Math.round(100 + 155 * intensity);
                    green = 10;
                    blue = Math.round(100 + 129 * intensity);
                } else {
                    red = 40; green = 25; blue = 65;
                }

                pixels[pi]     = red;
                pixels[pi + 1] = green;
                pixels[pi + 2] = blue;
                pixels[pi + 3] = 255;
            }
        }

        const offscreen = new OffscreenCanvas(W, H);
        const offCtx = offscreen.getContext('2d');
        offCtx.putImageData(imgData, 0, 0);

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    }
}

// ── WebSocket Connection ──
const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/stream`;
function connectWS() {
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        const badge = document.getElementById('ws-status');
        const text = document.getElementById('ws-status-text');
        badge.style.borderColor = 'var(--green)';
        text.style.color = 'var(--green)';
        text.textContent = '● Online (Python Backend)';
    };

    ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        document.getElementById('sim-clock').textContent = `T = ${d.t}`;
        renderGrid(d);
        if (d.comps) {
            renderComparisons(d.comps, d.t);
        }
    };

    ws.onclose = () => {
        const badge = document.getElementById('ws-status');
        const text = document.getElementById('ws-status-text');
        badge.style.borderColor = 'var(--red)';
        text.style.color = 'var(--red)';
        text.textContent = '● Offline';
        setTimeout(connectWS, 2000);
    };
}
connectWS();

// ── Static Graphs ──
function renderDecayCurves() {
    const canvas = document.getElementById('graphDecay');
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const W = canvas.width, H = canvas.height;
    const pad = { t: 10, r: 10, b: 20, l: 35 };
    const gw = W - pad.l - pad.r, gh = H - pad.t - pad.b;

    ctx.fillStyle = 'rgba(5, 8, 16, 0.8)';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + gh * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + gw, y); ctx.stroke();
    }
    
    const maxT = 200;
    const curves = [
        { fn: t => Math.exp(-decayLambda * t), color: '#00e5ff' },
        { fn: t => Math.pow(t + 1, -0.5), color: '#b388ff' },
        { fn: t => { const S = Math.min(1 + 0.3 * 5, 10); return Math.exp(-decayLambda / S * t); }, color: '#00e676' },
        { fn: t => t > 100 ? 0 : 1, color: '#ffab00' },
    ];

    for (const curve of curves) {
        ctx.beginPath();
        ctx.strokeStyle = curve.color;
        ctx.lineWidth = 2;
        ctx.shadowColor = curve.color;
        ctx.shadowBlur = 4;
        for (let i = 0; i <= 100; i++) {
            const t = (i / 100) * maxT;
            const x = pad.l + (i / 100) * gw;
            const y = pad.t + gh * (1 - curve.fn(t));
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}
renderDecayCurves();

function renderCoverageGraph() {
    const canvas = document.getElementById('graphCoverage');
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const W = canvas.width, H = canvas.height;
    const pad = { t: 10, r: 10, b: 10, l: 35 };
    const gw = W - pad.l - pad.r, gh = H - pad.t - pad.b;

    ctx.fillStyle = 'rgba(5, 8, 16, 0.8)';
    ctx.fillRect(0, 0, W, H);
    if (coverageHistory.length < 2) return;

    ctx.beginPath();
    const grad = ctx.createLinearGradient(pad.l, 0, pad.l + gw, 0);
    grad.addColorStop(0, '#00e5ff'); grad.addColorStop(1, '#b388ff');
    ctx.strokeStyle = grad; ctx.lineWidth = 2;
    ctx.shadowColor = '#00e5ff'; ctx.shadowBlur = 6;

    for (let i = 0; i < coverageHistory.length; i++) {
        const x = pad.l + (i / (MAX_HISTORY - 1)) * gw;
        const y = pad.t + gh * (1 - coverageHistory[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
}

function renderMemoryGraph() {
    const canvas = document.getElementById('graphMemory');
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const W = canvas.width, H = canvas.height;
    const pad = { t: 10, r: 10, b: 10, l: 35 };
    const gw = W - pad.l - pad.r, gh = H - pad.t - pad.b;

    ctx.fillStyle = 'rgba(5, 8, 16, 0.8)';
    ctx.fillRect(0, 0, W, H);
    if (memoryHistory.length < 2) return;

    const maxVal = Math.max(...memoryHistory, 1);
    ctx.beginPath();
    ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2;
    ctx.shadowColor = '#00e676'; ctx.shadowBlur = 6;

    for (let i = 0; i < memoryHistory.length; i++) {
        const x = pad.l + (i / (MAX_HISTORY - 1)) * gw;
        const y = pad.t + gh * (1 - memoryHistory[i] / maxVal);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
}
