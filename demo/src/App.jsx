import { useState, useEffect, useCallback, useRef } from "react";

const COLORS = {
  pin: { bg: "#FEF3C7", border: "#F59E0B", text: "#92400E", label: "#D97706" },
  recoverable: { bg: "#DBEAFE", border: "#3B82F6", text: "#1E3A5A", label: "#2563EB" },
  irreversible: { bg: "#F3E8FF", border: "#8B5CF6", text: "#4C1D95", label: "#7C3AED" },
  active: "#10B981",
  warm: "#F59E0B",
  archive: "#EF4444",
  bg: "#0F172A",
  card: "#1E293B",
  surface: "#334155",
  textPrimary: "#F1F5F9",
  textSecondary: "#94A3B8",
  accent: "#38BDF8",
};

const THETA1 = 0.6;
const THETA2 = 0.15;

const INITIAL_MEMORIES = [
  { id: 1, content: "Order ID: #58291", type: "pin", importance: 1.0, strength: 1.0, accessCount: 0, createdAt: 0, lastAccessed: 0, tier: "active", decayPath: [] },
  { id: 2, content: "User prefers dark mode UI", type: "recoverable", importance: 0.8, strength: 0.95, accessCount: 0, createdAt: 1, lastAccessed: 1, tier: "active", decayPath: [] },
  { id: 3, content: "Discussed React vs Vue tradeoffs", type: "recoverable", importance: 0.6, strength: 0.85, accessCount: 0, createdAt: 2, lastAccessed: 2, tier: "active", decayPath: [] },
  { id: 4, content: "User said: sounds good, let's do it", type: "irreversible", importance: 0.2, strength: 0.5, accessCount: 0, createdAt: 3, lastAccessed: 3, tier: "active", decayPath: [] },
  { id: 5, content: "API key: sk-prod-****", type: "pin", importance: 1.0, strength: 1.0, accessCount: 0, createdAt: 4, lastAccessed: 4, tier: "active", sensitivity: "sensitive", decayPath: [] },
  { id: 6, content: "Budget is $50k for Q3", type: "recoverable", importance: 0.75, strength: 0.8, accessCount: 0, createdAt: 5, lastAccessed: 5, tier: "active", decayPath: [] },
  { id: 7, content: "Hi, how are you today?", type: "irreversible", importance: 0.1, strength: 0.35, accessCount: 0, createdAt: 6, lastAccessed: 6, tier: "active", decayPath: [] },
  { id: 8, content: "Deploy to us-east-1 region", type: "recoverable", importance: 0.7, strength: 0.75, accessCount: 0, createdAt: 7, lastAccessed: 7, tier: "active", decayPath: [] },
];

function computeDecay(memory, tick) {
  if (memory.type === "pin") return memory.strength;
  const S0 = memory.importance * 10;
  const alpha = 0.3;
  const S = S0 * (1 + alpha * Math.log(1 + memory.accessCount));
  const beta = memory.tier === "active" ? 1.2 : 0.8;
  const dt = tick - memory.lastAccessed;
  if (dt <= 0) return memory.strength;
  const R = Math.exp(-Math.pow(dt / S, beta));
  const rho = 0.05;
  return rho + (1 - rho) * R;
}

function getTier(strength) {
  if (strength > THETA1) return "active";
  if (strength > THETA2) return "warm";
  return "archive";
}

/* ── Sub-components ─────────────────────────────────────────── */

function StrengthBar({ strength }) {
  const color = strength > THETA1 ? COLORS.active : strength > THETA2 ? COLORS.warm : COLORS.archive;
  return (
    <div className="relative w-full h-2 rounded-full overflow-hidden" style={{ backgroundColor: COLORS.surface }}>
      <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ width: `${Math.max(strength * 100, 2)}%`, backgroundColor: color }} />
      <div className="absolute top-0 h-full w-px" style={{ left: `${THETA1 * 100}%`, backgroundColor: "#ffffff30" }} />
      <div className="absolute top-0 h-full w-px" style={{ left: `${THETA2 * 100}%`, backgroundColor: "#ffffff20" }} />
    </div>
  );
}

function MemoryCard({ memory, onRecall }) {
  const typeColors = COLORS[memory.type];
  const tierLabel = memory.tier === "active" ? "ACTIVE" : memory.tier === "warm" ? "WARM" : "ARCHIVE";
  const tierColor = memory.tier === "active" ? COLORS.active : memory.tier === "warm" ? COLORS.warm : COLORS.archive;

  return (
    <div className="rounded-lg p-3 transition-all duration-500 border"
         style={{ backgroundColor: COLORS.card, borderColor: memory.strength > 0.5 ? typeColors.border + "60" : COLORS.surface }}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                style={{ backgroundColor: typeColors.bg, color: typeColors.label }}>
            {memory.type === "pin" ? "PIN" : memory.type === "recoverable" ? "RECOVERABLE" : "IRREVERSIBLE"}
          </span>
          <span className="text-xs font-mono px-1.5 py-0.5 rounded"
                style={{ backgroundColor: tierColor + "20", color: tierColor }}>
            {tierLabel}
          </span>
          {memory.sensitivity === "sensitive" && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-red-900/30 text-red-400">SENSITIVE</span>
          )}
        </div>
        {memory.type !== "pin" && memory.tier !== "archive" && (
          <button onClick={() => onRecall(memory.id)}
                  className="text-xs px-2 py-1 rounded font-medium transition-colors hover:opacity-80"
                  style={{ backgroundColor: COLORS.accent + "20", color: COLORS.accent }}>
            Recall
          </button>
        )}
      </div>
      <p className="text-sm mb-2 font-mono" style={{ color: COLORS.textPrimary, opacity: Math.max(memory.strength, 0.3) }}>
        {memory.content}
      </p>
      <StrengthBar strength={memory.strength} />
      <div className="flex justify-between mt-1.5">
        <span className="text-xs" style={{ color: COLORS.textSecondary }}>
          Strength: {(memory.strength * 100).toFixed(1)}%
        </span>
        <span className="text-xs" style={{ color: COLORS.textSecondary }}>
          Recalls: {memory.accessCount}
        </span>
      </div>
    </div>
  );
}

function DecayCurveChart({ memories, tick }) {
  const width = 320;
  const height = 160;
  const padding = { top: 10, right: 10, bottom: 25, left: 35 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const visibleMemories = memories.filter(m => m.type !== "pin" && m.tier !== "archive").slice(0, 4);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      <rect x={padding.left} y={padding.top} width={chartW} height={chartH} fill={COLORS.surface} rx="4" />
      {/* threshold lines */}
      <line x1={padding.left} y1={padding.top + chartH * (1 - THETA1)} x2={padding.left + chartW} y2={padding.top + chartH * (1 - THETA1)} stroke="#10B98140" strokeDasharray="4,4" />
      <text x={padding.left - 3} y={padding.top + chartH * (1 - THETA1) + 3} fill="#10B981" fontSize="8" textAnchor="end">{"\u03B8\u2081"}</text>
      <line x1={padding.left} y1={padding.top + chartH * (1 - THETA2)} x2={padding.left + chartW} y2={padding.top + chartH * (1 - THETA2)} stroke="#F59E0B40" strokeDasharray="4,4" />
      <text x={padding.left - 3} y={padding.top + chartH * (1 - THETA2) + 3} fill="#F59E0B" fontSize="8" textAnchor="end">{"\u03B8\u2082"}</text>

      {/* Y-axis labels */}
      {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
        <text key={v} x={padding.left - 3} y={padding.top + chartH * (1 - v) + 3} fill={COLORS.textSecondary} fontSize="7" textAnchor="end">{(v * 100).toFixed(0)}%</text>
      ))}

      {/* decay curves */}
      {visibleMemories.map((mem, mi) => {
        const colors = ["#38BDF8", "#A78BFA", "#F472B6", "#34D399"];
        const points = [];
        for (let t = Math.max(0, tick - 30); t <= tick; t++) {
          const s = computeDecay({ ...mem, strength: 1 }, t);
          const x = padding.left + ((t - Math.max(0, tick - 30)) / 30) * chartW;
          const y = padding.top + chartH * (1 - Math.min(s, 1));
          points.push(`${x},${y}`);
        }
        return (
          <g key={mem.id}>
            <polyline points={points.join(" ")} fill="none" stroke={colors[mi % 4]} strokeWidth="1.5" opacity="0.8" />
            <circle cx={parseFloat(points[points.length - 1].split(",")[0])} cy={parseFloat(points[points.length - 1].split(",")[1])} r="3" fill={colors[mi % 4]} />
          </g>
        );
      })}

      <text x={padding.left + chartW / 2} y={height - 3} fill={COLORS.textSecondary} fontSize="8" textAnchor="middle">Time (ticks)</text>
    </svg>
  );
}

function TierStats({ memories }) {
  const active = memories.filter(m => m.tier === "active");
  const warm = memories.filter(m => m.tier === "warm");
  const archive = memories.filter(m => m.tier === "archive");

  const tiers = [
    { name: "Active", count: active.length, color: COLORS.active, desc: "In context window" },
    { name: "Warm", count: warm.length, color: COLORS.warm, desc: "Vector DB" },
    { name: "Archive", count: archive.length, color: COLORS.archive, desc: "Audit trail" },
  ];

  return (
    <div className="flex gap-2">
      {tiers.map(t => (
        <div key={t.name} className="flex-1 rounded-lg p-2 text-center" style={{ backgroundColor: t.color + "15", border: `1px solid ${t.color}30` }}>
          <div className="text-xl font-bold" style={{ color: t.color }}>{t.count}</div>
          <div className="text-xs font-medium" style={{ color: t.color }}>{t.name}</div>
          <div className="text-xs mt-0.5" style={{ color: COLORS.textSecondary }}>{t.desc}</div>
        </div>
      ))}
    </div>
  );
}

function EventLog({ events }) {
  const logRef = useRef(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events]);

  return (
    <div ref={logRef} className="rounded-lg p-3 overflow-y-auto" style={{ backgroundColor: COLORS.card, maxHeight: "200px" }}>
      <div className="text-xs font-bold mb-2" style={{ color: COLORS.textSecondary }}>EVENT LOG</div>
      {events.length === 0 && <div className="text-xs italic" style={{ color: COLORS.textSecondary }}>Press Play to start simulation...</div>}
      {events.slice(-20).map((e, i) => (
        <div key={i} className="text-xs font-mono py-0.5 border-b" style={{ color: e.color || COLORS.textSecondary, borderColor: COLORS.surface }}>
          <span style={{ color: COLORS.textSecondary }}>[t={e.tick}]</span> {e.message}
        </div>
      ))}
    </div>
  );
}

/* ── Formula display ────────────────────────────────────────── */

function FormulaCard() {
  return (
    <div className="rounded-lg p-4 mb-5" style={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.surface}` }}>
      <div className="text-xs font-bold mb-2" style={{ color: COLORS.textSecondary }}>EBBINGHAUS DECAY MODEL</div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs font-mono" style={{ color: COLORS.textPrimary }}>
        <span>R(t) = exp(-(dt/S)^{"\u03B2"})</span>
        <span>S = S{"\u2080"} * (1 + {"\u03B1"} * ln(1 + n))</span>
        <span>{"\u03B2"}<sub>active</sub>=1.2 &nbsp; {"\u03B2"}<sub>warm</sub>=0.8</span>
        <span>{"\u03B8\u2081"}=0.6 &nbsp; {"\u03B8\u2082"}=0.15 &nbsp; {"\u03C1"}=0.05</span>
      </div>
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────── */

export default function App() {
  const [memories, setMemories] = useState(INITIAL_MEMORIES);
  const [tick, setTick] = useState(8);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [events, setEvents] = useState([]);
  const [nextId, setNextId] = useState(9);

  /* timer */
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      setTick(prev => prev + 1);
    }, 1000 / speed);
    return () => clearInterval(interval);
  }, [running, speed]);

  /* decay + migration */
  useEffect(() => {
    if (tick <= 8) return;
    setMemories(prev => {
      return prev.map(mem => {
        const newStrength = computeDecay(mem, tick);
        const newTier = mem.type === "pin" ? "active" : getTier(newStrength);
        const oldTier = mem.tier;

        if (newTier !== oldTier && mem.type !== "pin") {
          const msg = `"${mem.content.slice(0, 25)}..." migrated: ${oldTier.toUpperCase()} -> ${newTier.toUpperCase()}`;
          const color = newTier === "archive" ? COLORS.archive : COLORS.warm;
          setEvents(ev => [...ev, { tick, message: msg, color }]);
        }

        return {
          ...mem,
          strength: newStrength,
          tier: newTier,
          decayPath: [...mem.decayPath, { tick, strength: newStrength }],
        };
      });
    });
  }, [tick]);

  /* recall handler */
  const handleRecall = useCallback((id) => {
    setMemories(prev => prev.map(mem => {
      if (mem.id !== id) return mem;
      const newAccess = mem.accessCount + 1;
      const S0 = mem.importance * 10;
      const S = S0 * (1 + 0.3 * Math.log(1 + newAccess));
      const newStrength = Math.min(mem.strength + 0.25, 1.0);
      const newTier = getTier(newStrength);

      setEvents(ev => [...ev, {
        tick,
        message: `Recalled "${mem.content.slice(0, 25)}..." — strength ${(mem.strength * 100).toFixed(0)}% -> ${(newStrength * 100).toFixed(0)}% (S: ${S.toFixed(1)})`,
        color: COLORS.accent,
      }]);

      return { ...mem, strength: newStrength, accessCount: newAccess, lastAccessed: tick, tier: newTier };
    }));
  }, [tick]);

  /* add random memory */
  const handleAddMemory = useCallback(() => {
    const samples = [
      { content: "Meeting at 3pm tomorrow", type: "recoverable", importance: 0.6 },
      { content: "User ID: u_8f2k9x", type: "pin", importance: 1.0 },
      { content: "OK, got it, thanks", type: "irreversible", importance: 0.1 },
      { content: "Use PostgreSQL, not MySQL", type: "recoverable", importance: 0.8 },
      { content: "Patient diagnosis: confidential", type: "recoverable", importance: 0.9, sensitivity: "sensitive" },
      { content: "Refactor auth module by Friday", type: "recoverable", importance: 0.7 },
      { content: "That makes sense", type: "irreversible", importance: 0.15 },
      { content: "SSH key: ssh-ed25519 AAAA...", type: "pin", importance: 1.0, sensitivity: "sensitive" },
    ];
    const sample = samples[Math.floor(Math.random() * samples.length)];
    const newMem = {
      id: nextId,
      ...sample,
      strength: 0.95,
      accessCount: 0,
      createdAt: tick,
      lastAccessed: tick,
      tier: "active",
      decayPath: [],
    };
    setNextId(prev => prev + 1);
    setMemories(prev => [...prev, newMem]);
    setEvents(ev => [...ev, { tick, message: `Stored: "${sample.content}" [${sample.type.toUpperCase()}]`, color: COLORS.active }]);
  }, [tick, nextId]);

  /* reset */
  const handleReset = useCallback(() => {
    setMemories(INITIAL_MEMORIES);
    setTick(8);
    setRunning(false);
    setEvents([]);
    setNextId(9);
  }, []);

  const activeMems = memories.filter(m => m.tier === "active");
  const warmMems = memories.filter(m => m.tier === "warm");
  const archiveMems = memories.filter(m => m.tier === "archive");

  return (
    <div className="min-h-screen p-4 md:p-8" style={{ backgroundColor: COLORS.bg, color: COLORS.textPrimary }}>
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-1" style={{ color: COLORS.accent }}>EbbingContext</h1>
          <p className="text-sm mb-1" style={{ color: COLORS.textSecondary }}>
            Ebbinghaus Forgetting Curve-Based Context Management Engine for LLM Agents
          </p>
          <p className="text-xs" style={{ color: COLORS.textSecondary }}>
            Interactive simulation — watch memories decay, strengthen on recall, and migrate across storage tiers
          </p>
        </div>

        {/* Formula */}
        <FormulaCard />

        {/* Controls */}
        <div className="flex items-center justify-center gap-3 mb-5 flex-wrap">
          <button onClick={() => setRunning(!running)}
                  className="px-4 py-2 rounded-lg font-medium text-sm transition-colors"
                  style={{ backgroundColor: running ? "#EF444420" : COLORS.accent + "20", color: running ? "#EF4444" : COLORS.accent }}>
            {running ? "Pause" : "Play"}
          </button>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: COLORS.textSecondary }}>Speed:</span>
            {[1, 2, 5].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                      className="px-2 py-1 rounded text-xs font-mono"
                      style={{ backgroundColor: speed === s ? COLORS.accent + "30" : COLORS.surface, color: speed === s ? COLORS.accent : COLORS.textSecondary }}>
                {s}x
              </button>
            ))}
          </div>
          <button onClick={handleAddMemory} className="px-3 py-2 rounded-lg text-sm font-medium" style={{ backgroundColor: "#10B98120", color: "#10B981" }}>
            + Add Memory
          </button>
          <button onClick={handleReset} className="px-3 py-2 rounded-lg text-sm font-medium" style={{ backgroundColor: COLORS.surface, color: COLORS.textSecondary }}>
            Reset
          </button>
          <div className="px-3 py-1 rounded-lg font-mono text-sm" style={{ backgroundColor: COLORS.surface, color: COLORS.accent }}>
            t = {tick}
          </div>
        </div>

        {/* Stats + Chart */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
          <div>
            <div className="text-xs font-bold mb-2" style={{ color: COLORS.textSecondary }}>STORAGE TIERS</div>
            <TierStats memories={memories} />
          </div>
          <div>
            <div className="text-xs font-bold mb-2" style={{ color: COLORS.textSecondary }}>DECAY CURVES (non-pin, non-archived)</div>
            <div className="rounded-lg p-2" style={{ backgroundColor: COLORS.card }}>
              <DecayCurveChart memories={memories} tick={tick} />
            </div>
          </div>
        </div>

        {/* Memory Tiers */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.active }} />
              <span className="text-xs font-bold" style={{ color: COLORS.active }}>ACTIVE LAYER</span>
              <span className="text-xs" style={{ color: COLORS.textSecondary }}>strength &gt; {(THETA1 * 100).toFixed(0)}%</span>
            </div>
            <div className="space-y-2">
              {activeMems.length === 0 && <div className="text-xs italic p-3 rounded-lg" style={{ backgroundColor: COLORS.card, color: COLORS.textSecondary }}>No active memories</div>}
              {activeMems.map(m => <MemoryCard key={m.id} memory={m} onRecall={handleRecall} />)}
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.warm }} />
              <span className="text-xs font-bold" style={{ color: COLORS.warm }}>WARM LAYER</span>
              <span className="text-xs" style={{ color: COLORS.textSecondary }}>{(THETA2 * 100).toFixed(0)}% &lt; strength &le; {(THETA1 * 100).toFixed(0)}%</span>
            </div>
            <div className="space-y-2">
              {warmMems.length === 0 && <div className="text-xs italic p-3 rounded-lg" style={{ backgroundColor: COLORS.card, color: COLORS.textSecondary }}>No warm memories</div>}
              {warmMems.map(m => <MemoryCard key={m.id} memory={m} onRecall={handleRecall} />)}
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS.archive }} />
              <span className="text-xs font-bold" style={{ color: COLORS.archive }}>ARCHIVE LAYER</span>
              <span className="text-xs" style={{ color: COLORS.textSecondary }}>strength &le; {(THETA2 * 100).toFixed(0)}%</span>
            </div>
            <div className="space-y-2">
              {archiveMems.length === 0 && <div className="text-xs italic p-3 rounded-lg" style={{ backgroundColor: COLORS.card, color: COLORS.textSecondary }}>No archived memories</div>}
              {archiveMems.map(m => <MemoryCard key={m.id} memory={m} onRecall={handleRecall} />)}
            </div>
          </div>
        </div>

        {/* Event Log */}
        <EventLog events={events} />

        {/* How it works */}
        <div className="mt-4 p-3 rounded-lg text-xs" style={{ backgroundColor: COLORS.card, color: COLORS.textSecondary }}>
          <span className="font-bold" style={{ color: COLORS.textPrimary }}>How it works: </span>
          Press <b>Play</b> to start time. Watch memories decay based on their importance and type.
          Click <span style={{ color: COLORS.accent }}>Recall</span> to simulate a retrieval — the memory strengthens and its decay curve resets.
          <b> Pin</b> memories never decay. <b>Irreversible</b> memories decay without recovery.
          Memories migrate <span style={{ color: COLORS.active }}>Active</span> {"->"} <span style={{ color: COLORS.warm }}>Warm</span> {"->"} <span style={{ color: COLORS.archive }}>Archive</span> as strength drops below thresholds.
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-xs" style={{ color: COLORS.textSecondary }}>
          <a href="https://github.com/liainseptember-boop/ebbingcontext"
             target="_blank" rel="noopener noreferrer"
             className="hover:underline" style={{ color: COLORS.accent }}>
            GitHub
          </a>
          <span className="mx-2">|</span>
          <a href="https://github.com/liainseptember-boop/ebbingcontext/blob/main/ARCHITECTURE.md"
             target="_blank" rel="noopener noreferrer"
             className="hover:underline" style={{ color: COLORS.accent }}>
            Architecture
          </a>
          <span className="mx-2">|</span>
          <span>MIT License</span>
        </div>
      </div>
    </div>
  );
}
