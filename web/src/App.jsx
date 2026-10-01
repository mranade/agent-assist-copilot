import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import HistoryView, { SentimentTag } from "./HistoryView.jsx";
import InsightsView from "./InsightsView.jsx";
import CardBody, { FeedbackButtons } from "./Card.jsx";

const VIEWS = [
  { id: "live", label: "Live call" },
  { id: "history", label: "Call history" },
  { id: "insights", label: "Insights" },
];

// The view lives in the URL hash (#history, #insights) so it survives a reload.
function viewFromHash() {
  const id = window.location.hash.slice(1);
  return VIEWS.some((v) => v.id === id) ? id : "live";
}

const MODES = [
  { id: "playback", label: "Sample call" },
  { id: "mic", label: "Live mic" },
  { id: "type", label: "Type" },
];

export default function App() {
  const [status, setStatus] = useState(null);
  const [calls, setCalls] = useState([]);
  const [mode, setMode] = useState("playback");
  const [callId, setCallId] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [playAudio, setPlayAudio] = useState(true);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [pinned, setPinned] = useState(null); // index of a customer turn whose card is shown
  const [view, setViewState] = useState(viewFromHash); // "live" | "history" | "insights"
  const [historyId, setHistoryId] = useState(null);
  const [historyTurn, setHistoryTurn] = useState(null); // turn to scroll to when a call opens
  const [historyKey, setHistoryKey] = useState(0); // bump to reload the history list
  const playingRef = useRef(false);
  const ended = summary !== null; // a saved call is read-only until "New call"
  const locked = busy || ended;

  useEffect(() => {
    const onHash = () => setViewState(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function setView(next) {
    setViewState(next);
    window.history.replaceState(null, "", next === "live" ? window.location.pathname : `#${next}`);
  }

  useEffect(() => {
    Promise.all([api.status(), api.calls()])
      .then(([s, c]) => {
        setStatus(s);
        setCalls(c);
        setCallId(c[0]?.id ?? "");
      })
      .catch((e) => setError(`Can't reach the server: ${e.message}`));
  }, []);

  // Whisper loads in the background on first start (it may download the model),
  // so keep checking until it settles on ready, failed, or off.
  useEffect(() => {
    if (status?.stt_state !== "loading") return;
    const timer = setInterval(() => api.status().then(setStatus).catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, [status?.stt_state]);

  const call = calls.find((c) => c.id === callId);
  const hasCard = (t) => t.outcome === "suggested" || t.outcome === "escalated";
  const cards = turns.filter(hasCard);
  const pinnedCard = pinned !== null ? turns[pinned] : null;
  const shown = pinnedCard ?? withLaterProgress(turns);

  async function ensureSession() {
    if (sessionId) return sessionId;
    const { session_id } = await api.newSession(mode === "playback" ? callId : "live");
    setSessionId(session_id);
    return session_id;
  }

  function resetCall(nextCallId = callId) {
    playingRef.current = false;
    setPlaying(false);
    setSessionId(null);
    setTurns([]);
    setCursor(0);
    setSummary(null);
    setError(null);
    setPinned(null);
    setCallId(nextCallId);
  }

  function addTurn(result) {
    setPinned(null); // a new turn always brings the view back to live
    setTurns((prev) => [...prev, result]);
  }

  function pinTurn(index) {
    if (ended) return;
    setPinned(index === pinned ? null : index);
  }

  function openHistory(id = null, turn = null) {
    setHistoryId(id);
    setHistoryTurn(turn);
    setView("history");
  }

  // The agent rates a card. During the call the rating goes to the session;
  // once the call is saved, it updates the saved record.
  async function rateCard(index, value) {
    const previous = turns.find((t) => t.index === index)?.feedback ?? null;
    const setFeedback = (v) =>
      setTurns((prev) => prev.map((t) => (t.index === index ? { ...t, feedback: v } : t)));
    setFeedback(value);
    try {
      if (sessionId) await api.rateSessionTurn(sessionId, index, value);
      else if (summary) await api.rateSavedTurn(summary.id, index, value);
      setHistoryKey((k) => k + 1);
    } catch (e) {
      setFeedback(previous);
      setError(`Couldn't save the rating: ${e.message}`);
    }
  }

  async function playTurn(sid, index) {
    const turn = call.turns[index];
    const audioDone = playAudio
      ? new Promise((resolve) => {
          const audio = new Audio(turn.audio_url);
          audio.onended = resolve;
          audio.onerror = resolve;
          audio.play().catch(resolve);
        })
      : Promise.resolve();
    const [result] = await Promise.all([api.playbackTurn(sid, call.id, index), audioDone]);
    addTurn(result);
    setCursor(index + 1);
  }

  async function play() {
    if (!call) return;
    setError(null);
    playingRef.current = true;
    setPlaying(true);
    try {
      const sid = await ensureSession();
      for (let i = cursor; i < call.turns.length && playingRef.current; i++) {
        await playTurn(sid, i);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      playingRef.current = false;
      setPlaying(false);
    }
  }

  async function step() {
    if (!call || cursor >= call.turns.length) return;
    setBusy(true);
    setError(null);
    try {
      await playTurn(await ensureSession(), cursor);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendText(text) {
    setBusy(true);
    setError(null);
    try {
      addTurn(await api.textTurn(await ensureSession(), text));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendAudio(blob) {
    setBusy(true);
    setError(null);
    try {
      addTurn(await api.audioTurn(await ensureSession(), blob));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function endCall() {
    if (!sessionId) return;
    playingRef.current = false;
    setBusy(true);
    try {
      const record = await api.endCall(sessionId);
      setSummary(record);
      setPinned(null);
      setSessionId(null);
      setHistoryKey((k) => k + 1);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-name">Agent Assist</div>
            <div className="brand-sub">Relay Fiber support desk · fictional demo</div>
          </div>
        </div>
        <nav className="nav" aria-label="Views">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className={view === v.id ? "nav-item active" : "nav-item"}
              aria-current={view === v.id ? "page" : undefined}
              onClick={() => setView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </nav>
        {status && (
          <div className="chips">
            <SpeechChip status={status} />
            <Chip tone="neutral" label={`Suggestions: ${status.generator}`} />
            <Chip tone="neutral" label={`${status.kb_articles} KB articles`} />
          </div>
        )}
      </header>

      {error && <div className="banner">{error}</div>}

      {view === "insights" ? (
        <InsightsView onOpenCall={openHistory} callInProgress={Boolean(sessionId) && turns.length > 0} />
      ) : view === "history" ? (
        <HistoryView
          selectedId={historyId}
          focusTurn={historyTurn}
          onSelect={(id) => {
            setHistoryId(id);
            setHistoryTurn(null);
          }}
          refreshKey={historyKey}
        />
      ) : (
      <main className="layout">
        <section className="panel controls">
          <h2>Call input</h2>
          <div className="tabs" role="tablist">
            {MODES.map((m) => (
              <button
                key={m.id}
                role="tab"
                aria-selected={mode === m.id}
                className={mode === m.id ? "tab active" : "tab"}
                onClick={() => setMode(m.id)}
                disabled={playing}
              >
                {m.label}
              </button>
            ))}
          </div>

          {mode === "playback" && (
            <PlaybackControls
              calls={calls}
              call={call}
              cursor={cursor}
              playing={playing}
              busy={locked}
              playAudio={playAudio}
              onPlayAudio={setPlayAudio}
              onSelect={(id) => resetCall(id)}
              onPlay={play}
              onPause={() => (playingRef.current = false)}
              onStep={step}
            />
          )}
          {mode === "mic" && (
            <MicControls speech={status} busy={locked} onClip={sendAudio} />
          )}
          {mode === "type" && <TypeControls busy={locked} onSend={sendText} />}

          <div className="call-actions">
            <button className="btn" onClick={endCall} disabled={!sessionId || locked || playing}>
              End call &amp; save
            </button>
            <button className="btn ghost" onClick={() => resetCall()} disabled={playing}>
              New call
            </button>
          </div>
        </section>

        <section className="panel transcript">
          <h2>Live transcript</h2>
          <Transcript
            turns={turns}
            pending={playing || busy}
            pinned={pinned}
            onPick={(i) => hasCard(turns[i]) && pinTurn(i)}
          />
        </section>

        <section className="panel assist">
          <h2>Assist</h2>
          {pinnedCard && !summary && (
            <button className="link" onClick={() => setPinned(null)}>
              ← Back to the latest suggestion
            </button>
          )}
          {summary ? (
            <SummaryCard record={summary} onOpen={() => openHistory(summary.id)} />
          ) : shown ? (
            <SuggestionCard card={shown} onRate={rateCard} />
          ) : (
            <p className="empty">
              Suggestions appear here as the customer talks. Small talk is ignored on purpose. Click
              any customer line with a dot to see the card it produced.
            </p>
          )}
          {cards.length > 1 && !summary && !pinnedCard && (
            <details className="history">
              <summary>Earlier suggestions ({cards.length - 1})</summary>
              {cards
                .slice(0, -1)
                .reverse()
                .map((c, i) => (
                  <SuggestionCard key={i} card={c} compact onRate={rateCard} />
                ))}
            </details>
          )}
        </section>
      </main>
      )}
    </div>
  );
}

function SpeechChip({ status }) {
  const model = status.stt_model;
  const chips = {
    ready: { tone: "ok", label: `Whisper ${model} ready` },
    loading: { tone: "neutral", label: `Loading Whisper ${model}…` },
    failed: { tone: "warn", label: "Whisper failed to load · script text" },
    off: { tone: "warn", label: "Speech-to-text off · script text" },
  };
  const chip = chips[status.stt_state] ?? chips.off;
  return (
    <span className={`chip ${chip.tone}`} title={status.stt_error || ""}>
      {chip.label}
    </span>
  );
}

function Chip({ tone, label }) {
  return <span className={`chip ${tone}`}>{label}</span>;
}

function PlaybackControls(props) {
  const { calls, call, cursor, playing, busy, playAudio } = props;
  const total = call?.turns.length ?? 0;
  const finished = cursor >= total;
  return (
    <div className="stack">
      <label className="field">
        <span>Sample call</span>
        <select value={call?.id ?? ""} onChange={(e) => props.onSelect(e.target.value)} disabled={playing}>
          {calls.map((c) => (
            <option key={c.id} value={c.id}>
              {c.title}
            </option>
          ))}
        </select>
      </label>
      {call && <p className="hint">{call.description}</p>}
      <div className="row">
        {playing ? (
          <button className="btn" onClick={props.onPause}>
            Pause after this turn
          </button>
        ) : (
          <button className="btn primary" onClick={props.onPlay} disabled={busy || finished}>
            {finished ? "Call finished" : cursor === 0 ? "Play call" : "Resume"}
          </button>
        )}
        <button className="btn ghost" onClick={props.onStep} disabled={playing || busy || finished}>
          Next turn
        </button>
      </div>
      <label className="check">
        <input type="checkbox" checked={playAudio} onChange={(e) => props.onPlayAudio(e.target.checked)} />
        Play audio (turn off to step through quickly)
      </label>
      <Progress value={cursor} max={total} />
    </div>
  );
}

function Progress({ value, max }) {
  return (
    <div className="progress" aria-label={`Turn ${value} of ${max}`}>
      <div className="progress-bar" style={{ width: max ? `${(value / max) * 100}%` : 0 }} />
      <span>
        Turn {value} of {max}
      </span>
    </div>
  );
}

function MicControls({ speech, busy, onClip }) {
  const [recording, setRecording] = useState(false);
  const [micError, setMicError] = useState(null);
  const recorderRef = useRef(null);
  const holdingRef = useRef(false); // is the button still pressed?

  async function start() {
    holdingRef.current = true;
    setMicError(null);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      holdingRef.current = false;
      setMicError(`Couldn't open the microphone: ${e.message}. Check the browser's mic permission for this page.`);
      return;
    }
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      if (chunks.length) onClip(new Blob(chunks, { type: recorder.mimeType }));
    };
    recorder.start();
    recorderRef.current = recorder;
    setRecording(true);
    // Opening the mic is async: if the button was released meanwhile, stop now.
    if (!holdingRef.current) stop();
  }

  function stop() {
    holdingRef.current = false;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    recorderRef.current = null;
    setRecording(false);
  }

  const state = speech?.stt_state ?? "off";
  if (state !== "ready") {
    const messages = {
      loading: (
        <>
          Whisper is loading. The first run downloads the <code>{speech?.stt_model}</code> model (about
          150 MB), so this can take a minute. Sample calls and typing work meanwhile.
        </>
      ),
      failed: (
        <>
          Whisper failed to load{speech?.stt_error ? `: ${speech.stt_error}` : ""}. Check your internet
          connection and that ffmpeg is installed, then restart. Sample calls and typing still work.
        </>
      ),
      off: (
        <>
          Live mic needs local Whisper. Install it with <code>pip install -e ".[speech]"</code> and restart
          the server. Sample-call playback and typing work without it.
        </>
      ),
    };
    return <p className="hint">{messages[state] ?? messages.off}</p>;
  }
  return (
    <div className="stack">
      <p className="hint">You play the customer. Hold the button while you speak, release to send.</p>
      <button
        className={recording ? "btn record live" : "btn record"}
        onPointerDown={start}
        onPointerUp={stop}
        onPointerLeave={() => holdingRef.current && stop()}
        disabled={busy}
      >
        {recording ? "Listening… release to send" : busy ? "Transcribing…" : "Hold to talk"}
      </button>
      {micError && <p className="hint mic-error">{micError}</p>}
    </div>
  );
}

function TypeControls({ busy, onSend }) {
  const [text, setText] = useState("");
  function submit(e) {
    e.preventDefault();
    if (!text.trim()) return;
    onSend(text.trim());
    setText("");
  }
  return (
    <form className="stack" onSubmit={submit}>
      <p className="hint">Backup mode: type what the customer says.</p>
      <textarea
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. My internet keeps dropping every evening"
      />
      <button className="btn primary" disabled={busy || !text.trim()}>
        Send as customer
      </button>
    </form>
  );
}

function Transcript({ turns, pending, pinned, onPick }) {
  const endRef = useRef(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [turns.length]);
  if (!turns.length) return <p className="empty">Start a sample call, use the mic, or type a line.</p>;
  return (
    <ol className="turns">
      {turns.map((t, i) => (
        <li
          key={i}
          className={`turn ${t.speaker} ${t.outcome === "suggested" || t.outcome === "escalated" ? "clickable" : ""} ${pinned === i ? "pinned" : ""}`}
          onClick={() => onPick(i)}
        >
          <div className="turn-meta">
            <span className="speaker">{t.speaker === "agent" ? "Agent" : "Customer"}</span>
            <span className="source">{t.transcript_source}</span>
            {t.outcome && <span className={`dot ${t.outcome}`} title={t.outcome} />}
            {t.steps_marked?.map((m) => (
              <span key={m.step} className="step-chip" title={m.text}>
                ✓ Step {m.step}
              </span>
            ))}
          </div>
          <p>{highlightRedactions(t.text)}</p>
        </li>
      ))}
      {pending && <li className="turn pending">…</li>}
      <li ref={endRef} aria-hidden="true" />
    </ol>
  );
}

function highlightRedactions(text) {
  return text.split(/(\[REDACTED\])/g).map((part, i) =>
    part === "[REDACTED]" ? (
      <mark key={i} className="redacted">
        redacted
      </mark>
    ) : (
      part
    )
  );
}

// The latest card, updated with anything agent turns did after it: they
// don't get cards of their own, but they tick checklist steps.
function withLaterProgress(turns) {
  let index = -1;
  turns.forEach((t, i) => {
    if (t.outcome === "suggested" || t.outcome === "escalated") index = i;
  });
  if (index < 0) return null;
  const card = turns[index];
  const later = turns.slice(index + 1);
  const laterTrace = later.flatMap((t) =>
    (t.trace ?? []).filter((x) => x.startsWith("track:")).map((x) => `${t.speaker} turn · ${x}`)
  );
  const latestPlan = [...later].reverse().find((t) => t.plan)?.plan;
  const plan = card.suggestion.plan;
  if (latestPlan && plan && latestPlan.kb_id === plan.kb_id) {
    return { ...card, suggestion: { ...card.suggestion, plan: latestPlan }, laterTrace };
  }
  return { ...card, laterTrace };
}

function SuggestionCard({ card, compact = false, onRate }) {
  const s = card.suggestion;
  const escalated = card.outcome === "escalated";
  return (
    <article className={`card ${escalated ? "escalated" : ""} ${compact ? "compact" : ""}`}>
      <div className="card-head">
        <span className={`badge ${escalated ? "warn" : "ok"}`}>{escalated ? "No KB match" : "Suggested"}</span>
        <h3>{s.headline}</h3>
      </div>
      {!compact && <p className="said">Customer: “{card.text}”</p>}
      <CardBody suggestion={s} escalated={escalated} compact={compact} />
      <div className="articles">
        {s.articles.map((a) => (
          <span key={a.kb_id} className="article">
            {a.kb_id} · {a.title}
          </span>
        ))}
      </div>
      {onRate && <FeedbackButtons value={card.feedback ?? null} onChange={(v) => onRate(card.index, v)} />}
      {!compact && (
        <details className="why">
          <summary>Why this card</summary>
          <dl className="metrics">
            <div>
              <dt>Top match</dt>
              <dd>{card.top_score?.toFixed(2)}</dd>
            </div>
            <div>
              <dt>KB coverage</dt>
              <dd>{Math.round((card.coverage ?? 0) * 100)}%</dd>
            </div>
            <div>
              <dt>Speech-to-text</dt>
              <dd>{card.transcript_source === "whisper" ? `${card.stt_ms} ms` : card.transcript_source}</dd>
            </div>
            <div>
              <dt>Assist</dt>
              <dd>{card.assist_ms} ms</dd>
            </div>
          </dl>
          <ol className="trace">
            {card.trace.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
          {card.laterTrace?.length > 0 && (
            <>
              <p className="trace-head">Since this card</p>
              <ol className="trace">
                {card.laterTrace.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </>
          )}
        </details>
      )}
    </article>
  );
}

function SummaryCard({ record, onOpen }) {
  const { summary, sentiment } = record;
  return (
    <article className="card summary">
      <div className="card-head">
        <span className="badge neutral">Saved to call history</span>
        <h3>Call summary</h3>
      </div>
      <div className="summary-sentiment">
        <SentimentTag label={sentiment.label} score={sentiment.score} large />
        <span className="hint">Customer sentiment {sentiment.trend} over the call</span>
      </div>
      <p className="note">{summary.note}</p>
      <p className="hint">
        Generated by: {summary.generated_by}. Account numbers and PINs are redacted before summarizing.
      </p>
      <button className="btn primary" onClick={onOpen}>
        Open in call history
      </button>
    </article>
  );
}
