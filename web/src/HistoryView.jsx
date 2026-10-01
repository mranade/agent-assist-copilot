import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import SentimentChart, { formatScore } from "./SentimentChart.jsx";
import CardBody, { FEEDBACK_LABELS, FeedbackButtons } from "./Card.jsx";

export default function HistoryView({ selectedId, onSelect, refreshKey, focusTurn = null }) {
  const [calls, setCalls] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .history()
      .then((list) => {
        setCalls(list);
        if (!selectedId && list.length) onSelect(list[0].id);
      })
      .catch((e) => setError(e.message));
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedId) return;
    setDetail(null);
    api.historyCall(selectedId).then(setDetail).catch((e) => setError(e.message));
  }, [selectedId]);

  // Ratings can change after the call: update the saved record, and undo on failure.
  async function rate(index, value) {
    const callId = detail.id;
    const previous = detail.turns[index].feedback ?? null;
    const setFeedback = (v) =>
      setDetail((d) =>
        d && d.id === callId ? { ...d, turns: d.turns.map((t, i) => (i === index ? { ...t, feedback: v } : t)) } : d
      );
    setFeedback(value);
    try {
      await api.rateSavedTurn(callId, index, value);
    } catch (e) {
      setFeedback(previous);
      setError(`Couldn't save the rating: ${e.message}`);
    }
  }

  if (error) return <div className="banner">{error}</div>;
  if (!calls) return <p className="empty pad">Loading call history…</p>;

  return (
    <main className="history-layout">
      <section className="panel call-list" aria-label="Past calls">
        <h2>Past calls ({calls.length})</h2>
        <ul>
          {calls.map((c) => (
            <li key={c.id}>
              <button
                className={c.id === selectedId ? "call-item active" : "call-item"}
                onClick={() => onSelect(c.id)}
              >
                <span className="call-title">{c.title}</span>
                <span className="call-meta">
                  {[c.agent, formatDate(c.started_at), formatDuration(c.duration_s)].filter(Boolean).join(" · ")}
                </span>
                <span className="call-tags">
                  <SentimentTag label={c.sentiment.label} score={c.sentiment.score} />
                  {c.open_items > 0 && (
                    <span className="tag warn">
                      {c.open_items} open item{c.open_items > 1 ? "s" : ""}
                    </span>
                  )}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
      {detail ? (
        <CallDetail key={detail.id} call={detail} focusTurn={focusTurn} onRate={rate} />
      ) : (
        <section className="panel"><p className="empty">Loading…</p></section>
      )}
    </main>
  );
}

function CallDetail({ call, focusTurn, onRate }) {
  const turnRefs = useRef({});
  const [focus, setFocus] = useState(focusTurn);
  const s = call.sentiment;
  const customerTurns = call.turns.filter((t) => t.speaker === "customer");
  const suggested = customerTurns.filter((t) => t.outcome === "suggested").length;
  const escalated = customerTurns.filter((t) => t.outcome === "escalated").length;

  function jumpTo(index) {
    setFocus(index);
    turnRefs.current[index]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // Opened from Insights: scroll to the line that was linked.
  useEffect(() => {
    if (focusTurn !== null && focusTurn !== undefined) jumpTo(focusTurn);
  }, []);

  return (
    <section className="panel call-detail" aria-label="Call details">
      <header className="detail-head">
        <h2 className="detail-title">{call.title}</h2>
        <p className="call-meta">
          {[call.agent && `Agent ${call.agent}`, formatDate(call.started_at), formatDuration(call.duration_s),
            call.source.startsWith("call-") ? "sample call" : call.source === "live" ? "live call" : "past call"]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </header>

      <div className="stats">
        <Stat label="Overall sentiment">
          <SentimentTag label={s.label} score={s.score} large />
        </Stat>
        <Stat label="Over the call">
          <span className="stat-value">{s.trend}</span>
          <span className="stat-sub">
            {formatScore(s.start)} → {formatScore(s.end)}
          </span>
        </Stat>
        <Stat label="Assist cards">
          <span className="stat-value">{suggested} suggested</span>
          <span className="stat-sub">{escalated} with no KB match</span>
        </Stat>
        <Stat label="Open items">
          <span className="stat-value">{call.summary.open_items.length}</span>
          <span className="stat-sub">{call.summary.open_items.length ? "needs follow-up" : "none"}</span>
        </Stat>
      </div>

      <div className="detail-grid">
        <article className="card summary">
          <div className="card-head">
            <span className="badge neutral">Wrap-up note</span>
            <h3>Call summary</h3>
          </div>
          <p className="note">{call.summary.note}</p>
          <p className="hint">Generated by: {call.summary.generated_by}</p>
        </article>

        <article className="card plain">
          <SentimentChart turns={call.turns} onPick={jumpTo} />
          {(s.most_negative || s.most_positive) && (
            <dl className="drivers">
              {s.most_negative && (
                <div>
                  <dt>Lowest point</dt>
                  <dd>“{s.most_negative}”</dd>
                </div>
              )}
              {s.most_positive && (
                <div>
                  <dt>Highest point</dt>
                  <dd>“{s.most_positive}”</dd>
                </div>
              )}
            </dl>
          )}
        </article>
      </div>

      <h2 className="section-label">Transcript</h2>
      <ol className="turns history-turns">
        {call.turns.map((t, i) => (
          <li
            key={i}
            ref={(el) => (turnRefs.current[i] = el)}
            className={`turn ${t.speaker} ${focus === i ? "pinned" : ""}`}
          >
            <div className="turn-meta">
              <span className="speaker">{t.speaker === "agent" ? "Agent" : "Customer"}</span>
              {t.speaker === "customer" && t.sentiment !== null && (
                <span className="score">
                  <i className={`key ${t.sentiment > 0 ? "pos" : t.sentiment < 0 ? "neg" : "zero"}`} />
                  {formatScore(t.sentiment)}
                </span>
              )}
              {t.steps_marked?.map((m) => (
                <span key={m.step} className="step-chip" title={`${m.kb_id} step ${m.step}: ${m.text}`}>
                  ✓ Step {m.step} of {m.of}
                </span>
              ))}
            </div>
            <p>{highlightRedactions(t.text)}</p>
            {t.headline && (
              <details className="turn-card">
                <summary>
                  {t.outcome === "escalated" ? "No KB match" : "Assist card"}: {t.headline}
                  {t.card?.plan && t.outcome !== "escalated" && (
                    <span className="summary-step">
                      {t.card.plan.next === null ? " · all steps done" : ` · step ${t.card.plan.next + 1} of ${t.card.plan.steps.length}`}
                    </span>
                  )}
                  {t.feedback && (
                    <span className={`rated ${t.feedback}`}> · {FEEDBACK_LABELS[t.feedback]}</span>
                  )}
                </summary>
                {t.card ? (
                  <div className="turn-card-body">
                    <CardBody suggestion={t.card} escalated={t.outcome === "escalated"} />
                  </div>
                ) : (
                  <ul className="lines">
                    {t.lines.map((line, j) => (
                      <li key={j}>{line}</li>
                    ))}
                  </ul>
                )}
                <FeedbackButtons value={t.feedback ?? null} onChange={(v) => onRate(i, v)} />
              </details>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Stat({ label, children }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      {children}
    </div>
  );
}

export function SentimentTag({ label, score, large = false }) {
  const kind = label === "positive" ? "pos" : label === "negative" ? "neg" : "zero";
  return (
    <span className={large ? "sentiment large" : "sentiment"}>
      <i className={`key ${kind}`} aria-hidden="true" />
      <span className="sentiment-label">{label[0].toUpperCase() + label.slice(1)}</span>
      <span className="sentiment-score">{formatScore(score)}</span>
    </span>
  );
}

export function highlightRedactions(text) {
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

function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(seconds) {
  if (seconds < 60) return "under a minute";
  return `${Math.round(seconds / 60)} min`;
}
