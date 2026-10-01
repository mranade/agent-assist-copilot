import { useEffect, useState } from "react";
import { api } from "./api.js";
import { formatScore } from "./SentimentChart.jsx";

// Across all saved calls: what the KB couldn't answer, which articles help,
// and which kinds of calls go badly. Everything is computed by the server
// from the call history each time this view opens.
export default function InsightsView({ onOpenCall, callInProgress = false }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.insights().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="banner">{error}</div>;
  if (!data) return <p className="empty pad">Loading insights…</p>;

  const { totals } = data;
  if (!totals.calls) return <p className="empty pad">No saved calls yet. End a call to see insights.</p>;

  return (
    <main className="insights">
      {callInProgress && (
        <p className="notice">
          A call is in progress. Insights count saved calls only, so its cards and ratings appear here after you
          click <strong>End call &amp; save</strong>.
        </p>
      )}
      <div className="stats insight-stats">
        <Tile label="Calls" value={totals.calls} sub={`${totals.customer_turns} customer lines`} />
        <Tile
          label="Got a suggestion"
          value={pct(totals.suggested_pct)}
          sub={`${totals.suggested} of ${totals.customer_turns} lines · ${totals.skipped} small talk`}
        />
        <Tile label="Escalated" value={pct(totals.escalated_pct)} sub={`${totals.escalated} with no KB coverage`} />
        <Tile
          label="Cards rated helpful"
          value={pct(totals.helpful_pct)}
          sub={`${totals.helpful} of ${totals.rated} rated · ${totals.cards} cards shown`}
        />
      </div>

      <section className="panel" aria-labelledby="gaps-head">
        <h2 id="gaps-head">Questions the KB couldn't answer</h2>
        <Gaps gaps={data.gaps} onOpenCall={onOpenCall} />
      </section>

      <div className="insights-grid">
        <section className="panel" aria-labelledby="articles-head">
          <h2 id="articles-head">Article performance</h2>
          <ArticleTable articles={data.articles} />
        </section>
        <section className="panel" aria-labelledby="topics-head">
          <h2 id="topics-head">Sentiment by topic</h2>
          <TopicChart topics={data.sentiment_by_topic} />
        </section>
      </div>
    </main>
  );
}

function Tile({ label, value, sub }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      <span className="stat-sub">{sub}</span>
    </div>
  );
}

function Gaps({ gaps, onOpenCall }) {
  if (!gaps.length) return <p className="empty">Every customer question got a card. No KB gaps so far.</p>;
  return (
    <>
      <p className="hint gaps-hint">
        Customer lines that escalated with “No KB coverage”, grouped when they share the words the KB has never
        used. Each is a missing article or a missing synonym.
      </p>
      <ol className="gaps">
        {gaps.map((g, i) => (
          <li key={i} className="gap">
            <div className="gap-head">
              <span className="gap-label">{g.label}</span>
              <span className={g.count > 1 ? "tag warn" : "tag"}>
                {g.count} time{g.count > 1 ? "s" : ""}
                {g.calls > 1 ? ` · ${g.calls} calls` : ""}
              </span>
            </div>
            <ul className="gap-examples">
              {g.examples.map((e) => (
                <li key={`${e.call_id}-${e.turn}`}>
                  <button className="gap-link" onClick={() => onOpenCall(e.call_id, e.turn)}>
                    <span className="gap-line">“{e.text}”</span>
                    <span className="call-meta">
                      {e.call_title} · {formatDate(e.started_at)} →
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {g.count > g.examples.length && (
              <p className="hint">and {g.count - g.examples.length} more</p>
            )}
          </li>
        ))}
      </ol>
    </>
  );
}

const COLUMNS = [
  { key: "title", label: "Article", numeric: false },
  { key: "shown", label: "Shown", numeric: true },
  { key: "helpful_pct", label: "Helpful", numeric: true },
  { key: "avg_sentiment", label: "Avg sentiment", numeric: true },
];

function ArticleTable({ articles }) {
  const [sort, setSort] = useState({ key: "shown", dir: "desc" });
  const used = articles.filter((a) => a.shown > 0);
  const unused = articles.filter((a) => a.shown === 0);

  const rows = [...used].sort((a, b) => {
    const x = a[sort.key];
    const y = b[sort.key];
    // Articles with no votes (or no calls) always sort last.
    if (x === null && y === null) return 0;
    if (x === null) return 1;
    if (y === null) return -1;
    const order = typeof x === "string" ? x.localeCompare(y) : x - y;
    return sort.dir === "asc" ? order : -order;
  });

  function sortBy(key, numeric) {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: numeric ? "desc" : "asc" }));
  }

  return (
    <>
      <div className="table-wrap">
        <table className="article-table">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  className={c.numeric ? "num" : ""}
                  aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                >
                  <button className="sort" onClick={() => sortBy(c.key, c.numeric)}>
                    {c.label}
                    <span className="sort-mark" aria-hidden="true">
                      {sort.key === c.key ? (sort.dir === "asc" ? "▲" : "▼") : ""}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => {
              const votes = a.helpful + a.not_helpful;
              return (
                <tr key={a.kb_id}>
                  <td>
                    <span className="kb-id">{a.kb_id}</span> {a.title}
                  </td>
                  <td className="num">{a.shown}</td>
                  <td className="num">
                    {votes ? (
                      <>
                        {pct(a.helpful_pct)}
                        <span className="cell-sub">
                          {a.helpful} of {votes}
                        </span>
                      </>
                    ) : (
                      <span className="cell-sub">no votes</span>
                    )}
                  </td>
                  <td className="num">
                    <span className="score">
                      <i className={`key ${keyClass(a.avg_sentiment)}`} aria-hidden="true" />
                      {formatScore(a.avg_sentiment)}
                    </span>
                    <span className="cell-sub">
                      {a.calls} call{a.calls === 1 ? "" : "s"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="hint table-foot">
        Shown counts cards whose content came from the article. Avg sentiment is the overall customer sentiment of
        the calls that used it.
        {unused.length > 0 && ` Never shown: ${unused.map((a) => a.kb_id).join(", ")}.`}
      </p>
    </>
  );
}

// Diverging bars from a zero line: blue right for positive calls, red left
// for negative ones, the same colors and rounded ends as the per-call chart.
// The score and call count are printed beside each bar, so color is never
// the only cue.
function TopicChart({ topics }) {
  return (
    <figure className="chart topic-chart">
      <figcaption>Average call sentiment, grouped by each call's main article. Worst first.</figcaption>
      <ul className="topic-rows">
        {topics.map((t) => {
          const v = t.avg_sentiment ?? 0;
          const width = `${Math.min(1, Math.abs(v)) * 50}%`;
          return (
            <li
              key={t.kb_id ?? "none"}
              className="topic-row"
              title={`${t.title}: ${formatScore(v)} across ${t.calls} call${t.calls === 1 ? "" : "s"}, ${t.negative} negative`}
            >
              <span className="topic-label">
                {t.kb_id && <span className="kb-id">{t.kb_id}</span>} {t.title}
              </span>
              <span className="topic-track" aria-hidden="true">
                <span className="topic-zero" />
                {v !== 0 && (
                  <span
                    className={`topic-bar ${v > 0 ? "pos" : "neg"}`}
                    style={v > 0 ? { left: "50%", width } : { right: "50%", width }}
                  />
                )}
              </span>
              <span className="topic-value">
                {formatScore(v)}
                <span className="cell-sub">
                  {t.calls} call{t.calls === 1 ? "" : "s"}
                  {t.negative ? ` · ${t.negative} negative` : ""}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
      <div className="legend">
        <span>
          <i className="key pos" /> Positive
        </span>
        <span>
          <i className="key neg" /> Negative
        </span>
        <span className="muted">Scale −1 to +1, zero in the middle.</span>
      </div>
    </figure>
  );
}

function keyClass(v) {
  if (v === null || v === undefined || Math.abs(v) < 0.1) return "zero";
  return v > 0 ? "pos" : "neg";
}

function pct(v) {
  return v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`;
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
