// The body of an assist card, shared by the live panel and the call history.
//
// A plan card (the call's active issue has a checklist) shows the next step,
// a line the agent could say, the checklist with done steps ticked, and an
// optional note from another article. Other cards show the most relevant
// KB lines.

export default function CardBody({ suggestion, escalated = false, compact = false }) {
  const plan = suggestion.plan;
  if (escalated || !plan) {
    return (
      <>
        <ul className="lines">
          {suggestion.lines.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
        {plan && !compact && (
          <div className="plan-kept">
            <span className="label">Your plan is unchanged</span>
            <Checklist plan={plan} />
          </div>
        )}
      </>
    );
  }
  return (
    <>
      <NextStep plan={plan} compact={compact} />
      {!compact && <Checklist plan={plan} />}
      {suggestion.note && <NoteBlock note={suggestion.note} compact={compact} />}
    </>
  );
}

export function NextStep({ plan, compact = false }) {
  const total = plan.steps.length;
  if (plan.next === null || plan.next === undefined) {
    return (
      <section className="next-step done">
        <span className="label">All {total} steps done</span>
        <p>Everything on the {plan.kb_id} checklist is covered.</p>
      </section>
    );
  }
  return (
    <>
      <section className="next-step">
        <span className="label">
          Next step · Step {plan.next + 1} of {total}
        </span>
        <p>{plan.steps[plan.next].text}</p>
      </section>
      {plan.say && !compact && (
        <section className="say">
          <span className="label">
            Say {plan.say_by === "claude" ? <span className="by">written by Claude</span> : null}
          </span>
          <p>“{plan.say}”</p>
        </section>
      )}
    </>
  );
}

export function Checklist({ plan }) {
  const done = plan.steps.filter((s) => s.done).length;
  return (
    <section className="checklist" aria-label={`${plan.kb_id} checklist, ${done} of ${plan.steps.length} done`}>
      <span className="label">
        Progress · {done} of {plan.steps.length} done
      </span>
      <ol>
        {plan.steps.map((s, i) => (
          <li
            key={i}
            className={s.done ? "done" : i === plan.next ? "current" : ""}
            title={s.because ? `Done: ${s.because}` : undefined}
          >
            <span className="box" aria-hidden="true">
              {s.done ? "✓" : i + 1}
            </span>
            <span className="step-text">{s.text}</span>
            {s.done && <span className="sr-only"> (done, {s.because})</span>}
          </li>
        ))}
      </ol>
    </section>
  );
}

// "Was this card useful?" Clicking the active choice again clears it.
export function FeedbackButtons({ value, onChange, disabled = false }) {
  const choices = [
    { id: "helpful", label: "Helpful" },
    { id: "not_helpful", label: "Not helpful" },
  ];
  return (
    <div className="feedback" role="group" aria-label="Rate this card">
      <span className="feedback-label">Was this card useful?</span>
      {choices.map((c) => (
        <button
          key={c.id}
          type="button"
          className={`feedback-btn ${c.id} ${value === c.id ? "active" : ""}`}
          aria-pressed={value === c.id}
          disabled={disabled}
          onClick={(e) => {
            e.stopPropagation();
            onChange(value === c.id ? null : c.id);
          }}
        >
          {c.label}
        </button>
      ))}
    </div>
  );
}

export const FEEDBACK_LABELS = { helpful: "rated helpful", not_helpful: "rated not helpful" };

function NoteBlock({ note, compact }) {
  return (
    <aside className={`note-block ${note.kind === "Policy" ? "policy" : ""}`}>
      <span className="label">
        {note.kind} · {note.kb_id} {note.title}
      </span>
      <ul>
        {(compact ? note.lines.slice(0, 1) : note.lines).map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>
    </aside>
  );
}
