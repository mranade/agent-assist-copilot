import { useEffect, useRef, useState } from "react";

// Diverging columns: one per customer turn, growing up (positive, blue) or
// down (negative, red) from a zero baseline. Direction is carried by position
// as well as color, so the chart never relies on color alone.
const HEIGHT = 140;
const PAD_TOP = 12;
const PAD_BOTTOM = 22;
const AXIS_W = 28;
const MAX_BAR = 24;

export default function SentimentChart({ turns, onPick }) {
  const [hover, setHover] = useState(null);
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(360);

  // Draw at the container's real pixel width so text and bars stay their
  // designed size instead of being scaled up or down with the panel.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(200, entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const points = turns
    .map((t, i) => ({ ...t, index: i }))
    .filter((t) => t.speaker === "customer");
  if (!points.length) return null;

  const plotH = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const zeroY = PAD_TOP + plotH / 2;
  const band = (width - AXIS_W) / points.length;
  const barW = Math.min(MAX_BAR, band * 0.5);
  const y = (v) => zeroY - (v * plotH) / 2;

  return (
    <figure className="chart">
      <figcaption>Customer sentiment by turn</figcaption>
      <div className="chart-wrap" ref={wrapRef}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label="Customer sentiment for each customer turn, from -1 to +1"
        >
          {[1, 0, -1].map((v) => (
            <g key={v}>
              <line
                x1={AXIS_W}
                x2={width}
                y1={y(v)}
                y2={y(v)}
                className={v === 0 ? "axis-zero" : "grid"}
              />
              <text x={AXIS_W - 6} y={y(v) + 4} className="tick" textAnchor="end">
                {v > 0 ? "+1" : v < 0 ? "−1" : "0"}
              </text>
            </g>
          ))}
          {points.map((p, i) => {
            const cx = AXIS_W + band * i + band / 2;
            const v = p.sentiment ?? 0;
            const h = Math.max(Math.abs(y(v) - zeroY), v === 0 ? 0 : 2);
            const up = v >= 0;
            return (
              <g
                key={p.index}
                onMouseEnter={() => setHover({ ...p, cx, n: i + 1 })}
                onMouseLeave={() => setHover(null)}
                onClick={() => onPick?.(p.index)}
                className="bar-group"
              >
                <rect x={cx - band / 2} y={PAD_TOP} width={band} height={plotH} className="hit" />
                {v === 0 ? (
                  <rect x={cx - barW / 2} y={zeroY - 1} width={barW} height={2} className="bar-neutral" />
                ) : (
                  <path
                    d={roundedBar(cx - barW / 2, zeroY, barW, h, up)}
                    className={up ? "bar-pos" : "bar-neg"}
                  />
                )}
                <text x={cx} y={HEIGHT - 6} className="tick" textAnchor="middle">
                  {i + 1}
                </text>
              </g>
            );
          })}
        </svg>
        {hover && (
          <div
            className="tooltip"
            // Anchor to the bar, but flip to the other side near the edges so
            // the tooltip never gets squeezed or clipped.
            style={
              hover.cx > width * 0.6
                ? { right: `${width - hover.cx + 14}px` }
                : { left: `${hover.cx + 14}px` }
            }
            role="status"
          >
            <strong>
              Turn {hover.n}: {formatScore(hover.sentiment)}
            </strong>
            <span>{truncate(hover.text, 90)}</span>
          </div>
        )}
      </div>
      <div className="legend">
        <span>
          <i className="key pos" /> Positive
        </span>
        <span>
          <i className="key neg" /> Negative
        </span>
        <span className="muted">Customer turns, in order. Hover for the line.</span>
      </div>
    </figure>
  );
}

// A column with a 4px rounded data end and a square end at the baseline.
function roundedBar(x, zeroY, w, h, up) {
  const r = Math.min(4, h, w / 2);
  if (up) {
    const top = zeroY - h;
    return `M${x},${zeroY} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top + r} V${zeroY} Z`;
  }
  const bottom = zeroY + h;
  return `M${x},${zeroY} V${bottom - r} Q${x},${bottom} ${x + r},${bottom} H${x + w - r} Q${x + w},${bottom} ${x + w},${bottom - r} V${zeroY} Z`;
}

export function formatScore(v) {
  if (v === null || v === undefined) return "–";
  const s = Math.abs(v).toFixed(2);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : "0.00";
}

function truncate(text, n) {
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}
