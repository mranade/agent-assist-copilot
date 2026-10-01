async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(detail);
  }
  return res.json();
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  status: () => request("/api/status"),
  calls: () => request("/api/calls"),
  newSession: (source) => request("/api/sessions", json({ source })),
  endCall: (sid) => request(`/api/sessions/${sid}/end`, { method: "POST" }),
  history: () => request("/api/history"),
  historyCall: (id) => request(`/api/history/${id}`),
  insights: () => request("/api/insights"),
  // value: "helpful" | "not_helpful" | null (clears the rating)
  rateSessionTurn: (sid, index, value) =>
    request(`/api/sessions/${sid}/turns/${index}/feedback`, json({ value })),
  rateSavedTurn: (callId, index, value) =>
    request(`/api/history/${callId}/turns/${index}/feedback`, json({ value })),
  playbackTurn: (sid, callId, index) =>
    request(`/api/sessions/${sid}/turns/playback`, json({ call_id: callId, index })),
  textTurn: (sid, text) =>
    request(`/api/sessions/${sid}/turns/text`, json({ speaker: "customer", text })),
  audioTurn: (sid, blob) => {
    const form = new FormData();
    form.append("file", blob, "clip.webm");
    return request(`/api/sessions/${sid}/turns/audio`, { method: "POST", body: form });
  },
};
