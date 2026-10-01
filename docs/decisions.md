# Design decisions

The reasoning behind each choice, including the ones that didn't fully work.

## Retrieval: TF-IDF over KB sections

The knowledge base is 20 short articles split into 27 sections, written in
consistent product vocabulary ("gateway", "late fee", "technician"). For that
shape of corpus, TF-IDF with cosine similarity is fast (single-digit
milliseconds per lookup), needs no model or API key, and is fully
explainable: you can point at the shared terms that produced a match.

Chunking is by H2 section, with the article title prefixed to every chunk.
"Retention offers" alone is ambiguous; "Cancellation and retention offers >
Retention offers" isn't.

The cost is paraphrase. Customers say "cutting out", "relocating", "send
back"; the KB says "drops", "moving", "return". Five of the eight held-out
misses are this.

## Bridging customer vocabulary to KB vocabulary

Two cheap steps in `text.py`:

1. **A synonym map** ("router"/"modem" → "gateway", "refund" → "credit").
2. **Snowball stemming**, so "cancel", "cancelling", and "cancellation"
   match.

The synonym list is intentionally short. While building it, I noticed I had
started adding entries taken from the held-out eval set ("money back", "come
out", "send back"). That would have quietly inflated the eval score, so I
removed them. Every remaining entry comes from the sample calls or is an
obvious domain synonym. In production, the list would grow from the
escalation log: each "no KB coverage" that turns out to be a wording gap
becomes a new entry.

## Grading on two signals

The first version gated on similarity alone and it couldn't work: the
out-of-scope question "Do you have a TV and streaming bundle?" scored 0.127
against the KB, higher than a real cancellation question at 0.120. No
threshold separates those.

That still holds after the sample calls were rewritten. In the seeded
history, "Do you have a streaming bundle, like TV channels with the
internet?" scores 0.29, the same as "I'd like to cancel my account" (0.29).
Only coverage (25%) stops it. The bundle question in sample call 3 now fails
both checks (similarity 0.11, coverage 17%), so the seeded line is the better
example of why coverage is needed.

The fix was a second signal, **coverage**: the share of the customer's
meaningful words that the KB has ever used. "TV", "streaming", and "bundle"
appear nowhere in the KB, so coverage is 17%. A real question about
cancelling has coverage of 50–100%.

A suggestion is shown only if similarity ≥ 0.13 **and** coverage ≥ 0.30.
Both thresholds are midpoints between the in-scope and out-of-scope values
in the three sample calls. They were not tuned on the held-out set.

## Retry with conversation memory

Customers refer back: "And what do I do with the router you gave me?" If
the first retrieval is weak and an earlier customer turn exists, the agent
retries once with the previous turn prepended.

The subtle part: coverage is always judged on the **new** utterance alone.
Otherwise the borrowed context would lend its vocabulary to an off-topic
question and make it look answerable. There's a test for exactly this
(`test_borrowed_context_cannot_make_off_topic_question_answerable`).

## Triage: a fix that came from the eval

The first triage rule treated "short and not phrased as a question" as
small talk. The first eval run showed that rule silently dropped problem
statements like "the internet is completely down" and "there's a red light
on my modem", which are the most important lines in a support call.

The rule now says: small talk means *not a request* and *mentions nothing
the KB has ever talked about*. This change came after seeing held-out
results, so I'm flagging it. It was a logic fix, not threshold tuning; the
grading thresholds didn't move. Held-out overall went from 62% to 69%.

## What building the call history exposed

The past calls in the history are real pipeline output: their transcripts
run through the same agent, summary, and sentiment code as a live call.
Doing that surfaced three bugs the sample calls hadn't:

1. **Emotion diluted coverage.** "My internet has been completely down since
   yesterday morning... This is ridiculous" escalated. "Ridiculous",
   "yesterday", and "morning" counted as topics the KB doesn't cover, pulling
   coverage to 0.29, just under the floor. Coverage (and triage) now use
   *topic* words only: general emotion words from VADER's lexicon are left
   out. Support words that carry emotion, like "down", "slow", and "cancel",
   are kept, since they're also the topic. My first version of this got the
   split wrong: it kept "ridiculous" because I had added it to the sentiment
   lexicon. The fix was two lexicons, support terms and emotion terms.
2. **Thank-yous weren't small talk.** "Perfect, that's great. Thanks for
   making it so easy" produced a "No KB coverage" card, and the summary
   listed it as an open item. "Making" matched a KB word. Generic verbs
   (make, get, take, look) are now filler, and emotion words don't count as
   topics, so the line is correctly skipped.
3. **An empty card escalated a good match.** After the vocabulary changes,
   "the internet is completely down, nothing works at all" retrieved KB-101's
   overview section first. That section is only a "Use this when..." note for
   KB authors, which the suggester filters out, so the card had no lines, and
   the grounding check correctly refused an empty card. The suggester now
   falls through to the next retrieved section with usable lines.

Held-out score across these changes: 69% before, 65% after the first
attempt (bug 3 surfaced), 73% after all three fixes. None of the fixes were
driven by the held-out lines; they came from the seeded calls. The
thresholds themselves never moved.

## Call sentiment

**Model.** VADER, a lexicon-based model: offline, microseconds per line,
and every score is traceable to the words that produced it. The UI shows the
lowest and highest lines for every call so the rating is never a black box.

**Domain tuning.** Stock VADER misreads support calls:

Lines from the sample calls:

| Line | Stock VADER | Tuned | Why stock is wrong |
|---|---|---|---|
| "Okay, but this has been going on for over a week. Can I get some kind of credit on my bill for all the drops?" | +0.53 | −0.55 | "credit" and "kind" are positive in everyday English |
| "Honestly I'm also thinking about just cancelling because the price is kind of high..." | +0.30 | −0.36 | "honestly" and "kind" read positive; "cancel" isn't in the lexicon |
| "My internet keeps dropping every night, like it cuts out for a minute or two..." | −0.25 | −0.54 | barely registers the clearest complaint an ISP gets |

`sentiment.py` neutralizes billing and hedge words and adds weight to
outage and churn words.

**Call-level score.** A weighted average of the customer's lines that carry
any sentiment (greetings score zero and would dilute it), with quadratic
weights so the end of the call dominates. Most people calling support start
out unhappy; how they sound when they hang up is the better signal of
whether the call helped. Linear weights weren't enough: they rated a call
that ended "Great, thank you so much" as neutral. The trend (improved,
worsened, steady) compares the first third of the call with the last third.

**Limits.** No sarcasm ("great, another outage" reads positive), no
labeled evaluation set yet, and the weighting and thresholds were chosen by
looking at the six demo calls. The next step is to label a few dozen calls,
measure against them, and compare an LLM rating that explains itself.

## Extractive suggestions by default, Claude optional

The default suggester copies the most relevant steps from the matched
article, verbatim. That means the demo runs with no API key, and grounding
is guaranteed by construction.

Its weakness is that it only sees shared words. For "Is there any discount
for autopay?" it finds the right article but misses the "$5 off per month"
sentence, because that sentence doesn't contain "discount".

With `ANTHROPIC_API_KEY` set, Claude drafts the suggestion from the
retrieved excerpts, which reads better and uses the whole article. Because
generated text can drift, it gets a stricter check: every line must cite a
KB id, and every cited id must be one that was actually retrieved. If not,
the turn escalates. The graph and UI don't change between modes.

(Since the change described next, this applies to articles without a
checklist. For troubleshooting and process articles, the card shows the next
checklist step instead, and Claude only writes the "Say" line.)

## Cards that guide the agent: active issue and checklists

**Why the cards weren't useful.** The extractive suggester picked the KB
sentences sharing the most words with the customer's latest line. For "my
internet keeps dropping every night" in call-01, that surfaced KB-101's
"Likely causes" section (evening congestion, 2.4 GHz interference), not its
"Steps for the agent". The agent learned why it might be happening, not what
to do. And every card was built from one line only: when Dana asked for a
credit, the card switched to KB-109 and the troubleshooting guidance
disappeared, although the problem wasn't fixed yet.

**Checklists.** Troubleshooting and process articles (KB-101, 102, 103, 108,
112, 113, 115) now have a numbered "Steps for the agent" list. `steps.py`
parses it: the "Steps for the agent" section, or else the article's first
numbered list. Policies and prices didn't change; a few existing steps were
reworded or split so their first sentence is something the agent can say
(KB-101 step 5 now ends at "schedule a technician visit", and the ticket note
is its own sentence).

**Active issue.** Each call keeps one active issue (`CallState`), set from the
first customer line whose best match has a checklist. A later line about a
different article only replaces it when that's clearly a new issue. All four
of these must hold, and the trace says which one didn't:

1. the new article has its own checklist (a policy article like KB-109 never
   takes over),
2. the active checklist doesn't link to it (KB-101 step 5 says "see KB-115",
   so a technician-fee question is part of the same issue),
3. it's a strong match, score ≥ `NEW_ISSUE_SCORE` (0.18), and
4. the line no longer matches the active article (its best section scores
   below the usual 0.13 floor).

Otherwise the new article becomes a note on the same card: **Policy** if it
has no checklist, **Also relevant** if it does. The credit question now
shows the KB-101 plan with a KB-109 Policy note under it. An escalation card
("No KB coverage") keeps the plan visible too.

**Step tracking.** After every turn, *including agent turns* (which still
never get their own card), the active checklist is checked:

- A turn completes a step when it shares at least 2 topic words with it (the
  same `topic_tokens` used for coverage).
- Words that appear in more than half of a checklist's steps don't count.
  "Drops" is in most KB-101 steps, so it says nothing about which one was
  done.
- Only an agent turn can perform a step. A customer turn can only answer a
  step that starts with "Ask" ("None of my devices are wired" answers "Ask
  whether drops happen on wired devices too").
- A turn completes at most one step: the first pending one, in checklist
  order, that clears the bar.

Every decision is in the trace, with the words that matched: `track: KB-101
step 2 of 5 done (customer answered it); the turn shares "devices", "wired"`.
The live card updates on agent turns, and "Why this card" lists the
tracking decisions made since the card appeared.

**The card.** Next step (verbatim KB text), a "Say" line, the checklist with
done steps ticked ("Step 3 of 5"), and the note. In extractive mode the Say
line is the step's first sentence inside a fixed template ("Ask whether X" →
"Can you tell me whether X?"; "Have the customer X" → "Could you X?"). The
template adds framing words only, never facts. With Claude, Claude writes it
from the step and must cite that step's KB id. If it doesn't, the template
line is used, and the card says which one you're seeing.

**How the rules were set, and what changed while building them.** Everything
here was set by replaying the three sample calls, which I rewrote for this
change (17 turns each). Three corrections came out of that, and I'm listing
them because they're the kind of thing that fails silently:

- My first cut excluded words used by *half or more* of a checklist's steps.
  On 4-step lists that removed "leaving" from KB-113's "Ask why they are
  leaving", so the agent asking why never completed it. Now it's *more than*
  half.
- My first cut completed the step with the *most* shared words. "Can I ask
  why you're leaving? Is it the price?" then completed the retention *offer*
  (it mentions price and service) instead of the question. Now it's the
  first pending step in order.
- I also tracked the line that raised a new issue against its new checklist.
  A unit test caught "I'm moving to a new address" completing "Ask for the
  new address". That line mentions the address without giving it, so this
  was removed.

`NEW_ISSUE_SCORE` is the weakest number in the project. The sample calls
contain one real new-issue line (call-03, moving → cancelling, score 0.20),
so 0.18 rests on one data point. Its job is narrow: rules 1, 2, and 4 do
most of the work, and the score only stops a weak match from replacing the
plan.

**What the eval says.** The held-out numbers didn't move (19/26). They score
single lines with no call state, so the active issue and step tracking never
apply there. The cards look different, though: a lines card showed the top
two articles, and a plan card shows the checklist plus at most one note, so
several held-out lines now show one article instead of two. None of those
changes flipped a verdict. The demo calls went from 13/13 to 24/24 customer
lines, but the calls and the rules were written together, so that number
shows the design works as intended. It doesn't show that it generalizes.

**Limits.**

- Word overlap can't tell *doing* a step from *talking about* it. "If you
  cancel, service ends at the end of your billing cycle" completes "explain
  that service ends...", which is right; "I won't run a line check" would
  complete "run a line check", which is wrong.
- No branching. KB-101 step 1 says "if the line check shows errors, skip to
  step 5"; the card still walks through steps 2–4. Retention offers are one
  combined step for the same reason.
- One step per turn. An agent who does two things in one breath gets credit
  for the first.
- The Say templates are stiff ("I can offer you a retention option: 20% off
  your plan for 3 months if you cite price...").
- The runner-up note can be noise. On call-02's first card, the late-fee
  plan gets a KB-107 "Policy" note about what the bill shows, because the
  customer said "charged" and "month".
- Writing the calls surfaced a sentiment miss: VADER scores "Yeah, whatever.
  Bye." as positive (+0.30). That's sarcasm, the blind spot listed under
  sentiment. I cut the line from call-03 instead of tuning the lexicon for it.

## Insights: closing the loop between calls and the KB

**Why.** Every earlier section ends the same way: the escalation log shows
what to fix, the synonym list should grow from it, and confident wrong
answers are the dangerous kind. Until now nothing collected that evidence.
Escalations were buried in individual call records, and nobody could say
whether a card helped. The Insights view is that missing half of the loop:
calls produce evidence, and the evidence says what to change in the KB.

It needs two inputs, because the two kinds of miss look different:

- **Safe misses escalate**, so they can be found automatically. Every "No
  KB coverage" line is listed and grouped, so a question that keeps coming
  back (the TV and streaming bundle) stands out from a one-off.
- **Confident wrong answers don't escalate.** Nothing in the pipeline knows
  the KB-116 card for "student discount" is wrong; the grading checks
  passed. Only the agent on the call knows, so the card asks them. A
  "Not helpful" vote is the only signal this project has for the dangerous
  kind of miss, and article performance is where it adds up.

Ratings are optional and can be changed after the call, from the call
history. Escalation cards can be rated too: a "Not helpful" on "No KB
coverage" suggests the agent thought the KB should have covered it.

**How gap grouping works.** For each escalated line, take its topic words
(`topic_tokens`, the same words the coverage check uses) and keep the ones
the KB has never used. Those are exactly the words that pulled coverage
under 0.30, so they say what the question was about that the KB lacks. For
"Do you guys have a TV and streaming bundle I could add to lower the total
cost?" that's *streaming, bundle, lower, total, cost*; "add" is dropped
because the KB uses it. Two lines are the same gap when they share at least
two of these words (one, if either line has only one). Groups are built
transitively, and a group is labelled with the words at least half its
lines share, in the customer's own spelling. A line that escalated for
another reason (every word is in the KB, but grounding failed) falls back to
all its topic words.

**Where it's crude.**

- **Words, not meaning.** "Do you have cable TV packages?" and "Can I get
  Netflix through you?" are the same gap to a person and share no words, so
  they are two groups. Embeddings would group them; they are the obvious
  next step, and the same next step as for retrieval.
- **Short words vanish.** Tokens under three letters are dropped everywhere,
  so "TV" never counts. The bundle lines group on "streaming" and "bundle"
  only because both customers happened to say them.
- **Transitive chaining.** If A shares two words with B, and B shares two
  different words with C, all three become one group even if A and C share
  nothing. With a handful of escalations this hasn't happened; with
  thousands it would, and clustering with a real threshold would be needed.
- **The two-word bar is a guess**, set so the two bundle lines group and a
  line that only shares "streaming" with them doesn't (there's a test for
  each). It was not tuned on anything larger.
- **Sentiment by topic uses the call's main article**, the one credited on
  the most cards. A call about a TV bundle that also asked about plans is
  filed under whatever answered the plans question (in the seed data, the
  wrong KB-118 card). A call that only escalated lands in "No KB article".
- **Small numbers.** Eight seeded calls give one or two calls per topic, so
  the percentages move a lot with every call. The view shows the counts
  next to every rate for that reason.
- **Ratings are self-reported and sparse.** Agents rate some cards, not
  all, and a busy agent rates the ones that annoyed them. The helpful rate
  is among rated cards, and the tile says how many that is.

**Seed data.** Two short calls were added so the view isn't empty on first
start: a second TV and streaming bundle question (so the gap list shows a
*repeated* question), and a student-discount question. The student question
doesn't escalate: it gets the autopay card, the known confident wrong
answer from the eval, and the seeded agent rated it "Not helpful". That is
deliberate: it shows that confident wrong answers surface through ratings
and not through the gap list. The seeded ratings are the one hand-written
part of the history; the cards they rate are real pipeline output.

## Speech-to-text: local Whisper, per turn

Call audio is sensitive, so transcription runs locally with faster-whisper
instead of a cloud API. Each customer turn is transcribed as a whole clip.
That's simple and accurate, but it adds a pause after the customer stops
talking. Streaming partial transcripts is on the roadmap.

If Whisper isn't installed, playback mode falls back to the script text and
the header shows it. That's the backup plan for a live demo.

## Redaction before storage

Verification (KB-117) means customers read out account digits and PINs.
`redact.py` masks any run of four or more digits and anything spoken after
"PIN" *before* the text is stored, displayed, summarized, or sent to an
LLM. It's deliberately simple; production would add card-number detection
(Luhn), names, and addresses.

## What the evaluation says

Held-out, end to end: 19/26 (73%). Retrieval alone finds the right article
in its top 3 for 85% of in-scope lines, so most of the gap is between
"found it" and "confident enough to show it".

The misses split into:

- **4 safe misses**: in scope, but escalated because of wording gaps. The
  agent handles it manually, and the escalation log shows what to fix.
- **3 confident wrong answers**: the dangerous type. "Student discount"
  surfaced the autopay discount because "discount" matched strongly. "What
  plans do you have and how much are they" surfaced the data-usage policy
  alongside plan changes instead of the pricing article.

The next fix targets the second group: require the customer's
distinguishing noun ("student", "channels") to appear in the matched
section, and add embedding-based retrieval alongside TF-IDF for
paraphrases.

## Why MCP isn't in this version

The original tool I built this after was a UI for human agents, and that's
what this is. MCP matters once *other AI systems* need the same grounded
guidance: a CRM copilot, a self-service chatbot, Claude. The agent loop is
already separated from the HTTP layer, so exposing it as MCP tools
(`suggest_for_utterance`, `search_kb`, `summarize_call`, `report_kb_gap`) is
a new front door, not a rewrite. `report_kb_gap` is the interesting one:
it would let any consuming agent feed the same escalation log that improves
the KB.
