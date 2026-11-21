# Comment Quality Rubric (Task #5)

This rubric defines how we score the quality of a single comment (reply).
Hand labelers and the LLM labeler both use it, so their scores can be
compared directly.

## What gets scored

- **Unit:** one reply.
- **Context shown to the scorer:** the comment that started the thread and
  the comment being replied to. (In CGA-CMV, a conversation starts at a
  top-level comment; the original CMV post is not in the dataset.)
- **Scale:** each dimension is scored 0, 1, or 2. Reasoning may also be
  `N/A`.
- **Conversation-level quality** is not labeled separately. It is the
  average of a conversation's reply scores on each dimension (ignoring
  `N/A`). The dataset's civil/derailed label is kept as a separate sanity
  check: derailed conversations should average lower on civility.

## Dimensions

### Civility: is it respectful?

| Score | Meaning |
|---|---|
| 0 | Insults, mocks, or attacks a **person** (name-calling, "you're a liar", contempt aimed at the other user). |
| 1 | Dismissive, sarcastic, or condescending, but no direct personal attack. Includes mocking or caricaturing the other person's **argument**. |
| 2 | Respectful, even if it disagrees strongly or is blunt. |

### Relevance: is it on topic?

| Score | Meaning |
|---|---|
| 0 | Off topic. |
| 1 | Loosely related or a tangent, or mostly about the fight between users rather than the topic. |
| 2 | Directly responds to the thread or the comment it replies to. |

### Reasoning: does it support its claims?

| Score | Meaning |
|---|---|
| 0 | Makes a claim with no support. |
| 1 | Some support: one reason, an example, or a link. |
| 2 | Clear reasons, evidence, or a counterexample that connects to its point. |
| N/A | Makes no claim (e.g. a plain question, "thanks!"). |

### Constructiveness: does it move the discussion forward?

| Score | Meaning |
|---|---|
| 0 | Shuts the discussion down or adds nothing. |
| 1 | Adds a little: brief agreement, a simple or repeated question, or a bare counterpoint with no support. |
| 2 | Adds something new: information, a supported counterpoint, a challenging question, or a concession. |

## Rules for edge cases

1. **Argument vs. person.** Mocking an argument caps civility at 1.
   Attacking the person makes it 0.
2. **Profanity.** Swearing that is not aimed at anyone does not lower
   civility by itself.
3. **Score the comment on its own.** If the other user was rude first, the
   reply is still scored on what it says. Provocation does not excuse an
   insult.
4. **Dismissal.** A reply that brushes off the other person's point ("So
   what?") gets constructiveness 0, even if it adds a claim.
5. **Score what is written,** not what the dataset says happened to it
   (e.g. moderator removal).
6. **Insults at people outside the conversation** (public figures, groups,
   or a generic "you") cap civility at 1. Civility 0 is only for attacks on
   a user taking part in the conversation.

## Worked examples (CGA-CMV train split)

Scores are Civility / Relevance / Reasoning / Constructiveness.

| Utterance ID | Reply (shortened) | Scores | Why |
|---|---|---|---|
| `dqqbr1w` | "If someone wrongs me civilly, and I sue, and am awarded damages… would that be theft? It would seem to fit your definition." | 2 / 2 / 2 / 2 | Respectful counterexample that tests the other user's definition. |
| `dx6kivk` | "It's often not explained… nothing in the first Star Wars explains how the Force works…" | 2 / 2 / 2 / 2 | Direct answer with concrete examples. |
| `d7hh4ph` | Flips the parent's argument onto men, then "See? You can play this game any way you like. What's the point, though?" | 1 / 2 / 2 / 2 | Strong reasoning; mocks the argument, not the person. |
| `dt287ga` | "And white people have all the money." | 2 / 2 / 0 / 1 | Blunt but not hostile; bare counterpoint. |
| `dkc4vcp` | "So who can transwomen fairly compete with?" | 2 / 2 / N/A / 1 | Plain question that repeats an earlier one. |
| `e4rdy4d` | "So what? Russia has more common values with us than leftists do" | 1 / 2 / 0 / 0 | Dismissive; unsupported claim. |
| `deax47k` | "So, literally every single thing he's done… has been a sign of the apocalypse?" | 1 / 2 / 0 / 0 | Sarcastic caricature of the other user's view. |
| `dbamq2d` | A news link, then "you just immediately get shitty… would be hilarious if it wasn't so sad." | 0 / 1 / 1 / 0 | Personal attack even though provoked; link counts as some support. |
