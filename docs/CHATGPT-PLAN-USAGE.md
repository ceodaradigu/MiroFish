# ChatGPT plan usage in this local fork

This fork adds an **optional, local-only** Sign in with ChatGPT route for an
eligible ChatGPT plan. It does not turn a ChatGPT subscription into an OpenAI
API key. OpenAI's authorization page decides whether the signed-in account can
use this integration. The app never asks for a password or a token.

The integration follows OpenAI's [open-source sign-in guide](https://developers.openai.com/siwc/token-sharing-open-source/sign-in.md)
and [model/inference guide](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference.md).
It targets MiroFish running on the same computer, bound to `127.0.0.1`. It is
not configured for a hosted production service.

## Local preparation

1. Install the locked dependencies from the repository root:
   `npm ci --ignore-scripts`, `npm ci --prefix frontend --ignore-scripts`,
   and `cd backend; uv sync --frozen`.
2. In the ignored root `.env`, set `CHATGPT_PLAN_MODE=1`,
   `MEMORY_BACKEND=local`,
   `FLASK_HOST=127.0.0.1`, `FLASK_PORT=5001`, and `FLASK_DEBUG=False`.
   `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_NAME` are managed in memory by
   this mode and do not need values in `.env`.
3. Start the backend with `cd backend; uv run python run.py` and the frontend
   from another terminal with `npm run dev --prefix frontend`. Open
   `http://127.0.0.1:3000/` **on the same computer**.
4. Select **Continue with ChatGPT**. Finish the sign-in and consent directly
   on OpenAI's page. Return to MiroFish, refresh the status, and select a model
   listed for the account.
   OpenAI binds an issued client ID to the account and workspace used at
   registration. If its page says the account cannot access this application,
   stop that authorization. Repeating it with the same client ID will not
   grant access. If the saved registration belongs to a different account or
   workspace from the one you originally intended to use, choose **Register
   another account or workspace**.
   This starts OpenAI's separate registration flow and preserves the original
   client ID in the local file. The interface does not currently switch back
   to an older registration. Only a verified identity can become the active
   account. OpenAI
   still decides whether the intended account may use this integration; stop
   if it denies the new registration too. Do not use the separate registration
   to retry the same denied account and workspace. Do not erase a saved
   registration or reuse its client ID for another account or workspace.
5. The local graph is stored in the ignored `backend/uploads/local_graph.db`.
   `ZEP_API_KEY` is not needed in local mode. The original Zep Cloud route is
   still available by selecting `MEMORY_BACKEND=zep` and configuring its key.

The authorization and refresh tokens live only in the backend process's
memory. Restarting the backend requires reconnecting. An ignored local
registration file contains a host identifier, an OpenAI client identifier,
and the verified account subject; it contains no access or refresh token.
Simulation workers receive a fresh localhost bearer string, which the backend
checks at `/v1/chat/completions`; the OAuth token stays in the backend.

## Compatibility and current limits

MiroFish and CAMEL/OASIS call the Chat Completions shape, while this OpenAI
plan route uses the Responses API. The localhost adapter translates text,
supported JSON mode, function calls and their history. It sends `store:false`
and `stream:true` upstream. Unsupported request shapes are rejected; sampling
and output-token controls that the plan route forbids are omitted with a log
warning. Results may differ from the original MiroFish provider path.

The selected model must appear in the signed-in account's visible model list.
Model availability and usage limits depend on OpenAI's current eligibility
rules; see its [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations.md).
The local graph uses the selected model to extract entities and relationships
from the scenario; those text fragments therefore go to OpenAI under the
account's plan. Search starts with local BM25 keyword ranking. It does not
replicate Zep's temporal conflict handling or semantic ranking, and the local
SQLite file is not encrypted. See [local memory details](LOCAL-MEMORY.md).

The simulation process stays open for optional agent interviews after its
rounds finish. The UI offers an explicit **Finish run and close interviews**
step to let graph ingestion drain and enable the report. Interviews after
that step are unavailable; the post-report interview flow has not been
verified in this local mode.

This code has automated, credential-free tests. On 21 November 2026, a real
account completed OpenAI's consent on a local machine, listed its available
models, and completed the guide's minimal streamed inference request. A local
hypothetical workshop scenario then completed ontology generation, local graph
ingestion, five simulated profiles, ten rounds on two simulated platforms
(28 actions), and a four-section report. The first ontology attempt exposed a
JSON-mode validation issue: the adapter moved a system JSON instruction into
`instructions`, while the Responses API required a JSON instruction in
`input`. The adapter now supplies it there. In this run the local graph had
13 entity nodes but no extracted edges, so graph relationships did not add
evidence to the report. The run demonstrates a working local path for this
bounded scenario, not general reliability or predictive accuracy. Simulated
agents provide hypotheses, never real customer evidence.

Keep scenario notes and any real business data in a local ignored folder. Do
not commit `.env`, `backend/uploads/`, logs, OAuth artifacts, or private
scenario content to this public fork.
