# Local graph memory for this fork

Set `MEMORY_BACKEND=local` in the ignored root `.env` to use a SQLite graph at
`backend/uploads/local_graph.db` instead of Zep Cloud. No Zep account, key,
plan, or payment is needed in this mode. The default `MEMORY_BACKEND=zep`
preserves the original Cloud path.

Each project records the backend used when it was created. Changing
`MEMORY_BACKEND` does not move an existing graph between Zep and SQLite. To
open or delete an older project, select its original backend again. The app
rejects graph operations when a project's recorded backend differs from the
active one, so a switch cannot silently discard its graph reference.

This is a **local approximation**, not a copy of Zep's service. It stores
source episodes, extracted entities, relations, batch state, and simulation
activity on this computer. It asks the selected language model to extract
entities and relations from each source fragment; in ChatGPT plan mode, those
fragments are sent to OpenAI through the official Responses flow. Local memory
does not mean fully offline inference.

Search defaults to local keyword ranking (`MEMORY_EMBEDDINGS=0`) and does not
download an embedding model. It does not reproduce Zep's temporal conflict
resolution or cross-encoder reranking. Simulation activity is kept as
searchable episode text by default (`MEMORY_EXTRACT_ACTIVITY=0`), so it may not
become a structured graph relation. Empty or failed source extraction must
stop graph building instead of producing a fictitious success. Results will
therefore differ from Zep Cloud and should be treated as simulated hypotheses.

The SQLite file is ignored by Git but **is not encrypted**. Use the prepared
public or fictional Tallerna scenario, not customer records. Deleting a graph
or project does **not** delete saved simulations, reports, logs, backups, or
the whole SQLite file, and is not secure erasure. Never commit the database,
`.env`, logs, or source notes to the public fork.

## Provenance

The local adapter in `backend/app/utils/local_graph.py` and its optional
embedding helpers in `backend/app/utils/embeddings.py` are adapted from
[DarkWorldCoder/market-mirror at f4b83625](https://github.com/DarkWorldCoder/market-mirror/tree/f4b83625e9df025a07f1d411cd35f8cd53b93056),
licensed under AGPL-3.0. This fork retains the project's AGPL-3.0 license.
The adapter was changed to fit current MiroFish and this private local trial;
the source project does not certify these changes or the simulation results.
