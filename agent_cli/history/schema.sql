-- The history: every conversation of every working folder, and each message of it, in the order it was written.

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    working_folder TEXT NOT NULL,
    -- The start of the user's first message, which names the conversation in the list to resume.
    title TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- The tokens the model counted in its context when the last turn ended, for the cut after a resume.
    tokens_in_context INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    -- In a reply that asked for tools: the calls, as the JSON list the model gets back.
    tool_calls TEXT,
    -- In the result of a tool: the call it answers, and what the screen showed of it.
    tool_call_id TEXT,
    tool_summary TEXT,
    tool_ending TEXT
);

-- What the search of earlier turns finds: the text of each turn it can bring back, and the vector of that text.
CREATE TABLE IF NOT EXISTS excerpts (
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    turn INTEGER NOT NULL,
    -- The model that made the vector: the vectors of two embedding models cannot be compared.
    embedding_model TEXT NOT NULL,
    text TEXT NOT NULL,
    -- The numbers of the vector, each a 32-bit float.
    vector BLOB NOT NULL,
    PRIMARY KEY (conversation_id, turn, embedding_model)
);

CREATE INDEX IF NOT EXISTS conversations_of_a_folder ON conversations (working_folder, updated_at);
CREATE INDEX IF NOT EXISTS messages_of_a_conversation ON messages (conversation_id, id);
