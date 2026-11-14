# agent-CLI

A terminal agent that runs any GGUF model directly on your machine, or any model on OpenRouter. Models are configured through environment variables, and you switch between them without leaving the session.

> **Status:** early development. A session with a local model works: the model loads in the background, alone or split with RPC servers, and its reply appears on the screen piece by piece. `/model` switches between the local model and the ones on OpenRouter, and `/think` chooses how much the model in use thinks. Tools and history are not implemented yet.

## The idea

- **Local first.** By default the CLI opens the model file and generates the response itself. There is no server and no API, and nothing leaves your machine.
- **Any GGUF model.** The project was born to run ADA, the author's own model, but nothing in the code knows which model is running. Whatever changes from one model to another comes from outside, through an environment variable.
- **OpenRouter when you want it.** The same CLI can instead send the conversation to a model on OpenRouter. It is only used when you configure it, and when a local model is configured too, the session starts on the local one.
- **Switch models inside the session.** `/model` lists the models you configured, local and API, and switches between them. The conversation continues, and every model gets the same tools.
- **An agent, not just a chat.** The model can ask for a tool (read a file, run a command), and the CLI is the one that executes it.
- **Your own system prompt.** A variable points to a text file with the system prompt, so you change how the agent behaves without touching the code.

## How it works

```
environment variables ──► load the .gguf ──► session opens

inside the session, for each message:

  you type
      │
      ▼
  the conversation is formatted the way the model expects
      │
      ▼
  the model generates the response, piece by piece
      │
      ├── text ──────────► shown on screen
      │
      └── tool request ──► the CLI runs it and hands the result back
                           to the model, which generates again

  everything said goes into the history, saved in the database
```

With an API model there is no model to load, and the formatting and the generation happen on the API's side: the CLI sends the conversation and receives the response piece by piece. Everything else is the same.

The conversation does not belong to any model. When `/model` switches to another one, the next model receives the same conversation and continues from where the previous one stopped.

The parts, by the role each one plays:

| Part | What it does |
|---|---|
| Configuration | Reads the environment variables. It is the only place that reads them. |
| Model catalog | Lists the models that are available, the local one and the API ones, and hands over the one in use. |
| Loading | Local model only. Opens the `.gguf` with llama.cpp, on one machine or split across two. |
| Conversation format | Local model only. Formats the conversation the way that model expects, using the chat template stored inside the `.gguf` itself. |
| Generation | Turns the conversation into a response and returns it piece by piece, either with the local model or through the API. It never prints anything. |
| Conversation | Keeps the messages, which start with the system prompt, and runs each turn: asks for the generation and handles tool requests. |
| Tools | Detects when the model asked for a tool, runs it and returns the result. The same tools for every model. |
| History | Saves and restores conversations in the database. |
| Commands | Session commands, which start with `/`. The first one is `/model`. |
| Screen | The terminal session: reads what you type and shows what arrives. |

## Configuration

Everything is set through environment variables, in a `.env` file that stays out of git. Copy `.env.example` to `.env` and fill in the values.

| Variable | Value | Required |
|---|---|---|
| `AGENT_CLI_MODEL_PATH` | Path to the `.gguf` file. A relative path starts at the project folder. | For the local model |
| `AGENT_CLI_RPC_SERVERS` | `ip:port` of the other machine. When empty, everything runs on a single machine. | No |
| `AGENT_CLI_API_KEY` | Secret key of your OpenRouter account | For API models |
| `AGENT_CLI_API_MODELS` | Names of the models on OpenRouter, written as OpenRouter lists them, separated by commas. The first one is the default. | For API models |
| `AGENT_CLI_SYSTEM_PROMPT_PATH` | Path to a text file with the system prompt. A relative path starts at the project folder. When empty, the project's default prompt is used. A file with no text is an error. | No |

`/model` offers every model that is configured: the local one when `AGENT_CLI_MODEL_PATH` is filled in, and each name in `AGENT_CLI_API_MODELS`. The session starts on the local model when there is one, otherwise on the first API model.

## Design decisions

| Decision | Why |
|---|---|
| Local by default, in a single process | One command and it runs, with no second program to install and keep alive. No open port on the machine. And the part worth learning (tokens, context, generation) lives in this project's code. |
| GGUF models only, for local inference | It is the format in which practically every open model is distributed. A single loader covers all of them. |
| Python with `llama-cpp-python` | Generation runs in C++ inside llama.cpp, so the language is not the bottleneck. Python organizes the conversation and the screen. |
| Inference code written from scratch | Code written for one specific model carries that model's quirks (conversation format, tool format). Here nothing may depend on which model it is. |
| A local model gets the largest context its machines hold, never above the one it was trained with | The context is not a setting: nobody should need to know that number to use a model. llama.cpp reserves the memory of the whole context when a model loads, and recent models are trained with contexts that take more memory than most machines have. So when the session opens, the CLI loads the model once with a small context and measures, on this machine's GPU and on each RPC server, how much memory the weights take and how much each token of context takes. From what each device has free it computes the largest context that fits, leaving half a GiB free on each, and loads the model with it. A model that leaves room for fewer than 4096 tokens is refused with an error instead; one trained with less than that only has to fit whole. Two limits come from what llama.cpp reports. A machine with no GPU is not measured, so there the model gets the context it was trained with. And on a Mac the free memory counts only what the CLI itself holds, not what other programs use: a second session opened beside the first sizes its context as if it were alone. An API model gets the size OpenRouter reports for it. |
| A model that can think is told to think | The chat template of some models has a switch, `enable_thinking`, that lets the model write its thinking before the answer; the CLI formats the conversation with the model's own template and turns that switch on. A template with no such switch ignores it. The thinking arrives at the start of the reply, between `<think>` and `</think>`, the marks models of several families use; they are the only format fixed in the code, and the screen uses them to show the thinking apart from the answer. A small model may never end its thinking: `esc` stops it. With an API model the switch is OpenRouter's: every request asks for the reasoning effort chosen with `/think`, which starts at the highest the model accepts, the thinking arrives apart from the answer, and the CLI puts it between the same marks. The thinking of earlier replies is never sent back to a provider. |
| Flash attention is always on | Without it llama.cpp keeps a table of attention scores that grows with the context. A long context then takes far more memory, and each token no longer takes a fixed amount, so the context that fits could not be computed from a small measurement. One case is known not to work: llama.cpp does not ask an RPC server what it can compute, so a server whose backend has no flash attention for the size of the model's attention heads stops at the first reply, and the session stops with it. |
| A single variable to split across two machines | If there is an address, it splits; if it is empty, it does not. A second on/off variable would store the same information twice and allow combinations that make no sense. |
| API models as an opt-in | They allow comparing a local model with a hosted one in the same CLI, with the same conversation and the same tools. There is no variable to choose between the two: the CLI offers whatever is configured. When both are, the session starts on the local model, because with an API the conversation leaves the machine. |
| OpenRouter as the only API | It serves models from almost every provider with a single key, and it reports the context size of each model, which the CLI needs to know where to cut the conversation. Most APIs do not report it. A model served by several providers gets the smallest context among them, because OpenRouter picks the provider of each request. |
| API requests only go to providers that keep no data | Every request tells OpenRouter to use only the providers it lists as Zero Data Retention and that do not train on what they receive. The rule is written in the code, not in a variable: a model no such provider serves is refused, and a mistyped name fails instead of reaching a provider that keeps the conversation. |
| The oldest turns leave in a block when the context fills | Above 80% of the context, whole turns leave, oldest first, until the conversation takes about half of it. The model only reports the size of the whole context, so each message is taken to hold the share of it that matches the share of the text it holds. The system prompt never leaves, and neither does part of a turn. One block at a time, instead of one message every turn, keeps the start of the conversation the same for many turns, so llama.cpp reuses what it already processed. When the model refuses the conversation anyway, the message is sent again with no earlier turns; a message too long even alone is not sent, and the earlier turns left out to try it come back. |
| Variables prefixed with `AGENT_CLI_` | The CLI only uses an API when a variable written for this CLI says so. A key exported for another tool, such as `OPENROUTER_API_KEY`, is never picked up by accident. |
| Switching models inside the session | Different tasks suit different models, and loading a local model is slow, so restarting the CLI just to switch would waste that time. |
| The same tools for every model | A model is chosen for how well it does the task, not for what it is allowed to do. |
| The system prompt in a text file, with its path in a variable | Each user writes their own without touching the code. A file takes a prompt of any size, which would be awkward to write and to read inside `.env`. |
| A database only for history | It is the only thing that has to survive after the CLI closes. |

### The accepted cost

A local model has to be loaded into memory, and it leaves memory when the CLI closes. That is why the CLI is a **session you enter and stay in**, not a command you call for each question. Every open terminal loads the model again.

Opening a session loads the model twice: first with a small context, to measure the memory it takes, then with the context that fits. On one machine the second load reuses what the system has just read from disk. Split across two machines it adds a few seconds.

With an API model, what the model thinks is billed as output, and a model starts with as much thinking as it accepts until `/think` chooses less.

## Running on two machines

When the model does not fit on one machine, llama.cpp can split its layers with another one over the network (RPC). This works for any GGUF, with the one exception described under flash attention in the design decisions, and by default llama.cpp divides the layers according to the free memory of each machine.

The other machine must be running `ggml-rpc-server`, a program that ships with llama.cpp. The CLI does not start it for you.

> **Warning:** `ggml-rpc-server` has no authentication, and llama.cpp itself warns never to run it on an open network. Use it only on a network you trust. By default the CLI runs on a single machine.

## Installation

Requires Python 3.12.

`llama-cpp-python` is compiled at install time, and the flag depends on your machine:

| Machine | Flag |
|---|---|
| Mac (Apple Silicon) | `-DGGML_METAL=on` |
| NVIDIA GPU | `-DGGML_CUDA=on` |
| Splitting across two machines | add `-DGGML_RPC=on` |

```bash
python3.12 -m venv .venv && source .venv/bin/activate
CMAKE_ARGS="-DGGML_METAL=on -DGGML_RPC=on" pip install --no-cache-dir -r requirements.txt
pip install -e .
```

`--no-cache-dir` forces pip to compile again instead of reusing an old build made with a different flag.

`pip install -e .` creates the `agentcli` command. It runs the code of this folder and reads the `.env` of this folder, whatever directory you call it from.

## Running

With the virtual environment active:

```bash
agentcli
```

The screen opens at once and the model loads in the background; the message box unlocks when it is ready. The banner then shows the model and the context it got, in tokens.

| Key | What it does |
|---|---|
| `enter` | Sends the message. |
| `esc` | Stops the reply being written. What was written stays in the conversation. |
| `ctrl+q` | Quits. During the load of the model, the session closes as soon as the load ends. |

A message that starts with a slash, with the command right after it, is a command: the session runs it and the model never receives it. While you type one the message box changes color and suggests the rest (`→` accepts). A slash followed by a space, or anywhere else in the message, is ordinary text.

| Command | What it does |
|---|---|
| `/model` | Lists the configured models, numbered, and marks the one in use. |
| `/model <number or name>` | Switches to that model. The conversation continues with it, and a local model that is replaced leaves memory. |
| `/think` | Tells how much the model in use thinks and the choices it has. |
| `/think <choice>` | Chooses it: `on` or `off` for a local model, an effort from `none` to `max` for a model on OpenRouter. Each model keeps its choice during the session. |

What the model thinks before it answers is shown above the reply, in a muted color. The status line shows how much of the model's context the conversation takes, as a share and in tokens.

## Security

- The `.env` file stays out of git. A machine address is private data, and an API key is a secret.
- No path, private address or key is written in the code. The only address in it is OpenRouter's.
- An API model is only used when no local model is configured or when you pick one with `/model`. From then on, the conversation, including the system prompt, is sent to OpenRouter, and only to providers that keep no data.
- A system prompt with personal content belongs in a file that stays out of git.
- Splitting across two machines is off by default, for the reason explained above.
- The model only asks; the CLI is the one that executes. A tool that changes something (writing a file, running a command) asks for confirmation first.
- Whatever the model reads from a file or from a command's output is data, not an instruction.
