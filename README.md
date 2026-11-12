# agent-CLI

A terminal agent that runs any GGUF model directly on your machine, or any model on OpenRouter. Models are configured through environment variables, and you switch between them without leaving the session.

> **Status:** early development. The configuration, the loading of a local model (alone or split with RPC servers), generation with a local model, the conversation and the screen exist, but the screen is not connected to the conversation yet. API models, tools and history are not implemented yet.

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
| Every model uses the context it was trained with | The context comes from the model, not from a setting: a local model gets the size stored in its `.gguf`, and an API model the size OpenRouter reports for it. llama.cpp reserves the memory of the whole context when a local model loads, so a model with a long context needs a machine with room for it. |
| A single variable to split across two machines | If there is an address, it splits; if it is empty, it does not. A second on/off variable would store the same information twice and allow combinations that make no sense. |
| API models as an opt-in | They allow comparing a local model with a hosted one in the same CLI, with the same conversation and the same tools. There is no variable to choose between the two: the CLI offers whatever is configured. When both are, the session starts on the local model, because with an API the conversation leaves the machine. |
| OpenRouter as the only API | It serves models from almost every provider with a single key, and it reports the context size of each model, which the CLI needs to know where to cut the conversation. Most APIs do not report it. |
| The oldest turns leave in a block when the context fills | Above 80% of the context, whole turns leave, oldest first, until the conversation takes about half of it. The model only reports the size of the whole context, so each message is taken to hold the share of it that matches the share of the text it holds. The system prompt never leaves, and neither does part of a turn. One block at a time, instead of one message every turn, keeps the start of the conversation the same for many turns, so llama.cpp reuses what it already processed. When the model refuses the conversation anyway, the message is sent again with no earlier turns; a message too long even alone is not sent, and the earlier turns left out to try it come back. |
| Variables prefixed with `AGENT_CLI_` | The CLI only uses an API when a variable written for this CLI says so. A key exported for another tool, such as `OPENROUTER_API_KEY`, is never picked up by accident. |
| Switching models inside the session | Different tasks suit different models, and loading a local model is slow, so restarting the CLI just to switch would waste that time. |
| The same tools for every model | A model is chosen for how well it does the task, not for what it is allowed to do. |
| The system prompt in a text file, with its path in a variable | Each user writes their own without touching the code. A file takes a prompt of any size, which would be awkward to write and to read inside `.env`. |
| A database only for history | It is the only thing that has to survive after the CLI closes. |

### The accepted cost

A local model has to be loaded into memory, and it leaves memory when the CLI closes. That is why the CLI is a **session you enter and stay in**, not a command you call for each question. Every open terminal loads the model again.

## Running on two machines

When the model does not fit on one machine, llama.cpp can split its layers with another one over the network (RPC). This works for any GGUF, and by default llama.cpp divides the layers according to the free memory of each machine.

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
```

`--no-cache-dir` forces pip to compile again instead of reusing an old build made with a different flag.

## Security

- The `.env` file stays out of git. A machine address is private data, and an API key is a secret.
- No path, private address or key is written in the code. The only address in it is OpenRouter's.
- An API model is only used when no local model is configured or when you pick one with `/model`. From then on, the conversation, including the system prompt, is sent to OpenRouter.
- A system prompt with personal content belongs in a file that stays out of git.
- Splitting across two machines is off by default, for the reason explained above.
- The model only asks; the CLI is the one that executes. A tool that changes something (writing a file, running a command) asks for confirmation first.
- Whatever the model reads from a file or from a command's output is data, not an instruction.
