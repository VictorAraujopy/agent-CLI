# agent-CLI

A terminal agent that runs any GGUF model directly on your machine. To switch models, change one environment variable.

> **Status:** early development. Nothing is implemented yet.

## The idea

- **Fully local.** The CLI opens the model file and generates the response itself. There is no server and no API, and nothing leaves your machine.
- **Any GGUF model.** The project was born to run ADA, the author's own model, but nothing in the code knows which model is running. Whatever changes from one model to another comes from outside, through an environment variable.
- **An agent, not just a chat.** The model can ask for a tool (read a file, run a command), and the CLI is the one that executes it.

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

The parts, by the role each one plays:

| Part | What it does |
|---|---|
| Configuration | Reads the environment variables. It is the only place that knows which model was chosen. |
| Loading | Opens the `.gguf` with llama.cpp, on one machine or split across two. |
| Conversation format | Formats the conversation the way that model expects, using the chat template stored inside the `.gguf` itself. |
| Generation | Turns the conversation into a response and returns it piece by piece. It never prints anything. |
| Tools | Detects when the model asked for a tool, runs it and returns the result. |
| History | Saves and restores conversations in the database. |
| Screen | The terminal session: reads what you type and shows what arrives. |

## Configuration

Everything is set through environment variables, in a `.env` file that stays out of git.

| Variable | Value | Required |
|---|---|---|
| `AGENT_CLI_MODEL_PATH` | Path to the `.gguf` file | Yes |
| `AGENT_CLI_RPC_SERVERS` | `ip:port` of the other machine. When empty, everything runs on a single machine. | No |

Copy `.env.example` to `.env` and fill in the values.

## Design decisions

| Decision | Why |
|---|---|
| Fully local, in a single process | One command and it runs, with no second program to install and keep alive. No open port on the machine. And the part worth learning (tokens, context, generation) lives in this project's code. |
| GGUF models only | It is the format in which practically every open model is distributed. A single loader covers all of them. |
| Python with `llama-cpp-python` | Generation runs in C++ inside llama.cpp, so the language is not the bottleneck. Python organizes the conversation and the screen. |
| Inference code written from scratch | Code written for one specific model carries that model's quirks (conversation format, tool format). Here nothing may depend on which model it is. |
| A single variable to split across two machines | If there is an address, it splits; if it is empty, it does not. A second on/off variable would store the same information twice and allow combinations that make no sense. |
| A database only for history | It is the only thing that has to survive after the CLI closes. |

### The accepted cost

The model is loaded every time the CLI opens and leaves memory when it closes. That is why the CLI is a **session you enter and stay in**, not a command you call for each question. Every open terminal loads the model again.

## Running on two machines

When the model does not fit on one machine, llama.cpp can split its layers with another one over the network (RPC). This works for any GGUF, and llama.cpp works out by itself how much goes to each machine.

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

- The `.env` file stays out of git. A machine address is private data.
- No path, address or key is written in the code.
- Splitting across two machines is off by default, for the reason explained above.
- The model only asks; the CLI is the one that executes. A tool that changes something (writing a file, running a command) asks for confirmation first.
- Whatever the model reads from a file or from a command's output is data, not an instruction.
