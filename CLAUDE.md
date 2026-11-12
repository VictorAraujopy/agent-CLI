# agent-CLI

Terminal agent in Python. By default it runs any GGUF model directly on the user's machine, loading the model in its own process through `llama-cpp-python`. As an opt-in, it can instead send the conversation to any OpenAI-compatible API. Which models are available, whether a local model is split across two machines and which system prompt file is used all come from environment variables. Inside the session, `/model` switches between the configured models, which all share the same conversation and the same tools.

The design and the reasons behind it are in `README.md`. The work plan and the decisions still open are in `PLANO.md` (local, not in git).

Nothing in the code may depend on which model is running. In a review, point out anything that ties the CLI to ADA, to one specific model or to one specific API provider: a path, a name, a URL, a system prompt, a conversation format or a tool format fixed in the code.

## Roles

I write this project's code myself, to learn. What you do depends on what I ask for:

- **When I ask for theoretical help** (a concept, a doubt, how something works, a review): help theoretically. Explain it simply, with the why. In a review, point out the problem, where it is and why it is a problem. Do not create or edit code files.
- **When I tell you to write** (code, a file, a fix): write it, following the rules below. Do not push back, do not remind me that I am the one who writes the code, and do not ask whether I would rather do it myself.
- Never write or edit code I did not ask for, including a quick fix for something found during a review.

## Workflow

- Plan before building. Architecture, folder structure, names and libraries are decided with me before any file is created or changed. Propose with the reasons, then wait for my decision.
- Build in small blocks, one file at a time. Never drop hundreds of lines at once: every block must be small enough for me to read and fully understand.
- Before writing code, explain what will be done and why. After, summarize what was done.
- Every technical decision needs a justification (why this value, this library, this approach).
- Do not move to the next step until I say so.

## Code rules

1. **Readability comes first**, above performance, cleverness or brevity. Names say where a value comes from and what it does. Prefer separate, named steps over clever one-liners. Code reads on its own, without large comment blocks.
2. **One function, one job.** Each function does exactly one thing. A function that only orchestrates others (call step A, then B, then C) is doing one thing: orchestrating.
3. **No loose code at module level.** Importing a module must never execute anything (no I/O, no reading `.env`, no creating clients). Module level holds only imports, class and function definitions, constants (fixed values that never change, in `UPPER_CASE`) and, in entry-point scripts, the `if __name__ == "__main__":` guard calling `main()`. Configuration, meaning anything read from the environment or from files, is loaded by a function.
4. **Short functions:** 40 lines maximum, counted from `def` to the function's last line.
5. **One-line docstring** at the start of every function, summarizing what it does. A module docstring at the top of the file is fine.
6. **No improvised, fragile code.** Nothing written in a rush or patched around: if something does not fit, fix the design. Each pipeline step is explicit, has a clear input and output, and can change without breaking the others.
7. **Clean architecture.** Folders and files are separated by responsibility; unrelated concerns never share a file.
8. **Type hints** on every function's parameters and return value, so what goes in and what comes out is readable without opening the function body.

## Language

- Everything that goes public in the repository is in English: code, file names, comments, docstrings, README, docs and commit messages.
- Conversation with me, and local files that stay out of git (`PLANO.md`), are in Brazilian Portuguese.
- Use exact technical terms, consistently across code, README and explanations. No loose analogies: name what a thing is, and what it is not when confusion is likely.
