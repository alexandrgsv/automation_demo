# Computer Vision Team Skills

Shared English-language workflows for computer vision engineering. These skills supply reusable methods; each project's AGENTS.md supplies its concrete architecture, commands, data contracts, and acceptance requirements.

| Skill | Purpose |
| --- | --- |
| `cv-plan` | Turn a request into an evidence-based implementation and validation plan. |
| `cv-analyze` | Investigate quality, data, pipeline, and performance failures. |
| `cv-develop` | Implement changes and prepare an independent validation handoff. |
| `cv-test` | Coordinate a separate testing agent and close the defect loop. |

## Platform compatibility

These skills use the open [Agent Skills](https://agentskills.io/home) format: each folder contains a `SKILL.md` with `name`, `description`, and Markdown instructions. They can be reused on platforms that support this format. The table lists documented integrations, not a claim that this repository has been tested on every platform or an exhaustive list of all compatible products.

| Platform | Project installation directory | Personal installation directory | Documentation |
| --- | --- | --- | --- |
| OpenAI Codex | `.agents/skills/` | `~/.agents/skills/` | [Skills](https://learn.chatgpt.com/docs/build-skills) |
| Cursor (cursor.ai / cursor.com) | `.cursor/skills/` or `.agents/skills/` | `~/.cursor/skills/` or `~/.agents/skills/` | [Agent Skills](https://cursor.com/docs/skills) |
| VS Code + GitHub Copilot | `.github/skills/` | `~/.copilot/skills/` | [Agent Skills in VS Code](https://code.visualstudio.com/docs/agent-customization/agent-skills) |
| GitHub Copilot CLI and Copilot app | `.github/skills/` | `~/.copilot/skills/` | [Copilot skills](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills) |
| GitHub Copilot cloud agent and code review | `.github/skills/` committed to the repository | Use repository skills | [Copilot skills](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills) |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` | [Extend Claude with skills](https://code.claude.com/docs/en/skills) |
| Gemini CLI | `.gemini/skills/` or `.agents/skills/` | `~/.gemini/skills/` or `~/.agents/skills/` | [Managing Agent Skills](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/using-agent-skills.md) |
| OpenCode | `.opencode/skills/` or `.agents/skills/` | `~/.config/opencode/skills/` or `~/.agents/skills/` | [Agent Skills](https://docs.opencode.ai/docs/skills/) |
| Roo Code in VS Code | `.roo/skills/` | `~/.roo/skills/` | [Skills](https://docs.roocode.com/features/skills) |
| Claude web / desktop and Cowork | Upload each skill through the product's custom skills UI | Managed in your Claude account | [Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude) |
| Other agents implementing Agent Skills | Follow the platform's skill discovery or import instructions | Platform-specific | [Agent Skills specification](https://agentskills.io/specification) |

Copy or link each complete `cv-*` folder into the selected directory, preserving paths such as `.cursor/skills/cv-plan/SKILL.md`. For upload-based products, package and import each skill as required by that product. Simply cloning this repository does not install its nested skill folders. VS Code support depends on the selected AI extension or agent; the editor alone does not execute skills.

Format compatibility does not guarantee the same tools or results. Code changes and evaluation require access to the target repository, shell, dependencies, and data. In hosted chat products, only the workflow steps supported by the available environment can run. `cv-test` and the independent validation step in `cv-develop` require separate-agent tooling; without it, report independent validation as unavailable and identify any local checks separately. Read-only code review agents cannot perform the full implementation workflow.

Discovery paths, enabled features, and invocation syntax can vary by product version and organization policy. Start a fresh chat, confirm that the four skills are discovered, and try a small task before adopting them for a team. Installation makes skills available; it does not guarantee automatic selection.

## Install locally

Clone this repository to a stable location. On Linux or macOS, run the following from the repository root to link each skill into Codex's user-level discovery directory:

```bash
mkdir -p "$HOME/.agents/skills"
for skill in "$PWD"/cv-team-skills/skills/*; do
  ln -s "$skill" "$HOME/.agents/skills/$(basename "$skill")"
done
```

Existing destinations are deliberately not overwritten. Inspect a name collision before replacing anything. Keep the clone at the same path while using the links. On systems without symlink support, copy the four skill folders into the user skill directory and replace those copies when updating.

Codex detects skill changes automatically; restart if they do not appear. Do not install team skills into Codex's bundled `.system` directory.

## Use

The following examples use Codex invocation syntax. In Cursor and Claude Code, use slash commands such as `/cv-plan`; in VS Code, select the skill from the chat `/` menu when available. On other platforms, use their skill picker or explicitly ask the agent to load the named skill.

Explicit requests make the intended workflow clear:

```text
Use $cv-plan to plan adding instance segmentation to this inference service.
Use $cv-analyze to investigate false alarms after the camera configuration change.
Use $cv-develop to implement the agreed preprocessing fix.
Use $cv-test to validate the current changes with a separate testing agent.
```

Automatic selection is also allowed when a skill's description matches the task. The testing workflow requires separate-agent tooling; otherwise it reports that independent validation is unavailable and distinguishes local checks from an independent pass. The tester does not modify implementation files.

## Project policy

Adapt [AGENTS.md.example](AGENTS.md.example) and place it at the target project's root as `AGENTS.md`. Merge with an existing policy rather than overwriting it. Fill in actual project commands, contracts, artifact locations, and acceptance criteria. The example is intentionally not named `AGENTS.md` so it does not impose CV project rules on this repository.

On platforms that do not load `AGENTS.md`, adapt the policy into their supported project instruction mechanism (for example, `CLAUDE.md`, `GEMINI.md`, Cursor rules, or Copilot repository instructions). Ensure the agent reads it before starting work.

Agree on a model and configure it through your chosen agent or platform setup. Mentioning a model in a policy file does not change the runtime model. Local instructions, available tools, and model choice can still affect results; required checks belong in CI.

## Updates and ownership

Maintain these files through the team's normal code-review process and nominate an owner. For a reproducible rollout, select an agreed repository commit or release tag and have each developer check out that revision in their clone. Linked skills then use that checkout. If following a branch, fetch and fast-forward explicitly; local clones do not update themselves.

Record the selected revision in team onboarding or project documentation. Start a fresh session after changing a project policy to ensure its instruction chain is rebuilt. Do not assume an installed copy tracks Git changes automatically.

Before publishing changes, validate each skill's frontmatter with Codex's skill-creator validator, inspect the workflow boundaries, and try representative tasks. Structural validation alone does not establish model behavior or CV correctness.
