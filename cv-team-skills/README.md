# Computer Vision Team Skills

Shared English-language workflows for computer vision engineering. These skills supply reusable methods; each project's AGENTS.md supplies its concrete architecture, commands, data contracts, and acceptance requirements.

| Skill | Purpose |
| --- | --- |
| `cv-plan` | Turn a request into an evidence-based implementation and validation plan. |
| `cv-analyze` | Investigate quality, data, pipeline, and performance failures. |
| `cv-develop` | Implement changes and prepare an independent validation handoff. |
| `cv-test` | Coordinate a separate testing agent and close the defect loop. |

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

Agree on a model and configure it through your Codex setup. Mentioning a model in a policy file does not change the runtime model. Local instructions, available tools, and model choice can still affect results; required checks belong in CI.

## Updates and ownership

Maintain these files through the team's normal code-review process and nominate an owner. For a reproducible rollout, select an agreed repository commit or release tag and have each developer check out that revision in their clone. Linked skills then use that checkout. If following a branch, fetch and fast-forward explicitly; local clones do not update themselves.

Record the selected revision in team onboarding or project documentation. Start a fresh session after changing a project policy to ensure its instruction chain is rebuilt. Do not assume an installed copy tracks Git changes automatically.

Before publishing changes, validate each skill's frontmatter with Codex's skill-creator validator, inspect the workflow boundaries, and try representative tasks. Structural validation alone does not establish model behavior or CV correctness.
