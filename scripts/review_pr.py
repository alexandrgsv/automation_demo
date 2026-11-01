"""Build a Markdown pull-request review from the deterministic audit tools.

The script reads ``origin/<base>...HEAD``, runs the same pipeline as
``POST /api/v1/pr-review``, and writes ``pr_review_comment.md`` for the
workflow comment step.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_SERVICE_ROOT: Path = _REPO_ROOT / "agent-service"
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from app.schemas import PRReviewInput, PRReviewResponse
from app.tools.pr_review_tool import audit_code_changes

_COMMENT_PATH: Path = _REPO_ROOT / "pr_review_comment.md"
_DIFF_LIMIT: int = 10000


def get_git_diff(base_ref: str) -> str:
    """Return the triple-dot diff from the base branch to ``HEAD``.

    Parameters
    ----------
    base_ref : str
        Base branch name without the ``origin/`` prefix.

    Returns
    -------
    str
        Unified diff text. Empty when git cannot produce a diff.
    """
    try:
        raw: bytes = subprocess.check_output(
            ["git", "diff", f"origin/{base_ref}...HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        detail: str = exc.output.decode("utf-8", errors="replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"git diff failed: {detail}", file=sys.stderr)
        return ""
    return raw.decode("utf-8", errors="replace")


def render_comment(report: PRReviewResponse) -> str:
    """Format one audit as the pull-request comment body.

    Parameters
    ----------
    report : PRReviewResponse
        Pipeline result for this diff.

    Returns
    -------
    str
        Markdown comment. A clean diff uses a green mark. Any finding uses red.
    """
    review = report.review
    status_icon: str = "🟢" if review.can_auto_merge else "🔴"
    if review.detected_issues:
        issues_md: str = "\n".join(f"- ⚠️ {issue}" for issue in review.detected_issues)
    else:
        issues_md = "- None (Clean diff)"
    if review.suggested_tests:
        tests_md: str = "\n".join(f"- `pytest {name}`" for name in review.suggested_tests)
    else:
        tests_md = "- None"
    return f"""### {status_icon} AI Code Review & Triage (PR #{report.pr_id})

**Risk Level:** `{review.risk_level}` | **Estimated Time Saved:** `{review.review_time_saved_minutes} min`

**Recommendation:** `{review.recommendation}`

#### Detected Issues:
{issues_md}

#### Suggested Test Stubs:
{tests_md}

*Automated check by Agent Service via GitHub Actions.*
"""


def write_comment(body: str) -> None:
    """Write the comment body where the workflow expects it.

    Parameters
    ----------
    body : str
        Markdown to publish on the pull request.
    """
    _COMMENT_PATH.write_text(body, encoding="utf-8")


def main() -> None:
    """Read the pull-request diff and write the review comment.

    An empty diff still writes a short comment and exits 0. A diff that the
    audit refuses to auto-merge exits 1 after the comment file is written, so
    the GitHub check fails while the comment step can still publish it.
    """
    pr_id: int = int(os.getenv("PR_NUMBER", "1"))
    base_ref: str = os.getenv("BASE_REF", "main")
    diff: str = get_git_diff(base_ref)
    if not diff.strip():
        print("Diff is empty. Skipping review.")
        write_comment(f"### AI Code Review & Triage (PR #{pr_id})\n\nDiff is empty. No findings.\n")
        return

    payload: PRReviewInput = PRReviewInput(
        pr_id=pr_id,
        file_path="git-diff",
        diff_content=diff[:_DIFF_LIMIT],
    )
    report: PRReviewResponse = audit_code_changes(payload)
    write_comment(render_comment(report))
    print(f"Wrote {_COMMENT_PATH}")
    if not report.review.can_auto_merge:
        print(
            f"Review requested changes (risk {report.review.risk_level}).",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
