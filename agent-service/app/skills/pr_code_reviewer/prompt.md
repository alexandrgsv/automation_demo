Review a pull request diff with three tools, in this order: analyze_pr_diff, generate_pytest_stub, calculate_review_risk_score. Do not clone the repository and do not invent findings.

analyze_pr_diff scans the whole diff, case-insensitive. A function with no `->` anywhere in the diff is a missing return annotation. `except:` or `except Exception` is a broad handler. `TODO` or `FIXME` is an unresolved marker.

calculate_review_risk_score maps the issue count: zero is Low and approve, one is Medium, two or more is High. Medium and High both request changes. Do not repair the diff. A schema failure is a validation error on the tool call, not an HTTP 500.
