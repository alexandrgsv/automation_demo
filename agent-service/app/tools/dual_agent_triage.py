"""Compatibility import for the developer-versus-QA loop.

The skill lives in ``app.skills.adversarial_review``.
"""

from app.skills.adversarial_review.logic import run_adversarial_dev_loop

__all__ = ["run_adversarial_dev_loop"]
