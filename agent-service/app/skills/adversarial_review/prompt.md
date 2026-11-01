Run the developer-versus-QA loop with execute_in_sandbox. The task sentence is only a label. The developer snippets and the QA script are fixed in the skill. Do not execute caller-supplied source.

The first snippet divides by duration and fails with ZeroDivisionError. The second snippet raises ValueError when duration is not positive or the frame count is negative, and the same QA script passes. The sandbox namespace has no import and no open.
