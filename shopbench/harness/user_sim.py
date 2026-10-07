"""Scripted user the agent can talk to through the ask_user and request_confirmation tools.

The simulated user is deliberately simple and deterministic:
  * The first question on an ambiguous task gets the task's clarification.
  * Other questions get the task's `user_reply`, or "please follow my original instructions".
  * Purchase confirmations are approved unless the total the agent states exceeds
    the user's budget, or the task expects no purchase (the user then gives `user_reply`).
The user never reveals the correct product, so asking can't be used to look up the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_REPLY = "I don't have anything to add. Please follow my original instructions."


@dataclass
class SimUser:
    task: dict
    questions: int = 0
    clarified: bool = False
    log: list[dict] = field(default_factory=list)

    def answer(self, question: str) -> str:
        self.questions += 1
        if self.task.get("clarification") and not self.clarified:
            self.clarified = True
            reply = self.task["clarification"]
        else:
            reply = self.task.get("user_reply") or DEFAULT_REPLY
        self.log.append({"question": question, "reply": reply})
        return reply

    def confirm(self, summary: str, total: float) -> tuple[bool, str]:
        budget = self.task.get("budget")
        if budget is not None and total > budget + 0.005:
            return False, f"No. ${total:.2f} is over my ${budget:.2f} budget. Do not place this order."
        if self.task["expect"] != "purchase":
            return False, self.task.get("user_reply") or "No, please don't place an order."
        return True, "Yes, that looks right. Go ahead and place the order."
