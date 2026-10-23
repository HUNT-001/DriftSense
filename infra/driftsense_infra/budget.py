"""Budget stack: monthly COST budget with $40/$80 actual and 100% forecast alerts."""

from __future__ import annotations

from aws_cdk import Annotations, Stack
from aws_cdk import aws_budgets as budgets
from constructs import Construct


def _notification(email: str, *, kind: str, threshold: float, threshold_type: str):
    return budgets.CfnBudget.NotificationWithSubscribersProperty(
        notification=budgets.CfnBudget.NotificationProperty(
            comparison_operator="GREATER_THAN",
            notification_type=kind,
            threshold=threshold,
            threshold_type=threshold_type,
        ),
        subscribers=[budgets.CfnBudget.SubscriberProperty(address=email, subscription_type="EMAIL")],
    )


class BudgetStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        budget_usd: float = 100.0,
        alert_email: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        notifications = None
        if alert_email:
            notifications = [
                _notification(alert_email, kind="ACTUAL", threshold=40, threshold_type="ABSOLUTE_VALUE"),
                _notification(alert_email, kind="ACTUAL", threshold=80, threshold_type="ABSOLUTE_VALUE"),
                _notification(alert_email, kind="FORECASTED", threshold=100, threshold_type="PERCENTAGE"),
            ]
        else:
            Annotations.of(self).add_warning_v2(
                "driftsense:budget-no-email",
                "No 'alert_email' context given: the monthly budget is created WITHOUT notifications. "
                "Re-synth with -c alert_email=<address> to get the $40/$80/forecast alerts.",
            )

        self.budget = budgets.CfnBudget(
            self,
            "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="driftsense-monthly",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=budget_usd, unit="USD"),
            ),
            notifications_with_subscribers=notifications,
        )
