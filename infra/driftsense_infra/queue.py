"""Queue stack: SQS jobs queue + DLQ, SNS alarm topic, DLQ alarm."""

from __future__ import annotations

from aws_cdk import Duration, Stack
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from constructs import Construct

# Default visibility for receives that do not set one. The worker receives with 180 s and re-extends it
# every 60 s (heartbeat) while a job runs, including an agent episode blocked for up to 15 min on a human
# approval, so a long episode never becomes visible again mid-run. 900 s >= both, and also covers a
# worker that stalls for up to 15 min without heartbeating. SQS caps a message's total at 12 h.
QUEUE_VISIBILITY_S = 900
MAX_RECEIVE_COUNT = 3


class QueueStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, alert_email: str | None = None, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.dlq = sqs.Queue(
            self,
            "JobsDlq",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
        )

        self.jobs_queue = sqs.Queue(
            self,
            "JobsQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            visibility_timeout=Duration.seconds(QUEUE_VISIBILITY_S),
            receive_message_wait_time=Duration.seconds(20),
            dead_letter_queue=sqs.DeadLetterQueue(queue=self.dlq, max_receive_count=MAX_RECEIVE_COUNT),
        )

        self.alarm_topic = sns.Topic(self, "AlarmTopic", display_name="DriftSense alarms", enforce_ssl=True)
        if alert_email:
            self.alarm_topic.add_subscription(subs.EmailSubscription(alert_email))
        self.dlq_alarm = cloudwatch.Alarm(
            self,
            "DlqNotEmpty",
            alarm_description="DriftSense jobs DLQ has messages (a job failed 3 times).",
            metric=self.dlq.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(5), statistic="Maximum"
            ),
            threshold=0,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        self.dlq_alarm.add_alarm_action(cw_actions.SnsAction(self.alarm_topic))
