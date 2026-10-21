"""Assemble the DriftSense CDK app (shared by app.py and tests/infra)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from .api import ApiStack
from .aspects import Arm64Lambdas
from .bench import BenchStack
from .budget import BudgetStack
from .config import PREFIX, REGION, Settings
from .nag import ScopedNagPlugin
from .network import NetworkStack
from .observability import ObservabilityStack
from .oidc import GithubOidcStack
from .queue import QueueStack
from .storage import StorageStack
from .worker import WorkerStack


@dataclass
class DriftSenseApp:
    app: cdk.App
    network: NetworkStack
    storage: StorageStack
    queue: QueueStack
    api: ApiStack
    budget: BudgetStack
    observability: ObservabilityStack
    oidc: GithubOidcStack
    worker: WorkerStack | None = None
    bench: BenchStack | None = None

    def stacks(self) -> list[cdk.Stack]:
        return [c for c in self.app.node.children if isinstance(c, cdk.Stack)]


def build_app(context: dict | None = None, outdir: str | None = None) -> DriftSenseApp:
    app = cdk.App(context=context or {}, outdir=outdir)
    settings = Settings.from_node(app.node)
    account = os.environ.get("CDK_DEFAULT_ACCOUNT")
    if account:
        # Pre-seed the AZ context lookup (what cdk.context.json would cache) so a synth with a concrete
        # account never needs AWS credentials; the VPC uses exactly these zones.
        key = f"availability-zones:account={account}:region={REGION}"
        if app.node.try_get_context(key) is None:
            app.node.set_context(key, list(settings.azs))

    env = cdk.Environment(account=account, region=REGION)

    network = NetworkStack(
        app,
        f"{PREFIX}-Network",
        env=env,
        interface_endpoints=settings.interface_endpoints,
        availability_zones=settings.azs,
        description="DriftSense: public-only VPC, no NAT, gateway endpoints, worker SG",
    )
    storage = StorageStack(
        app,
        f"{PREFIX}-Storage",
        env=env,
        description="DriftSense: UI bucket + CloudFront, data bucket, jobs table",
    )
    queue = QueueStack(
        app,
        f"{PREFIX}-Queue",
        env=env,
        alert_email=settings.alert_email,
        description="DriftSense: jobs queue + DLQ, alarm topic",
    )
    api = ApiStack(
        app,
        f"{PREFIX}-Api",
        env=env,
        table=storage.table,
        jobs_queue=queue.jobs_queue,
        data_bucket=storage.data_bucket,
        ui_origin=storage.ui_origin,
        engine=settings.engine,
        engine_git_sha=settings.engine_git_sha,
        job_ttl_days=settings.job_ttl_days,
        approval_timeout_s=settings.approval_timeout_s,
        log_level=settings.log_level,
        api_asset=settings.api_asset,
        description="DriftSense: HTTP API + arm64 Lambda",
    )
    budget = BudgetStack(
        app,
        f"{PREFIX}-Budget",
        env=env,
        budget_usd=settings.budget_usd,
        alert_email=settings.alert_email,
        description="DriftSense: monthly cost budget",
    )

    worker = None
    bench = None
    if settings.cool_ami_id:
        worker = WorkerStack(
            app,
            f"{PREFIX}-Worker",
            env=env,
            settings=settings,
            vpc=network.vpc,
            worker_sg=network.worker_sg,
            table=storage.table,
            jobs_queue=queue.jobs_queue,
            data_bucket=storage.data_bucket,
            description=f"DriftSense: COOL worker ASG (AMI {settings.cool_ami_id})",
        )
    else:
        _warn(
            network,
            "driftsense:no-cool-ami",
            "No 'cool_ami_id' context: the Worker and Bench stacks are NOT synthesized. "
            "Pass -c cool_ami_id=ami-... (COOL 3.1 AMI from the Marketplace subscription).",
        )

    observability = ObservabilityStack(
        app,
        f"{PREFIX}-Observability",
        env=env,
        http_api=api.http_api,
        api_url=api.api_url,
        api_function=api.function,
        jobs_queue=queue.jobs_queue,
        dlq=queue.dlq,
        alarm_topic=queue.alarm_topic,
        worker_asg=worker.asg if worker else None,
        description="DriftSense: dashboard, 30-min canary, alarms",
    )
    oidc = GithubOidcStack(
        app,
        f"{PREFIX}-GithubOidc",
        env=env,
        github_repo=settings.github_repo,
        existing_provider_arn=settings.github_oidc_provider_arn,
        description="DriftSense: GitHub Actions OIDC deploy role (main only)",
    )

    if settings.bench:
        if not settings.cool_ami_id:
            _warn(network, "driftsense:bench-no-ami", "bench=true ignored: it needs -c cool_ami_id=ami-...")
        else:
            run_id = settings.bench_run_id
            if not run_id:
                run_id = "adhoc"
                _warn(
                    network, "driftsense:bench-no-run-id", "bench=true without bench_run_id: using 'adhoc'."
                )
            bench = BenchStack(
                app,
                f"{PREFIX}-Bench",
                env=env,
                settings=settings,
                vpc=network.vpc,
                worker_sg=network.worker_sg,
                data_bucket=storage.data_bucket,
                run_id=run_id,
                description=f"DriftSense: ephemeral COOL benchmark run {run_id} (self-terminating)",
            )

    cdk.Tags.of(app).add("project", "driftsense")
    cdk.Tags.of(app).add("owner", "B")
    cdk.Aspects.of(app).add(Arm64Lambdas())
    # cdk-nag 3.x: a policy-validation plugin; synth fails on any unacknowledged finding.
    cdk.Validations.of(app).add_plugins(ScopedNagPlugin(AwsSolutionsChecks(app, verbose=True), app))

    return DriftSenseApp(
        app=app,
        network=network,
        storage=storage,
        queue=queue,
        api=api,
        budget=budget,
        observability=observability,
        oidc=oidc,
        worker=worker,
        bench=bench,
    )


def _warn(scope: cdk.Stack, warning_id: str, message: str) -> None:
    cdk.Annotations.of(scope).add_warning_v2(warning_id, message)
    print(f"WARNING: {message}", file=sys.stderr)
