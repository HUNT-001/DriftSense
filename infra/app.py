#!/usr/bin/env python3
"""DriftSense Pilot CDK app entry point (``cdk.json`` runs ``python app.py``)."""

from driftsense_infra.build import build_app

if __name__ == "__main__":
    build_app().app.synth()
