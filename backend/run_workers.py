#!/usr/bin/env python3
"""
Unified Worker Manager
======================

Runs all workers as subprocesses in a single container.
Handles graceful shutdown, restart on failure, and logging.

Usage:
    python run_workers.py                    # Run all workers
    python run_workers.py --only api         # Just API server
    python run_workers.py --only extraction  # Just extraction workers
    python run_workers.py --only viz         # Just weave visualization
    python run_workers.py --workers 2        # 2 of each worker type
    python run_workers.py --no-viz           # Skip visualization server

Workers:
    - api: FastAPI server (uvicorn)
    - extraction: Page content extraction (N instances)
    - knowledge: Entity extraction + Wikidata linking (N instances)
    - weaver: Surface L2 identity clustering (replaces event worker)
    - inquiry: Inquiry resolution checker
    - viz: Weave topology D3 visualization (port 8080)
"""

import os
import sys
import signal
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import subprocess

# Load .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
log = logging.getLogger('worker-manager')


@dataclass
class WorkerConfig:
    name: str
    command: List[str]
    instances: int = 1
    restart_on_failure: bool = True
    env: Optional[Dict[str, str]] = None


class WorkerManager:
    """Manages multiple worker processes."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True

    def get_worker_configs(self, args) -> List[WorkerConfig]:
        """Build worker configs based on args."""
        configs = []

        # Check if we should skip API
        skip_api = getattr(args, 'no_api', False)

        # API server
        if not skip_api and args.only in (None, 'api', 'all'):
            configs.append(WorkerConfig(
                name='api',
                command=['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'],
                instances=1,
                restart_on_failure=True,
            ))

        # Extraction workers
        if args.only in (None, 'extraction', 'all'):
            for i in range(args.workers):
                configs.append(WorkerConfig(
                    name=f'extraction-{i+1}',
                    command=['python', 'run_extraction_worker.py'],
                    env={'WORKER_ID': str(i+1)},
                ))

        # Knowledge workers
        if args.only in (None, 'knowledge', 'all'):
            for i in range(args.workers):
                configs.append(WorkerConfig(
                    name=f'knowledge-{i+1}',
                    command=['python', 'run_knowledge_worker.py'],
                    env={'WORKER_NAME': f'knowledge-{i+1}'},
                ))

        # Principled Weaver (L2/L3/L4 topology with poll mode)
        if args.only in (None, 'weaver', 'all'):
            configs.append(WorkerConfig(
                name='weaver',
                command=['python', '-m', 'workers.principled_weaver', '--poll'],
            ))

        # Inquiry resolver (new)
        if args.only in (None, 'inquiry', 'all'):
            configs.append(WorkerConfig(
                name='inquiry',
                command=['python', 'run_inquiry_resolver.py'],
            ))

        # Weave visualization server (debug UI on port 8080)
        if args.only in (None, 'viz', 'all') and not getattr(args, 'no_viz', False):
            configs.append(WorkerConfig(
                name='weave-viz',
                command=['python', '-m', 'debug.weave_viz'],
                restart_on_failure=True,
            ))

        return configs

    def start_worker(self, config: WorkerConfig) -> Optional[subprocess.Popen]:
        """Start a single worker process."""
        env = os.environ.copy()
        if config.env:
            env.update(config.env)

        try:
            proc = subprocess.Popen(
                config.command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            log.info(f"Started {config.name} (PID {proc.pid})")
            return proc
        except Exception as e:
            log.error(f"Failed to start {config.name}: {e}")
            return None

    async def stream_output(self, name: str, proc: subprocess.Popen):
        """Stream process output with prefix."""
        try:
            for line in proc.stdout:
                print(f"[{name}] {line}", end='', flush=True)
        except:
            pass

    async def monitor_process(self, config: WorkerConfig, proc: subprocess.Popen):
        """Monitor a process and restart if needed."""
        name = config.name

        while self.running:
            # Check if process is still running
            ret = proc.poll()

            if ret is not None:
                log.warning(f"{name} exited with code {ret}")

                if config.restart_on_failure and self.running:
                    log.info(f"Restarting {name} in 5 seconds...")
                    await asyncio.sleep(5)

                    if self.running:
                        proc = self.start_worker(config)
                        if proc:
                            self.processes[name] = proc
                            asyncio.create_task(self.stream_output(name, proc))
                        else:
                            break
                else:
                    break

            await asyncio.sleep(1)

    def shutdown(self, signum=None, frame=None):
        """Gracefully shutdown all workers."""
        log.info("Shutting down all workers...")
        self.running = False

        for name, proc in self.processes.items():
            if proc.poll() is None:
                log.info(f"Stopping {name} (PID {proc.pid})")
                proc.terminate()

        # Wait for graceful shutdown
        for name, proc in self.processes.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning(f"Force killing {name}")
                proc.kill()

        log.info("All workers stopped")

    async def run(self, args):
        """Run all configured workers."""
        configs = self.get_worker_configs(args)

        if not configs:
            log.error("No workers configured")
            return

        log.info(f"Starting {len(configs)} worker(s)...")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

        # Start all workers
        tasks = []
        for config in configs:
            proc = self.start_worker(config)
            if proc:
                self.processes[config.name] = proc
                tasks.append(asyncio.create_task(self.stream_output(config.name, proc)))
                tasks.append(asyncio.create_task(self.monitor_process(config, proc)))

        if not self.processes:
            log.error("No workers started")
            return

        log.info(f"All workers running. Press Ctrl+C to stop.")

        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

        self.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Unified Worker Manager')
    parser.add_argument('--only', choices=['api', 'extraction', 'knowledge', 'weaver', 'inquiry', 'viz', 'all'],
                        help='Run only specific worker type')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of extraction/knowledge worker instances (default: 1)')
    parser.add_argument('--no-api', action='store_true',
                        help='Skip API server (run workers only)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip weave visualization server')
    args = parser.parse_args()

    manager = WorkerManager()
    asyncio.run(manager.run(args))


if __name__ == '__main__':
    main()
