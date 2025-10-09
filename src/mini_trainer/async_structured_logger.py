# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
import asyncio
import json
import threading
import torch.distributed as dist

# Third Party
import aiofiles
from rich.console import Console

# Local imports
from mini_trainer import wandb_wrapper
from mini_trainer.wandb_wrapper import check_wandb_available



class AsyncStructuredLogger:
    def __init__(self, file_name="training_log.jsonl", use_wandb=False):
        self.file_name = file_name
        
        # wandb init is a special case -- if it is requested but unavailable,
        # we should error out early
        if use_wandb:
            check_wandb_available("initialize wandb")
        self.use_wandb = use_wandb

        # Rich console for prettier output (force_terminal=True works with subprocess streaming)
        self.console = Console(force_terminal=True, force_interactive=False)

        self.logs = []
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=self._run_event_loop, args=(self.loop,), daemon=True
        )
        t.start()
        asyncio.run_coroutine_threadsafe(self._initialize_log_file(), self.loop)

    def _run_event_loop(self, loop):
        asyncio.set_event_loop(loop)  #
        loop.run_forever()

    async def _initialize_log_file(self):
        self.logs = []
        try:
            async with aiofiles.open(self.file_name, "r") as f:
                async for line in f:
                    if line.strip():  # Avoid empty lines
                        self.logs.append(json.loads(line.strip()))
        except FileNotFoundError:
            # File does not exist but the first log will create it.
            pass

    async def log(self, data):
        """logs a dictionary as a new line in a jsonl file with a timestamp"""
        try:
            if not isinstance(data, dict):
                raise ValueError("Logged data must be a dictionary")
            data["timestamp"] = datetime.now().isoformat()
            self.logs.append(data)
            await self._write_logs_to_file(data)
            
            # log to wandb if enabled and wandb is initialized, but only log this on the MAIN rank
            # wandb already handles timestamps so no need to include
            if self.use_wandb and dist.get_rank() == 0:
                wandb_data = {k: v for k, v in data.items() if k != "timestamp"}
                wandb_wrapper.log(wandb_data)
        except Exception as e:
            print(f"\033[1;38;2;0;255;255mError logging data: {e}\033[0m")

    async def _write_logs_to_file(self, data):
        """appends to the log instead of writing the whole log each time"""
        async with aiofiles.open(self.file_name, "a") as f:
            await f.write(json.dumps(data, indent=None) + "\n")

    def _create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create a rich-style progress bar that works with subprocess streaming.
        
        Args:
            current: Current step number
            total: Total steps
            width: Width of the progress bar in characters
            
        Returns:
            String representation of progress bar with Rich markup
        """
        filled = int(width * current / total)
        # Use Rich color markup for a prettier bar
        bar = '━' * filled + '╺' if filled < width else '━' * width
        empty = '─' * (width - filled - (1 if filled < width else 0))
        return f"[cyan]{bar}[/cyan][dim]{empty}[/dim]"
    
    def log_sync(self, data: dict):
        """runs the log coroutine non-blocking
        
        Args:
            data: Dictionary of metrics to log. Will automatically print a Rich-styled
                  progress bar if step and steps_per_epoch are present.
        """
        if not isinstance(data, dict):
            raise ValueError("Logged data must be a dictionary")

        # Print to console synchronously, but only on rank 0
        # to avoid duplicate outputs in distributed training
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            data_with_timestamp = {**data, "timestamp": datetime.now().isoformat()}
            
            # Print the JSON using Rich for syntax highlighting
            self.console.print_json(json.dumps(data_with_timestamp))
            
            # Print a rich-styled progress bar after the JSON (prints as new line each time)
            # This works correctly with subprocess streaming
            if 'step' in data and 'steps_per_epoch' in data and 'epoch' in data:
                current_step_in_epoch = (data['step'] - 1) % data['steps_per_epoch'] + 1
                progress_pct = current_step_in_epoch / data['steps_per_epoch'] * 100
                bar = self._create_progress_bar(current_step_in_epoch, data['steps_per_epoch'])
                
                # Format like tqdm with Rich markup: Epoch 1: [━━━━━━━╺────] 85% │ 164/192 │ loss: 1.36 │ lr: 2.0e-05 │ 40706 tok/s
                progress_line = (
                    f"[bold blue]Epoch {data['epoch'] + 1}:[/bold blue] {bar} "
                    f"[yellow]{progress_pct:3.0f}%[/yellow] │ "
                    f"[white]{current_step_in_epoch}/{data['steps_per_epoch']}[/white] │ "
                    f"[green]loss:[/green] [white]{data['loss']:.4f}[/white] │ "
                    f"[green]lr:[/green] [white]{data['lr']:.2e}[/white] │ "
                    f"[magenta]{data['tokens_per_second']:.0f}[/magenta] [dim]tok/s[/dim]"
                )
                self.console.print(progress_line)

        # Run async logging for file and wandb
        asyncio.run_coroutine_threadsafe(self.log(data), self.loop)

    def __repr__(self):
        return f"<AsyncStructuredLogger(file_name={self.file_name})>"
