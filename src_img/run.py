from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm.auto import tqdm

from .pipeline.main import PicSJTAgent


class SJTRunner:
    """Utility wrapper orchestrating ``PicSJTAgent`` executions for multiple tasks."""

    def __init__(
        self,
        tasks: Optional[Iterable[dict[str, Any]]] = None,
        ref_name: Optional[str] = None,
        ref_img: Optional[Image.Image] = None,
        model: str = "gpt-5",
        output_dir: str = "outputs",
    ) -> None:
        self.tasks = list(tasks) if tasks is not None else None
        self.ref_name = ref_name
        self.ref_img = ref_img
        self.model = model
        self.output_dir = output_dir

    @staticmethod
    def _task_label(task: dict[str, Any], index: int) -> str:
        return str(task.get("fname") or task.get("id") or index)

    def _task_output_targets(
        self,
        task: dict[str, Any],
        index: int,
        resolved_output_dir: str,
    ) -> tuple[Path, str]:
        label = self._task_label(task, index)
        task_output_dir = Path(task.get("output_dir", resolved_output_dir))
        output_fname = task.get("fname", label)
        return task_output_dir, output_fname

    def _has_existing_output(
        self,
        task: dict[str, Any],
        index: int,
        resolved_output_dir: str,
    ) -> bool:
        output_dir, output_fname = self._task_output_targets(task, index, resolved_output_dir)
        json_path = output_dir / f"{output_fname}.json"
        img_dir = output_dir / f"{output_fname}_imgs"
        return json_path.exists() and img_dir.exists() and any(img_dir.iterdir())

    def _resolve_inputs(
        self,
        tasks: Optional[Iterable[dict[str, Any]]],
        ref_name: Optional[str],
        ref_img: Optional[Image.Image],
        model: Optional[str],
        output_dir: Optional[str],
    ) -> tuple[list[dict[str, Any]], str, Image.Image, str, str]:
        resolved_tasks = list(tasks) if tasks is not None else (self.tasks or [])
        if not resolved_tasks:
            raise ValueError("tasks must be provided either during initialization or at call time")

        resolved_ref_name = ref_name if ref_name is not None else self.ref_name
        if resolved_ref_name is None:
            raise ValueError("ref_name must be provided either during initialization or at call time")

        resolved_ref_img = ref_img if ref_img is not None else self.ref_img
        if resolved_ref_img is None:
            raise ValueError("ref_img must be provided either during initialization or at call time")

        resolved_model = model if model is not None else self.model
        resolved_output_dir = output_dir if output_dir is not None else self.output_dir

        return resolved_tasks, resolved_ref_name, resolved_ref_img, resolved_model, resolved_output_dir

    def _create_agent(
        self,
        task: dict[str, Any],
        index: int,
        ref_name: str,
        ref_img: Image.Image,
        model: str,
        output_dir: str,
        agent_overrides: Optional[dict[str, Any]] = None,
    ) -> PicSJTAgent:
        agent_kwargs: dict[str, Any] = {
            "model": task.get("model", model),
            "situ": task["situ"],
            "ref_name": task.get("ref_name", ref_name),
            "ref_viz": task.get("ref_img", ref_img),
            "trait": task["trait"],
            "output_dir": task.get("output_dir", output_dir),
            "output_fname": task.get("fname", self._task_label(task, index)),
        }
        if agent_overrides:
            agent_kwargs.update(agent_overrides)
        agent_kwargs.update(task.get("agent_kwargs", {}))
        return PicSJTAgent(**agent_kwargs)

    @staticmethod
    def _merge_run_kwargs(
        base_kwargs: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        merged = base_kwargs.copy()
        merged.update(task.get("run_kwargs", {}))
        return merged

    def cook(
        self,
        tasks: Optional[Iterable[dict[str, Any]]] = None,
        ref_name: Optional[str] = None,
        ref_img: Optional[Image.Image] = None,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        *,
        verbose: bool = False,
        verbose_leave: Optional[bool] = None,
        save: Optional[bool] = None,
        agent_overrides: Optional[dict[str, Any]] = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        retry_backoff: float = 1.5,
        continue_on_error: bool = True,
        error_callback: Optional[Callable[[str, Exception, int], None]] = None,
        skip_generated: bool = False,
    ) -> dict[str, Any]:
        tasks_list, resolved_ref_name, resolved_ref_img, resolved_model, resolved_output_dir = self._resolve_inputs(
            tasks,
            ref_name,
            ref_img,
            model,
            output_dir,
        )

        base_run_kwargs: dict[str, Any] = {"verbose": verbose}
        if verbose_leave is not None:
            base_run_kwargs["verbose_leave"] = verbose_leave
        if save is not None:
            base_run_kwargs["save"] = save

        results: dict[str, Any] = {}
        pbar = tqdm(tasks_list, desc="Cooking tasks")
        for index, task in enumerate(pbar):
            label = self._task_label(task, index)

            if skip_generated and self._has_existing_output(task, index, resolved_output_dir):
                out_dir_path, fname = self._task_output_targets(task, index, resolved_output_dir)
                results[label] = {
                    "__skipped__": {
                        "reason": "existing_output",
                        "output_dir": str(out_dir_path),
                        "output_fname": fname,
                    }
                }
                pbar.set_postfix_str(f"Skipped {label}")
                continue

            agent = self._create_agent(
                task,
                index,
                resolved_ref_name,
                resolved_ref_img,
                resolved_model,
                resolved_output_dir,
                agent_overrides=agent_overrides,
            )
            run_kwargs = self._merge_run_kwargs(base_run_kwargs, task)

            attempt = 0
            current_delay = retry_delay
            while True:
                try:
                    results[label] = agent.run(**run_kwargs)
                except Exception as exc:  # pragma: no cover - pass task context upward
                    attempt += 1
                    if error_callback:
                        error_callback(label, exc, attempt)
                    if attempt > max_retries:
                        if continue_on_error:
                            results[label] = {
                                "__error__": {
                                    "message": str(exc),
                                    "exception_type": exc.__class__.__name__,
                                    "attempts": attempt,
                                }
                            }
                            pbar.set_postfix_str(f"Failed {label}")
                            break
                        raise RuntimeError(f"Failed to process task '{label}'") from exc
                    if current_delay > 0:
                        time.sleep(current_delay)
                        current_delay *= retry_backoff if retry_backoff is not None else 1.0
                    continue
                else:
                    pbar.set_postfix_str(f"Done {label}")
                    break

        return results

    async def _cook_async(
        self,
        tasks: Optional[Iterable[dict[str, Any]]] = None,
        ref_name: Optional[str] = None,
        ref_img: Optional[Image.Image] = None,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        *,
        verbose: bool = False,
        verbose_leave: Optional[bool] = None,
        save: Optional[bool] = None,
        agent_overrides: Optional[dict[str, Any]] = None,
        concurrency: Optional[int] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        retry_backoff: float = 1.5,
        continue_on_error: bool = True,
        error_callback: Optional[Callable[[str, Exception, int], None]] = None,
        skip_generated: bool = False,
    ) -> dict[str, Any]:
        (
            tasks_list,
            resolved_ref_name,
            resolved_ref_img,
            resolved_model,
            resolved_output_dir,
        ) = self._resolve_inputs(tasks, ref_name, ref_img, model, output_dir)

        total_tasks = len(tasks_list)
        if total_tasks == 0:
            return {}

        limit = total_tasks if concurrency is None else max(1, min(concurrency, total_tasks))
        semaphore = asyncio.Semaphore(limit)

        base_run_kwargs: dict[str, Any] = {"verbose": verbose}
        if verbose_leave is not None:
            base_run_kwargs["verbose_leave"] = verbose_leave
        if save is not None:
            base_run_kwargs["save"] = save

        results: dict[str, Any] = {self._task_label(task, idx): None for idx, task in enumerate(tasks_list)}
        skipped = 0
        completed = 0

        async def process(index: int, task: dict[str, Any]) -> tuple[str, Any]:
            label = self._task_label(task, index)

            # Create agent outside semaphore (lightweight operation)
            agent = self._create_agent(
                task,
                index,
                resolved_ref_name,
                resolved_ref_img,
                resolved_model,
                resolved_output_dir,
                agent_overrides=agent_overrides,
            )
            run_kwargs = self._merge_run_kwargs(base_run_kwargs, task)

            def run_agent() -> Any:
                return agent.run(**run_kwargs)

            # Only acquire semaphore for the actual agent execution
            async with semaphore:
                attempt = 0
                current_delay = retry_delay
                while True:
                    try:
                        result = await asyncio.to_thread(run_agent)
                    except Exception as exc:  # pragma: no cover - provide context
                        attempt += 1
                        if error_callback:
                            error_callback(label, exc, attempt)
                        if attempt > max_retries:
                            if continue_on_error:
                                result = {
                                    "__error__": {
                                        "message": str(exc),
                                        "exception_type": exc.__class__.__name__,
                                        "attempts": attempt,
                                    }
                                }
                                break
                            raise RuntimeError(f"Failed to process task '{label}'") from exc
                        if current_delay > 0:
                            await asyncio.sleep(current_delay)
                            current_delay *= retry_backoff if retry_backoff is not None else 1.0
                        continue
                    else:
                        break

            return label, result

        async_jobs: list[asyncio.Task[tuple[str, Any]]] = []
        for index, task in enumerate(tasks_list):
            label = self._task_label(task, index)
            if skip_generated and self._has_existing_output(task, index, resolved_output_dir):
                skipped += 1
                out_dir_path, fname = self._task_output_targets(task, index, resolved_output_dir)
                results[label] = {
                    "__skipped__": {
                        "reason": "existing_output",
                        "output_dir": str(out_dir_path),
                        "output_fname": fname,
                    }
                }
                if progress_callback:
                    progress_callback(skipped, total_tasks, f"跳过任务 {label}")
                continue
            async_jobs.append(asyncio.create_task(process(index, task)))

        if not async_jobs:
            return results

        pending = asyncio.as_completed(async_jobs)
        if show_progress:
            pending = tqdm(pending, total=total_tasks, initial=skipped, desc="Cooking tasks")

        try:
            for future in pending:
                label, outcome = await future
                results[label] = outcome
                completed += 1
                if progress_callback:
                    status = "失败" if isinstance(outcome, dict) and "__error__" in outcome else "完成"
                    progress_callback(skipped + completed, total_tasks, f"{status}任务 {label}")
        except Exception:
            for task in async_jobs:
                task.cancel()
            await asyncio.gather(*async_jobs, return_exceptions=True)
            raise

        return results

    def cook_async(
        self,
        tasks: Optional[Iterable[dict[str, Any]]] = None,
        ref_name: Optional[str] = None,
        ref_img: Optional[Image.Image] = None,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        async def _runner() -> dict[str, Any]:
            return await self._cook_async(
                tasks=tasks,
                ref_name=ref_name,
                ref_img=ref_img,
                model=model,
                output_dir=output_dir,
                **kwargs,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(_runner())
