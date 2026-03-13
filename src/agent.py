import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import sys
import traceback
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.tasks import TaskUpdater
from a2a.types import (DataPart, FilePart, FileWithBytes, Message, Part, Role,
                       Task, TaskState, TextPart)
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../osworld"))
import lib_run_single
from desktop_env.desktop_env import DesktopEnv

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../osworld/evaluation_examples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")


class A2AClientAgent:
    """
    Drop-in replacement for PromptAgent that delegates to a remote A2A server.

    predict sends a message with parts:
      - TextPart: the instruction
      - FilePart: screenshot (image/png, base64)
      - TextPart: accessibility tree (if present)

    The server responds with an artifact containing:
      - TextPart: raw LLM response text
      - DataPart: {"actions": list[str]}
    """

    def __init__(self, url: str):
        self._url = url
        self._context_id: str | None = None

    def reset(self, _logger: Any = None, vm_ip: Any = None, **kwargs: Any) -> None:
        # Generate a fresh context_id so the server treats this as a new task
        self._context_id = uuid4().hex

    def predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        return asyncio.run(self._async_predict(instruction, obs))

    async def _async_predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, (bytes, bytearray)):
            screenshot = base64.b64encode(screenshot).decode("ascii")

        parts: list[Part] = [Part(root=TextPart(text=instruction))]
        if screenshot:
            parts.append(Part(root=FilePart(
                file=FileWithBytes(bytes=screenshot, mime_type="image/png"),
            )))
        a11y_tree = obs.get("accessibility_tree")
        if a11y_tree:
            parts.append(Part(root=DataPart(data={"accessibility_tree": a11y_tree})))
        terminal = obs.get("terminal")
        if terminal:
            parts.append(Part(root=DataPart(data={"terminal": terminal})))

        msg = Message(
            kind="message",
            role=Role.user,
            parts=parts,
            message_id=uuid4().hex,
            context_id=self._context_id,
        )
        async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self._url)
            card = await resolver.get_agent_card()
            client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(card)
            last_event: tuple[Task, Any] | Message | None = None
            async for event in client.send_message(msg):
                last_event = event
            assert last_event is not None

        task: Task | Message = last_event[0] if isinstance(last_event, tuple) else last_event

        parts_out: list[Part] = []
        if isinstance(task, Task) and task.artifacts:
            for artifact in task.artifacts:
                parts_out.extend(artifact.parts)
        elif isinstance(task, Message):
            parts_out = task.parts

        llm_response = ""
        actions: list[str] = []
        for part in parts_out:
            if isinstance(part.root, TextPart):
                llm_response = part.root.text
            elif isinstance(part.root, DataPart):
                actions = part.root.data.get("actions", [])

        return llm_response, actions


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = ["agent"]
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Replace example code below with your agent logic
        # Use request.participants to get participant agent URLs by role
        # Use request.config for assessment parameters
        # Defaults and logic from osworld/scripts/python/run_multienv.py
        action_space = request.config.get("action_space", "pyautogui")
        observation_type = request.config.get("observation_type", "screenshot")
        require_a11y_tree = observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"]
        sleep_after_execution = request.config.get("sleep_after_execution", 0.0)
        max_steps = request.config.get("max_steps", 15)
        test_all_meta_name = request.config.get("test_all_meta_name", "test_nogdrive")
        test_all_meta_name_path = os.path.join(EXAMPLES_DIR, f"{test_all_meta_name}.json")
        num_workers = request.config.get("num_workers", 3)

        agent_url = str(request.participants["agent"])

        args = argparse.Namespace(
            sleep_after_execution=sleep_after_execution,
            result_dir=RESULTS_DIR,
        )

        await updater.update_status(
            TaskState.working, new_agent_text_message("Running evaluation...")
        )

        with open(test_all_meta_name_path, "r", encoding="utf-8") as f:
            tasks: dict[str, list[str]] = json.load(f)

        all_examples: list[tuple[str, str]] = [
            (domain, eid)
            for domain, eids in tasks.items()
            for eid in eids
        ]
        total = len(all_examples)
        loop = asyncio.get_event_loop()

        work_queue: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        for item in all_examples:
            work_queue.put(item)

        results: list[tuple[str, str, float]] = []

        def run_worker() -> None:
            agent = A2AClientAgent(url=agent_url)
            # Hardcoded env config for our container setup
            env = DesktopEnv(
                provider_name="qemu",
                action_space=action_space,
                headless=True,
                require_a11y_tree=require_a11y_tree,
            )
            try:
                while True:
                    try:
                        domain, example_id = work_queue.get_nowait()
                    except queue.Empty:
                        break
                    config_file = os.path.join(EXAMPLES_DIR, f"examples/{domain}/{example_id}.json")
                    with open(config_file, "r", encoding="utf-8") as f:
                        example = json.load(f)
                    example_result_dir = os.path.join(RESULTS_DIR, domain, example_id)
                    os.makedirs(example_result_dir, exist_ok=True)
                    scores: list[float] = []
                    try:
                        lib_run_single.run_single_example(
                            agent,
                            env,
                            example,
                            max_steps,
                            example["instruction"],
                            args,
                            example_result_dir,
                            scores,
                        )
                    except Exception as e:
                        print(f"Error in {domain}/{example_id}:")
                        traceback.print_exc(file=sys.stdout)
                        asyncio.run_coroutine_threadsafe(
                            updater.update_status(TaskState.working, new_agent_text_message(f"Error in {domain}/{example_id}: {e}")),
                            loop,
                        ).result()
                    score = scores[0] if scores else 0.0

                    async def record() -> None:
                        results.append((domain, example_id, score))
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(f"[{len(results)}/{total}] {domain}/{example_id}: {score:.2f}"),
                        )

                    asyncio.run_coroutine_threadsafe(record(), loop).result()
            finally:
                env.close()

        await asyncio.gather(*[asyncio.to_thread(run_worker) for _ in range(num_workers)])

        domain_scores: dict[str, list[float]] = {}
        for domain, example_id, score in results:
            domain_scores.setdefault(domain, []).append(score)

        all_scores = [s for ss in domain_scores.values() for s in ss]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        per_domain = {d: sum(ss) / len(ss) if ss else 0.0 for d, ss in domain_scores.items()}

        await updater.add_artifact(
            parts=[
                Part(root=DataPart(data={
                    # structured assessment results
                    "overall": overall,
                    "per_domain": per_domain,
                }))
            ],
            name="Result",
        )
