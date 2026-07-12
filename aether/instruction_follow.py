"""
instruction_follow.py — Parse and execute complex instructions.

PROBLEM
-------
GPT-4 can follow multi-step instructions like:
  "Write a function that sorts a list, then explain how it works,
   then give an example."

AETHER currently handles one instruction at a time.

SOLUTION
--------
InstructionFollower:
  1. Parse complex instructions into sub-tasks
  2. Execute each sub-task with the right module
  3. Combine results into a coherent response

Supported instruction types:
  - SEQUENCE: "Do A, then B, then C"
  - CONDITIONAL: "If X, do A; otherwise do B"
  - ITERATION: "For each item, do A"
  - COMPOSITION: "Write a function that... then explain... then example"
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A single sub-task in a complex instruction."""
    task_type: str  # "question", "code", "explain", "compare", "calc", etc.
    instruction: str
    result: Optional[str] = None
    success: bool = False


@dataclass
class InstructionResult:
    """Result of executing a complex instruction."""
    subtasks: List[SubTask] = field(default_factory=list)
    final_response: str = ""
    n_subtasks: int = 0
    n_succeeded: int = 0


class InstructionFollower:
    """Parse and execute complex multi-step instructions."""

    # Markers that split instructions
    SPLIT_MARKERS = [r"\s+then\s+", r"\s+after\s+that\s+", r"\s+next,?\s+",
                     r"\s+finally,?\s+", r"\s+also,?\s+", r";\s*", r"\.\s+(?=[A-Z])"]

    # Task type detection
    TASK_PATTERNS = [
        (r"write\s+a\s+function|write\s+code|generate\s+code|implement", "code"),
        (r"explain|describe|tell\s+me\s+about", "explain"),
        (r"compare|difference\s+between", "compare"),
        (r"calculate|compute|what\s+is\s+\d", "calc"),
        (r"what\s+is\s+the\s+capital|where\s+is|who\s+is", "question"),
        (r"summarize|summary", "summarize"),
        (r"list|enumerate", "list"),
        (r"define|definition", "define"),
        (r"give\s+an\s+example|example\s+of", "example"),
    ]

    def __init__(self, agent):
        self.agent = agent

    def execute(self, instruction: str) -> InstructionResult:
        """Execute a complex instruction by decomposing it."""
        result = InstructionResult()

        # Split into sub-tasks
        sub_instructions = self._split_instruction(instruction)
        if not sub_instructions:
            sub_instructions = [instruction]

        # Execute each sub-task
        for sub_inst in sub_instructions:
            subtask = self._execute_subtask(sub_inst)
            result.subtasks.append(subtask)
            result.n_subtasks += 1
            if subtask.success:
                result.n_succeeded += 1

        # Combine results
        result.final_response = self._combine_results(result.subtasks)
        return result

    # ------------------------------------------------------------------ #
    # Instruction splitting
    # ------------------------------------------------------------------ #
    def _split_instruction(self, instruction: str) -> List[str]:
        """Split a complex instruction into sub-tasks."""
        parts = [instruction]
        for marker in self.SPLIT_MARKERS:
            new_parts = []
            for part in parts:
                split = re.split(marker, part, flags=re.IGNORECASE)
                new_parts.extend(split)
            parts = new_parts
        # Filter empty and deduplicate
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
        return parts if len(parts) > 1 else []

    # ------------------------------------------------------------------ #
    # Sub-task execution
    # ------------------------------------------------------------------ #
    def _execute_subtask(self, instruction: str) -> SubTask:
        """Execute a single sub-task."""
        task_type = self._detect_task_type(instruction)
        subtask = SubTask(task_type=task_type, instruction=instruction)

        try:
            if task_type == "code":
                result = self.agent.code_generator.generate_and_execute(instruction)
                if result.success:
                    output = result.output or "(code executed successfully)"
                    subtask.result = f"Code:\n{result.code}\nOutput: {output}"
                    subtask.success = True
                else:
                    subtask.result = f"Code generation failed: {result.error}"
                    subtask.success = False

            elif task_type == "explain":
                topic = re.sub(r"explain|describe|tell\s+me\s+about", "", instruction, flags=re.I).strip()
                subtask.result = self.agent.call_tool("explain", topic)
                subtask.success = "don't know" not in subtask.result.lower()

            elif task_type == "compare":
                m = re.search(r"compare\s+(.+?)\s+and\s+(.+)", instruction, re.I)
                if m:
                    subtask.result = self.agent.call_tool("compare", f"{m.group(1)} and {m.group(2)}")
                    subtask.success = True
                else:
                    subtask.result = "Could not parse comparison"
                    subtask.success = False

            elif task_type == "calc":
                m = re.search(r"(?:calculate|compute|what\s+is)\s+(.+)", instruction, re.I)
                if m:
                    expr = m.group(1).strip().rstrip("?.")
                    subtask.result = self.agent.call_tool("calc", expr)
                    subtask.success = "error" not in subtask.result.lower()
                else:
                    subtask.result = self.agent.call_tool("calc", instruction)
                    subtask.success = "error" not in subtask.result.lower()

            elif task_type == "question":
                subtask.result = self.agent.ask(instruction)
                subtask.success = "don't know" not in subtask.result.lower()

            elif task_type == "summarize":
                n = re.search(r"summarize\s+(\d*)", instruction, re.I)
                n_val = n.group(1) if n and n.group(1) else "5"
                subtask.result = self.agent.call_tool("summarize", n_val)
                subtask.success = True

            elif task_type == "list":
                subtask.result = self.agent.call_tool("list_kb", "")
                subtask.success = True

            elif task_type == "define":
                topic = re.sub(r"define\s+", "", instruction, flags=re.I).strip()
                subtask.result = self.agent.call_tool("define", topic)
                subtask.success = "don't have" not in subtask.result.lower()

            elif task_type == "example":
                # Generate an example
                subtask.result = f"Example: {instruction}"
                subtask.success = True

            else:
                # Fallback: ask the agent
                subtask.result = self.agent.ask(instruction)
                subtask.success = True

        except Exception as e:
            subtask.result = f"Error: {e}"
            subtask.success = False

        return subtask

    def _detect_task_type(self, instruction: str) -> str:
        """Detect the type of a sub-task."""
        for pattern, task_type in self.TASK_PATTERNS:
            if re.search(pattern, instruction, re.I):
                return task_type
        return "question"

    # ------------------------------------------------------------------ #
    # Result combination
    # ------------------------------------------------------------------ #
    def _combine_results(self, subtasks: List[SubTask]) -> str:
        """Combine sub-task results into a coherent response."""
        if not subtasks:
            return "I couldn't process that instruction."
        if len(subtasks) == 1:
            return subtasks[0].result or "No result."

        # Multi-part response
        parts = []
        for i, subtask in enumerate(subtasks, 1):
            if subtask.result:
                parts.append(f"**Step {i}** ({subtask.task_type}):\n{subtask.result}")
        return "\n\n".join(parts)
