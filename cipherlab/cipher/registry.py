from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .components_builtin import Component, builtins


class ComponentRegistry:
    def __init__(self):
        self._components: Dict[str, Component] = builtins()

    def get(self, component_id: str) -> Component:
        if component_id not in self._components:
            raise KeyError(f"Unknown component_id: {component_id}")
        return self._components[component_id]

    def list(self) -> List[Component]:
        return list(self._components.values())

    def list_by_kind(self, kind: str, *, arch: Optional[str] = None) -> List[Component]:
        kind = kind.upper()
        out = [c for c in self._components.values() if c.kind == kind]
        if arch:
            arch = arch.upper()
            out = [c for c in out if arch in c.compatible_arch]
        out.sort(key=lambda c: c.component_id)
        return out

    def exists(self, component_id: str) -> bool:
        return component_id in self._components
