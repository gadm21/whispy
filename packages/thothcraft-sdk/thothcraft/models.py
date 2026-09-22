"""Models uploaded to Brain and pull-based device deployments."""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from .errors import NotFoundError

if TYPE_CHECKING:
    from .client import Client


class Model:
    def __init__(self, client: Client, info: dict[str, Any]) -> None:
        self._client = client
        self.info = info

    @property
    def id(self) -> int:
        return int(self.info['id'])

    @property
    def name(self) -> str:
        return str(self.info.get('name', self.id))

    def deploy(self, device_id: str, config: dict[str, Any] | None = None) -> Deployment:
        return self._client.deploy_model(self.id, device_id, config)

    def __repr__(self) -> str:
        return f'Model(id={self.id}, name={self.name!r})'


class Deployment:
    def __init__(self, client: Client, info: dict[str, Any]) -> None:
        self._client = client
        self.info = info

    @property
    def id(self) -> str:
        return str(self.info['deployment_id'])

    @property
    def status(self) -> str:
        return str(self.info.get('status', 'pending'))

    def refresh(self) -> Deployment:
        for deployment in self._client.deployments():
            if deployment.id == self.id:
                self.info = deployment.info
                return self
        raise NotFoundError(f'Deployment {self.id} no longer exists')

    def wait(self, timeout: float = 180, poll_s: float = 2) -> Deployment:
        """Poll for delivered/declined; raise TimeoutError when the deadline expires.

        A declined deployment is returned with ``status == 'declined'``.
        The HTTP request timeout can extend the deadline by one request.
        """
        if timeout < 0 or poll_s <= 0:
            raise ValueError('timeout must be nonnegative and poll_s positive')
        deadline = time.monotonic() + timeout
        while self.status not in {'delivered', 'declined', 'failed', 'cancelled'}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Deployment {self.id} still {self.status}')
            self.refresh()
            if self.status in {'delivered', 'declined', 'failed', 'cancelled'}:
                break
            time.sleep(min(poll_s, max(0, deadline - time.monotonic())))
        return self

    def cancel(self) -> dict[str, Any]:
        return self._client.cancel_deployment(self.id)

    def set_active(self, enabled: bool) -> dict[str, Any]:
        return self._client.set_deployment_active(self.id, enabled)

    def __repr__(self) -> str:
        return f'Deployment(id={self.id!r}, status={self.status!r})'
