from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from threading import Thread, Lock
from time import sleep

class LivePanel:
    def __init__(self, title="Live Messages", refresh_per_second=4):
        self.messages = []
        self.title = title
        self._lock = Lock()
        self._live = Live(self._render_panel(), refresh_per_second=refresh_per_second, screen=False)
        self._running = False

    def _render_panel(self):
        with self._lock:
            return Panel(Text("\n".join(self.messages), justify="left"), title=self.title)

    def add_message(self, message: str):
        with self._lock:
            self.messages.append(message)
            self._live.update(self._render_panel())

    def start(self):
        self._running = True
        self._live.start()

    def stop(self):
        self._running = False
        self._live.stop()

# Example usage
if __name__ == "__main__":
    panel = LivePanel()
    panel.start()

    def simulate_messages():
        for i in range(1, 11):
            panel.add_message(f"Message {i}")
            sleep(1)
        panel.stop()

    Thread(target=simulate_messages).start()

