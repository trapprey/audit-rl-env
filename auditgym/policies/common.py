from ..reconcile import Books, finding


def act(name, **args):
    return {"name": name, "args": args}


class TableReader:
    def __init__(self, tables):
        self.queue = list(tables)
        self.cursor = None
        self.data = {t: [] for t in tables}
        self.pending = None

    @property
    def done(self):
        return not self.queue

    def next_action(self):
        self.pending = self.queue[0]
        return act("list_records", table=self.pending, cursor=self.cursor, limit=50)

    def absorb(self, observation):
        if self.pending is None:
            return
        table, self.pending = self.pending, None
        result = observation.get("result")
        if not isinstance(result, dict) or "records" not in result:
            self.queue.pop(0)
            self.cursor = None
            return
        self.data[table].extend(result["records"])
        self.cursor = result["next_cursor"]
        if self.cursor is None:
            self.queue.pop(0)
