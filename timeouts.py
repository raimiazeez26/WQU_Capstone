from rq.timeouts import BaseDeathPenalty, JobTimeoutException


class WindowsDeathPenalty(BaseDeathPenalty):
    def setup_death_penalty(self):
        pass

    def cancel_death_penalty(self):
        pass

    def handle_death_penalty(self, *args):
        raise JobTimeoutException('Job exceeded maximum timeout value')
