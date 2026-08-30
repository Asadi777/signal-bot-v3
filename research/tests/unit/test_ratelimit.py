from prepump.net.ratelimit import TokenBucket


def test_bucket_waits_once_capacity_is_spent():
    now = [0.0]
    slept = []

    def sleeper(seconds):
        slept.append(seconds)
        now[0] += seconds

    bucket = TokenBucket(rate_per_second=2.0, capacity=1.0, clock=lambda: now[0], sleeper=sleeper)
    assert bucket.acquire() == 0.0  # first token is free
    waited = bucket.acquire()
    assert waited > 0 and slept
    assert abs(waited - 0.5) < 1e-9  # 2 rps means half a second per token


def test_wait_is_accumulated_for_the_manifest():
    now = [0.0]

    def sleeper(seconds):
        now[0] += seconds

    bucket = TokenBucket(rate_per_second=1.0, clock=lambda: now[0], sleeper=sleeper)
    bucket.acquire()
    bucket.acquire()
    bucket.acquire()
    assert bucket.total_wait_seconds > 0
