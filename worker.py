# import os
# import redis
# from rq import Worker, Queue, Connection
# from timeouts import WindowsDeathPenalty
# from rq.worker import SimpleWorker
#
# class WindowsWorker(SimpleWorker):
#     death_penalty_class = WindowsDeathPenalty
#
# listen = ['default']
#
# redis_url = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
# # print('From worker - REDIS_URL = {redis_url}')
# conn = redis.from_url(redis_url)
#
# if __name__ == '__main__':
#     with Connection(conn):
#         worker = WindowsWorker(list(map(Queue, listen)))
#         worker.work()
#
#

# DEPLOYMENT
import os
import redis
from rq import Worker, Queue, Connection
from timeouts import WindowsDeathPenalty
from rq.worker import SimpleWorker
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

class WindowsWorker(SimpleWorker):
    death_penalty_class = WindowsDeathPenalty

listen = ['default']

redis_url = urlparse(os.getenv("REDISCLOUD_URL"))
conn = redis.Redis(host=redis_url.hostname, port=redis_url.port, password=redis_url.password, ssl=True, ssl_cert_reqs=None) #redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = WindowsWorker(list(map(Queue, listen)))
        worker.work()

