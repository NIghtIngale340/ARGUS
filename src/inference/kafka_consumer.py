import os

from faust import App


app = App(
    "argus-consumer",
    broker=os.getenv("KAFKA_BOOTSTRAP", "kafka://kafka:9092"),
    store=os.getenv("REDIS_URL", "redis://redis:6379/0"),
)

raw_logs = app.topic("argus.raw-logs", value_serializer="json")


@app.agent(raw_logs)
async def consume(stream):
    async for event in stream:
        yield event