import os
import asyncio
import json
from azure.iot.hub import IoTHubRegistryManager
from azure.eventhub.aio import EventHubConsumerClient
import time

class Request:

    def __init__(self, device: str, cmd: str, parameters: dict):
        self.device = device
        self.cmd = cmd
        self.parameters = parameters

class Response:

    def __init__(self, task: str, completion: float, status: dict):
        self.task = task
        self.completion = completion
        self.status = status


class EdgeBridge:

    def __init__(self, credentials):
        # convert date-day-time to requestID
        self.requestId = int(time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.resp = None

        #IOT_HUB_CONNECTION_STR = os.getenv("IOT_HUB_CONNECTION_STRING")
        IOT_HUB_CONNECTION_STR = credentials["iot"]["IOT_HUB_CONNECTION_STRING"]
        self.registry_manager = IoTHubRegistryManager(IOT_HUB_CONNECTION_STR)
        print(IOT_HUB_CONNECTION_STR)
        #EVENT_HUB_CONNECTION_STR = os.getenv("EVENT_HUB_CONNECTION_STRING")
        EVENT_HUB_CONNECTION_STR = credentials["iot"]["EVENT_HUB_CONNECTION_STRING"]
        print(EVENT_HUB_CONNECTION_STR)
        #EVENT_HUB_NAME = os.getenv("EVENT_HUB_NAME")
        EVENT_HUB_NAME = credentials["iot"]["EVENT_HUB_NAME"]
        print(EVENT_HUB_NAME)
        self.resp_handler = EventHubConsumerClient.from_connection_string(
            EVENT_HUB_CONNECTION_STR,
            consumer_group="$Default",
            eventhub_name=EVENT_HUB_NAME
        )

    async def on_event(self, partition_context, event):
        print("timestamp: ", event.enqueued_time)
        msg = json.loads(event.body_as_str(encoding="UTF-8"))
        if msg["responseId"] == self.requestId:
            self.resp = Response(msg["task"], msg["completion"], msg["status"])
            print("resp: ", self.resp.status)
            await partition_context.update_checkpoint(event)

    async def receive_from_hub(self):
        async with self.resp_handler:
            await self.resp_handler.receive(on_event=self.on_event, starting_position="@latest", starting_position_inclusive=True)

    async def wait_results(self):
        recv_task = asyncio.ensure_future(self.receive_from_hub())
        while True:
            await asyncio.sleep(.02)
            if self.resp is not None:
                recv_task.cancel()
                break

    async def run(self, req: Request) -> Response:
        self.resp = None

        props = {}
        props.update(contentType = "application/json")

        data = {
            "requestId": self.requestId,
            "command": req.cmd,
            "parameters": req.parameters
        }
        self.registry_manager.send_c2d_message(req.device, json.dumps(data), properties=props)
        await self.wait_results()

        return self.resp