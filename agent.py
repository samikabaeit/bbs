import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
import smtplib
from email.mime.text import MIMEText
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, google, silero
from livekit.agents.voice.room_io import RoomInputOptions
from livekit_ext.vad import MultilingualModel

logger = logging.getLogger("doorman-ai")
logger.setLevel(logging.INFO)

@dataclass
class VisitorData:
    apartment_number: Optional[str] = None
    resident_name: Optional[str] = None
    visitor_name: Optional[str] = None
    visit_reason: Optional[str] = None
    confirmed: bool = False

    def summarize(self) -> str:
        return yaml.dump({
            "apartment_number": self.apartment_number or "unknown",
            "resident_name": self.resident_name or "unknown",
            "visitor_name": self.visitor_name or "unknown",
            "visit_reason": self.visit_reason or "unknown",
        })

RunContext_T = RunContext[VisitorData]

def send_email(summary: str):
    msg = MIMEText(summary)
    msg["Subject"] = "Visitor Notification"
    msg["From"] = "doorman@building.local"
    msg["To"] = "admin@admin.admin"

    with smtplib.SMTP("localhost") as server:
        server.send_message(msg)

class DoormanAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a doorman assistant. Your task is to ask visitors for:\n"
                "- Apartment number\n- Resident name\n- Visitor name\n- Reason for visit\n"
                "After collecting, confirm and send to the admin via email."
            ),
            tools=[update_apartment, update_resident, update_visitor, update_reason, confirm_visit],
            tts=google.TTS(),
        )

    async def on_enter(self):
        logger.info("Doorman agent started")
        await self.say("Welcome. Please tell me the apartment number you are visiting.")

@function_tool()
async def update_apartment(
    apartment: Annotated[str, Field(description="The apartment number")],
    context: RunContext_T,
) -> str:
    context.userdata.apartment_number = apartment
    return f"Apartment number recorded as {apartment}."

@function_tool()
async def update_resident(
    name: Annotated[str, Field(description="The resident's name")],
    context: RunContext_T,
) -> str:
    context.userdata.resident_name = name
    return f"Resident name recorded as {name}."

@function_tool()
async def update_visitor(
    name: Annotated[str, Field(description="The visitor's name")],
    context: RunContext_T,
) -> str:
    context.userdata.visitor_name = name
    return f"Visitor name recorded as {name}."

@function_tool()
async def update_reason(
    reason: Annotated[str, Field(description="The reason for the visit")],
    context: RunContext_T,
) -> str:
    context.userdata.visit_reason = reason
    return f"Reason for visit recorded as: {reason}."

@function_tool()
async def confirm_visit(context: RunContext_T) -> str:
    userdata = context.userdata
    if not all([userdata.apartment_number, userdata.resident_name, userdata.visitor_name, userdata.visit_reason]):
        return "Some information is still missing. Please provide all required details."

    summary = userdata.summarize()
    send_email(summary)
    userdata.confirmed = True
    return f"Thank you. The following information has been sent to admin:\n{summary}"

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[VisitorData](
        userdata=VisitorData(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            temperature=0.5,
            instructions="Translate user input into Hindi."
        ),
        tts=google.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=DoormanAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
