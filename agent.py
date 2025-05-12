import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional
import yaml
import smtplib
from email.mime.text import MIMEText
from pydantic import Field
import requests  # For backend API calls

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, google, silero
from livekit.agents.voice.room_io import RoomInputOptions
from livekit_ext.vad import MultilingualModel

# Configure logging
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
    """Send an email notification with the visitor details."""
    msg = MIMEText(summary)
    msg["Subject"] = "Visitor Notification"
    msg["From"] = "doorman@building.local"
    msg["To"] = "admin@admin.admin"

    try:
        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


class DoormanAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a doorman assistant. Your task is to ask visitors for:\n"
                "- Apartment number\n- Resident name\n"
                "Check the backend if the resident exists.\n"
                "If the resident exists, proceed to:\n"
                "- Collect Visitor name\n- Reason for visit\n"
                "If not, end the conversation nicely."
            ),
            tools=[update_apartment, update_resident, check_resident, update_visitor, update_reason, confirm_visit],
            tts=google.TTS(),
        )

    async def on_enter(self):
        """Welcome the visitor and start the conversation."""
        logger.info("Doorman agent started")
        await self.say("Welcome. Please tell me the apartment number you are visiting.")


@function_tool()
async def update_apartment(
    apartment: Annotated[str, Field(description="The apartment number")],
    context: RunContext_T,
) -> str:
    context.userdata.apartment_number = apartment
    return f"Apartment number recorded as {apartment}. Please provide the resident's name."


@function_tool()
async def update_resident(
    name: Annotated[str, Field(description="The resident's name")],
    context: RunContext_T,
) -> str:
    context.userdata.resident_name = name
    return f"Resident name recorded as {name}. Checking if the resident exists."


@function_tool()
async def check_resident(context: RunContext_T) -> str:
    """Check the backend if the resident exists."""
    apartment = context.userdata.apartment_number
    resident = context.userdata.resident_name

    # Replace with the actual backend API endpoint
    backend_url = f"http://backend.local/api/residents?apartment={apartment}&name={resident}"

    try:
        response = requests.get(backend_url)
        response.raise_for_status()
        data = response.json()

        if data.get("exists"):
            return "Resident found. Please provide the visitor's name."
        else:
            return "Sorry, the resident does not exist. Have a good day!"
    except requests.RequestException as e:
        logger.error(f"Error checking resident: {e}")
        return "Sorry, there was an error checking the resident's details. Please try again later."


@function_tool()
async def update_visitor(
    name: Annotated[str, Field(description="The visitor's name")],
    context: RunContext_T,
) -> str:
    context.userdata.visitor_name = name
    return f"Visitor name recorded as {name}. Please provide the reason for the visit."


@function_tool()
async def update_reason(
    reason: Annotated[str, Field(description="The reason for the visit")],
    context: RunContext_T,
) -> str:
    context.userdata.visit_reason = reason
    return f"Reason for visit recorded as: {reason}."


@function_tool()
async def confirm_visit(context: RunContext_T) -> str:
    """Confirm the visit and send the details to the admin."""
    userdata = context.userdata
    if not all([userdata.apartment_number, userdata.resident_name, userdata.visitor_name, userdata.visit_reason]):
        return "Some information is still missing. Please provide all required details."

    summary = userdata.summarize()
    send_email(summary)
    userdata.confirmed = True
    return f"Thank you. The following information has been sent to admin:\n{summary}"


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent session."""
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
