"""
MCP Server for Health Agent
===========================
This file defines the tools that the health agent can call AFTER triage.

MCP (Model Context Protocol) is a standard that lets AI agents call external
tools in a structured way. This server exposes healthcare-related actions
that only trigger after the agent has made a triage decision.

To run this server:
    python mcp_server.py

The agent will connect to this server and call tools like:
- log_triage_result: Save triage outcome to records
- send_urgent_alert: Notify urgent care (for urgent cases)
- schedule_followup: Book a follow-up appointment
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime
import json

# Initialize the MCP server with a name
# This name identifies your server to clients
mcp = FastMCP("health-agent-tools")


# =============================================================================
# TOOL 1: Log Triage Result
# =============================================================================
# This tool saves the triage outcome. In production, this would write to a
# database or EHR system. For MVP, we write to a local JSON file.

@mcp.tool()
def log_triage_result(
    patient_message: str,
    intake_data: dict,
    triage_decision: str,
    is_urgent: bool,
    ai_response: str
) -> dict:
    """
    Log the triage result to persistent storage.

    Args:
        patient_message: The original health concern from the patient
        intake_data: Collected intake information (symptoms, severity, etc.)
        triage_decision: The final triage category (urgent/non-urgent)
        is_urgent: Whether immediate care is needed
        ai_response: The response given to the patient

    Returns:
        Confirmation with log ID and timestamp
    """
    # Create a log entry
    log_entry = {
        "id": f"TRI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "patient_message": patient_message,
        "intake_data": intake_data,
        "triage_decision": triage_decision,
        "is_urgent": is_urgent,
        "ai_response": ai_response[:500],  # Truncate for storage
    }

    # In production: write to database/EHR
    # For MVP: append to a local JSON log file
    log_file = "triage_logs.json"
    try:
        with open(log_file, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[MCP] Logged triage result: {log_entry['id']}")

    return {
        "success": True,
        "log_id": log_entry["id"],
        "timestamp": log_entry["timestamp"],
        "message": "Triage result logged successfully"
    }


# =============================================================================
# TOOL 2: Send Urgent Alert
# =============================================================================
# This tool triggers when a patient needs urgent care. In production, this
# would send a real alert (SMS, page a nurse, notify ER). For MVP, we simulate.

@mcp.tool()
def send_urgent_alert(
    patient_message: str,
    intake_summary: str,
    urgency_reason: str
) -> dict:
    """
    Send an alert for urgent cases requiring immediate medical attention.
    Only called when triage determines urgent care is needed.

    Args:
        patient_message: The original health concern
        intake_summary: Summary of collected symptoms
        urgency_reason: Why this was flagged as urgent (e.g., "neurological symptoms")

    Returns:
        Confirmation that alert was sent
    """
    alert = {
        "alert_id": f"URG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "type": "URGENT_TRIAGE",
        "patient_concern": patient_message,
        "intake_summary": intake_summary,
        "urgency_reason": urgency_reason,
    }

    # In production: Send to urgent care system, page nurse, SMS, etc.
    # For MVP: Log to console and file
    print(f"\n{'='*60}")
    print(f"[MCP] URGENT ALERT TRIGGERED")
    print(f"Alert ID: {alert['alert_id']}")
    print(f"Reason: {urgency_reason}")
    print(f"{'='*60}\n")

    # Save alert to file
    alert_file = "urgent_alerts.json"
    try:
        with open(alert_file, "r") as f:
            alerts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        alerts = []

    alerts.append(alert)

    with open(alert_file, "w") as f:
        json.dump(alerts, f, indent=2)

    return {
        "success": True,
        "alert_id": alert["alert_id"],
        "message": "Urgent alert sent to care team"
    }


# =============================================================================
# TOOL 3: Schedule Follow-up
# =============================================================================
# For non-urgent cases, offer to schedule a follow-up appointment.

@mcp.tool()
def schedule_followup(
    patient_message: str,
    recommended_timeframe: str,
    visit_type: str
) -> dict:
    """
    Schedule a follow-up appointment for non-urgent cases.

    Args:
        patient_message: The original health concern
        recommended_timeframe: When to follow up (e.g., "within 1 week")
        visit_type: Type of visit (e.g., "primary_care", "specialist")

    Returns:
        Confirmation with appointment details
    """
    appointment = {
        "appointment_id": f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "created_at": datetime.now().isoformat(),
        "patient_concern": patient_message,
        "recommended_timeframe": recommended_timeframe,
        "visit_type": visit_type,
        "status": "pending_confirmation"
    }

    # In production: Integrate with scheduling system
    # For MVP: Log to file
    appt_file = "appointments.json"
    try:
        with open(appt_file, "r") as f:
            appointments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        appointments = []

    appointments.append(appointment)

    with open(appt_file, "w") as f:
        json.dump(appointments, f, indent=2)

    print(f"[MCP] Follow-up scheduled: {appointment['appointment_id']}")

    return {
        "success": True,
        "appointment_id": appointment["appointment_id"],
        "message": f"Follow-up {visit_type} visit recommended {recommended_timeframe}"
    }


# =============================================================================
# Run the server
# =============================================================================
if __name__ == "__main__":
    print("Starting Health Agent MCP Server...")
    print("Tools available: log_triage_result, send_urgent_alert, schedule_followup")
    print("-" * 50)
    # This runs the server using stdio transport (standard for MCP)
    mcp.run()