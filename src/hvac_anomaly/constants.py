"""Column names used by the RTU experiment."""

TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Fault Detection Ground Truth"
ROOM_IDS = ("102", "103", "104", "105", "106", "202", "203", "204", "205", "206")
TARGET_COLUMNS = tuple(f"Terminal: Room {room} Air Temperature" for room in ROOM_IDS)

