# ANSI helpers for terminals; optional HTML mapping for dashboards (e.g. Gradio).

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"

RESET = "\033[0m"

# Map ANSI prefix pairs -> hex colours for HTML rendering
mapper = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00dd00",
    BG_BLACK + YELLOW: "#dddd00",
    BG_BLACK + BLUE: "#0000ee",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00dddd",
    BG_BLACK + WHITE: "#87CEEB",
    BG_BLUE + WHITE: "#ff7800",
    BG_GREEN + WHITE: "#00cc66",
    BG_RED + WHITE: "#ff4444",
    BG_MAGENTA + WHITE: "#cc44ff",
    BG_CYAN + WHITE: "#00cccc",
}


def reformat(message: str) -> str:
    """Convert ANSI escape codes to HTML spans for log rendering in browsers."""
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    message = message.replace(RESET, "</span>")
    return message
