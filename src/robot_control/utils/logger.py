def robot_says_phase(phase: str):
    """Utility function to print the current phase of the robot in a visually distinct way."""

    hashtags = "#" * (len(phase) + 10)

    print("\n" f"{hashtags}\n" f"#    {phase}    #\n" f"{hashtags}")


def robot_says(message: str):
    """Utility function to print a message from the robot in a visually distinct way."""

    print(f"\n" f"{message}\n")
