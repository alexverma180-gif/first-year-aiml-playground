
import subprocess
import sys
from pathlib import Path
import time
import signal

def profile_app():
    """
    Profiles the Streamlit app using cProfile and saves the output to a .pstats file.
    """
    # Define the output file for the profiling stats
    output_file = "profile_output.pstats"

    # Construct the command to run the profiler
    command = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        output_file,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
    ]

    try:
        print(f"Running profiler with command: {' '.join(command)}")
        # Use Popen to run the streamlit app in a new process
        process = subprocess.Popen(command, cwd="first-year-aiml-playground", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for a few seconds to let the app initialize and run
        print("Profiling for 15 seconds...")
        time.sleep(15)

        # Send SIGINT to the process to trigger a graceful shutdown
        print("Stopping profiler...")
        process.send_signal(signal.SIGINT)

        # Wait for the process to terminate
        stdout, stderr = process.communicate(timeout=15) # Add a timeout to communicate

        print("Profiler process terminated.")

        output_path = Path('first-year-aiml-playground') / output_file
        if output_path.exists():
            print(f"Profiling data saved to {output_path}")
        else:
            print("Error: Profiling output file not found!")

        if stdout:
            print("\n--- Profiler STDOUT ---")
            print(stdout)
        # Streamlit prints shutdown messages to stderr, so we'll only print if there's an error
        if stderr and "Traceback" in stderr:
             print("\n--- Profiler STDERR ---")
             print(stderr)

    except subprocess.TimeoutExpired:
        print("Timeout expired while waiting for profiler to shut down. Killing process.")
        process.kill()
        stdout, stderr = process.communicate()
        print("Profiler killed.")
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Make sure Streamlit is installed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    profile_app()
