#!/bin/bash

# Check for at least 2 arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <username> <host> [session_id]"
  exit 1
fi

USERNAME="$1"
HOST="$2"
SESSION_NAME="${3:-session_0}"     # Default to 'session_0' if not provided

# Start SSH reverse tunnel in background
ssh -R 5555:localhost:5559 "${USERNAME}@${HOST}" -N &
SSH_PID=$!

# Cleanup on exit or interrupt
trap "echo 'Terminating...'; kill $SSH_PID; exit" INT TERM EXIT

# Run local ROS launch command with session_id
roslaunch fep_rl_experiment bringup_real.launch 