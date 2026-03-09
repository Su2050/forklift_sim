#!/bin/bash
echo "Killing zombie train.py and kit processes..."
pgrep -af "train.py" | awk '{print $1}' | xargs -r kill -9
pgrep -af "kit" | awk '{print $1}' | xargs -r kill -9
echo "Done."
