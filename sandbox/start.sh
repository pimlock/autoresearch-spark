#!/bin/bash
# Entrypoint for autoresearch-spark sandbox.
#
# Drops into claude in the autoresearch directory.
# For persistent sessions with zellij, use start-with-zellij.sh instead.
#
# Usage:
#   openshell sandbox create --from Dockerfile --gpu -- sandbox/start.sh

# Initialize git repo if not already set up (program.md expects git for branching)
if [ ! -d /sandbox/autoresearch/.git ]; then
    git config --global user.email "sandbox@autoresearch"
    git config --global user.name "autoresearch"
    git -C /sandbox/autoresearch init -q
    git -C /sandbox/autoresearch add -A
    git -C /sandbox/autoresearch commit -q -m "initial commit"
fi

cd /sandbox/autoresearch
exec claude --dangerously-skip-permissions
